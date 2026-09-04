from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from time import monotonic
from typing import Any, ClassVar, Literal
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

from cayu._validation import (
    canonical_durable_json_bytes,
    json_utf8_size_within_limit,
    revalidate_model_input,
)
from cayu.evals.calibration import (
    EVAL_JUDGE_CALIBRATION_MAX_BYTES,
    EvalJudgeCalibrationReportV1,
    eval_judge_calibration_report_from_json,
    eval_judge_calibration_report_to_json,
)
from cayu.evals.capacity import EVAL_MAX_CONCURRENCY
from cayu.evals.corpus import (
    EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE,
    EVAL_CORPUS_MAX_BYTES,
    EVAL_CORPUS_MAX_CASES,
    EVAL_CORPUS_MAX_MESSAGES_PER_CASE,
    EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS,
    EVAL_CORPUS_MAX_SUITES,
    EVAL_CORPUS_MAX_TIMEOUT_SECONDS,
    EVAL_CORPUS_MAX_TRIALS,
    EvalCorpusDocument,
    EvalCorpusInspectionV1,
    EvalCorpusSuiteInspectionV1,
    EvalSuiteSpec,
    _bounded_durable_text,
    _model_python_input,
    _portable_id,
    _sha256_revision,
    eval_suite_trial_policy,
    inspect_eval_corpus,
)
from cayu.evals.execution import (
    CORPUS_EXECUTION_RESULT_MAX_BYTES,
    CorpusExecutionResult,
)
from cayu.evals.execution_profiles import (
    EvalExecutionProfileBindingV1,
    EvalExecutionProfileV1,
)
from cayu.evals.models import EvalTrialResult
from cayu.evals.published import _validate_published_eval_run_for_corpus
from cayu.evals.result_contract import _EvalTrialPublicData
from cayu.evals.results import (
    CAPTURED_EVALUATION_RESULT_MAX_BYTES,
    CapturedEvaluationResultV1,
    EvalResultOrigin,
    EvalResultTargetIdentityV1,
    captured_evaluation_result_from_json,
    eval_result_projection,
    validate_captured_result_for_corpus,
)
from cayu.evals.revisions import eval_trial_result_revision
from cayu.evals.scenario import (
    EVAL_SCENARIO_MAX_ARTIFACT_REQUIREMENTS,
    EVAL_SCENARIO_MAX_BYTES,
    EVAL_SCENARIO_MAX_EVENTS,
    EVAL_SCENARIO_MAX_SECRET_REQUIREMENTS,
    EvalScenarioDocumentV2,
    EvalScenarioInspectionV2,
    eval_scenario_from_json,
    inspect_eval_scenario,
)
from cayu.evals.suite_authoring import (
    EVAL_SUITE_AUTHORING_MAX_BYTES,
    EvalCaseDefinition,
    EvalCaseDefinitionV1,
    EvalCaseDefinitionV2,
    EvalScenarioStimulusV1,
    EvalSuiteDocument,
    eval_suite_document_from_json,
    validate_expected_eval_suite_revision,
)
from cayu.evals.trial_policy import EVAL_SUITE_MAX_CONCURRENCY, EvalSuiteRunExposureV1
from cayu.runtime.config import MAX_STEPS
from cayu.runtime.invocation import (
    InvocationOrigin,
    InvocationOriginTrust,
    SessionExecutionSource,
    copy_invocation_origin,
)
from cayu.runtime.stop_policy import RunLimits, copy_run_limits

EVAL_STORE_DEFAULT_PAGE_SIZE = 50
EVAL_STORE_MAX_PAGE_SIZE = 200
EVAL_STORE_DEFAULT_PAGE_BYTES = 1 << 20
EVAL_STORE_MAX_PAGE_BYTES = 8 << 20
EVAL_STORE_MAX_CURSOR_BYTES = 1_024
EVAL_STORE_MAX_IDENTIFIER_CHARS = 128
EVAL_STORE_MAX_LEASE_SECONDS = 3_600
EVAL_STORE_MAX_CLAIM_TARGETS = 128
EVAL_RUN_MIN_OBSERVATION_INTERVAL_SECONDS = 0.05
EVAL_RUN_MAX_OBSERVATION_INTERVAL_SECONDS = 1.0
EVAL_RUN_MAX_TERMINAL_WAIT_SECONDS = 300.0
_EVAL_STORE_MAX_BIGINT = 2**63 - 1
EVAL_RUN_INVOCATION_MAX_BYTES = 64 << 10
EVAL_SCENARIO_PROGRESS_MAX_BYTES = 256 << 10
EVAL_RUN_TRIAL_CHECKPOINTS_MAX_BYTES = CORPUS_EXECUTION_RESULT_MAX_BYTES
EVAL_RUN_TRIAL_CHECKPOINTS_MAX_ITEMS = EVAL_CORPUS_MAX_CASES * EVAL_CORPUS_MAX_TRIALS

_STORE_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z", re.ASCII)
_CURSOR_VERSION = 1

logger = logging.getLogger(__name__)


class _EvalStoreModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        protected_namespaces=(),
        revalidate_instances="always",
    )


def _store_identifier(value: str, field_name: str) -> str:
    value = _bounded_durable_text(
        value,
        field_name,
        max_chars=EVAL_STORE_MAX_IDENTIFIER_CHARS,
        nonblank=True,
        clean=True,
    )
    if _STORE_IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must start with an ASCII letter or digit and contain only "
            "ASCII letters, digits, '.', '_', ':', or '-'."
        )
    return value


def _idempotency_key(value: str, field_name: str) -> str:
    # Persist only the caller/server-computed digest, never a raw HTTP
    # Idempotency-Key that may itself contain credential material.
    return _sha256_revision(value, field_name)


def _aware_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
    return value.astimezone(UTC)


def _bounded_positive_seconds(value: float, field_name: str, *, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be a number.")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be finite and greater than zero.")
    if value > maximum:
        raise ValueError(f"{field_name} cannot exceed {maximum:g} seconds.")
    return value


def _exact_model(value: BaseModel, model_type: type[BaseModel], field_name: str):
    if type(value) is not model_type:
        raise TypeError(f"{field_name} must be an exact {model_type.__name__}.")
    return model_type.model_validate(_model_python_input(value))


def _wire_model_bytes(value: BaseModel) -> bytes:
    return json.dumps(
        value.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class EvalStoreResultTooLarge(ValueError):
    """A bounded eval-store read cannot fit the requested byte ceiling."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        super().__init__(f"Eval store result exceeds the {max_bytes}-byte safety limit.")


class EvalCorpusConflict(ValueError):
    """One immutable corpus revision resolves to contradictory stored content."""


class EvalScenarioConflict(ValueError):
    """One immutable scenario revision resolves to contradictory stored content."""


class EvalAuthoredSuiteConflict(ValueError):
    """One immutable authored-suite revision resolves to contradictory content."""


class EvalAuthoredSuiteReferenceError(ValueError):
    """An authored suite references an unavailable or incompatible scenario."""


class EvalJudgeCalibrationConflict(ValueError):
    """A calibration revision or run identity resolves to contradictory content."""


class EvalStorePublicationRejected(ValueError):
    """Public eval data could not cross the active credential-redaction boundary."""


class EvalRunAdmissionConflict(ValueError):
    """A run id or idempotency key is already bound to another request."""


class EvalRunStateConflict(ValueError):
    """A requested run lifecycle transition is not valid from current state."""


class EvalRunClaimLost(RuntimeError):
    """A worker no longer owns the live fenced claim required for a mutation."""


class EvalStoreTransientContention(RuntimeError):
    """A bounded durable eval-store write exhausted its transient contention budget."""


class EvalResultConflict(ValueError):
    """One immutable result revision resolves to contradictory stored content."""


class EvalBaselineConflict(ValueError):
    """A baseline mutation lost its compare-and-swap or idempotency contract."""


def _require_publication_safe(
    document: dict[str, Any],
    *,
    redact_json: Callable[[Any], Any],
    resource_name: str,
) -> None:
    if not callable(redact_json):
        raise TypeError("redact_json must be callable.")
    try:
        redacted = redact_json(document)
    except Exception:
        raise EvalStorePublicationRejected(
            f"{resource_name} could not cross the credential-redaction boundary."
        ) from None
    if type(redacted) is not dict or redacted != document:
        raise EvalStorePublicationRejected(
            f"{resource_name} contains a configured workload secret."
        )


def _prepare_corpus_for_store(
    corpus: EvalCorpusDocument,
    *,
    redact_json: Callable[[Any], Any],
) -> tuple[EvalCorpusDocument, bytes]:
    """Validate and serialize one corpus after a fail-closed credential scan."""

    validated = _exact_model(corpus, EvalCorpusDocument, "corpus")
    document = validated.model_dump(mode="json")
    _require_publication_safe(
        document,
        redact_json=redact_json,
        resource_name="Eval corpus",
    )
    wire_document = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(wire_document) > EVAL_CORPUS_MAX_BYTES:
        raise EvalStoreResultTooLarge(EVAL_CORPUS_MAX_BYTES)
    return validated, wire_document


def _prepare_scenario_for_store(
    scenario: EvalScenarioDocumentV2,
    *,
    redact_json: Callable[[Any], Any],
) -> tuple[EvalScenarioDocumentV2, bytes]:
    """Validate and serialize one scenario after a fail-closed credential scan."""

    validated = _exact_model(scenario, EvalScenarioDocumentV2, "scenario")
    document = validated.model_dump(mode="json")
    _require_publication_safe(
        document,
        redact_json=redact_json,
        resource_name="Eval scenario",
    )
    wire_document = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(wire_document) > EVAL_SCENARIO_MAX_BYTES:
        raise EvalStoreResultTooLarge(EVAL_SCENARIO_MAX_BYTES)
    return validated, wire_document


def _prepare_authored_suite_for_store(
    document: EvalSuiteDocument,
    *,
    redact_json: Callable[[Any], Any],
) -> tuple[EvalSuiteDocument, bytes]:
    """Validate and serialize one authored suite after credential scanning."""

    validated = validate_expected_eval_suite_revision(document, document.revision)
    payload = validated.model_dump(mode="json")
    _require_publication_safe(
        payload,
        redact_json=redact_json,
        resource_name="Authored eval suite",
    )
    wire_document = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(wire_document) > EVAL_SUITE_AUTHORING_MAX_BYTES:
        raise EvalStoreResultTooLarge(EVAL_SUITE_AUTHORING_MAX_BYTES)
    return validated, wire_document


def _prepare_judge_calibration_for_store(
    report: EvalJudgeCalibrationReportV1,
    *,
    redact_json: Callable[[Any], Any],
) -> tuple[EvalJudgeCalibrationReportV1, bytes]:
    """Validate and scan one complete fixed-evidence calibration report."""

    if type(report) is not EvalJudgeCalibrationReportV1:
        raise TypeError("report must be an exact EvalJudgeCalibrationReportV1.")
    validated = EvalJudgeCalibrationReportV1.model_validate(_model_python_input(report))
    payload = validated.model_dump(mode="json")
    _require_publication_safe(
        payload,
        redact_json=redact_json,
        resource_name="Judge calibration report",
    )
    wire = eval_judge_calibration_report_to_json(validated).encode("utf-8")
    if len(wire) > EVAL_JUDGE_CALIBRATION_MAX_BYTES:
        raise EvalStoreResultTooLarge(EVAL_JUDGE_CALIBRATION_MAX_BYTES)
    return validated, wire


def _prepare_run_request_for_store(
    request: EvalRunRequest,
    *,
    redact_json: Callable[[Any], Any],
) -> EvalRunRequest:
    validated = _exact_model(request, EvalRunRequest, "request")
    _require_publication_safe(
        validated.model_dump(mode="json"),
        redact_json=redact_json,
        resource_name="Eval run request",
    )
    return validated


def _prepare_result_for_store(
    result: CorpusExecutionResult,
    *,
    redact_json: Callable[[Any], Any],
) -> tuple[CorpusExecutionResult, bytes]:
    validated = _exact_model(result, CorpusExecutionResult, "result")
    document = validated.model_dump(mode="json")
    _require_publication_safe(
        document,
        redact_json=redact_json,
        resource_name="Eval result",
    )
    wire_document = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(wire_document) > CORPUS_EXECUTION_RESULT_MAX_BYTES:
        raise EvalStoreResultTooLarge(CORPUS_EXECUTION_RESULT_MAX_BYTES)
    return validated, wire_document


def _prepare_captured_result_for_store(
    result: CapturedEvaluationResultV1,
    corpus: EvalCorpusDocument,
    *,
    redact_json: Callable[[Any], Any],
) -> tuple[CapturedEvaluationResultV1, bytes]:
    validated = validate_captured_result_for_corpus(result, corpus)
    document = validated.model_dump(mode="json")
    _require_publication_safe(
        document,
        redact_json=redact_json,
        resource_name="Captured eval result",
    )
    wire_document = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(wire_document) > CAPTURED_EVALUATION_RESULT_MAX_BYTES:
        raise EvalStoreResultTooLarge(CAPTURED_EVALUATION_RESULT_MAX_BYTES)
    return validated, wire_document


class EvalRunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_EVAL_RUN_STATUSES = frozenset(
    {EvalRunStatus.COMPLETED, EvalRunStatus.FAILED, EvalRunStatus.CANCELLED}
)


class EvalRunFailureCode(StrEnum):
    """Safe terminal diagnostics; arbitrary exception text is never persisted."""

    TARGET_UNAVAILABLE = "target_unavailable"
    CORPUS_UNAVAILABLE = "corpus_unavailable"
    EXECUTION_FAILED = "execution_failed"
    WORKER_INTERRUPTED = "worker_interrupted"


class EvalCatalogQuery(_EvalStoreModel):
    target_key: StrictStr | None = None
    cursor: StrictStr | None = None
    limit: StrictInt = Field(
        default=EVAL_STORE_DEFAULT_PAGE_SIZE,
        ge=1,
        le=EVAL_STORE_MAX_PAGE_SIZE,
    )
    max_result_bytes: StrictInt = Field(
        default=EVAL_STORE_DEFAULT_PAGE_BYTES,
        ge=1_024,
        le=EVAL_STORE_MAX_PAGE_BYTES,
    )

    @field_validator("target_key")
    @classmethod
    def validate_target_key(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _portable_id(value, info.field_name)

    @field_validator("cursor")
    @classmethod
    def validate_cursor(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_cursor(value, info.field_name)


class EvalScenarioCatalogQuery(_EvalStoreModel):
    target_key: StrictStr | None = None
    scenario_id: StrictStr | None = None
    cursor: StrictStr | None = None
    limit: StrictInt = Field(
        default=EVAL_STORE_DEFAULT_PAGE_SIZE,
        ge=1,
        le=EVAL_STORE_MAX_PAGE_SIZE,
    )
    max_result_bytes: StrictInt = Field(
        default=EVAL_STORE_DEFAULT_PAGE_BYTES,
        ge=1_024,
        le=EVAL_STORE_MAX_PAGE_BYTES,
    )

    @field_validator("target_key", "scenario_id")
    @classmethod
    def validate_ids(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _portable_id(value, info.field_name)

    @field_validator("cursor")
    @classmethod
    def validate_cursor(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_cursor(value, info.field_name)


class EvalAuthoredSuiteCatalogQuery(_EvalStoreModel):
    target_key: StrictStr | None = None
    suite_id: StrictStr | None = None
    cursor: StrictStr | None = None
    limit: StrictInt = Field(
        default=EVAL_STORE_DEFAULT_PAGE_SIZE,
        ge=1,
        le=EVAL_STORE_MAX_PAGE_SIZE,
    )
    max_result_bytes: StrictInt = Field(
        default=EVAL_STORE_DEFAULT_PAGE_BYTES,
        ge=1_024,
        le=EVAL_STORE_MAX_PAGE_BYTES,
    )

    @field_validator("target_key", "suite_id")
    @classmethod
    def validate_ids(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _portable_id(value, info.field_name)

    @field_validator("cursor")
    @classmethod
    def validate_cursor(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_cursor(value, info.field_name)


class EvalCaseCatalogQuery(_EvalStoreModel):
    corpus_revision: StrictStr
    suite_id: StrictStr
    cursor: StrictStr | None = None
    limit: StrictInt = Field(
        default=EVAL_STORE_DEFAULT_PAGE_SIZE,
        ge=1,
        le=EVAL_STORE_MAX_PAGE_SIZE,
    )
    max_result_bytes: StrictInt = Field(
        default=EVAL_STORE_DEFAULT_PAGE_BYTES,
        ge=1_024,
        le=EVAL_STORE_MAX_PAGE_BYTES,
    )

    @field_validator("corpus_revision")
    @classmethod
    def validate_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("suite_id")
    @classmethod
    def validate_suite_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("cursor")
    @classmethod
    def validate_cursor(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_cursor(value, info.field_name)


class EvalSuiteCatalogQuery(_EvalStoreModel):
    corpus_revision: StrictStr
    cursor: StrictStr | None = None
    limit: StrictInt = Field(
        default=EVAL_STORE_DEFAULT_PAGE_SIZE,
        ge=1,
        le=EVAL_STORE_MAX_PAGE_SIZE,
    )
    max_result_bytes: StrictInt = Field(
        default=EVAL_STORE_DEFAULT_PAGE_BYTES,
        ge=1_024,
        le=EVAL_STORE_MAX_PAGE_BYTES,
    )

    @field_validator("corpus_revision")
    @classmethod
    def validate_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("cursor")
    @classmethod
    def validate_cursor(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_cursor(value, info.field_name)


class EvalRunQuery(_EvalStoreModel):
    target_key: StrictStr | None = None
    status: EvalRunStatus | None = None
    corpus_revision: StrictStr | None = None
    cursor: StrictStr | None = None
    limit: StrictInt = Field(
        default=EVAL_STORE_DEFAULT_PAGE_SIZE,
        ge=1,
        le=EVAL_STORE_MAX_PAGE_SIZE,
    )
    max_result_bytes: StrictInt = Field(
        default=EVAL_STORE_DEFAULT_PAGE_BYTES,
        ge=1_024,
        le=EVAL_STORE_MAX_PAGE_BYTES,
    )

    @field_validator("target_key")
    @classmethod
    def validate_target_key(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _portable_id(value, info.field_name)

    @field_validator("corpus_revision")
    @classmethod
    def validate_revision(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_revision(value, info.field_name)

    @field_validator("cursor")
    @classmethod
    def validate_cursor(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_cursor(value, info.field_name)


class EvalCorpusCatalogEntry(_EvalStoreModel):
    revision: StrictStr
    target_key: StrictStr
    evidence_policy_revision: StrictStr
    pricing_profile_fingerprint: StrictStr | None = None
    suite_count: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_SUITES)
    case_count: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_CASES)
    assertion_count: StrictInt = Field(
        ge=1,
        le=EVAL_CORPUS_MAX_CASES * EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE,
    )
    expanded_assertion_result_count: StrictInt = Field(
        ge=1,
        le=EVAL_CORPUS_MAX_SUITES * EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS,
    )
    document_bytes: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_BYTES)
    created_at: datetime

    @field_validator(
        "revision",
        "evidence_policy_revision",
        "pricing_profile_fingerprint",
    )
    @classmethod
    def validate_revision_fields(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_revision(value, info.field_name)

    @field_validator("target_key")
    @classmethod
    def validate_target_key(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime, info) -> datetime:
        return _aware_utc(value, info.field_name)

    @model_validator(mode="after")
    def validate_counts(self) -> EvalCorpusCatalogEntry:
        if not (
            self.case_count
            <= self.assertion_count
            <= self.case_count * EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE
        ):
            raise ValueError("Eval corpus catalog assertion count is impossible.")
        if self.expanded_assertion_result_count < self.assertion_count:
            raise ValueError("Eval corpus catalog expanded assertion count is impossible.")
        return self


class EvalScenarioCatalogEntry(_EvalStoreModel):
    revision: StrictStr
    id: StrictStr
    target_key: StrictStr
    name: StrictStr
    description: StrictStr | None = None
    event_count: StrictInt = Field(ge=1, le=EVAL_SCENARIO_MAX_EVENTS)
    input_event_count: StrictInt = Field(ge=1, le=EVAL_SCENARIO_MAX_EVENTS)
    approval_checkpoint_count: StrictInt = Field(ge=0, le=EVAL_SCENARIO_MAX_EVENTS)
    message_count: StrictInt = Field(
        ge=1,
        le=EVAL_SCENARIO_MAX_EVENTS * 32,
    )
    part_count: StrictInt = Field(
        ge=1,
        le=EVAL_SCENARIO_MAX_EVENTS * 32 * 32,
    )
    artifact_requirement_count: StrictInt = Field(
        ge=0,
        le=EVAL_SCENARIO_MAX_ARTIFACT_REQUIREMENTS,
    )
    secret_requirement_count: StrictInt = Field(
        ge=0,
        le=EVAL_SCENARIO_MAX_SECRET_REQUIREMENTS,
    )
    document_bytes: StrictInt = Field(ge=1, le=EVAL_SCENARIO_MAX_BYTES)
    created_at: datetime

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("id", "target_key")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=2_048,
            nonblank=True,
            clean=True,
        )

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime, info) -> datetime:
        return _aware_utc(value, info.field_name)

    @model_validator(mode="after")
    def validate_counts(self) -> EvalScenarioCatalogEntry:
        if self.input_event_count + self.approval_checkpoint_count != self.event_count:
            raise ValueError("Eval scenario catalog event counts are inconsistent.")
        if self.message_count < self.input_event_count:
            raise ValueError("Eval scenario catalog message count is impossible.")
        if self.part_count < self.message_count:
            raise ValueError("Eval scenario catalog part count is impossible.")
        return self


class EvalAuthoredSuiteCatalogEntry(_EvalStoreModel):
    revision: StrictStr
    id: StrictStr
    suite_revision: StrictStr
    target_key: StrictStr
    name: StrictStr
    description: StrictStr | None = None
    case_count: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_CASES)
    assertion_count: StrictInt = Field(
        ge=1,
        le=EVAL_CORPUS_MAX_CASES * EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE,
    )
    simple_input_count: StrictInt = Field(ge=0, le=EVAL_CORPUS_MAX_CASES)
    scenario_count: StrictInt = Field(ge=0, le=EVAL_CORPUS_MAX_CASES)
    trials: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_TRIALS)
    timeout_seconds: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_TIMEOUT_SECONDS)
    document_bytes: StrictInt = Field(ge=1, le=EVAL_SUITE_AUTHORING_MAX_BYTES)
    created_at: datetime

    @field_validator("revision", "suite_revision")
    @classmethod
    def validate_revisions(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("id", "target_key")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=2_048,
            nonblank=True,
            clean=True,
        )

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime, info) -> datetime:
        return _aware_utc(value, info.field_name)

    @model_validator(mode="after")
    def validate_counts(self) -> EvalAuthoredSuiteCatalogEntry:
        if self.simple_input_count + self.scenario_count != self.case_count:
            raise ValueError("Authored eval suite stimulus counts are inconsistent.")
        if not (
            self.case_count
            <= self.assertion_count
            <= self.case_count * EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE
        ):
            raise ValueError("Authored eval suite assertion count is impossible.")
        if self.assertion_count * self.trials > EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS:
            raise ValueError("Authored eval suite expanded assertion count is impossible.")
        return self


class EvalSuiteCatalogEntry(_EvalStoreModel):
    corpus_revision: StrictStr
    id: StrictStr
    revision: StrictStr
    name: StrictStr
    description: StrictStr | None = None
    case_count: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_CASES)
    assertion_count: StrictInt = Field(
        ge=1,
        le=EVAL_CORPUS_MAX_CASES * EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE,
    )
    trials: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_TRIALS)
    timeout_seconds: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_TIMEOUT_SECONDS)

    @field_validator("corpus_revision", "revision")
    @classmethod
    def validate_revision_fields(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=2_048,
            nonblank=True,
            clean=True,
        )

    @model_validator(mode="after")
    def validate_counts(self) -> EvalSuiteCatalogEntry:
        if not (
            self.case_count
            <= self.assertion_count
            <= self.case_count * EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE
        ):
            raise ValueError("Eval suite catalog assertion count is impossible.")
        if self.assertion_count * self.trials > EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS:
            raise ValueError("Eval suite catalog expanded assertion count is impossible.")
        return self


class EvalCaseCatalogEntry(_EvalStoreModel):
    corpus_revision: StrictStr
    id: StrictStr
    revision: StrictStr
    suite_id: StrictStr
    name: StrictStr
    description: StrictStr | None = None
    message_count: StrictInt = Field(ge=0, le=EVAL_CORPUS_MAX_MESSAGES_PER_CASE)
    assertion_count: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE)

    @field_validator("corpus_revision", "revision")
    @classmethod
    def validate_revision_fields(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("id", "suite_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=2_048,
            nonblank=True,
            clean=True,
        )


class EvalCorpusCatalogPage(_EvalStoreModel):
    items: tuple[EvalCorpusCatalogEntry, ...] = Field(max_length=EVAL_STORE_MAX_PAGE_SIZE)
    next_cursor: StrictStr | None = None
    has_more: StrictBool = False

    @model_validator(mode="after")
    def validate_page(self) -> EvalCorpusCatalogPage:
        _validate_page_boundary(self.items, self.next_cursor, self.has_more)
        if len({item.revision for item in self.items}) != len(self.items):
            raise ValueError("Eval corpus catalog page contains duplicate revisions.")
        expected = list(self.items)
        expected.sort(key=lambda item: item.revision)
        expected.sort(key=lambda item: item.created_at, reverse=True)
        if list(self.items) != expected:
            raise ValueError("Eval corpus catalog page is not in keyset order.")
        if self.has_more:
            assert self.next_cursor is not None
            timestamp, revision, target_key = _decode_cursor(
                self.next_cursor,
                "corpora",
                ("created_at", "revision", "target_key"),
            )
            if (timestamp, revision) != (
                self.items[-1].created_at.isoformat(),
                self.items[-1].revision,
            ):
                raise ValueError("Eval corpus catalog cursor does not follow its last item.")
            if target_key and any(item.target_key != target_key for item in self.items):
                raise ValueError("Eval corpus catalog cursor filter does not match its items.")
        return self


class EvalScenarioCatalogPage(_EvalStoreModel):
    items: tuple[EvalScenarioCatalogEntry, ...] = Field(max_length=EVAL_STORE_MAX_PAGE_SIZE)
    next_cursor: StrictStr | None = None
    has_more: StrictBool = False

    @model_validator(mode="after")
    def validate_page(self) -> EvalScenarioCatalogPage:
        _validate_page_boundary(self.items, self.next_cursor, self.has_more)
        if len({item.revision for item in self.items}) != len(self.items):
            raise ValueError("Eval scenario catalog page contains duplicate revisions.")
        expected = list(self.items)
        expected.sort(key=lambda item: item.revision)
        expected.sort(key=lambda item: item.created_at, reverse=True)
        if list(self.items) != expected:
            raise ValueError("Eval scenario catalog page is not in keyset order.")
        if self.has_more:
            assert self.next_cursor is not None
            timestamp, revision, target_key, scenario_id = _decode_cursor(
                self.next_cursor,
                "scenarios",
                ("created_at", "revision", "target_key", "scenario_id"),
            )
            if (timestamp, revision) != (
                self.items[-1].created_at.isoformat(),
                self.items[-1].revision,
            ):
                raise ValueError("Eval scenario catalog cursor does not follow its last item.")
            if target_key and any(item.target_key != target_key for item in self.items):
                raise ValueError("Eval scenario catalog cursor filter does not match its items.")
            if scenario_id and any(item.id != scenario_id for item in self.items):
                raise ValueError("Eval scenario catalog cursor filter does not match its items.")
        return self


class EvalAuthoredSuiteCatalogPage(_EvalStoreModel):
    items: tuple[EvalAuthoredSuiteCatalogEntry, ...] = Field(max_length=EVAL_STORE_MAX_PAGE_SIZE)
    next_cursor: StrictStr | None = None
    has_more: StrictBool = False

    @model_validator(mode="after")
    def validate_page(self) -> EvalAuthoredSuiteCatalogPage:
        _validate_page_boundary(self.items, self.next_cursor, self.has_more)
        if len({item.revision for item in self.items}) != len(self.items):
            raise ValueError("Authored eval suite catalog page contains duplicate revisions.")
        expected = list(self.items)
        expected.sort(key=lambda item: item.revision)
        expected.sort(key=lambda item: item.created_at, reverse=True)
        if list(self.items) != expected:
            raise ValueError("Authored eval suite catalog page is not in keyset order.")
        if self.has_more:
            assert self.next_cursor is not None
            timestamp, revision, target_key, suite_id = _decode_cursor(
                self.next_cursor,
                "authored_suites",
                ("created_at", "revision", "target_key", "suite_id"),
            )
            if (timestamp, revision) != (
                self.items[-1].created_at.isoformat(),
                self.items[-1].revision,
            ):
                raise ValueError(
                    "Authored eval suite catalog cursor does not follow its last item."
                )
            if target_key and any(item.target_key != target_key for item in self.items):
                raise ValueError(
                    "Authored eval suite catalog cursor filter does not match its items."
                )
            if suite_id and any(item.id != suite_id for item in self.items):
                raise ValueError(
                    "Authored eval suite catalog cursor filter does not match its items."
                )
        return self


class EvalSuiteCatalogPage(_EvalStoreModel):
    items: tuple[EvalSuiteCatalogEntry, ...] = Field(max_length=EVAL_STORE_MAX_PAGE_SIZE)
    next_cursor: StrictStr | None = None
    has_more: StrictBool = False

    @model_validator(mode="after")
    def validate_page(self) -> EvalSuiteCatalogPage:
        _validate_page_boundary(self.items, self.next_cursor, self.has_more)
        if self.items:
            revision = self.items[0].corpus_revision
            if any(item.corpus_revision != revision for item in self.items):
                raise ValueError("Eval suite catalog page mixes corpus revisions.")
            if tuple(item.id for item in self.items) != tuple(
                sorted({item.id for item in self.items})
            ):
                raise ValueError("Eval suite catalog page is not in unique keyset order.")
        if self.has_more and self.next_cursor != _suite_cursor(self.items[-1]):
            raise ValueError("Eval suite catalog cursor does not follow its last item.")
        return self


class EvalCaseCatalogPage(_EvalStoreModel):
    items: tuple[EvalCaseCatalogEntry, ...] = Field(max_length=EVAL_STORE_MAX_PAGE_SIZE)
    next_cursor: StrictStr | None = None
    has_more: StrictBool = False

    @model_validator(mode="after")
    def validate_page(self) -> EvalCaseCatalogPage:
        _validate_page_boundary(self.items, self.next_cursor, self.has_more)
        if self.items:
            boundary = (self.items[0].corpus_revision, self.items[0].suite_id)
            if any((item.corpus_revision, item.suite_id) != boundary for item in self.items):
                raise ValueError("Eval case catalog page mixes corpus suites.")
            if tuple(item.id for item in self.items) != tuple(
                sorted({item.id for item in self.items})
            ):
                raise ValueError("Eval case catalog page is not in unique keyset order.")
        if self.has_more and self.next_cursor != _case_cursor(self.items[-1]):
            raise ValueError("Eval case catalog cursor does not follow its last item.")
        return self


class EvalRunCostBudget(_EvalStoreModel):
    """One server-priced cost ceiling applied independently to each eval-trial session."""

    max_estimated_cost: Decimal = Field(gt=0)
    currency: StrictStr = Field(default="USD", min_length=1, max_length=16)

    @field_validator("max_estimated_cost")
    @classmethod
    def validate_cost(cls, value: Decimal, info) -> Decimal:
        if not value.is_finite():
            raise ValueError(f"{info.field_name} must be finite.")
        return value

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value.upper(),
            info.field_name,
            max_chars=16,
            nonblank=True,
            clean=True,
        )


class EvalScenarioArtifactReference(_EvalStoreModel):
    """One immutable scenario requirement-to-artifact launch selection."""

    requirement_id: StrictStr
    artifact_id: StrictStr

    @field_validator("requirement_id")
    @classmethod
    def validate_requirement_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_id(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=512,
            nonblank=True,
            clean=True,
        )


class EvalScenarioRunInvocation(_EvalStoreModel):
    """Authority-free scenario launch facts frozen into one durable run."""

    schema_version: Literal[1] = 1
    scenario_revision: StrictStr
    binding_revision: StrictStr
    authored_suite_revision: StrictStr | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    authored_case_revision: StrictStr | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    environment_name: StrictStr | None = None
    trials: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_TRIALS)
    timeout_seconds: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_TIMEOUT_SECONDS)
    artifact_references: tuple[EvalScenarioArtifactReference, ...] = Field(
        default_factory=tuple,
        max_length=EVAL_SCENARIO_MAX_ARTIFACT_REQUIREMENTS,
    )

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator(
        "scenario_revision",
        "binding_revision",
        "authored_suite_revision",
        "authored_case_revision",
    )
    @classmethod
    def validate_revisions(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_revision(value, info.field_name)

    @field_validator("environment_name")
    @classmethod
    def validate_environment_name(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )

    @model_validator(mode="after")
    def validate_artifact_references(self) -> EvalScenarioRunInvocation:
        requirement_ids = tuple(item.requirement_id for item in self.artifact_references)
        if requirement_ids != tuple(sorted(set(requirement_ids))):
            raise ValueError("Scenario artifact references must be unique and sorted.")
        if (self.authored_suite_revision is None) != (self.authored_case_revision is None):
            raise ValueError(
                "Authored scenario runs require both suite and case revision identities."
            )
        return self


class EvalScenarioTrialPhase(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    AWAITING_RESUME = "awaiting_resume"
    COMPLETED = "completed"
    ERROR = "error"


class EvalScenarioTrialFailureCode(StrEnum):
    EXECUTION_FAILED = "execution_failed"
    EXECUTION_PROFILE_CHANGED = "execution_profile_changed"
    UNEXPECTED_SESSION_STATE = "unexpected_session_state"
    EXPECTED_APPROVAL_UNAVAILABLE = "expected_approval_unavailable"
    EXPECTED_USER_INPUT_UNAVAILABLE = "expected_user_input_unavailable"


class EvalRunTrialCheckpoint(_EvalStoreModel):
    """One bounded terminal trial retained privately until atomic publication."""

    case_id: StrictStr
    result: EvalTrialResult
    public_data: _EvalTrialPublicData

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("result", mode="before")
    @classmethod
    def copy_result(cls, value: object) -> object:
        if type(value) is EvalTrialResult:
            return value.model_dump(mode="python", warnings=False)
        if isinstance(value, BaseModel):
            raise TypeError("result must be an exact EvalTrialResult or JSON object.")
        return value

    @field_validator("public_data", mode="before")
    @classmethod
    def copy_public_data(cls, value: object) -> object:
        if type(value) is _EvalTrialPublicData:
            return value.model_dump(mode="python")
        if isinstance(value, BaseModel):
            raise TypeError("public_data must be exact trial public data or a JSON object.")
        return value

    @model_validator(mode="after")
    def validate_private_checkpoint(self) -> EvalRunTrialCheckpoint:
        if (
            self.result.trajectory is not None
            or self.result.final_output
            or self.result.structured_output is not None
        ):
            raise ValueError("Durable eval trial checkpoints cannot retain raw candidate output.")
        return self

    @property
    def trial_number(self) -> int:
        return self.result.trial_number


def _validated_trial_checkpoints(
    values: tuple[EvalRunTrialCheckpoint, ...],
    *,
    expected_document_bytes: int | None = None,
) -> tuple[EvalRunTrialCheckpoint, ...]:
    if type(values) is not tuple:
        raise TypeError("trial checkpoints must be a tuple.")
    if len(values) > EVAL_RUN_TRIAL_CHECKPOINTS_MAX_ITEMS:
        raise ValueError("Eval run trial checkpoints exceed their item limit.")
    copied = tuple(
        _exact_model(value, EvalRunTrialCheckpoint, "trial checkpoint") for value in values
    )
    keys = tuple((item.case_id, item.trial_number) for item in copied)
    if keys != tuple(sorted(set(keys))):
        raise ValueError("Eval run trial checkpoints must be unique and sorted by slot.")
    document_bytes = sum(
        len(eval_run_trial_checkpoint_to_json(item).encode("utf-8")) for item in copied
    )
    if document_bytes > EVAL_RUN_TRIAL_CHECKPOINTS_MAX_BYTES:
        raise ValueError("Eval run trial checkpoints exceed their byte limit.")
    if expected_document_bytes is not None and document_bytes != expected_document_bytes:
        raise ValueError("Eval run trial checkpoint byte accounting is inconsistent.")
    return copied


def eval_run_trial_checkpoint_to_json(value: EvalRunTrialCheckpoint) -> str:
    validated = _exact_model(value, EvalRunTrialCheckpoint, "trial checkpoint")
    return json.dumps(
        validated.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def eval_run_trial_checkpoint_from_json(source: str) -> EvalRunTrialCheckpoint:
    if type(source) is not str:
        raise TypeError("trial checkpoint JSON must be a string.")
    raw = source.encode("utf-8")
    if not 1 <= len(raw) <= EVAL_RUN_TRIAL_CHECKPOINTS_MAX_BYTES:
        raise ValueError("Eval run trial checkpoint JSON exceeds its byte limit.")
    document = json.loads(source)
    if type(document) is not dict:
        raise ValueError("Eval run trial checkpoint JSON must contain an object.")
    return EvalRunTrialCheckpoint.model_validate(document)


@dataclass(frozen=True, slots=True)
class _PreparedEvalRunTrialCheckpoint:
    checkpoint: EvalRunTrialCheckpoint
    document: str
    document_bytes: int


def _prepare_trial_checkpoint_for_store(
    checkpoint: EvalRunTrialCheckpoint,
    *,
    redact_json: Callable[[Any], Any],
) -> _PreparedEvalRunTrialCheckpoint:
    checkpoint = _exact_model(checkpoint, EvalRunTrialCheckpoint, "checkpoint")
    _require_publication_safe(
        checkpoint.model_dump(mode="json"),
        redact_json=redact_json,
        resource_name="Eval trial checkpoint",
    )
    document = eval_run_trial_checkpoint_to_json(checkpoint)
    document_bytes = len(document.encode("utf-8"))
    if document_bytes > EVAL_RUN_TRIAL_CHECKPOINTS_MAX_BYTES:
        raise ValueError("Eval run trial checkpoint exceeds its byte limit.")
    return _PreparedEvalRunTrialCheckpoint(
        checkpoint=checkpoint,
        document=document,
        document_bytes=document_bytes,
    )


def _validate_trial_checkpoint_for_run(
    checkpoint: EvalRunTrialCheckpoint,
    request: EvalRunRequest,
    corpus: EvalCorpusDocument,
) -> EvalRunTrialCheckpoint:
    checkpoint = _exact_model(checkpoint, EvalRunTrialCheckpoint, "checkpoint")
    suite = next((item for item in corpus.suites if item.id == request.suite_id), None)
    if suite is None:
        raise EvalRunStateConflict("Eval run suite is unavailable for trial checkpointing.")
    case_ids = {case.id for case in corpus.cases if case.suite_id == suite.id}
    if checkpoint.case_id not in case_ids:
        raise EvalRunStateConflict("Eval trial checkpoint case does not belong to its run.")
    trial_count = eval_suite_trial_policy(suite).trial_count
    if checkpoint.trial_number > trial_count:
        raise EvalRunStateConflict("Eval trial checkpoint number exceeds its immutable policy.")
    return checkpoint


def _validate_trial_checkpoints_for_result(
    checkpoints: tuple[EvalRunTrialCheckpoint, ...],
    result: CorpusExecutionResult,
) -> None:
    """Bind retained private terminals to the exact public trials being published."""

    checkpoints = _validated_trial_checkpoints(checkpoints)
    if not checkpoints:
        return
    published_by_slot = {
        (case.case_id, trial.trial_number): trial
        for case in result.run.cases
        for trial in case.trials
    }
    for checkpoint in checkpoints:
        published = published_by_slot.get((checkpoint.case_id, checkpoint.trial_number))
        if (
            published is None
            or published.source_trial_revision != eval_trial_result_revision(checkpoint.result)
            or published.code is not checkpoint.public_data.diagnostic_code
            or published.output != checkpoint.public_data.output
        ):
            raise EvalRunStateConflict(
                "Published eval result contradicts a durable terminal trial checkpoint."
            )


class EvalScenarioApprovalDecisionRecord(_EvalStoreModel):
    decision: Literal["approve", "deny"]
    reason: StrictStr | None = Field(default=None, max_length=2_048)
    actor_id: StrictStr = Field(max_length=512)
    submitted_at: datetime

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=2_048,
            nonblank=True,
            clean=False,
        )

    @field_validator("actor_id")
    @classmethod
    def validate_actor_id(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=512,
            nonblank=True,
            clean=True,
        )

    @field_validator("submitted_at")
    @classmethod
    def validate_submitted_at(cls, value: datetime, info) -> datetime:
        return _aware_utc(value, info.field_name)


class EvalScenarioTrialProgress(_EvalStoreModel):
    trial_number: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_TRIALS)
    phase: EvalScenarioTrialPhase
    session_id: StrictStr | None = None
    next_event_sequence: StrictInt = Field(ge=0, le=EVAL_SCENARIO_MAX_EVENTS)
    pending_event_id: StrictStr | None = None
    pending_tool_name: StrictStr | None = None
    pending_input_id: StrictStr | None = None
    pending_resume_kind: Literal["user_input", "manual_recovery"] | None = None
    approval: EvalScenarioApprovalDecisionRecord | None = None
    failure_code: EvalScenarioTrialFailureCode | None = None

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _store_identifier(value, info.field_name)

    @field_validator("pending_event_id")
    @classmethod
    def validate_pending_event_id(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _portable_id(value, info.field_name)

    @field_validator("pending_tool_name")
    @classmethod
    def validate_pending_tool_name(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )

    @field_validator("pending_input_id")
    @classmethod
    def validate_pending_input_id(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=512,
            nonblank=True,
            clean=True,
        )

    @model_validator(mode="after")
    def validate_phase(self) -> EvalScenarioTrialProgress:
        awaiting_approval = self.phase is EvalScenarioTrialPhase.AWAITING_APPROVAL
        awaiting_resume = self.phase is EvalScenarioTrialPhase.AWAITING_RESUME
        if awaiting_approval != (
            self.pending_event_id is not None
            and self.pending_tool_name is not None
            and self.pending_input_id is None
            and self.pending_resume_kind is None
        ):
            raise ValueError("Approval-waiting scenario progress is inconsistent.")
        if awaiting_resume != (
            self.pending_event_id is not None
            and self.pending_tool_name is None
            and self.pending_resume_kind is not None
            and ((self.pending_resume_kind == "user_input") == (self.pending_input_id is not None))
        ):
            raise ValueError("Resume-waiting scenario progress is inconsistent.")
        if self.approval is not None and not awaiting_approval:
            raise ValueError("Only an approval-waiting trial may retain a decision.")
        if (self.phase is EvalScenarioTrialPhase.ERROR) != (self.failure_code is not None):
            raise ValueError("Scenario trial failure state is inconsistent.")
        if self.phase is not EvalScenarioTrialPhase.PENDING and self.session_id is None:
            raise ValueError("Started scenario trials require a concrete session id.")
        return self


def _scenario_progress_revision(document: Mapping[str, Any]) -> str:
    material = dict(document)
    material.pop("revision", None)
    return (
        "sha256:"
        + hashlib.sha256(
            canonical_durable_json_bytes(material, "eval scenario progress")
        ).hexdigest()
    )


class EvalScenarioRunProgress(_EvalStoreModel):
    schema_version: Literal[1] = 1
    revision: StrictStr
    scenario_revision: StrictStr
    binding_revision: StrictStr
    attempt: StrictInt = Field(ge=1, le=_EVAL_STORE_MAX_BIGINT)
    trials: tuple[EvalScenarioTrialProgress, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_TRIALS,
    )

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator("revision", "scenario_revision", "binding_revision")
    @classmethod
    def validate_revisions(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @model_validator(mode="after")
    def validate_progress(self) -> EvalScenarioRunProgress:
        expected_numbers = tuple(range(1, len(self.trials) + 1))
        if tuple(item.trial_number for item in self.trials) != expected_numbers:
            raise ValueError("Scenario trial progress must be contiguous and ordered.")
        if self.revision != _scenario_progress_revision(self.model_dump(mode="json")):
            raise ValueError("Scenario progress revision does not match its content.")
        if not json_utf8_size_within_limit(self, EVAL_SCENARIO_PROGRESS_MAX_BYTES):
            raise ValueError(
                f"Scenario progress exceeds {EVAL_SCENARIO_PROGRESS_MAX_BYTES} JSON bytes."
            )
        return self

    @classmethod
    def create(
        cls,
        *,
        scenario_revision: str,
        binding_revision: str,
        attempt: int,
        trials: tuple[EvalScenarioTrialProgress, ...],
    ) -> EvalScenarioRunProgress:
        document = {
            "schema_version": 1,
            "scenario_revision": scenario_revision,
            "binding_revision": binding_revision,
            "attempt": attempt,
            "trials": [item.model_dump(mode="json") for item in trials],
        }
        return cls(
            revision=_scenario_progress_revision(document),
            schema_version=1,
            scenario_revision=scenario_revision,
            binding_revision=binding_revision,
            attempt=attempt,
            trials=trials,
        )

    def replace_trial(self, trial: EvalScenarioTrialProgress) -> EvalScenarioRunProgress:
        if type(trial) is not EvalScenarioTrialProgress:
            raise TypeError("trial must be an exact EvalScenarioTrialProgress.")
        if not 1 <= trial.trial_number <= len(self.trials):
            raise ValueError("Scenario trial number is outside this run.")
        values = list(self.trials)
        values[trial.trial_number - 1] = EvalScenarioTrialProgress.model_validate(
            trial.model_dump(mode="python")
        )
        return EvalScenarioRunProgress.create(
            scenario_revision=self.scenario_revision,
            binding_revision=self.binding_revision,
            attempt=self.attempt,
            trials=tuple(values),
        )


def _scenario_progress_for_claim(
    progress: EvalScenarioRunProgress | None,
    *,
    scenario: EvalScenarioRunInvocation | None,
    attempt: int,
    terminal_trial_numbers: frozenset[int] = frozenset(),
) -> EvalScenarioRunProgress | None:
    """Fence resumable checkpoints into a new claim and reset unsafe work.

    Approval, user-input, and explicit session-resume pauses retain enough durable
    identity to continue after claim loss. Other stages may have lost their worker
    between any two provider/tool effects, so a new claim restarts those trials
    under a new session id and publication fence.
    """

    if progress is None or scenario is None:
        return None
    if (
        progress.scenario_revision != scenario.scenario_revision
        or progress.binding_revision != scenario.binding_revision
        or len(progress.trials) != scenario.trials
    ):
        raise EvalRunStateConflict("Scenario progress does not match its durable invocation.")
    trials = tuple(
        trial.model_copy(deep=True)
        if trial.phase
        in {
            EvalScenarioTrialPhase.AWAITING_APPROVAL,
            EvalScenarioTrialPhase.AWAITING_RESUME,
        }
        or (
            trial.phase in {EvalScenarioTrialPhase.COMPLETED, EvalScenarioTrialPhase.ERROR}
            and trial.trial_number in terminal_trial_numbers
        )
        else EvalScenarioTrialProgress(
            trial_number=trial.trial_number,
            phase=EvalScenarioTrialPhase.PENDING,
            next_event_sequence=0,
        )
        for trial in progress.trials
    )
    return EvalScenarioRunProgress.create(
        scenario_revision=scenario.scenario_revision,
        binding_revision=scenario.binding_revision,
        attempt=attempt,
        trials=trials,
    )


class EvalScenarioApprovalSubmission(_EvalStoreModel):
    expected_progress_revision: StrictStr
    trial_number: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_TRIALS)
    event_id: StrictStr
    decision: Literal["approve", "deny"]
    reason: StrictStr | None = Field(default=None, max_length=2_048)
    actor_id: StrictStr = Field(max_length=512)

    @field_validator("expected_progress_revision")
    @classmethod
    def validate_expected_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("event_id")
    @classmethod
    def validate_event_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return EvalScenarioApprovalDecisionRecord.validate_reason(value, info)

    @field_validator("actor_id")
    @classmethod
    def validate_actor_id(cls, value: str, info) -> str:
        return EvalScenarioApprovalDecisionRecord.validate_actor_id(value, info)


class EvalRunInvocation(_EvalStoreModel):
    """Durable, authority-free execution contractions and trusted caller provenance.

    The HTTP API constructs this value after authentication. It retains only the
    bounded subject/tenant projection needed to mint ordinary runtime invocation
    provenance after a worker restart; authentication claims never enter EvalStore.
    ``None`` bounds inherit the server-owned target request base.
    """

    schema_version: Literal[1] = 1
    source: SessionExecutionSource = SessionExecutionSource.SDK_RUN
    origin: InvocationOrigin | None = None
    max_steps: StrictInt | None = Field(default=None, ge=1, le=MAX_STEPS)
    limits: RunLimits | None = None
    cost_budget: EvalRunCostBudget | None = None
    execution_profile: EvalExecutionProfileBindingV1 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    execution_profile_snapshot: EvalExecutionProfileV1 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    admission_request_revision: StrictStr | None = Field(
        default=None,
        description=(
            "Secret-safe identity of the exact authenticated request that admitted this run."
        ),
        exclude_if=lambda value: value is None,
    )
    authored_suite_revision: StrictStr | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    authored_suite_selection_revision: StrictStr | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    authored_suite_launch_revision: StrictStr | None = Field(
        default=None,
        description=(
            "Secret-safe identity shared by every durable run admitted by one "
            "authored-suite launch."
        ),
        exclude_if=lambda value: value is None,
    )
    authored_suite_launch_lane: StrictInt | None = Field(
        default=None,
        ge=0,
        le=EVAL_SUITE_MAX_CONCURRENCY - 1,
        description="Concurrency lane shared by serialized runs from one suite launch.",
        exclude_if=lambda value: value is None,
    )
    authored_suite_exposure: EvalSuiteRunExposureV1 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    scenario: EvalScenarioRunInvocation | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator("origin", mode="before")
    @classmethod
    def copy_origin(cls, value: object) -> object:
        if value is None:
            return None
        if type(value) is InvocationOrigin:
            return copy_invocation_origin(value)
        return value

    @field_validator("limits", mode="before")
    @classmethod
    def copy_limits(cls, value: object) -> object:
        if value is None:
            return None
        if type(value) is RunLimits:
            return copy_run_limits(value)
        return value

    @field_validator("cost_budget", mode="before")
    @classmethod
    def copy_cost_budget(cls, value: object) -> object:
        if value is None:
            return None
        if type(value) is EvalRunCostBudget:
            return EvalRunCostBudget.model_validate(value.model_dump(mode="python"))
        return value

    @field_validator("execution_profile", mode="before")
    @classmethod
    def copy_execution_profile(cls, value: object) -> object:
        return revalidate_model_input(value, EvalExecutionProfileBindingV1)

    @field_validator("execution_profile_snapshot", mode="before")
    @classmethod
    def copy_execution_profile_snapshot(cls, value: object) -> object:
        return revalidate_model_input(value, EvalExecutionProfileV1)

    @field_validator("scenario", mode="before")
    @classmethod
    def copy_scenario(cls, value: object) -> object:
        if value is None:
            return None
        if type(value) is EvalScenarioRunInvocation:
            return EvalScenarioRunInvocation.model_validate(value.model_dump(mode="python"))
        return value

    @field_validator("authored_suite_exposure", mode="before")
    @classmethod
    def copy_authored_suite_exposure(cls, value: object) -> object:
        return revalidate_model_input(value, EvalSuiteRunExposureV1)

    @field_validator(
        "admission_request_revision",
        "authored_suite_revision",
        "authored_suite_selection_revision",
        "authored_suite_launch_revision",
    )
    @classmethod
    def validate_content_revisions(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_revision(value, info.field_name)

    @model_validator(mode="after")
    def validate_provenance(self) -> EvalRunInvocation:
        if self.source not in {
            SessionExecutionSource.SDK_RUN,
            SessionExecutionSource.HTTP_RUN,
        }:
            raise ValueError("Eval runs support only SDK or HTTP root invocation sources.")
        if self.origin is not None and (
            self.source is not SessionExecutionSource.HTTP_RUN
            or self.origin.trust is not InvocationOriginTrust.SERVER_VERIFIED
        ):
            raise ValueError("Eval run origins require server-verified HTTP provenance.")
        if self.limits is not None and self.limits.scope != "run":
            raise ValueError("Eval run invocation limits must use run scope.")
        if self.execution_profile_snapshot is not None and self.execution_profile is None:
            raise ValueError("An eval execution-profile snapshot requires its durable binding.")
        if self.execution_profile_snapshot is not None:
            assert self.execution_profile is not None
            runtime_profile = self.execution_profile.runtime_execution_profile
            candidate = self.execution_profile_snapshot.candidate
            if (
                self.execution_profile_snapshot.revision != self.execution_profile.profile_revision
                or candidate.runtime_execution_profile_schema_version
                != runtime_profile.schema_version
                or candidate.runtime_execution_profile_fingerprint != runtime_profile.fingerprint
            ):
                raise ValueError(
                    "Eval execution-profile snapshot conflicts with its durable binding."
                )
        if (self.authored_suite_revision is None) != (
            self.authored_suite_selection_revision is None
        ):
            raise ValueError(
                "Authored suite runs require both suite and selection revision identities."
            )
        if self.authored_suite_exposure is not None and self.authored_suite_revision is None:
            raise ValueError("Only authored suite runs may carry accepted work exposure.")
        if self.authored_suite_launch_revision is not None and self.authored_suite_revision is None:
            raise ValueError("Only authored suite runs may carry a launch revision.")
        if (self.authored_suite_launch_revision is None) != (
            self.authored_suite_launch_lane is None
        ):
            raise ValueError("Authored suite launch revision and lane must be paired.")
        if self.authored_suite_exposure is not None and self.authored_suite_launch_revision is None:
            raise ValueError("Authored suite exposure requires its durable launch revision.")
        if (
            self.authored_suite_exposure is not None
            and self.authored_suite_exposure.selection_revision
            != self.authored_suite_selection_revision
        ):
            raise ValueError("Authored suite exposure must match its immutable selection.")
        if (
            self.scenario is not None
            and self.scenario.authored_suite_revision != self.authored_suite_revision
        ):
            raise ValueError(
                "Authored scenario suite identity must match its parent run invocation."
            )
        if not json_utf8_size_within_limit(self, EVAL_RUN_INVOCATION_MAX_BYTES):
            raise ValueError(
                f"Eval run invocation exceeds {EVAL_RUN_INVOCATION_MAX_BYTES} JSON bytes."
            )
        return self


def eval_run_invocation_from_json(source: str) -> EvalRunInvocation:
    if type(source) is not str:
        raise TypeError("Eval run invocation JSON must be text.")
    if len(source.encode("utf-8")) > EVAL_RUN_INVOCATION_MAX_BYTES:
        raise ValueError(f"Eval run invocation exceeds {EVAL_RUN_INVOCATION_MAX_BYTES} JSON bytes.")
    return EvalRunInvocation.model_validate_json(source)


class EvalRunSpec(_EvalStoreModel):
    run_id: StrictStr
    corpus_revision: StrictStr
    target_key: StrictStr
    suite_id: StrictStr
    suite_revision: StrictStr
    max_concurrency: StrictInt = Field(ge=1, le=EVAL_MAX_CONCURRENCY)
    invocation: EvalRunInvocation = Field(default_factory=EvalRunInvocation)

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str, info) -> str:
        return _store_identifier(value, info.field_name)

    @field_validator("corpus_revision", "suite_revision")
    @classmethod
    def validate_revisions(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("target_key", "suite_id")
    @classmethod
    def validate_portable_ids(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("invocation", mode="before")
    @classmethod
    def copy_invocation(cls, value: object) -> object:
        if type(value) is EvalRunInvocation:
            return EvalRunInvocation.model_validate(value.model_dump(mode="python"))
        return value

    @property
    def id(self) -> str:
        return self.run_id


class EvalRunRequest(EvalRunSpec):
    idempotency_key: StrictStr = Field(repr=False)

    @field_validator("idempotency_key")
    @classmethod
    def validate_idempotency_key(cls, value: str, info) -> str:
        return _idempotency_key(value, info.field_name)

    def same_logical_request(self, other: EvalRunRequest) -> bool:
        if type(other) is not EvalRunRequest:
            return False
        return self.model_copy(update={"run_id": other.run_id}) == other


class EvalRunOwnership(_EvalStoreModel):
    epoch: StrictInt = Field(ge=1, le=_EVAL_STORE_MAX_BIGINT)
    lease_expires_at: datetime

    @field_validator("lease_expires_at")
    @classmethod
    def validate_lease_expires_at(cls, value: datetime, info) -> datetime:
        return _aware_utc(value, info.field_name)


class EvalRunObservation(_EvalStoreModel):
    """Small mutable-lifecycle projection that never rehydrates run invocation data."""

    run_id: StrictStr
    status: EvalRunStatus
    attempt_count: StrictInt = Field(ge=0, le=_EVAL_STORE_MAX_BIGINT)
    updated_at: datetime
    ownership: EvalRunOwnership | None = None

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str, info) -> str:
        return _store_identifier(value, info.field_name)

    @field_validator("updated_at")
    @classmethod
    def validate_updated_at(cls, value: datetime, info) -> datetime:
        return _aware_utc(value, info.field_name)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> EvalRunObservation:
        active = self.status in {EvalRunStatus.RUNNING, EvalRunStatus.CANCELLING}
        if active != (self.ownership is not None):
            raise ValueError("Only active eval run observations require ownership.")
        if self.ownership is not None and self.attempt_count != self.ownership.epoch:
            raise ValueError("Active eval run attempt count must match its ownership epoch.")
        return self

    @property
    def id(self) -> str:
        return self.run_id

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_EVAL_RUN_STATUSES


class EvalRunClaim(_EvalStoreModel):
    run_id: StrictStr
    claim_id: StrictStr = Field(repr=False)
    epoch: StrictInt = Field(ge=1, le=_EVAL_STORE_MAX_BIGINT)

    @field_validator("run_id", "claim_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return _store_identifier(value, info.field_name)


class EvalRunResultSummary(_EvalStoreModel):
    revision: StrictStr
    status: Literal["passed", "failed", "unavailable", "error"]
    score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    duration_ms: StrictInt = Field(ge=0, le=_EVAL_STORE_MAX_BIGINT)

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @model_validator(mode="after")
    def validate_score(self) -> EvalRunResultSummary:
        scored = self.status in {"passed", "failed"}
        if scored != (self.score is not None):
            raise ValueError("Eval run result summary score is inconsistent with its status.")
        return self


class EvalRunRecord(_EvalStoreModel):
    spec: EvalRunSpec
    status: EvalRunStatus
    attempt_count: StrictInt = Field(ge=0, le=_EVAL_STORE_MAX_BIGINT)
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    ownership: EvalRunOwnership | None = None
    result: EvalRunResultSummary | None = None
    failure_code: EvalRunFailureCode | None = None
    scenario_progress: EvalScenarioRunProgress | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    @field_validator(
        "created_at",
        "updated_at",
        "started_at",
        "finished_at",
        "cancel_requested_at",
    )
    @classmethod
    def validate_timestamps(cls, value: datetime | None, info) -> datetime | None:
        if value is None:
            return None
        return _aware_utc(value, info.field_name)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> EvalRunRecord:
        if self.updated_at < self.created_at:
            raise ValueError("Eval run updated_at cannot precede created_at.")
        for name in ("started_at", "finished_at", "cancel_requested_at"):
            value = getattr(self, name)
            if value is not None and value < self.created_at:
                raise ValueError(f"Eval run {name} cannot precede created_at.")
            if value is not None and value > self.updated_at:
                raise ValueError(f"Eval run {name} cannot follow updated_at.")
        terminal = self.status in TERMINAL_EVAL_RUN_STATUSES
        if terminal != (self.finished_at is not None):
            raise ValueError("Only terminal eval runs require finished_at.")
        if self.status in {EvalRunStatus.RUNNING, EvalRunStatus.CANCELLING}:
            if self.started_at is None or self.ownership is None:
                raise ValueError("Active eval runs require started_at and ownership.")
            if self.ownership.lease_expires_at <= self.updated_at:
                raise ValueError("Active eval run leases must follow updated_at.")
            if self.attempt_count != self.ownership.epoch:
                raise ValueError("Active eval run attempt count must match its ownership epoch.")
        elif self.ownership is not None:
            raise ValueError("Only active eval runs expose ownership.")
        if (self.started_at is None) != (self.attempt_count == 0):
            raise ValueError("Eval run attempt count must agree with whether execution started.")
        if (
            self.status in {EvalRunStatus.COMPLETED, EvalRunStatus.FAILED}
            and self.started_at is None
        ):
            raise ValueError("Executed terminal eval runs require started_at.")
        cancellation_status = self.status in {
            EvalRunStatus.CANCELLING,
            EvalRunStatus.CANCELLED,
        }
        if cancellation_status != (self.cancel_requested_at is not None):
            raise ValueError("Eval run cancellation state is inconsistent.")
        if self.status is EvalRunStatus.COMPLETED:
            if self.result is None or self.failure_code is not None:
                raise ValueError("Completed eval runs require only a result summary.")
        elif self.result is not None:
            raise ValueError("Only completed eval runs may reference a result.")
        if self.status is EvalRunStatus.FAILED:
            if self.failure_code is None:
                raise ValueError("Failed eval runs require a safe failure code.")
        elif self.failure_code is not None:
            raise ValueError("Only failed eval runs may carry a failure code.")
        scenario = self.spec.invocation.scenario
        if scenario is None:
            if self.scenario_progress is not None:
                raise ValueError("Corpus eval runs cannot expose scenario progress.")
        elif self.scenario_progress is not None:
            progress = self.scenario_progress
            if (
                progress.scenario_revision != scenario.scenario_revision
                or progress.binding_revision != scenario.binding_revision
                or len(progress.trials) != scenario.trials
                or progress.attempt > self.attempt_count
            ):
                raise ValueError("Scenario progress does not match its durable run invocation.")
            if (
                self.status in {EvalRunStatus.RUNNING, EvalRunStatus.CANCELLING}
                and progress.attempt != self.attempt_count
            ):
                raise ValueError("Active scenario progress must belong to the current claim epoch.")
        return self

    @property
    def id(self) -> str:
        return self.spec.run_id


def eval_run_observation(record: EvalRunRecord) -> EvalRunObservation:
    """Project a full public run record into its lightweight lifecycle shape."""

    record = _exact_model(record, EvalRunRecord, "record")
    return EvalRunObservation(
        run_id=record.id,
        status=record.status,
        attempt_count=record.attempt_count,
        updated_at=record.updated_at,
        ownership=record.ownership,
    )


class EvalRunLease(_EvalStoreModel):
    run: EvalRunRecord
    claim: EvalRunClaim = Field(repr=False)

    @model_validator(mode="after")
    def validate_binding(self) -> EvalRunLease:
        if self.run.id != self.claim.run_id:
            raise ValueError("Eval run lease claim does not match its run.")
        ownership = self.run.ownership
        if ownership is None or ownership.epoch != self.claim.epoch:
            raise ValueError("Eval run lease claim does not match active ownership.")
        return self


class EvalRunPage(_EvalStoreModel):
    items: tuple[EvalRunRecord, ...] = Field(max_length=EVAL_STORE_MAX_PAGE_SIZE)
    next_cursor: StrictStr | None = None
    has_more: StrictBool = False

    @model_validator(mode="after")
    def validate_page(self) -> EvalRunPage:
        _validate_page_boundary(self.items, self.next_cursor, self.has_more)
        if len({item.id for item in self.items}) != len(self.items):
            raise ValueError("Eval run page contains duplicate run ids.")
        expected = list(self.items)
        expected.sort(key=lambda item: item.id)
        expected.sort(key=lambda item: item.created_at, reverse=True)
        if list(self.items) != expected:
            raise ValueError("Eval run page is not in keyset order.")
        if self.has_more:
            assert self.next_cursor is not None
            timestamp, run_id, target_key, status, corpus_revision = _decode_cursor(
                self.next_cursor,
                "runs",
                ("created_at", "run_id", "target_key", "status", "corpus_revision"),
            )
            if (timestamp, run_id) != (
                self.items[-1].created_at.isoformat(),
                self.items[-1].id,
            ):
                raise ValueError("Eval run cursor does not follow its last item.")
            if status and any(str(item.status) != status for item in self.items):
                raise ValueError("Eval run cursor status filter does not match its items.")
            if target_key and any(item.spec.target_key != target_key for item in self.items):
                raise ValueError("Eval run cursor target filter does not match its items.")
            if corpus_revision and any(
                item.spec.corpus_revision != corpus_revision for item in self.items
            ):
                raise ValueError("Eval run cursor corpus filter does not match its items.")
        return self


class EvalResultRecord(_EvalStoreModel):
    """Bounded catalog metadata for one immutable captured or fresh result."""

    revision: StrictStr
    origin: EvalResultOrigin
    target: EvalResultTargetIdentityV1
    corpus_revision: StrictStr
    suite_id: StrictStr
    suite_revision: StrictStr
    status: Literal["passed", "failed", "unavailable", "error"]
    score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    document_bytes: StrictInt = Field(ge=1, le=CORPUS_EXECUTION_RESULT_MAX_BYTES)
    created_at: datetime

    @field_validator("revision", "corpus_revision", "suite_revision")
    @classmethod
    def validate_revisions(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("suite_id")
    @classmethod
    def validate_suite_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime, info) -> datetime:
        return _aware_utc(value, info.field_name)

    @model_validator(mode="after")
    def validate_result_summary(self) -> EvalResultRecord:
        if (self.status in {"passed", "failed"}) != (self.score is not None):
            raise ValueError("Eval result record status contradicts its score.")
        if (
            self.origin is EvalResultOrigin.CAPTURED_SESSION
            and self.document_bytes > CAPTURED_EVALUATION_RESULT_MAX_BYTES
        ):
            raise ValueError("Captured eval result record exceeds its document limit.")
        return self


class EvalResultQuery(_EvalStoreModel):
    """Bounded target-scoped immutable result catalog query."""

    target_key: StrictStr
    origin: EvalResultOrigin | None = None
    cursor: StrictStr | None = None
    limit: StrictInt = Field(
        default=EVAL_STORE_DEFAULT_PAGE_SIZE,
        ge=1,
        le=EVAL_STORE_MAX_PAGE_SIZE,
    )
    max_result_bytes: StrictInt = Field(
        default=EVAL_STORE_DEFAULT_PAGE_BYTES,
        ge=1_024,
        le=EVAL_STORE_MAX_PAGE_BYTES,
    )

    @field_validator("target_key")
    @classmethod
    def validate_target_key(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("cursor")
    @classmethod
    def validate_cursor(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_cursor(value, info.field_name)


class EvalResultPage(_EvalStoreModel):
    items: tuple[EvalResultRecord, ...] = Field(max_length=EVAL_STORE_MAX_PAGE_SIZE)
    next_cursor: StrictStr | None = None
    has_more: StrictBool = False

    @model_validator(mode="after")
    def validate_page(self) -> EvalResultPage:
        _validate_page_boundary(self.items, self.next_cursor, self.has_more)
        if len({item.revision for item in self.items}) != len(self.items):
            raise ValueError("Eval result page contains duplicate revisions.")
        expected = list(self.items)
        expected.sort(key=lambda item: item.revision)
        expected.sort(key=lambda item: item.created_at, reverse=True)
        if list(self.items) != expected:
            raise ValueError("Eval result page is not in keyset order.")
        if self.has_more:
            assert self.next_cursor is not None
            timestamp, revision, target_key, origin = _decode_cursor(
                self.next_cursor,
                "results",
                ("created_at", "revision", "target_key", "origin"),
            )
            if (timestamp, revision) != (
                self.items[-1].created_at.isoformat(),
                self.items[-1].revision,
            ):
                raise ValueError("Eval result cursor does not follow its last item.")
            if any(item.target.target_key != target_key for item in self.items):
                raise ValueError("Eval result cursor target filter does not match its items.")
            if origin and any(str(item.origin) != origin for item in self.items):
                raise ValueError("Eval result cursor origin filter does not match its items.")
        return self


class EvalBaselineKey(_EvalStoreModel):
    """Stable scope of one explicit baseline pointer."""

    target_key: StrictStr
    corpus_revision: StrictStr
    suite_id: StrictStr

    @field_validator("target_key", "suite_id")
    @classmethod
    def validate_portable_ids(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("corpus_revision")
    @classmethod
    def validate_corpus_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)


class EvalBaselineUpdate(_EvalStoreModel):
    """Actor-attributed idempotent compare-and-swap baseline request."""

    key: EvalBaselineKey
    result_revision: StrictStr
    expected_generation: StrictInt = Field(ge=0, le=_EVAL_STORE_MAX_BIGINT)
    operation_id: StrictStr
    actor_id: StrictStr

    @field_validator("result_revision", "operation_id")
    @classmethod
    def validate_revision_fields(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("actor_id")
    @classmethod
    def validate_actor_id(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=512,
            nonblank=True,
            clean=True,
        )


class EvalBaselineRecord(_EvalStoreModel):
    """Current mutable pointer to an immutable eval result."""

    key: EvalBaselineKey
    result_revision: StrictStr
    generation: StrictInt = Field(ge=1, le=_EVAL_STORE_MAX_BIGINT)
    updated_by: StrictStr
    updated_at: datetime

    @field_validator("result_revision")
    @classmethod
    def validate_result_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("updated_by")
    @classmethod
    def validate_updated_by(cls, value: str, info) -> str:
        return EvalBaselineUpdate.validate_actor_id(value, info)

    @field_validator("updated_at")
    @classmethod
    def validate_updated_at(cls, value: datetime, info) -> datetime:
        return _aware_utc(value, info.field_name)


class EvalBaselineMutationRecord(_EvalStoreModel):
    """Immutable audit fact for one committed baseline transition."""

    operation_id: StrictStr
    key: EvalBaselineKey
    expected_generation: StrictInt = Field(ge=0, le=_EVAL_STORE_MAX_BIGINT)
    previous_result_revision: StrictStr | None = None
    selected_result_revision: StrictStr
    resulting_generation: StrictInt = Field(ge=1, le=_EVAL_STORE_MAX_BIGINT)
    actor_id: StrictStr
    created_at: datetime

    @field_validator(
        "operation_id",
        "previous_result_revision",
        "selected_result_revision",
    )
    @classmethod
    def validate_revision_fields(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_revision(value, info.field_name)

    @field_validator("actor_id")
    @classmethod
    def validate_actor_id(cls, value: str, info) -> str:
        return EvalBaselineUpdate.validate_actor_id(value, info)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime, info) -> datetime:
        return _aware_utc(value, info.field_name)

    @model_validator(mode="after")
    def validate_generations(self) -> EvalBaselineMutationRecord:
        if self.resulting_generation != self.expected_generation + 1:
            raise ValueError("Baseline mutation generation does not follow its CAS expectation.")
        if (self.expected_generation == 0) != (self.previous_result_revision is None):
            raise ValueError("Baseline mutation prior state contradicts its expected generation.")
        return self


def _validate_page_boundary(items: tuple, next_cursor: str | None, has_more: bool) -> None:
    if has_more and (not items or next_cursor is None):
        raise ValueError("A continued eval-store page requires items and next_cursor.")
    if not has_more and next_cursor is not None:
        raise ValueError("A terminal eval-store page cannot carry next_cursor.")


def _bounded_cursor(value: str, field_name: str) -> str:
    value = _bounded_durable_text(
        value,
        field_name,
        max_chars=EVAL_STORE_MAX_CURSOR_BYTES,
        nonblank=True,
        clean=True,
    )
    if len(value.encode("utf-8")) > EVAL_STORE_MAX_CURSOR_BYTES:
        raise ValueError(f"{field_name} exceeds {EVAL_STORE_MAX_CURSOR_BYTES} UTF-8 bytes.")
    return value


def _encode_cursor(scope: str, values: Mapping[str, str]) -> str:
    document = {"v": _CURSOR_VERSION, "scope": scope, **values}
    raw = json.dumps(document, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    encoded = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    return _bounded_cursor(encoded, "cursor")


def _decode_cursor(cursor: str, scope: str, fields: tuple[str, ...]) -> tuple[str, ...]:
    cursor = _bounded_cursor(cursor, "cursor")
    try:
        padding = "=" * (-len(cursor) % 4)
        raw = base64.b64decode(cursor + padding, altchars=b"-_", validate=True)
        document = json.loads(raw)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid eval-store cursor.") from exc
    expected_keys = {"v", "scope", *fields}
    if (
        type(document) is not dict
        or set(document) != expected_keys
        or document.get("v") != _CURSOR_VERSION
        or document.get("scope") != scope
        or any(type(document.get(field)) is not str for field in fields)
    ):
        raise ValueError("Eval-store cursor does not match this query.")
    return tuple(document[field] for field in fields)


def _corpus_cursor(entry: EvalCorpusCatalogEntry, target_key: str | None) -> str:
    return _encode_cursor(
        "corpora",
        {
            "created_at": entry.created_at.isoformat(),
            "revision": entry.revision,
            "target_key": target_key or "",
        },
    )


def decode_corpus_cursor(cursor: str, target_key: str | None) -> tuple[datetime, str]:
    timestamp, revision, cursor_target_key = _decode_cursor(
        cursor,
        "corpora",
        ("created_at", "revision", "target_key"),
    )
    if cursor_target_key != (target_key or ""):
        raise ValueError("Eval-store corpus cursor does not match this query.")
    try:
        created_at = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ValueError("Invalid eval-store corpus cursor timestamp.") from exc
    return _aware_utc(created_at, "cursor created_at"), _sha256_revision(
        revision, "cursor revision"
    )


def _scenario_cursor(entry: EvalScenarioCatalogEntry, query: EvalScenarioCatalogQuery) -> str:
    return _encode_cursor(
        "scenarios",
        {
            "created_at": entry.created_at.isoformat(),
            "revision": entry.revision,
            "target_key": query.target_key or "",
            "scenario_id": query.scenario_id or "",
        },
    )


def decode_scenario_cursor(
    cursor: str,
    target_key: str | None,
    scenario_id: str | None,
) -> tuple[datetime, str]:
    timestamp, revision, cursor_target_key, cursor_scenario_id = _decode_cursor(
        cursor,
        "scenarios",
        ("created_at", "revision", "target_key", "scenario_id"),
    )
    if cursor_target_key != (target_key or "") or cursor_scenario_id != (scenario_id or ""):
        raise ValueError("Eval-store scenario cursor does not match this query.")
    try:
        created_at = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ValueError("Invalid eval-store scenario cursor timestamp.") from exc
    return _aware_utc(created_at, "cursor created_at"), _sha256_revision(
        revision,
        "cursor revision",
    )


def _authored_suite_cursor(
    entry: EvalAuthoredSuiteCatalogEntry,
    query: EvalAuthoredSuiteCatalogQuery,
) -> str:
    return _encode_cursor(
        "authored_suites",
        {
            "created_at": entry.created_at.isoformat(),
            "revision": entry.revision,
            "target_key": query.target_key or "",
            "suite_id": query.suite_id or "",
        },
    )


def decode_authored_suite_cursor(
    cursor: str,
    target_key: str | None,
    suite_id: str | None,
) -> tuple[datetime, str]:
    timestamp, revision, cursor_target_key, cursor_suite_id = _decode_cursor(
        cursor,
        "authored_suites",
        ("created_at", "revision", "target_key", "suite_id"),
    )
    if cursor_target_key != (target_key or "") or cursor_suite_id != (suite_id or ""):
        raise ValueError("Eval-store authored-suite cursor does not match this query.")
    try:
        created_at = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ValueError("Invalid eval-store authored-suite cursor timestamp.") from exc
    return _aware_utc(created_at, "cursor created_at"), _sha256_revision(
        revision,
        "cursor revision",
    )


def _suite_cursor(entry: EvalSuiteCatalogEntry) -> str:
    return _encode_cursor(
        "suites",
        {"corpus_revision": entry.corpus_revision, "suite_id": entry.id},
    )


def decode_suite_cursor(cursor: str, corpus_revision: str) -> str:
    revision, suite_id = _decode_cursor(
        cursor,
        "suites",
        ("corpus_revision", "suite_id"),
    )
    if revision != corpus_revision:
        raise ValueError("Eval-store suite cursor does not match this corpus.")
    return _portable_id(suite_id, "cursor suite_id")


def _case_cursor(entry: EvalCaseCatalogEntry) -> str:
    return _encode_cursor(
        "cases",
        {
            "corpus_revision": entry.corpus_revision,
            "suite_id": entry.suite_id,
            "case_id": entry.id,
        },
    )


def decode_case_cursor(cursor: str, corpus_revision: str, suite_id: str) -> str:
    revision, cursor_suite_id, case_id = _decode_cursor(
        cursor,
        "cases",
        ("corpus_revision", "suite_id", "case_id"),
    )
    if revision != corpus_revision or cursor_suite_id != suite_id:
        raise ValueError("Eval-store case cursor does not match this corpus suite.")
    return _portable_id(case_id, "cursor case_id")


def _run_cursor(record: EvalRunRecord, query: EvalRunQuery) -> str:
    return _encode_cursor(
        "runs",
        {
            "created_at": record.created_at.isoformat(),
            "run_id": record.id,
            "target_key": query.target_key or "",
            "status": "" if query.status is None else str(query.status),
            "corpus_revision": query.corpus_revision or "",
        },
    )


def _result_cursor(record: EvalResultRecord, query: EvalResultQuery) -> str:
    return _encode_cursor(
        "results",
        {
            "created_at": record.created_at.isoformat(),
            "revision": record.revision,
            "target_key": query.target_key,
            "origin": "" if query.origin is None else str(query.origin),
        },
    )


def decode_result_cursor(
    cursor: str,
    target_key: str,
    origin: EvalResultOrigin | None,
) -> tuple[datetime, str]:
    timestamp, revision, cursor_target_key, cursor_origin = _decode_cursor(
        cursor,
        "results",
        ("created_at", "revision", "target_key", "origin"),
    )
    if cursor_target_key != target_key or cursor_origin != ("" if origin is None else str(origin)):
        raise ValueError("Eval-store result cursor does not match this query.")
    try:
        created_at = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ValueError("Invalid eval-store result cursor timestamp.") from exc
    return _aware_utc(created_at, "cursor created_at"), _sha256_revision(
        revision,
        "cursor revision",
    )


def decode_run_cursor(
    cursor: str,
    target_key: str | None,
    status: EvalRunStatus | None,
    corpus_revision: str | None,
) -> tuple[datetime, str]:
    timestamp, run_id, cursor_target_key, cursor_status, cursor_corpus_revision = _decode_cursor(
        cursor,
        "runs",
        ("created_at", "run_id", "target_key", "status", "corpus_revision"),
    )
    if (
        cursor_target_key != (target_key or "")
        or cursor_status != ("" if status is None else str(status))
        or cursor_corpus_revision != (corpus_revision or "")
    ):
        raise ValueError("Eval-store run cursor does not match this query.")
    try:
        created_at = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ValueError("Invalid eval-store run cursor timestamp.") from exc
    return _aware_utc(created_at, "cursor created_at"), _store_identifier(run_id, "cursor run_id")


def corpus_catalog_entry(
    corpus: EvalCorpusDocument,
    *,
    created_at: datetime,
    document_bytes: int | None = None,
) -> EvalCorpusCatalogEntry:
    validated = _exact_model(corpus, EvalCorpusDocument, "corpus")
    inspection = inspect_eval_corpus(validated)
    size = len(_wire_model_bytes(validated)) if document_bytes is None else document_bytes
    return EvalCorpusCatalogEntry(
        revision=inspection.revision,
        target_key=inspection.target_key,
        evidence_policy_revision=inspection.evidence_policy_revision,
        pricing_profile_fingerprint=inspection.pricing_profile_fingerprint,
        suite_count=inspection.suite_count,
        case_count=inspection.case_count,
        assertion_count=inspection.assertion_count,
        expanded_assertion_result_count=inspection.expanded_assertion_result_count,
        document_bytes=size,
        created_at=created_at,
    )


def scenario_catalog_entry(
    scenario: EvalScenarioDocumentV2,
    *,
    created_at: datetime,
    document_bytes: int | None = None,
) -> EvalScenarioCatalogEntry:
    validated = _exact_model(scenario, EvalScenarioDocumentV2, "scenario")
    inspection = inspect_eval_scenario(validated)
    size = len(_wire_model_bytes(validated)) if document_bytes is None else document_bytes
    return EvalScenarioCatalogEntry(
        revision=inspection.revision,
        id=inspection.id,
        target_key=inspection.target_key,
        name=validated.name,
        description=validated.description,
        event_count=inspection.event_count,
        input_event_count=inspection.input_event_count,
        approval_checkpoint_count=inspection.approval_checkpoint_count,
        message_count=inspection.message_count,
        part_count=inspection.part_count,
        artifact_requirement_count=inspection.artifact_requirement_count,
        secret_requirement_count=inspection.secret_requirement_count,
        document_bytes=size,
        created_at=created_at,
    )


def suite_catalog_entries(
    corpus: EvalCorpusDocument,
) -> tuple[EvalSuiteCatalogEntry, ...]:
    validated = _exact_model(corpus, EvalCorpusDocument, "corpus")
    inspection = inspect_eval_corpus(validated)
    specs = {suite.id: suite for suite in validated.suites}
    return tuple(
        _suite_catalog_entry(validated.revision, specs[item.id], item) for item in inspection.suites
    )


def _suite_catalog_entry(
    corpus_revision: str,
    spec: EvalSuiteSpec,
    inspection: EvalCorpusSuiteInspectionV1,
) -> EvalSuiteCatalogEntry:
    return EvalSuiteCatalogEntry(
        corpus_revision=corpus_revision,
        id=spec.id,
        revision=spec.revision,
        name=spec.name,
        description=spec.description,
        case_count=inspection.case_count,
        assertion_count=inspection.assertion_count,
        trials=inspection.trials,
        timeout_seconds=inspection.timeout_seconds,
    )


def case_catalog_entries(corpus: EvalCorpusDocument) -> tuple[EvalCaseCatalogEntry, ...]:
    validated = _exact_model(corpus, EvalCorpusDocument, "corpus")
    return tuple(
        EvalCaseCatalogEntry(
            corpus_revision=validated.revision,
            id=case.id,
            revision=case.revision,
            suite_id=case.suite_id,
            name=case.name,
            description=case.description,
            message_count=0 if case.input is None else len(case.input.messages),
            assertion_count=len(case.assertions),
        )
        for case in validated.cases
    )


def _prepare_corpus_catalog_for_store(
    corpus: EvalCorpusDocument,
    *,
    redact_json: Callable[[Any], Any],
) -> tuple[
    EvalCorpusDocument,
    bytes,
    EvalCorpusInspectionV1,
    tuple[EvalSuiteCatalogEntry, ...],
    tuple[EvalCaseCatalogEntry, ...],
]:
    """Prepare one immutable corpus and its catalog projections in one CPU phase."""

    validated, document = _prepare_corpus_for_store(
        corpus,
        redact_json=redact_json,
    )
    return (
        validated,
        document,
        inspect_eval_corpus(validated),
        suite_catalog_entries(validated),
        case_catalog_entries(validated),
    )


def _prepare_scenario_catalog_for_store(
    scenario: EvalScenarioDocumentV2,
    *,
    redact_json: Callable[[Any], Any],
) -> tuple[EvalScenarioDocumentV2, bytes, EvalScenarioInspectionV2]:
    """Prepare one immutable scenario and its catalog projection in one CPU phase."""

    validated, document = _prepare_scenario_for_store(
        scenario,
        redact_json=redact_json,
    )
    return validated, document, inspect_eval_scenario(validated)


def authored_suite_catalog_entry(
    document: EvalSuiteDocument,
    *,
    created_at: datetime,
    document_bytes: int,
) -> EvalAuthoredSuiteCatalogEntry:
    validated = validate_expected_eval_suite_revision(document, document.revision)
    simple_input_count = sum(case.stimulus.kind == "simple_input" for case in validated.cases)
    return EvalAuthoredSuiteCatalogEntry(
        revision=validated.revision,
        id=validated.suite.id,
        suite_revision=validated.suite.revision,
        target_key=validated.target_key,
        name=validated.suite.name,
        description=validated.suite.description,
        case_count=len(validated.cases),
        assertion_count=sum(len(case.assertions) for case in validated.cases),
        simple_input_count=simple_input_count,
        scenario_count=len(validated.cases) - simple_input_count,
        trials=validated.suite.trial_request.trials,
        timeout_seconds=validated.suite.trial_request.timeout_seconds,
        document_bytes=document_bytes,
        created_at=created_at,
    )


def authored_suite_scenario_cases(
    document: EvalSuiteDocument,
) -> tuple[tuple[EvalCaseDefinition, EvalScenarioStimulusV1], ...]:
    validated = validate_expected_eval_suite_revision(document, document.revision)
    return tuple(
        (case, case.stimulus)
        for case in validated.cases
        if type(case.stimulus) is EvalScenarioStimulusV1
    )


def validate_authored_suite_scenario(
    document: EvalSuiteDocument,
    case: EvalCaseDefinition,
    scenario: EvalScenarioDocumentV2 | None,
) -> None:
    """Validate one content-addressed scenario reference without inferring fallback."""

    if type(case) not in {EvalCaseDefinitionV1, EvalCaseDefinitionV2}:
        raise TypeError("case must be an exact authored eval case definition.")
    if type(case.stimulus) is not EvalScenarioStimulusV1:
        raise TypeError("case must be an exact scenario-backed authored case definition.")
    reference = case.stimulus
    if scenario is None:
        raise EvalAuthoredSuiteReferenceError(
            f"Authored eval case {case.id!r} references an unavailable scenario revision."
        )
    if scenario.id != reference.scenario_id:
        raise EvalAuthoredSuiteReferenceError(
            f"Authored eval case {case.id!r} scenario ID does not match stored content."
        )
    if scenario.target_key != document.target_key:
        raise EvalAuthoredSuiteReferenceError(
            f"Authored eval case {case.id!r} scenario target does not match its suite."
        )
    if case.source is not None and case.source != scenario.source:
        raise EvalAuthoredSuiteReferenceError(
            f"Authored eval case {case.id!r} source does not match its scenario."
        )


def validate_run_request_for_corpus(
    request: EvalRunRequest,
    corpus: EvalCorpusDocument,
) -> None:
    if request.corpus_revision != corpus.revision:
        raise EvalRunAdmissionConflict("Eval run corpus revision does not match stored content.")
    if request.target_key != corpus.target_key:
        raise EvalRunAdmissionConflict("Eval run target key does not match its corpus.")
    suite = next((item for item in corpus.suites if item.id == request.suite_id), None)
    if suite is None:
        raise EvalRunAdmissionConflict(f"Eval suite not found: {request.suite_id}")
    if request.suite_revision != suite.revision:
        raise EvalRunAdmissionConflict("Eval run suite revision does not match its corpus.")
    if any(case.suite_id == request.suite_id and case.input is None for case in corpus.cases):
        raise EvalRunAdmissionConflict(
            "Captured-only eval cases cannot run until runnable input is authored."
        )


def validate_result_for_run(
    request: EvalRunRequest,
    result: CorpusExecutionResult,
    corpus: EvalCorpusDocument,
) -> CorpusExecutionResult:
    if type(result) is not CorpusExecutionResult:
        raise TypeError("result must be an exact CorpusExecutionResult.")
    if type(corpus) is not EvalCorpusDocument:
        raise TypeError("corpus must be an exact EvalCorpusDocument.")
    if result.target.target_key != request.target_key:
        raise EvalRunStateConflict("Eval result target key does not match its run request.")
    if result.run.corpus_revision != request.corpus_revision:
        raise EvalRunStateConflict("Eval result corpus revision does not match its run request.")
    if result.run.suite_id != request.suite_id:
        raise EvalRunStateConflict("Eval result suite id does not match its run request.")
    if result.run.suite_revision != request.suite_revision:
        raise EvalRunStateConflict("Eval result suite revision does not match its run request.")
    try:
        _validate_published_eval_run_for_corpus(corpus, result.run)
    except ValueError:
        raise EvalRunStateConflict(
            "Eval result does not match its immutable corpus suite contract."
        ) from None
    return result


def result_summary(result: CorpusExecutionResult) -> EvalRunResultSummary:
    return EvalRunResultSummary(
        revision=result.revision,
        status=result.run.status,
        score=result.run.score,
        duration_ms=result.run.duration_ms,
    )


def eval_result_record(
    result: CorpusExecutionResult | CapturedEvaluationResultV1,
    *,
    document_bytes: int,
    created_at: datetime,
) -> EvalResultRecord:
    if type(result) not in {CorpusExecutionResult, CapturedEvaluationResultV1}:
        raise TypeError(
            "result must be an exact CorpusExecutionResult or CapturedEvaluationResultV1."
        )
    projection = eval_result_projection(result)
    return EvalResultRecord(
        revision=projection.result_revision,
        origin=projection.origin,
        target=projection.target,
        corpus_revision=projection.corpus_revision,
        suite_id=projection.suite_id,
        suite_revision=projection.suite_revision,
        status=projection.status,
        score=projection.score,
        document_bytes=document_bytes,
        created_at=created_at,
    )


def _prepare_baseline_update_for_store(
    update: EvalBaselineUpdate,
    *,
    redact_json: Callable[[Any], Any],
) -> EvalBaselineUpdate:
    validated = _exact_model(update, EvalBaselineUpdate, "update")
    _require_publication_safe(
        validated.model_dump(mode="json"),
        redact_json=redact_json,
        resource_name="Eval baseline mutation",
    )
    return validated


def _validate_baseline_result(
    update: EvalBaselineUpdate,
    result: EvalResultRecord,
) -> None:
    if result.revision != update.result_revision:
        raise EvalBaselineConflict("Baseline result revision does not match stored content.")
    if (
        result.target.target_key != update.key.target_key
        or result.corpus_revision != update.key.corpus_revision
        or result.suite_id != update.key.suite_id
    ):
        raise EvalBaselineConflict("Baseline result does not belong to the requested scope.")


def run_spec(request: EvalRunRequest) -> EvalRunSpec:
    values = request.model_dump(mode="python")
    values.pop("idempotency_key")
    return EvalRunSpec.model_validate(values)


class EvalStore(ABC):
    """Bounded persistence for public eval corpora, run state, and safe results."""

    durable: ClassVar[bool] = False
    captured_results: ClassVar[bool] = False
    scenarios: ClassVar[bool] = False
    scenario_execution: ClassVar[bool] = False
    trial_checkpointing: ClassVar[bool] = False
    suite_authoring: ClassVar[bool] = False
    judge_calibrations: ClassVar[bool] = False

    @abstractmethod
    async def close(self) -> None:
        """Release store resources; process-local stores may perform no work."""

    @abstractmethod
    async def save_corpus(
        self,
        corpus: EvalCorpusDocument,
        *,
        redact_json: Callable[[Any], Any],
    ) -> EvalCorpusCatalogEntry:
        """Scan and atomically save one complete immutable corpus revision."""

    @abstractmethod
    async def load_corpus(
        self,
        revision: str,
        *,
        max_bytes: int = EVAL_CORPUS_MAX_BYTES,
    ) -> EvalCorpusDocument | None:
        """Load one corpus without crossing the caller's byte ceiling."""

    @abstractmethod
    async def list_corpora(
        self,
        query: EvalCatalogQuery | None = None,
    ) -> EvalCorpusCatalogPage:
        """List immutable corpus revisions in newest-first keyset order."""

    async def save_scenario(
        self,
        scenario: EvalScenarioDocumentV2,
        *,
        redact_json: Callable[[Any], Any],
    ) -> EvalScenarioCatalogEntry:
        """Scan and atomically save one immutable portable scenario revision."""

        del scenario, redact_json
        raise NotImplementedError("Eval scenario persistence is not supported.")

    async def load_scenario(
        self,
        revision: str,
        *,
        max_bytes: int = EVAL_SCENARIO_MAX_BYTES,
    ) -> EvalScenarioDocumentV2 | None:
        """Load one scenario without crossing the caller's byte ceiling."""

        del revision, max_bytes
        raise NotImplementedError("Eval scenario persistence is not supported.")

    async def list_scenarios(
        self,
        query: EvalScenarioCatalogQuery | None = None,
    ) -> EvalScenarioCatalogPage:
        """List immutable scenario revisions in newest-first keyset order."""

        del query
        raise NotImplementedError("Eval scenario persistence is not supported.")

    async def save_authored_suite(
        self,
        document: EvalSuiteDocument,
        *,
        redact_json: Callable[[Any], Any],
    ) -> EvalAuthoredSuiteCatalogEntry:
        """Scan and atomically save one reviewed immutable authored suite."""

        del document, redact_json
        raise NotImplementedError("Authored eval suite persistence is not supported.")

    async def load_authored_suite(
        self,
        revision: str,
        *,
        max_bytes: int = EVAL_SUITE_AUTHORING_MAX_BYTES,
    ) -> EvalSuiteDocument | None:
        """Load one authored suite without crossing the caller byte ceiling."""

        del revision, max_bytes
        raise NotImplementedError("Authored eval suite persistence is not supported.")

    async def list_authored_suites(
        self,
        query: EvalAuthoredSuiteCatalogQuery | None = None,
    ) -> EvalAuthoredSuiteCatalogPage:
        """List immutable authored suite revisions in newest-first keyset order."""

        del query
        raise NotImplementedError("Authored eval suite persistence is not supported.")

    async def save_judge_calibration(
        self,
        report: EvalJudgeCalibrationReportV1,
        *,
        redact_json: Callable[[Any], Any],
    ) -> EvalJudgeCalibrationReportV1:
        """Scan and atomically retain every trial in one completed calibration."""

        del report, redact_json
        raise NotImplementedError("Judge calibration persistence is not supported.")

    async def load_judge_calibration(
        self,
        revision: str,
        *,
        max_bytes: int = EVAL_JUDGE_CALIBRATION_MAX_BYTES,
    ) -> EvalJudgeCalibrationReportV1 | None:
        """Load one immutable completed calibration by content revision."""

        del revision, max_bytes
        raise NotImplementedError("Judge calibration persistence is not supported.")

    async def load_judge_calibration_by_run_id(
        self,
        run_id: str,
        *,
        max_bytes: int = EVAL_JUDGE_CALIBRATION_MAX_BYTES,
    ) -> EvalJudgeCalibrationReportV1 | None:
        """Recover one completed calibration after retry or server restart."""

        del run_id, max_bytes
        raise NotImplementedError("Judge calibration persistence is not supported.")

    @abstractmethod
    async def list_suites(self, query: EvalSuiteCatalogQuery) -> EvalSuiteCatalogPage:
        """List bounded suite projections for one corpus revision."""

    @abstractmethod
    async def list_cases(self, query: EvalCaseCatalogQuery) -> EvalCaseCatalogPage:
        """List bounded case projections for one corpus suite."""

    @abstractmethod
    async def admit_run(
        self,
        request: EvalRunRequest,
        *,
        redact_json: Callable[[Any], Any],
    ) -> EvalRunRecord:
        """Atomically admit or replay one logical run request."""

    @abstractmethod
    async def load_run_by_idempotency_key(
        self,
        idempotency_key: str,
    ) -> EvalRunRecord | None:
        """Load a previously admitted run by its server-scoped idempotency digest."""

    @abstractmethod
    async def load_run(self, run_id: str) -> EvalRunRecord | None:
        """Load one run's public lifecycle projection."""

    async def load_run_observation(self, run_id: str) -> EvalRunObservation | None:
        """Load mutable claim/status state without requiring immutable invocation data.

        Custom stores retain compatibility through this full-record fallback. Built-in
        stores override it with a small indexed projection.
        """

        record = await self.load_run(run_id)
        return None if record is None else eval_run_observation(record)

    async def wait_for_run_terminal(
        self,
        run_id: str,
        *,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = EVAL_RUN_MIN_OBSERVATION_INTERVAL_SECONDS,
        max_poll_interval_seconds: float = EVAL_RUN_MAX_OBSERVATION_INTERVAL_SECONDS,
    ) -> EvalRunObservation | None:
        """Wait within a strict deadline for a terminal run observation.

        Polling begins no faster than the public 50 ms floor and exponentially backs
        off to the caller's bounded ceiling. A missing run returns ``None``; deadline
        exhaustion raises ``TimeoutError``.
        """

        run_id = _store_identifier(run_id, "run_id")
        timeout_seconds = _bounded_positive_seconds(
            timeout_seconds,
            "timeout_seconds",
            maximum=EVAL_RUN_MAX_TERMINAL_WAIT_SECONDS,
        )
        poll_interval_seconds = _bounded_positive_seconds(
            poll_interval_seconds,
            "poll_interval_seconds",
            maximum=EVAL_RUN_MAX_OBSERVATION_INTERVAL_SECONDS,
        )
        max_poll_interval_seconds = _bounded_positive_seconds(
            max_poll_interval_seconds,
            "max_poll_interval_seconds",
            maximum=EVAL_RUN_MAX_OBSERVATION_INTERVAL_SECONDS,
        )
        interval = max(
            poll_interval_seconds,
            EVAL_RUN_MIN_OBSERVATION_INTERVAL_SECONDS,
        )
        if interval > max_poll_interval_seconds:
            raise ValueError(
                "max_poll_interval_seconds cannot be less than the effective polling floor."
            )

        deadline = monotonic() + timeout_seconds
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for eval run to finish: {run_id}")
            try:
                async with asyncio.timeout(remaining):
                    observation = await self.load_run_observation(run_id)
            except TimeoutError as exc:
                raise TimeoutError(f"Timed out waiting for eval run to finish: {run_id}") from exc
            if observation is None or observation.terminal:
                return observation
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for eval run to finish: {run_id}")
            await asyncio.sleep(min(interval, remaining))
            interval = min(interval * 2, max_poll_interval_seconds)

    @abstractmethod
    async def list_runs(self, query: EvalRunQuery | None = None) -> EvalRunPage:
        """List run projections in newest-first keyset order."""

    @abstractmethod
    async def claim_run(
        self,
        *,
        target_key: str | None = None,
        lease_seconds: int = 300,
    ) -> EvalRunLease | None:
        """Claim the oldest eligible queued or expired run for an optional target."""

    async def claim_run_for_targets(
        self,
        target_keys: tuple[str, ...],
        *,
        lease_seconds: int = 300,
    ) -> EvalRunLease | None:
        """Claim work only from a bounded approved target set.

        Custom stores inherit a compatible sequential fallback. Durable built-in
        stores override this with one indexed atomic claim across the whole set.
        """

        target_keys = _claim_target_keys(target_keys)
        lease_seconds = _lease_seconds(lease_seconds)
        for target_key in target_keys:
            lease = await self.claim_run(
                target_key=target_key,
                lease_seconds=lease_seconds,
            )
            if lease is not None:
                return lease
        return None

    @abstractmethod
    async def heartbeat_run(
        self,
        claim: EvalRunClaim,
        *,
        extend_seconds: int = 300,
    ) -> EvalRunRecord:
        """Extend one still-live fenced run claim."""

    async def heartbeat_run_observation(
        self,
        claim: EvalRunClaim,
        *,
        extend_seconds: int = 300,
    ) -> EvalRunObservation:
        """Extend a claim and return only its mutable lifecycle projection.

        Custom stores retain compatibility through the full-record heartbeat.
        Built-in stores override this method to avoid immutable rehydration.
        """

        return eval_run_observation(await self.heartbeat_run(claim, extend_seconds=extend_seconds))

    @abstractmethod
    async def request_cancel(self, run_id: str) -> EvalRunRecord:
        """Persist cancellation intent, terminalizing unclaimed queued work."""

    async def initialize_scenario_progress(
        self,
        claim: EvalRunClaim,
        progress: EvalScenarioRunProgress,
    ) -> EvalRunRecord:
        """Replace scenario progress for a newly claimed execution attempt."""

        del claim, progress
        raise NotImplementedError("Scenario execution progress is not supported.")

    async def update_scenario_trial(
        self,
        claim: EvalRunClaim,
        trial: EvalScenarioTrialProgress,
    ) -> EvalRunRecord:
        """Fenced update of one trial inside the current scenario attempt."""

        del claim, trial
        raise NotImplementedError("Scenario execution progress is not supported.")

    async def load_trial_checkpoints(
        self,
        claim: EvalRunClaim,
    ) -> tuple[EvalRunTrialCheckpoint, ...]:
        """Load terminal trial slots only while the exact run claim remains live."""

        del claim
        raise NotImplementedError("Durable eval trial checkpoints are not supported.")

    async def save_trial_checkpoint(
        self,
        claim: EvalRunClaim,
        checkpoint: EvalRunTrialCheckpoint,
        *,
        redact_json: Callable[[Any], Any],
    ) -> None:
        """Idempotently retain one terminal slot behind the live publication fence."""

        del claim, checkpoint, redact_json
        raise NotImplementedError("Durable eval trial checkpoints are not supported.")

    async def submit_scenario_approval(
        self,
        run_id: str,
        submission: EvalScenarioApprovalSubmission,
    ) -> EvalRunRecord:
        """CAS one fresh operator decision into the exact pending checkpoint."""

        del run_id, submission
        raise NotImplementedError("Scenario approval submission is not supported.")

    @abstractmethod
    async def publish_result(
        self,
        claim: EvalRunClaim,
        result: CorpusExecutionResult,
        *,
        redact_json: Callable[[Any], Any],
    ) -> EvalRunRecord:
        """Atomically publish exactly one immutable safe terminal result."""

    @abstractmethod
    async def fail_run(
        self,
        claim: EvalRunClaim,
        code: EvalRunFailureCode,
    ) -> EvalRunRecord:
        """Terminalize owned work with a closed, credential-free diagnostic."""

    @abstractmethod
    async def finish_cancel(self, claim: EvalRunClaim) -> EvalRunRecord:
        """Acknowledge cancellation after the owner has stopped execution."""

    @abstractmethod
    async def release_run(self, claim: EvalRunClaim) -> EvalRunRecord:
        """Release stopped owned work for retry, or finish requested cancellation."""

    @abstractmethod
    async def load_result(
        self,
        run_id: str,
        *,
        max_bytes: int = CORPUS_EXECUTION_RESULT_MAX_BYTES,
    ) -> CorpusExecutionResult | None:
        """Load one immutable safe result under an exact byte ceiling."""

    async def save_captured_result(
        self,
        corpus: EvalCorpusDocument,
        result: CapturedEvaluationResultV1,
        *,
        redact_json: Callable[[Any], Any],
    ) -> EvalResultRecord:
        """Atomically save one immutable corpus and its captured result.

        Existing custom V1 stores remain constructible. They opt into the new
        contract by overriding this method and ``captured_results``.
        """

        del corpus, result, redact_json
        raise NotImplementedError("Captured eval result persistence is not supported.")

    async def load_result_by_revision(
        self,
        revision: str,
        *,
        max_bytes: int = CORPUS_EXECUTION_RESULT_MAX_BYTES,
    ) -> CorpusExecutionResult | CapturedEvaluationResultV1 | None:
        """Load one immutable result independent of its captured/fresh origin."""

        del revision, max_bytes
        raise NotImplementedError("Origin-aware eval result reads are not supported.")

    async def load_result_record(self, revision: str) -> EvalResultRecord | None:
        """Load bounded metadata for one immutable result revision."""

        del revision
        raise NotImplementedError("Origin-aware eval result reads are not supported.")

    async def list_results(self, query: EvalResultQuery) -> EvalResultPage:
        """List target-scoped immutable result metadata in keyset order."""

        del query
        raise NotImplementedError("Origin-aware eval result catalogs are not supported.")

    async def set_baseline(
        self,
        update: EvalBaselineUpdate,
        *,
        redact_json: Callable[[Any], Any],
    ) -> EvalBaselineMutationRecord:
        """Commit or replay one actor-attributed baseline CAS mutation."""

        del update, redact_json
        raise NotImplementedError("Eval baseline persistence is not supported.")

    async def load_baseline(self, key: EvalBaselineKey) -> EvalBaselineRecord | None:
        """Load the current baseline pointer for one exact evaluation scope."""

        del key
        raise NotImplementedError("Eval baseline persistence is not supported.")

    async def load_baseline_mutation(
        self,
        operation_id: str,
    ) -> EvalBaselineMutationRecord | None:
        """Load one immutable baseline audit fact by its idempotency digest."""

        del operation_id
        raise NotImplementedError("Eval baseline persistence is not supported.")


@dataclass
class _MemoryRunState:
    request: EvalRunRequest
    status: EvalRunStatus
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    claim_id: str | None = None
    epoch: int = 0
    lease_expires_at: datetime | None = None
    result: CorpusExecutionResult | None = None
    failure_code: EvalRunFailureCode | None = None
    scenario_progress: EvalScenarioRunProgress | None = None
    trial_checkpoints: dict[tuple[str, int], EvalRunTrialCheckpoint] = field(default_factory=dict)
    trial_checkpoint_bytes: int = 0


@dataclass
class _MemoryEvalResultState:
    record: EvalResultRecord
    document: bytes


def _authored_suite_launch_predecessor_exists(
    candidate: _MemoryRunState,
    states: Iterable[_MemoryRunState],
) -> bool:
    launch_revision = candidate.request.invocation.authored_suite_launch_revision
    if launch_revision is None:
        return False
    candidate_order = (candidate.created_at, candidate.request.run_id)
    return any(
        state.request.invocation.authored_suite_launch_revision == launch_revision
        and state.request.invocation.authored_suite_launch_lane
        == candidate.request.invocation.authored_suite_launch_lane
        and state.status not in TERMINAL_EVAL_RUN_STATUSES
        and (state.created_at, state.request.run_id) < candidate_order
        for state in states
    )


class InMemoryEvalStore(EvalStore):
    """Explicitly ephemeral EvalStore for tests and process-local SDK use."""

    durable: ClassVar[bool] = False
    captured_results: ClassVar[bool] = True
    scenarios: ClassVar[bool] = True
    scenario_execution: ClassVar[bool] = True
    trial_checkpointing: ClassVar[bool] = True
    suite_authoring: ClassVar[bool] = True
    judge_calibrations: ClassVar[bool] = True

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
        claim_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._clock = clock or (lambda: datetime.now(UTC))
        self._claim_id_factory = claim_id_factory or (lambda: str(uuid4()))
        self._lock = asyncio.Lock()
        self._corpus_documents: dict[str, bytes] = {}
        self._corpora: dict[str, EvalCorpusCatalogEntry] = {}
        self._scenario_documents: dict[str, bytes] = {}
        self._scenarios: dict[str, EvalScenarioCatalogEntry] = {}
        self._authored_suite_documents: dict[str, bytes] = {}
        self._authored_suites: dict[str, EvalAuthoredSuiteCatalogEntry] = {}
        self._judge_calibration_documents: dict[str, bytes] = {}
        self._judge_calibration_revisions_by_run_id: dict[str, str] = {}
        self._suites: dict[str, tuple[EvalSuiteCatalogEntry, ...]] = {}
        self._cases: dict[str, tuple[EvalCaseCatalogEntry, ...]] = {}
        self._runs: dict[str, _MemoryRunState] = {}
        self._run_ids_by_idempotency_key: dict[str, str] = {}
        self._results: dict[str, _MemoryEvalResultState] = {}
        self._baselines: dict[tuple[str, str, str], EvalBaselineRecord] = {}
        self._baseline_mutations: dict[str, EvalBaselineMutationRecord] = {}

    def _now(self) -> datetime:
        return _aware_utc(self._clock(), "clock result")

    async def close(self) -> None:
        return None

    def _save_prepared_corpus(
        self,
        corpus: EvalCorpusDocument,
        document: bytes,
        suites: tuple[EvalSuiteCatalogEntry, ...],
        cases: tuple[EvalCaseCatalogEntry, ...],
        *,
        created_at: datetime,
    ) -> EvalCorpusCatalogEntry:
        existing = self._corpus_documents.get(corpus.revision)
        if existing is not None:
            if existing != document:
                raise EvalCorpusConflict(
                    f"Eval corpus revision {corpus.revision} has conflicting content."
                )
            return self._corpora[corpus.revision].model_copy(deep=True)
        entry = corpus_catalog_entry(
            corpus,
            created_at=created_at,
            document_bytes=len(document),
        )
        self._corpus_documents[corpus.revision] = document
        self._corpora[corpus.revision] = entry
        self._suites[corpus.revision] = suites
        self._cases[corpus.revision] = cases
        return entry.model_copy(deep=True)

    async def save_corpus(
        self,
        corpus: EvalCorpusDocument,
        *,
        redact_json: Callable[[Any], Any],
    ) -> EvalCorpusCatalogEntry:
        validated, document = _prepare_corpus_for_store(
            corpus,
            redact_json=redact_json,
        )
        suites = tuple(sorted(suite_catalog_entries(validated), key=lambda item: item.id))
        cases = tuple(sorted(case_catalog_entries(validated), key=lambda item: item.id))
        async with self._lock:
            return self._save_prepared_corpus(
                validated,
                document,
                suites,
                cases,
                created_at=self._now(),
            )

    async def load_corpus(
        self,
        revision: str,
        *,
        max_bytes: int = EVAL_CORPUS_MAX_BYTES,
    ) -> EvalCorpusDocument | None:
        revision = _sha256_revision(revision, "revision")
        max_bytes = _read_limit(max_bytes, hard_max=EVAL_CORPUS_MAX_BYTES)
        async with self._lock:
            document = self._corpus_documents.get(revision)
            if document is None:
                return None
            if len(document) > max_bytes:
                raise EvalStoreResultTooLarge(max_bytes)
            return EvalCorpusDocument.model_validate(json.loads(document))

    async def list_corpora(
        self,
        query: EvalCatalogQuery | None = None,
    ) -> EvalCorpusCatalogPage:
        query = _copy_query(query, EvalCatalogQuery)
        boundary = (
            decode_corpus_cursor(query.cursor, query.target_key)
            if query.cursor is not None
            else None
        )
        async with self._lock:
            items = [
                item
                for item in self._corpora.values()
                if query.target_key is None or item.target_key == query.target_key
            ]
            items.sort(key=lambda item: item.revision)
            items.sort(key=lambda item: item.created_at, reverse=True)
            if boundary is not None:
                created_at, revision = boundary
                items = [
                    item
                    for item in items
                    if item.created_at < created_at
                    or (item.created_at == created_at and item.revision > revision)
                ]
            return _bounded_corpus_page(items, query)

    async def save_scenario(
        self,
        scenario: EvalScenarioDocumentV2,
        *,
        redact_json: Callable[[Any], Any],
    ) -> EvalScenarioCatalogEntry:
        validated, document, _ = _prepare_scenario_catalog_for_store(
            scenario,
            redact_json=redact_json,
        )
        async with self._lock:
            existing = self._scenario_documents.get(validated.revision)
            if existing is not None:
                if existing != document:
                    raise EvalScenarioConflict(
                        f"Eval scenario revision {validated.revision} has conflicting content."
                    )
                return self._scenarios[validated.revision].model_copy(deep=True)
            entry = scenario_catalog_entry(
                validated,
                created_at=self._now(),
                document_bytes=len(document),
            )
            self._scenario_documents[validated.revision] = document
            self._scenarios[validated.revision] = entry
            return entry.model_copy(deep=True)

    async def load_scenario(
        self,
        revision: str,
        *,
        max_bytes: int = EVAL_SCENARIO_MAX_BYTES,
    ) -> EvalScenarioDocumentV2 | None:
        revision = _sha256_revision(revision, "revision")
        max_bytes = _read_limit(max_bytes, hard_max=EVAL_SCENARIO_MAX_BYTES)
        async with self._lock:
            document = self._scenario_documents.get(revision)
            if document is None:
                return None
            if len(document) > max_bytes:
                raise EvalStoreResultTooLarge(max_bytes)
            return eval_scenario_from_json(document.decode("utf-8"))

    async def list_scenarios(
        self,
        query: EvalScenarioCatalogQuery | None = None,
    ) -> EvalScenarioCatalogPage:
        query = _copy_query(query, EvalScenarioCatalogQuery)
        boundary = (
            decode_scenario_cursor(query.cursor, query.target_key, query.scenario_id)
            if query.cursor is not None
            else None
        )
        async with self._lock:
            items = [
                item
                for item in self._scenarios.values()
                if (query.target_key is None or item.target_key == query.target_key)
                and (query.scenario_id is None or item.id == query.scenario_id)
            ]
            items.sort(key=lambda item: item.revision)
            items.sort(key=lambda item: item.created_at, reverse=True)
            if boundary is not None:
                created_at, revision = boundary
                items = [
                    item
                    for item in items
                    if item.created_at < created_at
                    or (item.created_at == created_at and item.revision > revision)
                ]
            return _bounded_scenario_page(items, query)

    async def save_authored_suite(
        self,
        document: EvalSuiteDocument,
        *,
        redact_json: Callable[[Any], Any],
    ) -> EvalAuthoredSuiteCatalogEntry:
        validated, payload = _prepare_authored_suite_for_store(
            document,
            redact_json=redact_json,
        )
        async with self._lock:
            scenario_cases = authored_suite_scenario_cases(validated)
            scenario_by_revision: dict[str, EvalScenarioDocumentV2] = {}
            revisions = {reference.scenario_revision for _, reference in scenario_cases}
            for revision in revisions:
                scenario_payload = self._scenario_documents.get(revision)
                if scenario_payload is not None:
                    scenario_by_revision[revision] = eval_scenario_from_json(
                        scenario_payload.decode("utf-8")
                    )
            for case, reference in scenario_cases:
                validate_authored_suite_scenario(
                    validated,
                    case,
                    scenario_by_revision.get(reference.scenario_revision),
                )
            existing = self._authored_suite_documents.get(validated.revision)
            if existing is not None:
                if existing != payload:
                    raise EvalAuthoredSuiteConflict(
                        f"Authored eval suite revision {validated.revision} has "
                        "conflicting content."
                    )
                return self._authored_suites[validated.revision].model_copy(deep=True)
            entry = authored_suite_catalog_entry(
                validated,
                created_at=self._now(),
                document_bytes=len(payload),
            )
            self._authored_suite_documents[validated.revision] = payload
            self._authored_suites[validated.revision] = entry
            return entry.model_copy(deep=True)

    async def load_authored_suite(
        self,
        revision: str,
        *,
        max_bytes: int = EVAL_SUITE_AUTHORING_MAX_BYTES,
    ) -> EvalSuiteDocument | None:
        revision = _sha256_revision(revision, "revision")
        max_bytes = _read_limit(max_bytes, hard_max=EVAL_SUITE_AUTHORING_MAX_BYTES)
        async with self._lock:
            payload = self._authored_suite_documents.get(revision)
            if payload is None:
                return None
            if len(payload) > max_bytes:
                raise EvalStoreResultTooLarge(max_bytes)
            return eval_suite_document_from_json(payload.decode("utf-8"))

    async def list_authored_suites(
        self,
        query: EvalAuthoredSuiteCatalogQuery | None = None,
    ) -> EvalAuthoredSuiteCatalogPage:
        query = _copy_query(query, EvalAuthoredSuiteCatalogQuery)
        boundary = (
            decode_authored_suite_cursor(
                query.cursor,
                query.target_key,
                query.suite_id,
            )
            if query.cursor is not None
            else None
        )
        async with self._lock:
            items = [
                item
                for item in self._authored_suites.values()
                if (query.target_key is None or item.target_key == query.target_key)
                and (query.suite_id is None or item.id == query.suite_id)
            ]
            items.sort(key=lambda item: item.revision)
            items.sort(key=lambda item: item.created_at, reverse=True)
            if boundary is not None:
                created_at, revision = boundary
                items = [
                    item
                    for item in items
                    if item.created_at < created_at
                    or (item.created_at == created_at and item.revision > revision)
                ]
            return _bounded_authored_suite_page(items, query)

    async def save_judge_calibration(
        self,
        report: EvalJudgeCalibrationReportV1,
        *,
        redact_json: Callable[[Any], Any],
    ) -> EvalJudgeCalibrationReportV1:
        validated, payload = _prepare_judge_calibration_for_store(
            report,
            redact_json=redact_json,
        )
        async with self._lock:
            existing = self._judge_calibration_documents.get(validated.revision)
            run_revision = self._judge_calibration_revisions_by_run_id.get(validated.run_id)
            if existing is not None:
                if existing != payload or run_revision != validated.revision:
                    raise EvalJudgeCalibrationConflict(
                        "Judge calibration revision or run ID has conflicting content."
                    )
                return eval_judge_calibration_report_from_json(existing.decode("utf-8"))
            if run_revision is not None:
                raise EvalJudgeCalibrationConflict(
                    "Judge calibration revision or run ID has conflicting content."
                )
            self._judge_calibration_documents[validated.revision] = payload
            self._judge_calibration_revisions_by_run_id[validated.run_id] = validated.revision
            return validated.model_copy(deep=True)

    async def load_judge_calibration(
        self,
        revision: str,
        *,
        max_bytes: int = EVAL_JUDGE_CALIBRATION_MAX_BYTES,
    ) -> EvalJudgeCalibrationReportV1 | None:
        revision = _sha256_revision(revision, "revision")
        max_bytes = _read_limit(max_bytes, hard_max=EVAL_JUDGE_CALIBRATION_MAX_BYTES)
        async with self._lock:
            payload = self._judge_calibration_documents.get(revision)
            if payload is None:
                return None
            if len(payload) > max_bytes:
                raise EvalStoreResultTooLarge(max_bytes)
            return eval_judge_calibration_report_from_json(payload.decode("utf-8"))

    async def load_judge_calibration_by_run_id(
        self,
        run_id: str,
        *,
        max_bytes: int = EVAL_JUDGE_CALIBRATION_MAX_BYTES,
    ) -> EvalJudgeCalibrationReportV1 | None:
        run_id = _portable_id(run_id, "run_id")
        max_bytes = _read_limit(max_bytes, hard_max=EVAL_JUDGE_CALIBRATION_MAX_BYTES)
        async with self._lock:
            revision = self._judge_calibration_revisions_by_run_id.get(run_id)
            payload = None if revision is None else self._judge_calibration_documents.get(revision)
            if payload is None:
                return None
            if len(payload) > max_bytes:
                raise EvalStoreResultTooLarge(max_bytes)
            return eval_judge_calibration_report_from_json(payload.decode("utf-8"))

    async def list_suites(self, query: EvalSuiteCatalogQuery) -> EvalSuiteCatalogPage:
        query = _exact_model(query, EvalSuiteCatalogQuery, "query")
        boundary = (
            decode_suite_cursor(query.cursor, query.corpus_revision)
            if query.cursor is not None
            else None
        )
        async with self._lock:
            if query.corpus_revision not in self._corpus_documents:
                raise KeyError(f"Eval corpus not found: {query.corpus_revision}")
            items = [
                item
                for item in self._suites[query.corpus_revision]
                if boundary is None or item.id > boundary
            ]
            return _bounded_suite_page(items, query)

    async def list_cases(self, query: EvalCaseCatalogQuery) -> EvalCaseCatalogPage:
        query = _copy_query(query, EvalCaseCatalogQuery)
        boundary = (
            decode_case_cursor(query.cursor, query.corpus_revision, query.suite_id)
            if query.cursor is not None
            else None
        )
        async with self._lock:
            if query.corpus_revision not in self._corpus_documents:
                raise KeyError(f"Eval corpus not found: {query.corpus_revision}")
            if not any(suite.id == query.suite_id for suite in self._suites[query.corpus_revision]):
                raise KeyError(f"Eval suite not found: {query.suite_id}")
            items = [
                item
                for item in self._cases[query.corpus_revision]
                if item.suite_id == query.suite_id and (boundary is None or item.id > boundary)
            ]
            return _bounded_case_page(items, query)

    async def admit_run(
        self,
        request: EvalRunRequest,
        *,
        redact_json: Callable[[Any], Any],
    ) -> EvalRunRecord:
        request = _prepare_run_request_for_store(
            request,
            redact_json=redact_json,
        )
        async with self._lock:
            duplicate_id = self._run_ids_by_idempotency_key.get(request.idempotency_key)
            if duplicate_id is not None:
                existing = self._runs[duplicate_id]
                if not existing.request.same_logical_request(request):
                    raise EvalRunAdmissionConflict(
                        "Eval run idempotency key is already bound to another request."
                    )
                return self._record(existing)
            if request.run_id in self._runs:
                raise EvalRunAdmissionConflict(
                    f"Eval run id is already bound to another request: {request.run_id}"
                )
            document = self._corpus_documents.get(request.corpus_revision)
            if document is None:
                raise KeyError(f"Eval corpus not found: {request.corpus_revision}")
            validate_run_request_for_corpus(
                request,
                EvalCorpusDocument.model_validate(json.loads(document)),
            )
            now = self._now()
            state = _MemoryRunState(
                request=request,
                status=EvalRunStatus.QUEUED,
                created_at=now,
                updated_at=now,
            )
            self._runs[request.run_id] = state
            self._run_ids_by_idempotency_key[request.idempotency_key] = request.run_id
            return self._record(state)

    async def load_run_by_idempotency_key(
        self,
        idempotency_key: str,
    ) -> EvalRunRecord | None:
        idempotency_key = _idempotency_key(idempotency_key, "idempotency_key")
        async with self._lock:
            run_id = self._run_ids_by_idempotency_key.get(idempotency_key)
            return None if run_id is None else self._record(self._runs[run_id])

    async def load_run(self, run_id: str) -> EvalRunRecord | None:
        run_id = _store_identifier(run_id, "run_id")
        started_at = monotonic()
        try:
            async with self._lock:
                state = self._runs.get(run_id)
                return None if state is None else self._record(state)
        finally:
            logger.debug(
                "In-memory eval run fully rehydrated.",
                extra={
                    "cayu_eval_store_event": "full_run_rehydration",
                    "eval_store_kind": "memory",
                    "eval_run_id": run_id,
                    "duration_seconds": monotonic() - started_at,
                },
            )

    async def load_run_observation(self, run_id: str) -> EvalRunObservation | None:
        run_id = _store_identifier(run_id, "run_id")
        started_at = monotonic()
        try:
            async with self._lock:
                state = self._runs.get(run_id)
                return None if state is None else self._observation(state)
        finally:
            logger.debug(
                "In-memory eval run status observed.",
                extra={
                    "cayu_eval_store_event": "run_status_read",
                    "eval_store_kind": "memory",
                    "eval_run_id": run_id,
                    "duration_seconds": monotonic() - started_at,
                },
            )

    async def list_runs(self, query: EvalRunQuery | None = None) -> EvalRunPage:
        query = _copy_query(query, EvalRunQuery)
        boundary = (
            decode_run_cursor(
                query.cursor,
                query.target_key,
                query.status,
                query.corpus_revision,
            )
            if query.cursor is not None
            else None
        )
        async with self._lock:
            records = [
                self._record(state)
                for state in self._runs.values()
                if (query.status is None or state.status is query.status)
                and (query.target_key is None or state.request.target_key == query.target_key)
                and (
                    query.corpus_revision is None
                    or state.request.corpus_revision == query.corpus_revision
                )
            ]
            records.sort(key=lambda item: item.id)
            records.sort(key=lambda item: item.created_at, reverse=True)
            if boundary is not None:
                created_at, run_id = boundary
                records = [
                    item
                    for item in records
                    if item.created_at < created_at
                    or (item.created_at == created_at and item.id > run_id)
                ]
            return _bounded_run_page(records, query)

    async def claim_run(
        self,
        *,
        target_key: str | None = None,
        lease_seconds: int = 300,
    ) -> EvalRunLease | None:
        if target_key is not None:
            target_key = _portable_id(target_key, "target_key")
        lease_seconds = _lease_seconds(lease_seconds)
        async with self._lock:
            now = self._now()
            eligible = [
                state
                for state in self._runs.values()
                if (target_key is None or state.request.target_key == target_key)
                and state.epoch < _EVAL_STORE_MAX_BIGINT
                and (
                    state.status is EvalRunStatus.QUEUED
                    or (
                        state.status in {EvalRunStatus.RUNNING, EvalRunStatus.CANCELLING}
                        and state.lease_expires_at is not None
                        and state.lease_expires_at <= now
                    )
                )
                and not _authored_suite_launch_predecessor_exists(
                    state,
                    self._runs.values(),
                )
            ]
            if not eligible:
                return None
            state = min(eligible, key=lambda item: (item.created_at, item.request.run_id))
            state.status = (
                EvalRunStatus.CANCELLING
                if state.cancel_requested_at is not None
                else EvalRunStatus.RUNNING
            )
            state.started_at = state.started_at or now
            state.updated_at = now
            state.claim_id = _store_identifier(self._claim_id_factory(), "claim_id")
            state.epoch += 1
            state.lease_expires_at = now + timedelta(seconds=lease_seconds)
            state.scenario_progress = _scenario_progress_for_claim(
                state.scenario_progress,
                scenario=state.request.invocation.scenario,
                attempt=state.epoch,
                terminal_trial_numbers=frozenset(
                    trial_number for _case_id, trial_number in state.trial_checkpoints
                ),
            )
            record = self._record(state)
            return EvalRunLease(
                run=record,
                claim=EvalRunClaim(
                    run_id=record.id,
                    claim_id=state.claim_id,
                    epoch=state.epoch,
                ),
            )

    async def claim_run_for_targets(
        self,
        target_keys: tuple[str, ...],
        *,
        lease_seconds: int = 300,
    ) -> EvalRunLease | None:
        target_keys = _claim_target_keys(target_keys)
        lease_seconds = _lease_seconds(lease_seconds)
        approved = frozenset(target_keys)
        async with self._lock:
            now = self._now()
            eligible = [
                state
                for state in self._runs.values()
                if state.request.target_key in approved
                and state.epoch < _EVAL_STORE_MAX_BIGINT
                and (
                    state.status is EvalRunStatus.QUEUED
                    or (
                        state.status in {EvalRunStatus.RUNNING, EvalRunStatus.CANCELLING}
                        and state.lease_expires_at is not None
                        and state.lease_expires_at <= now
                    )
                )
                and not _authored_suite_launch_predecessor_exists(
                    state,
                    self._runs.values(),
                )
            ]
            if not eligible:
                return None
            state = min(eligible, key=lambda item: (item.created_at, item.request.run_id))
            state.status = (
                EvalRunStatus.CANCELLING
                if state.cancel_requested_at is not None
                else EvalRunStatus.RUNNING
            )
            state.started_at = state.started_at or now
            state.updated_at = now
            state.claim_id = _store_identifier(self._claim_id_factory(), "claim_id")
            state.epoch += 1
            state.lease_expires_at = now + timedelta(seconds=lease_seconds)
            state.scenario_progress = _scenario_progress_for_claim(
                state.scenario_progress,
                scenario=state.request.invocation.scenario,
                attempt=state.epoch,
                terminal_trial_numbers=frozenset(
                    trial_number for _case_id, trial_number in state.trial_checkpoints
                ),
            )
            record = self._record(state)
            return EvalRunLease(
                run=record,
                claim=EvalRunClaim(
                    run_id=record.id,
                    claim_id=state.claim_id,
                    epoch=state.epoch,
                ),
            )

    async def heartbeat_run(
        self,
        claim: EvalRunClaim,
        *,
        extend_seconds: int = 300,
    ) -> EvalRunRecord:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        extend_seconds = _lease_seconds(extend_seconds)
        async with self._lock:
            state = self._require_live_claim(claim)
            now = self._now()
            state.lease_expires_at = now + timedelta(seconds=extend_seconds)
            state.updated_at = now
            return self._record(state)

    async def heartbeat_run_observation(
        self,
        claim: EvalRunClaim,
        *,
        extend_seconds: int = 300,
    ) -> EvalRunObservation:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        extend_seconds = _lease_seconds(extend_seconds)
        async with self._lock:
            state = self._require_live_claim(claim)
            now = self._now()
            state.lease_expires_at = now + timedelta(seconds=extend_seconds)
            state.updated_at = now
            return self._observation(state)

    async def request_cancel(self, run_id: str) -> EvalRunRecord:
        run_id = _store_identifier(run_id, "run_id")
        async with self._lock:
            state = self._require_run(run_id)
            if state.status in TERMINAL_EVAL_RUN_STATUSES:
                return self._record(state)
            now = self._now()
            state.cancel_requested_at = state.cancel_requested_at or now
            state.updated_at = now
            claim_expired = (
                state.status in {EvalRunStatus.RUNNING, EvalRunStatus.CANCELLING}
                and state.lease_expires_at is not None
                and state.lease_expires_at <= now
            )
            if state.status is EvalRunStatus.QUEUED or claim_expired:
                state.status = EvalRunStatus.CANCELLED
                state.finished_at = now
                state.claim_id = None
                state.lease_expires_at = None
                state.trial_checkpoints.clear()
                state.trial_checkpoint_bytes = 0
            else:
                state.status = EvalRunStatus.CANCELLING
            return self._record(state)

    async def initialize_scenario_progress(
        self,
        claim: EvalRunClaim,
        progress: EvalScenarioRunProgress,
    ) -> EvalRunRecord:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        progress = _exact_model(progress, EvalScenarioRunProgress, "progress")
        async with self._lock:
            state = self._require_live_claim(claim)
            scenario = state.request.invocation.scenario
            if scenario is None:
                raise EvalRunStateConflict("Only scenario runs may initialize scenario progress.")
            if (
                progress.attempt != claim.epoch
                or progress.scenario_revision != scenario.scenario_revision
                or progress.binding_revision != scenario.binding_revision
                or len(progress.trials) != scenario.trials
            ):
                raise EvalRunStateConflict("Scenario progress does not match the claimed run.")
            state.scenario_progress = progress.model_copy(deep=True)
            state.updated_at = self._now()
            return self._record(state)

    async def load_trial_checkpoints(
        self,
        claim: EvalRunClaim,
    ) -> tuple[EvalRunTrialCheckpoint, ...]:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        async with self._lock:
            state = self._require_live_claim(claim)
            return _validated_trial_checkpoints(
                tuple(state.trial_checkpoints[key] for key in sorted(state.trial_checkpoints)),
                expected_document_bytes=state.trial_checkpoint_bytes,
            )

    async def save_trial_checkpoint(
        self,
        claim: EvalRunClaim,
        checkpoint: EvalRunTrialCheckpoint,
        *,
        redact_json: Callable[[Any], Any],
    ) -> None:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        prepared = await asyncio.to_thread(
            _prepare_trial_checkpoint_for_store,
            checkpoint,
            redact_json=redact_json,
        )
        async with self._lock:
            state = self._require_live_claim(claim)
            corpus = EvalCorpusDocument.model_validate(
                json.loads(self._corpus_documents[state.request.corpus_revision])
            )
            checkpoint = _validate_trial_checkpoint_for_run(
                prepared.checkpoint,
                state.request,
                corpus,
            )
            key = (checkpoint.case_id, checkpoint.trial_number)
            current = state.trial_checkpoints.get(key)
            if current is not None:
                if current == checkpoint:
                    return
                raise EvalRunStateConflict("Eval trial slot already has another terminal result.")
            if len(state.trial_checkpoints) >= EVAL_RUN_TRIAL_CHECKPOINTS_MAX_ITEMS:
                raise ValueError("Eval run trial checkpoints exceed their item limit.")
            if (
                state.trial_checkpoint_bytes + prepared.document_bytes
                > EVAL_RUN_TRIAL_CHECKPOINTS_MAX_BYTES
            ):
                raise ValueError("Eval run trial checkpoints exceed their byte limit.")
            state.trial_checkpoints[key] = checkpoint
            state.trial_checkpoint_bytes += prepared.document_bytes
            state.updated_at = self._now()

    async def update_scenario_trial(
        self,
        claim: EvalRunClaim,
        trial: EvalScenarioTrialProgress,
    ) -> EvalRunRecord:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        trial = _exact_model(trial, EvalScenarioTrialProgress, "trial")
        async with self._lock:
            state = self._require_live_claim(claim)
            progress = state.scenario_progress
            if progress is None or progress.attempt != claim.epoch:
                raise EvalRunStateConflict("Scenario progress is absent for this claim.")
            state.scenario_progress = progress.replace_trial(trial)
            state.updated_at = self._now()
            return self._record(state)

    async def submit_scenario_approval(
        self,
        run_id: str,
        submission: EvalScenarioApprovalSubmission,
    ) -> EvalRunRecord:
        run_id = _store_identifier(run_id, "run_id")
        submission = _exact_model(submission, EvalScenarioApprovalSubmission, "submission")
        async with self._lock:
            state = self._require_run(run_id)
            if state.status is not EvalRunStatus.RUNNING:
                raise EvalRunStateConflict("Scenario approval requires an active run.")
            progress = state.scenario_progress
            if progress is None or progress.revision != submission.expected_progress_revision:
                raise EvalRunStateConflict("Scenario progress changed before approval submission.")
            if submission.trial_number > len(progress.trials):
                raise EvalRunStateConflict("Scenario trial does not exist.")
            trial = progress.trials[submission.trial_number - 1]
            if (
                trial.phase is not EvalScenarioTrialPhase.AWAITING_APPROVAL
                or trial.pending_event_id != submission.event_id
                or trial.approval is not None
            ):
                raise EvalRunStateConflict("Scenario approval checkpoint is no longer pending.")
            updated_trial = trial.model_copy(
                update={
                    "approval": EvalScenarioApprovalDecisionRecord(
                        decision=submission.decision,
                        reason=submission.reason,
                        actor_id=submission.actor_id,
                        submitted_at=self._now(),
                    )
                },
                deep=True,
            )
            state.scenario_progress = progress.replace_trial(updated_trial)
            state.updated_at = self._now()
            return self._record(state)

    async def publish_result(
        self,
        claim: EvalRunClaim,
        result: CorpusExecutionResult,
        *,
        redact_json: Callable[[Any], Any],
    ) -> EvalRunRecord:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        validated_result, result_document = _prepare_result_for_store(
            result,
            redact_json=redact_json,
        )
        async with self._lock:
            state = self._require_run(claim.run_id)
            corpus = EvalCorpusDocument.model_validate(
                json.loads(self._corpus_documents[state.request.corpus_revision])
            )
            validated_result = validate_result_for_run(
                state.request,
                validated_result,
                corpus,
            )
            if state.status is EvalRunStatus.COMPLETED:
                if (
                    self._historical_claim_matches(state, claim)
                    and state.result == validated_result
                ):
                    return self._record(state)
                raise EvalRunStateConflict("Eval run already has another terminal result.")
            state = self._require_live_claim(claim)
            if state.status is not EvalRunStatus.RUNNING:
                raise EvalRunStateConflict("Only a running eval may publish a result.")
            _validate_trial_checkpoints_for_result(
                tuple(state.trial_checkpoints[key] for key in sorted(state.trial_checkpoints)),
                validated_result,
            )
            now = self._now()
            self._save_result_document(
                validated_result,
                result_document,
                created_at=now,
            )
            state.status = EvalRunStatus.COMPLETED
            state.result = validated_result
            state.trial_checkpoints.clear()
            state.trial_checkpoint_bytes = 0
            state.updated_at = now
            state.finished_at = now
            state.lease_expires_at = None
            return self._record(state)

    async def fail_run(
        self,
        claim: EvalRunClaim,
        code: EvalRunFailureCode,
    ) -> EvalRunRecord:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        if not isinstance(code, EvalRunFailureCode):
            raise TypeError("code must be an EvalRunFailureCode.")
        async with self._lock:
            state = self._require_run(claim.run_id)
            if state.status is EvalRunStatus.FAILED:
                if self._historical_claim_matches(state, claim) and state.failure_code is code:
                    return self._record(state)
                raise EvalRunStateConflict("Eval run already failed with another outcome.")
            state = self._require_live_claim(claim)
            if state.status is not EvalRunStatus.RUNNING:
                raise EvalRunStateConflict("Only a running eval may fail.")
            now = self._now()
            state.status = EvalRunStatus.FAILED
            state.failure_code = code
            state.trial_checkpoints.clear()
            state.trial_checkpoint_bytes = 0
            state.updated_at = now
            state.finished_at = now
            state.lease_expires_at = None
            return self._record(state)

    async def finish_cancel(self, claim: EvalRunClaim) -> EvalRunRecord:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        async with self._lock:
            state = self._require_run(claim.run_id)
            if state.status is EvalRunStatus.CANCELLED:
                if self._historical_claim_matches(state, claim):
                    return self._record(state)
                raise EvalRunStateConflict("Eval run was cancelled by another transition.")
            state = self._require_live_claim(claim)
            if state.status is not EvalRunStatus.CANCELLING:
                raise EvalRunStateConflict("Eval run has no cancellation intent to finish.")
            self._cancel_owned(state)
            return self._record(state)

    async def release_run(self, claim: EvalRunClaim) -> EvalRunRecord:
        claim = _exact_model(claim, EvalRunClaim, "claim")
        async with self._lock:
            state = self._require_live_claim(claim)
            if state.status is EvalRunStatus.CANCELLING:
                self._cancel_owned(state)
                return self._record(state)
            if state.status is not EvalRunStatus.RUNNING:
                raise EvalRunStateConflict("Only active eval work may be released.")
            state.status = EvalRunStatus.QUEUED
            state.updated_at = self._now()
            state.claim_id = None
            state.lease_expires_at = None
            return self._record(state)

    async def load_result(
        self,
        run_id: str,
        *,
        max_bytes: int = CORPUS_EXECUTION_RESULT_MAX_BYTES,
    ) -> CorpusExecutionResult | None:
        run_id = _store_identifier(run_id, "run_id")
        max_bytes = _read_limit(max_bytes, hard_max=CORPUS_EXECUTION_RESULT_MAX_BYTES)
        async with self._lock:
            state = self._runs.get(run_id)
            if state is None or state.result is None:
                return None
            if len(_wire_model_bytes(state.result)) > max_bytes:
                raise EvalStoreResultTooLarge(max_bytes)
            return CorpusExecutionResult.model_validate(_model_python_input(state.result))

    async def save_captured_result(
        self,
        corpus: EvalCorpusDocument,
        result: CapturedEvaluationResultV1,
        *,
        redact_json: Callable[[Any], Any],
    ) -> EvalResultRecord:
        validated_corpus, corpus_document = _prepare_corpus_for_store(
            corpus,
            redact_json=redact_json,
        )
        suites = tuple(sorted(suite_catalog_entries(validated_corpus), key=lambda item: item.id))
        cases = tuple(sorted(case_catalog_entries(validated_corpus), key=lambda item: item.id))
        validated_result, result_document = _prepare_captured_result_for_store(
            result,
            validated_corpus,
            redact_json=redact_json,
        )
        prepared_record = eval_result_record(
            validated_result,
            document_bytes=len(result_document),
            created_at=self._now(),
        )
        async with self._lock:
            now = self._now()
            existing_corpus = self._corpus_documents.get(validated_corpus.revision)
            if existing_corpus is not None and existing_corpus != corpus_document:
                raise EvalCorpusConflict(
                    f"Eval corpus revision {validated_corpus.revision} has conflicting content."
                )
            existing_result = self._results.get(prepared_record.revision)
            if existing_result is not None and (
                existing_result.document != result_document
                or existing_result.record.origin is not prepared_record.origin
            ):
                raise EvalResultConflict(
                    f"Eval result revision {prepared_record.revision} has conflicting content."
                )
            self._save_prepared_corpus(
                validated_corpus,
                corpus_document,
                suites,
                cases,
                created_at=now,
            )
            return self._save_result_document(
                validated_result,
                result_document,
                created_at=now,
            )

    async def load_result_by_revision(
        self,
        revision: str,
        *,
        max_bytes: int = CORPUS_EXECUTION_RESULT_MAX_BYTES,
    ) -> CorpusExecutionResult | CapturedEvaluationResultV1 | None:
        revision = _sha256_revision(revision, "revision")
        max_bytes = _read_limit(max_bytes, hard_max=CORPUS_EXECUTION_RESULT_MAX_BYTES)
        async with self._lock:
            state = self._results.get(revision)
            if state is None:
                return None
            if len(state.document) > max_bytes:
                raise EvalStoreResultTooLarge(max_bytes)
            source = state.document.decode("utf-8")
            if state.record.origin is EvalResultOrigin.CAPTURED_SESSION:
                return captured_evaluation_result_from_json(source)
            return CorpusExecutionResult.model_validate_json(source)

    async def load_result_record(self, revision: str) -> EvalResultRecord | None:
        revision = _sha256_revision(revision, "revision")
        async with self._lock:
            state = self._results.get(revision)
            return None if state is None else state.record.model_copy(deep=True)

    async def list_results(self, query: EvalResultQuery) -> EvalResultPage:
        query = _exact_model(query, EvalResultQuery, "query")
        boundary = (
            decode_result_cursor(query.cursor, query.target_key, query.origin)
            if query.cursor is not None
            else None
        )
        async with self._lock:
            records = [
                state.record.model_copy(deep=True)
                for state in self._results.values()
                if state.record.target.target_key == query.target_key
                and (query.origin is None or state.record.origin is query.origin)
            ]
            records.sort(key=lambda item: item.revision)
            records.sort(key=lambda item: item.created_at, reverse=True)
            if boundary is not None:
                created_at, revision = boundary
                records = [
                    item
                    for item in records
                    if item.created_at < created_at
                    or (item.created_at == created_at and item.revision > revision)
                ]
            return _bounded_result_page(records, query)

    async def set_baseline(
        self,
        update: EvalBaselineUpdate,
        *,
        redact_json: Callable[[Any], Any],
    ) -> EvalBaselineMutationRecord:
        update = _prepare_baseline_update_for_store(update, redact_json=redact_json)
        async with self._lock:
            replay = self._baseline_mutations.get(update.operation_id)
            if replay is not None:
                if not self._baseline_mutation_matches(replay, update):
                    raise EvalBaselineConflict(
                        "Baseline operation id is already bound to another mutation."
                    )
                return replay.model_copy(deep=True)
            result_state = self._results.get(update.result_revision)
            if result_state is None:
                raise KeyError(f"Eval result not found: {update.result_revision}")
            _validate_baseline_result(update, result_state.record)
            scope = self._baseline_scope(update.key)
            current = self._baselines.get(scope)
            current_generation = 0 if current is None else current.generation
            if current_generation != update.expected_generation:
                raise EvalBaselineConflict("Eval baseline generation changed.")
            if current_generation >= _EVAL_STORE_MAX_BIGINT:
                raise EvalBaselineConflict("Eval baseline generation is exhausted.")
            now = self._now()
            mutation = EvalBaselineMutationRecord(
                operation_id=update.operation_id,
                key=update.key,
                expected_generation=update.expected_generation,
                previous_result_revision=(None if current is None else current.result_revision),
                selected_result_revision=update.result_revision,
                resulting_generation=current_generation + 1,
                actor_id=update.actor_id,
                created_at=now,
            )
            self._baselines[scope] = EvalBaselineRecord(
                key=update.key,
                result_revision=update.result_revision,
                generation=mutation.resulting_generation,
                updated_by=update.actor_id,
                updated_at=now,
            )
            self._baseline_mutations[update.operation_id] = mutation
            return mutation.model_copy(deep=True)

    async def load_baseline(self, key: EvalBaselineKey) -> EvalBaselineRecord | None:
        key = _exact_model(key, EvalBaselineKey, "key")
        async with self._lock:
            record = self._baselines.get(self._baseline_scope(key))
            return None if record is None else record.model_copy(deep=True)

    async def load_baseline_mutation(
        self,
        operation_id: str,
    ) -> EvalBaselineMutationRecord | None:
        operation_id = _sha256_revision(operation_id, "operation_id")
        async with self._lock:
            mutation = self._baseline_mutations.get(operation_id)
            return None if mutation is None else mutation.model_copy(deep=True)

    def _save_result_document(
        self,
        result: CorpusExecutionResult | CapturedEvaluationResultV1,
        document: bytes,
        *,
        created_at: datetime,
    ) -> EvalResultRecord:
        record = eval_result_record(
            result,
            document_bytes=len(document),
            created_at=created_at,
        )
        existing = self._results.get(record.revision)
        if existing is not None:
            if existing.document != document or existing.record.origin is not record.origin:
                raise EvalResultConflict(
                    f"Eval result revision {record.revision} has conflicting content."
                )
            return existing.record.model_copy(deep=True)
        self._results[record.revision] = _MemoryEvalResultState(
            record=record,
            document=bytes(document),
        )
        return record.model_copy(deep=True)

    @staticmethod
    def _baseline_scope(key: EvalBaselineKey) -> tuple[str, str, str]:
        return key.target_key, key.corpus_revision, key.suite_id

    @staticmethod
    def _baseline_mutation_matches(
        mutation: EvalBaselineMutationRecord,
        update: EvalBaselineUpdate,
    ) -> bool:
        return (
            mutation.operation_id == update.operation_id
            and mutation.key == update.key
            and mutation.expected_generation == update.expected_generation
            and mutation.selected_result_revision == update.result_revision
            and mutation.actor_id == update.actor_id
        )

    def _require_run(self, run_id: str) -> _MemoryRunState:
        state = self._runs.get(run_id)
        if state is None:
            raise KeyError(f"Eval run not found: {run_id}")
        return state

    def _require_live_claim(self, claim: EvalRunClaim) -> _MemoryRunState:
        state = self._require_run(claim.run_id)
        now = self._now()
        if not self._historical_claim_matches(state, claim):
            raise EvalRunClaimLost("Eval run claim is no longer owned by this worker.")
        if state.status not in {EvalRunStatus.RUNNING, EvalRunStatus.CANCELLING}:
            raise EvalRunClaimLost("Eval run is no longer active.")
        if state.lease_expires_at is None or state.lease_expires_at <= now:
            raise EvalRunClaimLost("Eval run claim lease has expired.")
        return state

    @staticmethod
    def _historical_claim_matches(state: _MemoryRunState, claim: EvalRunClaim) -> bool:
        return state.claim_id == claim.claim_id and state.epoch == claim.epoch

    def _cancel_owned(self, state: _MemoryRunState) -> None:
        now = self._now()
        state.status = EvalRunStatus.CANCELLED
        state.cancel_requested_at = state.cancel_requested_at or now
        state.updated_at = now
        state.finished_at = now
        state.lease_expires_at = None
        state.trial_checkpoints.clear()
        state.trial_checkpoint_bytes = 0

    @staticmethod
    def _record(state: _MemoryRunState) -> EvalRunRecord:
        ownership = None
        if state.status in {EvalRunStatus.RUNNING, EvalRunStatus.CANCELLING}:
            assert state.claim_id is not None
            assert state.lease_expires_at is not None
            ownership = EvalRunOwnership(
                epoch=state.epoch,
                lease_expires_at=state.lease_expires_at,
            )
        return EvalRunRecord(
            spec=run_spec(state.request),
            status=state.status,
            attempt_count=state.epoch,
            created_at=state.created_at,
            updated_at=state.updated_at,
            started_at=state.started_at,
            finished_at=state.finished_at,
            cancel_requested_at=state.cancel_requested_at,
            ownership=ownership,
            result=None if state.result is None else result_summary(state.result),
            failure_code=state.failure_code,
            scenario_progress=(
                None
                if state.scenario_progress is None
                else state.scenario_progress.model_copy(deep=True)
            ),
        )

    @staticmethod
    def _observation(state: _MemoryRunState) -> EvalRunObservation:
        ownership = None
        if state.status in {EvalRunStatus.RUNNING, EvalRunStatus.CANCELLING}:
            assert state.lease_expires_at is not None
            ownership = EvalRunOwnership(
                epoch=state.epoch,
                lease_expires_at=state.lease_expires_at,
            )
        return EvalRunObservation(
            run_id=state.request.run_id,
            status=state.status,
            attempt_count=state.epoch,
            updated_at=state.updated_at,
            ownership=ownership,
        )


def _copy_query(value, model_type):
    if value is None:
        return model_type()
    return _exact_model(value, model_type, "query")


def _read_limit(value: int, *, hard_max: int) -> int:
    if type(value) is not int:
        raise TypeError("max_bytes must be an int.")
    if not 1 <= value <= hard_max:
        raise ValueError(f"max_bytes must be between 1 and {hard_max}.")
    return value


def _lease_seconds(value: int) -> int:
    if type(value) is not int:
        raise TypeError("lease seconds must be an int.")
    if not 1 <= value <= EVAL_STORE_MAX_LEASE_SECONDS:
        raise ValueError(f"lease seconds must be between 1 and {EVAL_STORE_MAX_LEASE_SECONDS}.")
    return value


def _claim_target_keys(value: tuple[str, ...]) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError("target_keys must be a tuple.")
    if not value:
        raise ValueError("target_keys cannot be empty.")
    if len(value) > EVAL_STORE_MAX_CLAIM_TARGETS:
        raise ValueError(
            f"target_keys cannot contain more than {EVAL_STORE_MAX_CLAIM_TARGETS} values."
        )
    validated = tuple(
        _portable_id(target_key, f"target_keys[{index}]") for index, target_key in enumerate(value)
    )
    if len(validated) != len(set(validated)):
        raise ValueError("target_keys must be unique.")
    return validated


def _bounded_page(items, *, limit: int, max_bytes: int, cursor):
    retained = []
    for index, item in enumerate(items[:limit]):
        if len(retained) == limit:
            break
        candidate = (*retained, item)
        candidate_has_more = index + 1 < len(items)
        candidate_cursor = cursor(candidate[-1]) if candidate_has_more else None
        candidate_page = {
            "items": candidate,
            "next_cursor": candidate_cursor,
            "has_more": candidate_has_more,
        }
        if not json_utf8_size_within_limit(candidate_page, max_bytes):
            if not retained:
                raise EvalStoreResultTooLarge(max_bytes)
            break
        retained.append(item)
    has_more = len(retained) < len(items)
    next_cursor = cursor(retained[-1]) if has_more and retained else None
    return tuple(retained), next_cursor, has_more


def _bounded_corpus_page(
    items: list[EvalCorpusCatalogEntry],
    query: EvalCatalogQuery,
) -> EvalCorpusCatalogPage:
    retained, next_cursor, has_more = _bounded_page(
        items,
        limit=query.limit,
        max_bytes=query.max_result_bytes,
        cursor=lambda item: _corpus_cursor(item, query.target_key),
    )
    return EvalCorpusCatalogPage(
        items=retained,
        next_cursor=next_cursor,
        has_more=has_more,
    )


def _bounded_scenario_page(
    items: list[EvalScenarioCatalogEntry],
    query: EvalScenarioCatalogQuery,
) -> EvalScenarioCatalogPage:
    retained, next_cursor, has_more = _bounded_page(
        items,
        limit=query.limit,
        max_bytes=query.max_result_bytes,
        cursor=lambda item: _scenario_cursor(item, query),
    )
    return EvalScenarioCatalogPage(
        items=retained,
        next_cursor=next_cursor,
        has_more=has_more,
    )


def _bounded_authored_suite_page(
    items: list[EvalAuthoredSuiteCatalogEntry],
    query: EvalAuthoredSuiteCatalogQuery,
) -> EvalAuthoredSuiteCatalogPage:
    retained, next_cursor, has_more = _bounded_page(
        items,
        limit=query.limit,
        max_bytes=query.max_result_bytes,
        cursor=lambda item: _authored_suite_cursor(item, query),
    )
    return EvalAuthoredSuiteCatalogPage(
        items=retained,
        next_cursor=next_cursor,
        has_more=has_more,
    )


def _bounded_case_page(
    items: list[EvalCaseCatalogEntry],
    query: EvalCaseCatalogQuery,
) -> EvalCaseCatalogPage:
    retained, next_cursor, has_more = _bounded_page(
        items,
        limit=query.limit,
        max_bytes=query.max_result_bytes,
        cursor=_case_cursor,
    )
    return EvalCaseCatalogPage(items=retained, next_cursor=next_cursor, has_more=has_more)


def _bounded_suite_page(
    items: list[EvalSuiteCatalogEntry],
    query: EvalSuiteCatalogQuery,
) -> EvalSuiteCatalogPage:
    retained, next_cursor, has_more = _bounded_page(
        items,
        limit=query.limit,
        max_bytes=query.max_result_bytes,
        cursor=_suite_cursor,
    )
    return EvalSuiteCatalogPage(items=retained, next_cursor=next_cursor, has_more=has_more)


def _bounded_run_page(items: list[EvalRunRecord], query: EvalRunQuery) -> EvalRunPage:
    retained, next_cursor, has_more = _bounded_page(
        items,
        limit=query.limit,
        max_bytes=query.max_result_bytes,
        cursor=lambda item: _run_cursor(item, query),
    )
    return EvalRunPage(items=retained, next_cursor=next_cursor, has_more=has_more)


def _bounded_result_page(
    items: list[EvalResultRecord],
    query: EvalResultQuery,
) -> EvalResultPage:
    retained, next_cursor, has_more = _bounded_page(
        items,
        limit=query.limit,
        max_bytes=query.max_result_bytes,
        cursor=lambda item: _result_cursor(item, query),
    )
    return EvalResultPage(items=retained, next_cursor=next_cursor, has_more=has_more)


__all__ = [
    "EVAL_RUN_INVOCATION_MAX_BYTES",
    "EVAL_SCENARIO_PROGRESS_MAX_BYTES",
    "EVAL_STORE_DEFAULT_PAGE_BYTES",
    "EVAL_STORE_DEFAULT_PAGE_SIZE",
    "EVAL_STORE_MAX_CLAIM_TARGETS",
    "EVAL_STORE_MAX_CURSOR_BYTES",
    "EVAL_STORE_MAX_IDENTIFIER_CHARS",
    "EVAL_STORE_MAX_LEASE_SECONDS",
    "EVAL_STORE_MAX_PAGE_BYTES",
    "EVAL_STORE_MAX_PAGE_SIZE",
    "TERMINAL_EVAL_RUN_STATUSES",
    "EvalAuthoredSuiteCatalogEntry",
    "EvalAuthoredSuiteCatalogPage",
    "EvalAuthoredSuiteCatalogQuery",
    "EvalAuthoredSuiteConflict",
    "EvalAuthoredSuiteReferenceError",
    "EvalCaseCatalogEntry",
    "EvalCaseCatalogPage",
    "EvalCaseCatalogQuery",
    "EvalCatalogQuery",
    "EvalCorpusCatalogEntry",
    "EvalCorpusCatalogPage",
    "EvalCorpusConflict",
    "EvalResultPage",
    "EvalResultQuery",
    "EvalRunAdmissionConflict",
    "EvalRunClaim",
    "EvalRunClaimLost",
    "EvalRunCostBudget",
    "EvalRunFailureCode",
    "EvalRunInvocation",
    "EvalRunLease",
    "EvalRunOwnership",
    "EvalRunPage",
    "EvalRunQuery",
    "EvalRunRecord",
    "EvalRunRequest",
    "EvalRunResultSummary",
    "EvalRunSpec",
    "EvalRunStateConflict",
    "EvalRunStatus",
    "EvalScenarioApprovalDecisionRecord",
    "EvalScenarioApprovalSubmission",
    "EvalScenarioArtifactReference",
    "EvalScenarioCatalogEntry",
    "EvalScenarioCatalogPage",
    "EvalScenarioCatalogQuery",
    "EvalScenarioConflict",
    "EvalScenarioRunInvocation",
    "EvalScenarioRunProgress",
    "EvalScenarioTrialFailureCode",
    "EvalScenarioTrialPhase",
    "EvalScenarioTrialProgress",
    "EvalStore",
    "EvalStorePublicationRejected",
    "EvalStoreResultTooLarge",
    "EvalStoreTransientContention",
    "EvalSuiteCatalogEntry",
    "EvalSuiteCatalogPage",
    "EvalSuiteCatalogQuery",
    "InMemoryEvalStore",
]
