from __future__ import annotations

import asyncio
import base64
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
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

from cayu._validation import json_utf8_size_within_limit
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
    inspect_eval_corpus,
)
from cayu.evals.execution import (
    CORPUS_EXECUTION_MAX_CONCURRENCY,
    CORPUS_EXECUTION_RESULT_MAX_BYTES,
    CorpusExecutionResult,
)
from cayu.evals.published import _validate_published_eval_run_for_corpus
from cayu.evals.results import (
    CAPTURED_EVALUATION_RESULT_MAX_BYTES,
    CapturedEvaluationResultV1,
    EvalResultOrigin,
    EvalResultTargetIdentityV1,
    captured_evaluation_result_from_json,
    eval_result_projection,
    validate_captured_result_for_corpus,
)
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
_EVAL_STORE_MAX_BIGINT = 2**63 - 1
EVAL_RUN_INVOCATION_MAX_BYTES = 64 << 10

_STORE_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z", re.ASCII)
_CURSOR_VERSION = 1


class _EvalStoreModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
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


class EvalStorePublicationRejected(ValueError):
    """Public eval data could not cross the active credential-redaction boundary."""


class EvalRunAdmissionConflict(ValueError):
    """A run id or idempotency key is already bound to another request."""


class EvalRunStateConflict(ValueError):
    """A requested run lifecycle transition is not valid from current state."""


class EvalRunClaimLost(RuntimeError):
    """A worker no longer owns the live fenced claim required for a mutation."""


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
    max_steps: StrictInt | None = Field(default=None, ge=1, le=256)
    limits: RunLimits | None = None
    cost_budget: EvalRunCostBudget | None = None

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
    max_concurrency: StrictInt = Field(ge=1, le=CORPUS_EXECUTION_MAX_CONCURRENCY)
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
        return self

    @property
    def id(self) -> str:
        return self.spec.run_id


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
    async def load_run(self, run_id: str) -> EvalRunRecord | None:
        """Load one run's public lifecycle projection."""

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

    @abstractmethod
    async def request_cancel(self, run_id: str) -> EvalRunRecord:
        """Persist cancellation intent, terminalizing unclaimed queued work."""

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


@dataclass
class _MemoryEvalResultState:
    record: EvalResultRecord
    document: bytes


class InMemoryEvalStore(EvalStore):
    """Explicitly ephemeral EvalStore for tests and process-local SDK use."""

    durable: ClassVar[bool] = False
    captured_results: ClassVar[bool] = True
    scenarios: ClassVar[bool] = True

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

    async def load_run(self, run_id: str) -> EvalRunRecord | None:
        run_id = _store_identifier(run_id, "run_id")
        async with self._lock:
            state = self._runs.get(run_id)
            return None if state is None else self._record(state)

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
            else:
                state.status = EvalRunStatus.CANCELLING
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
            now = self._now()
            self._save_result_document(
                validated_result,
                result_document,
                created_at=now,
            )
            state.status = EvalRunStatus.COMPLETED
            state.result = validated_result
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
    "EVAL_STORE_DEFAULT_PAGE_BYTES",
    "EVAL_STORE_DEFAULT_PAGE_SIZE",
    "EVAL_STORE_MAX_CLAIM_TARGETS",
    "EVAL_STORE_MAX_CURSOR_BYTES",
    "EVAL_STORE_MAX_IDENTIFIER_CHARS",
    "EVAL_STORE_MAX_LEASE_SECONDS",
    "EVAL_STORE_MAX_PAGE_BYTES",
    "EVAL_STORE_MAX_PAGE_SIZE",
    "TERMINAL_EVAL_RUN_STATUSES",
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
    "EvalScenarioCatalogEntry",
    "EvalScenarioCatalogPage",
    "EvalScenarioCatalogQuery",
    "EvalScenarioConflict",
    "EvalStore",
    "EvalStorePublicationRejected",
    "EvalStoreResultTooLarge",
    "EvalSuiteCatalogEntry",
    "EvalSuiteCatalogPage",
    "EvalSuiteCatalogQuery",
    "InMemoryEvalStore",
]
