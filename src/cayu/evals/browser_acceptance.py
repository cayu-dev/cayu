"""Versioned browser acceptance manifests, receipts, and portable reports.

The generic eval runner remains responsible for executing Cayu applications.
This module owns only the browser-specific, bounded projection needed to state
what was exercised and whether the observed browser outcome matched the pinned
expectation.
"""

from __future__ import annotations

import asyncio
import errno
import html
import json
import os
import platform
import tempfile
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self, TypeVar
from urllib.parse import urlsplit

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
    compact_json_utf8_size,
    copy_durable_json_object,
    require_durable_clean_nonblank,
    require_unicode_scalar_text,
    revalidate_model_input,
)
from cayu.build_provenance import RuntimeBuildProvenance, current_runtime_build_provenance
from cayu.core.events import EventType
from cayu.egress import EgressAuthorityIdentity
from cayu.evals.browser_acceptance_fixture import BrowserAcceptanceFixtureV1
from cayu.evals.corpus import (
    EvaluationEvidencePolicySpec,
    PricingProfileIdentityV1,
    _content_revision,
    pricing_profile_identity,
)
from cayu.evals.evidence import AssertionEvidenceView, project_assertion_evidence_view
from cayu.evals.models import EvalStatus, EvalTrialResult
from cayu.evals.runner import EvalPlan, EvalSuite, run_eval_suite
from cayu.evals.testing import ScriptedModelProvider
from cayu.runners import PINNED_BROWSER_SESSION_WORKLOAD
from cayu.runtime.costs import PriceBook
from cayu.runtime.usage import SessionUsageSummary
from cayu.tools.browser_session import (
    DEFAULT_BROWSER_SESSION_MAX_ARTIFACT_BYTES,
    BrowserSessionTool,
)
from cayu.tools.webbridge import WebBridge, WebBridgeProfileKind

BROWSER_ACCEPTANCE_SCHEMA_VERSION = 1
BROWSER_ACCEPTANCE_MANIFEST_MAX_BYTES = 8 << 20
BROWSER_ACCEPTANCE_REPORT_MAX_BYTES = 32 << 20
BROWSER_ACCEPTANCE_HTML_MAX_BYTES = 32 << 20
BROWSER_ACCEPTANCE_MAX_CASES = 256
BROWSER_ACCEPTANCE_MAX_TRIALS = 32
BROWSER_ACCEPTANCE_MAX_ROWS = 4_096
BROWSER_ACCEPTANCE_MAX_OPERATIONS_PER_ROW = 256
BROWSER_ACCEPTANCE_MAX_ARTIFACTS_PER_ROW = 128
BROWSER_ACCEPTANCE_MAX_REQUEST_SUMMARIES_PER_ROW = 256
BROWSER_ACCEPTANCE_MAX_ORIGINS = 64
BROWSER_ACCEPTANCE_MAX_OPERATIONS_PER_CASE = 64
BROWSER_ACCEPTANCE_MAX_CHECKPOINTS_PER_CASE = 32
BROWSER_ACCEPTANCE_MAX_ERROR_CATEGORIES = 64
BROWSER_ACCEPTANCE_MAX_TRUNCATION_CATEGORIES = 32
BROWSER_ACCEPTANCE_MAX_ARTIFACT_BYTES_PER_OPERATION = DEFAULT_BROWSER_SESSION_MAX_ARTIFACT_BYTES

_REFUSAL_ERRORS = frozenset(
    {
        "actionability_failed",
        "allocation_lost",
        "authority_expired",
        "fetch_failed",
        "incompatible_profile",
        "operation_conflict",
        "operation_not_dispatched",
        "policy_denied",
        "resource_exhausted",
        "restoration_required",
        "session_closed",
        "stale_observation",
        "unknown_element",
        "unknown_page",
        "unknown_session",
    }
)

_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    hide_input_in_errors=True,
    revalidate_instances="always",
    validate_default=True,
)

_ContentBoundModel = TypeVar("_ContentBoundModel", bound=BaseModel)


def _clean(value: str, field_name: str, *, maximum: int = 256) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    value = require_unicode_scalar_text(value, field_name)
    if len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{field_name} exceeds {maximum} UTF-8 bytes.")
    return value


def _revision(value: str, field_name: str) -> str:
    if (
        len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{field_name} must be a sha256 revision.")
    return value


def _fingerprint(value: str, field_name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be lowercase SHA-256 hex.")
    return value


def _identity_fingerprint(value: Mapping[str, Any], field_name: str) -> str:
    """Return a domain-separated portable fingerprint for bounded identity material."""

    return _content_revision({"identity": value}, field_name)[7:]


def _estimated_cost(value: str) -> tuple[Decimal, str]:
    parts = value.split(" ")
    if len(parts) != 2 or not parts[1].isalpha() or not parts[1].isupper():
        raise ValueError("max_estimated_cost must use '<decimal> <CURRENCY>'.")
    try:
        amount = Decimal(parts[0])
    except InvalidOperation as exc:
        raise ValueError("max_estimated_cost has an invalid decimal amount.") from exc
    if not amount.is_finite() or amount <= 0 or format(amount, "f") != parts[0]:
        raise ValueError("max_estimated_cost must use a positive canonical decimal amount.")
    return amount, parts[1]


def _validate_cost_evidence(value: dict[str, Any], field_name: str) -> dict[str, Any]:
    for currency, amount in value.items():
        if (
            not currency.isalpha()
            or not currency.isupper()
            or len(currency.encode("utf-8")) > 16
            or type(amount) is not str
        ):
            raise ValueError(f"{field_name} contains malformed cost evidence.")
        try:
            parsed = Decimal(amount)
        except InvalidOperation as exc:
            raise ValueError(f"{field_name} contains malformed cost evidence.") from exc
        if not parsed.is_finite() or parsed < 0 or format(parsed, "f") != amount:
            raise ValueError(f"{field_name} contains malformed cost evidence.")
    return value


def _copy_model(value: object, model_type: type[BaseModel]) -> object:
    return revalidate_model_input(value, model_type)


def _build_content_bound(
    model_type: type[_ContentBoundModel],
    values: dict[str, Any],
    field_name: str,
) -> _ContentBoundModel:
    """Apply model defaults before deriving a content-bound revision."""

    placeholder = "sha256:" + ("0" * 64)
    draft = model_type.model_construct(revision=placeholder, **values)
    material = draft.model_dump(mode="json", exclude={"revision"}, warnings="error")
    return model_type(
        revision=_content_revision(material, field_name),
        **values,
    )


class BrowserAcceptanceMode(StrEnum):
    DETERMINISTIC = "deterministic"
    LIVE_PUBLIC = "live_public"
    LIVE_AUTHENTICATED = "live_authenticated"


class BrowserAcceptanceCaseCategory(StrEnum):
    SUCCESS = "success"
    REFUSAL = "refusal"
    LIMIT = "limit"
    CRASH = "crash"
    CANCELLATION = "cancellation"
    RECOVERY = "recovery"
    AMBIGUITY = "ambiguity"
    ADVERSARIAL = "adversarial"
    CAPABILITY = "capability"


class BrowserAcceptanceState(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    PASSED = "passed"
    FAILED = "failed"
    REFUSED = "refused"
    AMBIGUOUS = "ambiguous"
    UNAVAILABLE = "unavailable"


class BrowserAcceptanceConformance(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


class BrowserAcceptanceSemanticState(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE = "unavailable"


class BrowserAcceptanceInfrastructureState(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class BrowserAcceptanceCompletionState(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class BrowserAcceptanceDiagnosticState(StrEnum):
    CAPTURED = "captured"
    UNAVAILABLE = "unavailable"
    NOT_REQUESTED = "not_requested"


class BrowserAcceptanceAgentReportState(StrEnum):
    CLAIMED_SUCCESS = "claimed_success"
    CLAIMED_FAILURE = "claimed_failure"
    ABSENT = "absent"
    UNAVAILABLE = "unavailable"


class BrowserAcceptanceVariabilityState(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    STABLE = "stable"
    VARIABLE = "variable"
    INCOMPLETE = "incomplete"


class BrowserAcceptanceOperationState(StrEnum):
    INTENT = "intent"
    DISPATCHED = "dispatched"
    TERMINAL = "terminal"
    OPERATION_NOT_DISPATCHED = "operation_not_dispatched"
    OUTCOME_AMBIGUOUS = "outcome_ambiguous"


class BrowserAcceptanceAccessState(StrEnum):
    AVAILABLE = "available"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class BrowserAllocationDisposition(StrEnum):
    LIVE = "live"
    RETIRED = "retired"
    UNCERTAIN = "uncertain"
    UNAVAILABLE = "unavailable"


class BrowserAcceptanceSemanticOracle(StrEnum):
    OBSERVATION = "observation"
    FIXTURE_EFFECT = "fixture_effect"
    ARTIFACT = "artifact"
    STABLE_ERROR = "stable_error"
    PUBLIC_SCHEMA_UNSUPPORTED = "public_schema_unsupported"
    RECOVERY_STATE = "recovery_state"


class BrowserAcceptanceFaultScenario(StrEnum):
    CANCEL_AFTER_INTENT = "cancel_after_intent"
    CANCEL_AFTER_DISPATCHED = "cancel_after_dispatched"
    CANCEL_BEFORE_TERMINAL = "cancel_before_terminal"
    CANCEL_AFTER_ARTIFACT = "cancel_after_artifact"
    CANCEL_AFTER_TERMINAL = "cancel_after_terminal"
    PROCESS_AFTER_INTENT = "process_after_intent"
    PROCESS_AFTER_DISPATCHED = "process_after_dispatched"
    PROCESS_BEFORE_TERMINAL = "process_before_terminal"
    PROCESS_AFTER_ARTIFACT = "process_after_artifact"
    PROCESS_AFTER_TERMINAL = "process_after_terminal"
    BROWSER_BEFORE_DISPATCH = "browser_before_dispatch"
    BROWSER_ALLOCATION_LOSS = "browser_allocation_loss"
    BROWSER_DURING_EXECUTION = "browser_during_execution"
    BROWSER_AFTER_EFFECT = "browser_after_effect"
    BROWSER_DURING_CLEANUP = "browser_during_cleanup"
    BROWSER_ACTIVE_PAGE_CRASH = "browser_active_page_crash"
    BROWSER_BACKGROUND_PAGE_CRASH = "browser_background_page_crash"
    ACKNOWLEDGEMENT_LOSS = "acknowledgement_loss"


class BrowserAcceptanceLimitsV1(BaseModel):
    model_config = _MODEL_CONFIG

    max_destinations: StrictInt = Field(ge=1, le=64)
    max_browser_operations: StrictInt = Field(ge=1, le=16_384)
    max_model_steps: StrictInt = Field(ge=1, le=1_024)
    max_wall_time_ms: StrictInt = Field(ge=1, le=3_600_000)
    max_artifact_bytes: StrictInt = Field(ge=1, le=1 << 30)
    max_concurrency: StrictInt = Field(ge=1, le=32)
    max_input_tokens: StrictInt | None = Field(default=None, ge=1)
    max_output_tokens: StrictInt | None = Field(default=None, ge=1)
    max_estimated_cost: StrictStr | None = Field(default=None, max_length=128)

    @field_validator("max_estimated_cost")
    @classmethod
    def validate_cost(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        value = _clean(value, info.field_name, maximum=128)
        _estimated_cost(value)
        return value


class BrowserAcceptanceCaseV1(BaseModel):
    model_config = _MODEL_CONFIG

    case_id: StrictStr
    revision: StrictStr
    category: BrowserAcceptanceCaseCategory
    expected_state: BrowserAcceptanceState
    semantic_oracle: BrowserAcceptanceSemanticOracle
    semantic_success_required: StrictBool = True
    fault_scenario: BrowserAcceptanceFaultScenario | None = None
    required: StrictBool = True
    fixture_route: StrictStr | None = None
    operations: tuple[StrictStr, ...] = Field(
        min_length=1,
        max_length=BROWSER_ACCEPTANCE_MAX_OPERATIONS_PER_CASE,
    )
    screenshot_checkpoints: tuple[StrictStr, ...] = Field(
        default=(),
        max_length=BROWSER_ACCEPTANCE_MAX_CHECKPOINTS_PER_CASE,
    )
    oracle_parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str, info) -> str:
        return _clean(value, info.field_name, maximum=128)

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str, info) -> str:
        return _revision(value, info.field_name)

    @field_validator("fixture_route")
    @classmethod
    def validate_fixture_route(cls, value: str | None, info) -> str | None:
        return None if value is None else _clean(value, info.field_name, maximum=256)

    @field_validator("operations")
    @classmethod
    def validate_operations(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        # Repeated operations are decision-bearing for exact replay and capacity cases.
        return tuple(_clean(item, info.field_name, maximum=128) for item in value)

    @field_validator("screenshot_checkpoints")
    @classmethod
    def validate_checkpoints(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        normalized = tuple(_clean(item, info.field_name, maximum=128) for item in value)
        if len(normalized) != len(set(normalized)):
            raise ValueError("screenshot_checkpoints must not contain duplicates.")
        return normalized

    @field_validator("oracle_parameters", mode="before")
    @classmethod
    def copy_oracle_parameters(cls, value: object) -> dict[str, Any]:
        return copy_durable_json_object(value, "oracle_parameters")

    @model_validator(mode="after")
    def validate_case(self) -> Self:
        if self.expected_state is BrowserAcceptanceState.UNSUPPORTED and (
            self.semantic_oracle is not BrowserAcceptanceSemanticOracle.PUBLIC_SCHEMA_UNSUPPORTED
            or self.semantic_success_required
        ):
            raise ValueError(
                "Unsupported cases require the public-schema oracle and no semantic success."
            )
        if self.semantic_success_required != (self.expected_state is BrowserAcceptanceState.PASSED):
            raise ValueError("Only cases expecting a pass may require semantic task success.")
        if (
            self.category
            in {
                BrowserAcceptanceCaseCategory.CRASH,
                BrowserAcceptanceCaseCategory.CANCELLATION,
            }
            and self.fault_scenario is None
        ):
            raise ValueError("Crash and cancellation cases require an executable fault scenario.")
        material = self.model_dump(mode="json", exclude={"revision"})
        if self.revision != _content_revision(material, "browser acceptance case"):
            raise ValueError("Browser acceptance case revision does not match its content.")
        return self

    @classmethod
    def build(cls, **values: Any) -> BrowserAcceptanceCaseV1:
        return _build_content_bound(
            cls,
            dict(values),
            "browser acceptance case",
        )


class BrowserAcceptanceManifestV1(BaseModel):
    model_config = _MODEL_CONFIG

    record_type: Literal["cayu.browser-acceptance-manifest"] = "cayu.browser-acceptance-manifest"
    schema_version: Literal[1] = BROWSER_ACCEPTANCE_SCHEMA_VERSION
    revision: StrictStr
    corpus_revision: StrictStr
    suite_id: StrictStr
    mode: BrowserAcceptanceMode
    enabled: StrictBool = True
    trial_count: StrictInt = Field(ge=1, le=BROWSER_ACCEPTANCE_MAX_TRIALS)
    allowed_origins: tuple[StrictStr, ...] = Field(
        min_length=1,
        max_length=BROWSER_ACCEPTANCE_MAX_ORIGINS,
    )
    limits: BrowserAcceptanceLimitsV1
    cases: tuple[BrowserAcceptanceCaseV1, ...] = Field(
        min_length=1,
        max_length=BROWSER_ACCEPTANCE_MAX_CASES,
    )

    @field_validator("revision", "corpus_revision")
    @classmethod
    def validate_revision(cls, value: str, info) -> str:
        return _revision(value, info.field_name)

    @field_validator("suite_id")
    @classmethod
    def validate_suite_id(cls, value: str, info) -> str:
        return _clean(value, info.field_name, maximum=128)

    @field_validator("allowed_origins")
    @classmethod
    def validate_origins(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        normalized = tuple(_clean(item, info.field_name, maximum=512) for item in value)
        if normalized != tuple(sorted(set(normalized))):
            raise ValueError("allowed_origins must be unique and sorted.")
        return normalized

    @field_validator("limits", mode="before")
    @classmethod
    def copy_limits(cls, value: object) -> object:
        return _copy_model(value, BrowserAcceptanceLimitsV1)

    @field_validator("cases", mode="before")
    @classmethod
    def copy_cases(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            return value
        return tuple(_copy_model(item, BrowserAcceptanceCaseV1) for item in value)

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        case_ids = tuple(item.case_id for item in self.cases)
        if case_ids != tuple(sorted(set(case_ids))):
            raise ValueError("Browser acceptance cases must be unique and sorted by case_id.")
        origin_hosts = _manifest_origin_hosts(self.allowed_origins)
        if len(origin_hosts) > self.limits.max_destinations:
            raise ValueError("Browser acceptance origins exceed the destination ceiling.")
        if self.mode is BrowserAcceptanceMode.DETERMINISTIC:
            if self.trial_count != 1 or self.limits.max_concurrency != 1:
                raise ValueError("Deterministic acceptance requires one serial trial per case.")
            if (
                self.limits.max_input_tokens is not None
                or self.limits.max_output_tokens is not None
            ):
                raise ValueError(
                    "Deterministic acceptance cannot declare live model token budgets."
                )
            if self.limits.max_estimated_cost is not None:
                raise ValueError("Deterministic acceptance cannot declare live model cost.")
        if self.mode is BrowserAcceptanceMode.LIVE_PUBLIC:
            if not self.enabled:
                raise ValueError("The live-public manifest must remain explicitly runnable.")
            if (
                self.limits.max_input_tokens is None
                or self.limits.max_output_tokens is None
                or self.limits.max_estimated_cost is None
            ):
                raise ValueError("Live-public acceptance requires finite model budgets.")
        if self.mode is BrowserAcceptanceMode.LIVE_AUTHENTICATED and self.enabled:
            raise ValueError("Authenticated browser acceptance is not enabled in schema v1.")
        material = self.model_dump(mode="json", exclude={"revision"})
        if self.revision != _content_revision(material, "browser acceptance manifest"):
            raise ValueError("Browser acceptance manifest revision does not match its content.")
        if (
            compact_json_utf8_size(self.model_dump(mode="json"))
            > BROWSER_ACCEPTANCE_MANIFEST_MAX_BYTES
        ):
            raise ValueError("Browser acceptance manifest exceeds its byte bound.")
        return self

    @classmethod
    def build(cls, **values: Any) -> BrowserAcceptanceManifestV1:
        return _build_content_bound(
            cls,
            dict(values),
            "browser acceptance manifest",
        )


class BrowserAcceptanceRuntimeIdentityV1(BaseModel):
    model_config = _MODEL_CONFIG

    revision: StrictStr
    runtime_build_provenance: RuntimeBuildProvenance
    browser_protocol: StrictStr
    browser_worker_version: StrictStr
    playwright_version: StrictStr
    chromium_identity: StrictStr | None = None
    runner_fingerprint: StrictStr
    workload_fingerprint: StrictStr
    egress_fingerprint: StrictStr
    artifact_store_fingerprint: StrictStr
    execution_profile_fingerprint: StrictStr
    execution_suite_fingerprint: StrictStr
    provider_name: StrictStr
    model: StrictStr
    pricing_profile_fingerprint: StrictStr | None = None
    cost_currencies: tuple[StrictStr, ...] = Field(default=(), max_length=32)
    platform_system: StrictStr
    platform_machine: StrictStr
    python_version: StrictStr

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str, info) -> str:
        return _revision(value, info.field_name)

    @field_validator(
        "runner_fingerprint",
        "workload_fingerprint",
        "egress_fingerprint",
        "artifact_store_fingerprint",
        "execution_profile_fingerprint",
        "execution_suite_fingerprint",
        "pricing_profile_fingerprint",
    )
    @classmethod
    def validate_fingerprint(cls, value: str | None, info) -> str | None:
        return None if value is None else _fingerprint(value, info.field_name)

    @field_validator(
        "browser_protocol",
        "browser_worker_version",
        "playwright_version",
        "chromium_identity",
        "provider_name",
        "model",
        "platform_system",
        "platform_machine",
        "python_version",
    )
    @classmethod
    def validate_text(cls, value: str | None, info) -> str | None:
        return None if value is None else _clean(value, info.field_name, maximum=256)

    @field_validator("cost_currencies")
    @classmethod
    def validate_cost_currencies(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        cleaned = tuple(_clean(item, info.field_name, maximum=16) for item in value)
        if cleaned != tuple(sorted(set(cleaned))) or any(
            not item.isalpha() or not item.isupper() for item in cleaned
        ):
            raise ValueError("cost_currencies must be unique sorted uppercase identifiers.")
        return cleaned

    @field_validator("runtime_build_provenance", mode="before")
    @classmethod
    def copy_build_provenance(cls, value: object) -> object:
        return _copy_model(value, RuntimeBuildProvenance)

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if (self.pricing_profile_fingerprint is None) != (not self.cost_currencies):
            raise ValueError(
                "Browser acceptance pricing identity and currencies must be present together."
            )
        material = self.model_dump(mode="json", exclude={"revision"})
        if self.revision != _content_revision(material, "browser acceptance runtime identity"):
            raise ValueError("Browser acceptance runtime identity revision is invalid.")
        return self

    @classmethod
    def build(cls, **values: Any) -> BrowserAcceptanceRuntimeIdentityV1:
        return _build_content_bound(
            cls,
            dict(values),
            "browser acceptance runtime identity",
        )


def _run_identity_revision(
    manifest: BrowserAcceptanceManifestV1,
    runtime_identity: BrowserAcceptanceRuntimeIdentityV1,
) -> str:
    """Bind admission identity while allowing Chromium evidence to arrive in trial one."""

    runtime_material = runtime_identity.model_dump(
        mode="json",
        exclude={"revision", "chromium_identity"},
    )
    return _content_revision(
        {
            "manifest_revision": manifest.revision,
            "runtime_identity": runtime_material,
            "mode": manifest.mode.value,
        },
        "browser acceptance run identity",
    )


class BrowserAcceptanceArtifactEvidenceV1(BaseModel):
    model_config = _MODEL_CONFIG

    artifact_id: StrictStr
    artifact_revision: StrictStr
    kind: StrictStr
    content_type: StrictStr
    size_bytes: StrictInt = Field(ge=0)

    @field_validator("artifact_revision")
    @classmethod
    def validate_revision(cls, value: str, info) -> str:
        return _revision(value, info.field_name)

    @field_validator("artifact_id", "kind", "content_type")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _clean(value, info.field_name, maximum=256)


class BrowserAcceptanceRequestSummaryV1(BaseModel):
    model_config = _MODEL_CONFIG

    sequence: StrictInt = Field(ge=1, le=BROWSER_ACCEPTANCE_MAX_REQUEST_SUMMARIES_PER_ROW)
    method: StrictStr
    destination_revision: StrictStr
    route_revision: StrictStr
    outcome: StrictStr
    status_code: StrictInt | None = Field(default=None, ge=100, le=599)
    request_bytes: StrictInt | None = Field(default=None, ge=0)
    response_bytes: StrictInt | None = Field(default=None, ge=0)
    truncated: StrictBool = False

    @field_validator("destination_revision", "route_revision")
    @classmethod
    def validate_revision(cls, value: str, info) -> str:
        return _revision(value, info.field_name)

    @field_validator("method", "outcome")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _clean(value, info.field_name, maximum=512)


class BrowserAcceptanceOperationEvidenceV1(BaseModel):
    model_config = _MODEL_CONFIG

    sequence: StrictInt = Field(ge=1, le=BROWSER_ACCEPTANCE_MAX_OPERATIONS_PER_ROW)
    invocation_revision: StrictStr
    operation: StrictStr
    state: BrowserAcceptanceOperationState
    error_category: StrictStr | None = None
    allocation_disposition: BrowserAllocationDisposition
    target_revision: StrictStr | None = None
    observed_target_revision: StrictStr | None = None
    observation_revision: StrictStr | None = None
    ref_count: StrictInt | None = Field(default=None, ge=0)
    snapshot_bytes: StrictInt | None = Field(default=None, ge=0)
    load_state: StrictStr | None = None
    access_state: BrowserAcceptanceAccessState | None = None
    truncation: tuple[StrictStr, ...] = Field(
        default=(),
        max_length=BROWSER_ACCEPTANCE_MAX_TRUNCATION_CATEGORIES,
    )
    artifacts: tuple[BrowserAcceptanceArtifactEvidenceV1, ...] = Field(
        default=(),
        max_length=BROWSER_ACCEPTANCE_MAX_ARTIFACTS_PER_ROW,
    )

    @field_validator("invocation_revision")
    @classmethod
    def validate_invocation_revision(cls, value: str, info) -> str:
        return _revision(value, info.field_name)

    @field_validator("target_revision", "observed_target_revision")
    @classmethod
    def validate_target_revision(cls, value: str | None, info) -> str | None:
        return None if value is None else _revision(value, info.field_name)

    @field_validator(
        "operation",
        "error_category",
        "observation_revision",
        "load_state",
    )
    @classmethod
    def validate_text(cls, value: str | None, info) -> str | None:
        return None if value is None else _clean(value, info.field_name, maximum=256)

    @field_validator("truncation")
    @classmethod
    def validate_truncation(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        normalized = tuple(_clean(item, info.field_name, maximum=128) for item in value)
        if normalized != tuple(sorted(set(normalized))):
            raise ValueError("Operation truncation categories must be unique and sorted.")
        return normalized

    @field_validator("artifacts", mode="before")
    @classmethod
    def copy_artifacts(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            return value
        return tuple(_copy_model(item, BrowserAcceptanceArtifactEvidenceV1) for item in value)


class BrowserAcceptanceUsageV1(BaseModel):
    model_config = _MODEL_CONFIG

    model_steps: StrictInt | None = Field(default=None, ge=0)
    browser_operations: StrictInt | None = Field(default=None, ge=0)
    input_tokens: StrictInt | None = Field(default=None, ge=0)
    output_tokens: StrictInt | None = Field(default=None, ge=0)
    total_tokens: StrictInt | None = Field(default=None, ge=0)
    estimated_costs: dict[str, StrictStr] = Field(default_factory=dict)
    unpriced_model_steps: dict[str, StrictInt] = Field(default_factory=dict)

    @field_validator("estimated_costs", "unpriced_model_steps", mode="before")
    @classmethod
    def copy_cost_evidence(cls, value: object, info) -> dict[str, Any]:
        copied = copy_durable_json_object(value, info.field_name)
        if len(copied) > 32:
            raise ValueError(f"{info.field_name} contains too many currencies.")
        if info.field_name == "estimated_costs":
            return _validate_cost_evidence(copied, info.field_name)
        if any(
            type(currency) is not str
            or not currency.isalpha()
            or not currency.isupper()
            or len(currency.encode("utf-8")) > 16
            or type(count) is not int
            or count < 0
            for currency, count in copied.items()
        ):
            raise ValueError("unpriced_model_steps contains malformed evidence.")
        return dict(sorted(copied.items()))


class BrowserAcceptanceFaultEvidenceV1(BaseModel):
    model_config = _MODEL_CONFIG

    scenario: BrowserAcceptanceFaultScenario
    boundary_observed: StrictBool
    cancellation_delivered: StrictBool = False
    process_loss_observed: StrictBool = False
    recovered_in_fresh_app: StrictBool = False
    browser_dispatches: StrictInt = Field(ge=0, le=BROWSER_ACCEPTANCE_MAX_OPERATIONS_PER_CASE)

    @model_validator(mode="after")
    def validate_signal(self) -> Self:
        is_cancellation = self.scenario.value.startswith("cancel_")
        is_process_loss = self.scenario.value.startswith("process_")
        if self.cancellation_delivered != is_cancellation:
            raise ValueError("Fault evidence cancellation signal conflicts with its scenario.")
        if self.process_loss_observed != is_process_loss:
            raise ValueError("Fault evidence process-loss signal conflicts with its scenario.")
        if self.recovered_in_fresh_app and not is_process_loss:
            raise ValueError("Only process-loss scenarios may claim fresh-app recovery.")
        return self


class BrowserAcceptanceDiagnosticV1(BaseModel):
    model_config = _MODEL_CONFIG

    state: BrowserAcceptanceDiagnosticState
    error_code: StrictStr | None = None
    fixture_route_observed: StrictBool | None = None
    fixture_route_request_count: StrictInt | None = Field(default=None, ge=0)
    browser_dispatches: StrictInt | None = Field(
        default=None,
        ge=0,
        le=BROWSER_ACCEPTANCE_MAX_OPERATIONS_PER_ROW,
    )
    fixture_effects: dict[StrictStr, StrictInt] = Field(default_factory=dict)
    chromium_identity: StrictStr | None = None
    fault: BrowserAcceptanceFaultEvidenceV1 | None = None
    operations: tuple[BrowserAcceptanceOperationEvidenceV1, ...] = Field(
        default=(),
        max_length=BROWSER_ACCEPTANCE_MAX_OPERATIONS_PER_ROW,
    )
    requests: tuple[BrowserAcceptanceRequestSummaryV1, ...] = Field(
        default=(),
        max_length=BROWSER_ACCEPTANCE_MAX_REQUEST_SUMMARIES_PER_ROW,
    )
    truncated_categories: tuple[StrictStr, ...] = Field(
        default=(),
        max_length=BROWSER_ACCEPTANCE_MAX_TRUNCATION_CATEGORIES,
    )

    @field_validator("error_code")
    @classmethod
    def validate_error_code(cls, value: str | None, info) -> str | None:
        return None if value is None else _clean(value, info.field_name, maximum=128)

    @field_validator("chromium_identity")
    @classmethod
    def validate_chromium_identity(cls, value: str | None, info) -> str | None:
        return None if value is None else _clean(value, info.field_name, maximum=256)

    @field_validator("operations", mode="before")
    @classmethod
    def copy_operations(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            return value
        return tuple(_copy_model(item, BrowserAcceptanceOperationEvidenceV1) for item in value)

    @field_validator("fault", mode="before")
    @classmethod
    def copy_fault(cls, value: object) -> object:
        return None if value is None else _copy_model(value, BrowserAcceptanceFaultEvidenceV1)

    @field_validator("requests", mode="before")
    @classmethod
    def copy_requests(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            return value
        return tuple(_copy_model(item, BrowserAcceptanceRequestSummaryV1) for item in value)

    @field_validator("truncated_categories")
    @classmethod
    def validate_truncation(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        normalized = tuple(_clean(item, info.field_name, maximum=128) for item in value)
        if normalized != tuple(sorted(set(normalized))):
            raise ValueError("Diagnostic truncation categories must be unique and sorted.")
        return normalized

    @field_validator("fixture_effects", mode="before")
    @classmethod
    def copy_fixture_effects(cls, value: object) -> dict[str, Any]:
        copied = copy_durable_json_object(value, "fixture_effects")
        if len(copied) > 64 or any(
            type(key) is not str or type(count) is not int or count < 0
            for key, count in copied.items()
        ):
            raise ValueError("fixture_effects contains malformed bounded counts.")
        return dict(sorted(copied.items()))

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if (self.state is BrowserAcceptanceDiagnosticState.UNAVAILABLE) != (
            self.error_code is not None
        ):
            raise ValueError("Unavailable diagnostics require exactly one stable error code.")
        if self.state is BrowserAcceptanceDiagnosticState.NOT_REQUESTED and (
            self.operations or self.requests or self.truncated_categories or self.fault is not None
        ):
            raise ValueError("Diagnostics that were not requested cannot carry evidence.")
        return self


def _browser_acceptance_row_id(
    *,
    run_identity_revision: str,
    case_revision: str,
    trial_number: int,
    attempt_number: int,
) -> str:
    return _content_revision(
        {
            "run_identity_revision": run_identity_revision,
            "case_revision": case_revision,
            "trial_number": trial_number,
            "attempt_number": attempt_number,
        },
        "browser acceptance row identity",
    )


class BrowserAcceptanceTrialReceiptV1(BaseModel):
    model_config = _MODEL_CONFIG

    record_type: Literal["cayu.browser-acceptance-trial"] = "cayu.browser-acceptance-trial"
    schema_version: Literal[1] = BROWSER_ACCEPTANCE_SCHEMA_VERSION
    revision: StrictStr
    row_id: StrictStr
    run_identity_revision: StrictStr
    case_id: StrictStr
    case_revision: StrictStr
    trial_number: StrictInt = Field(ge=1, le=BROWSER_ACCEPTANCE_MAX_TRIALS)
    attempt_number: StrictInt = Field(default=1, ge=1, le=1_000)
    expected_state: BrowserAcceptanceState
    observed_state: BrowserAcceptanceState
    conformance: BrowserAcceptanceConformance
    semantic_state: BrowserAcceptanceSemanticState
    infrastructure_state: BrowserAcceptanceInfrastructureState
    completion_state: BrowserAcceptanceCompletionState
    agent_report_state: BrowserAcceptanceAgentReportState
    started_at: datetime
    completed_at: datetime
    elapsed_ms: StrictInt = Field(ge=0)
    usage: BrowserAcceptanceUsageV1
    error_categories: dict[StrictStr, StrictInt] = Field(default_factory=dict)
    truncation_categories: dict[StrictStr, StrictInt] = Field(default_factory=dict)
    diagnostic: BrowserAcceptanceDiagnosticV1

    @field_validator("revision", "row_id", "run_identity_revision", "case_revision")
    @classmethod
    def validate_revision(cls, value: str, info) -> str:
        return _revision(value, info.field_name)

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str, info) -> str:
        return _clean(value, info.field_name, maximum=128)

    @field_validator("started_at", "completed_at")
    @classmethod
    def validate_timestamp(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value

    @field_validator("usage", mode="before")
    @classmethod
    def copy_usage(cls, value: object) -> object:
        return _copy_model(value, BrowserAcceptanceUsageV1)

    @field_validator("diagnostic", mode="before")
    @classmethod
    def copy_diagnostic(cls, value: object) -> object:
        return _copy_model(value, BrowserAcceptanceDiagnosticV1)

    @field_validator("error_categories", "truncation_categories", mode="before")
    @classmethod
    def copy_count_map(cls, value: object, info) -> dict[str, Any]:
        copied = copy_durable_json_object(value, info.field_name)
        maximum = (
            BROWSER_ACCEPTANCE_MAX_ERROR_CATEGORIES
            if info.field_name == "error_categories"
            else BROWSER_ACCEPTANCE_MAX_TRUNCATION_CATEGORIES
        )
        if len(copied) > maximum:
            raise ValueError(f"{info.field_name} contains too many categories.")
        return copied

    @model_validator(mode="after")
    def validate_receipt(self) -> Self:
        if self.completed_at < self.started_at:
            raise ValueError("Browser acceptance trial completed_at precedes started_at.")
        expected_elapsed = max(
            int((self.completed_at - self.started_at).total_seconds() * 1_000),
            0,
        )
        if self.elapsed_ms != expected_elapsed:
            raise ValueError("Browser acceptance elapsed_ms conflicts with timestamps.")
        expected_row_id = _browser_acceptance_row_id(
            run_identity_revision=self.run_identity_revision,
            case_revision=self.case_revision,
            trial_number=self.trial_number,
            attempt_number=self.attempt_number,
        )
        if self.row_id != expected_row_id:
            raise ValueError("Browser acceptance row identity is invalid.")
        semantic_conforms = (
            self.semantic_state is BrowserAcceptanceSemanticState.PASSED
            if self.expected_state is BrowserAcceptanceState.PASSED
            else self.semantic_state
            in {
                BrowserAcceptanceSemanticState.PASSED,
                BrowserAcceptanceSemanticState.NOT_APPLICABLE,
            }
        )
        expected_conformance = (
            BrowserAcceptanceConformance.INCOMPLETE
            if self.infrastructure_state is BrowserAcceptanceInfrastructureState.UNAVAILABLE
            or self.completion_state is BrowserAcceptanceCompletionState.INCOMPLETE
            or self.diagnostic.state is BrowserAcceptanceDiagnosticState.UNAVAILABLE
            else BrowserAcceptanceConformance.PASSED
            if self.observed_state is self.expected_state and semantic_conforms
            else BrowserAcceptanceConformance.FAILED
        )
        if self.conformance is not expected_conformance:
            raise ValueError("Browser acceptance conformance conflicts with retained evidence.")
        material = self.model_dump(mode="json", exclude={"revision"})
        if self.revision != _content_revision(material, "browser acceptance trial"):
            raise ValueError("Browser acceptance trial revision does not match its content.")
        return self

    @classmethod
    def build(cls, **values: Any) -> BrowserAcceptanceTrialReceiptV1:
        material = dict(values)
        material["row_id"] = _browser_acceptance_row_id(
            run_identity_revision=material["run_identity_revision"],
            case_revision=material["case_revision"],
            trial_number=material["trial_number"],
            attempt_number=material.get("attempt_number", 1),
        )
        diagnostic = material["diagnostic"]
        diagnostic_state = (
            diagnostic.state
            if isinstance(diagnostic, BrowserAcceptanceDiagnosticV1)
            else diagnostic["state"]
        )
        expected_state = BrowserAcceptanceState(material["expected_state"])
        observed_state = BrowserAcceptanceState(material["observed_state"])
        semantic_state = BrowserAcceptanceSemanticState(material["semantic_state"])
        infrastructure_state = BrowserAcceptanceInfrastructureState(
            material["infrastructure_state"]
        )
        completion_state = BrowserAcceptanceCompletionState(material["completion_state"])
        diagnostic_state = BrowserAcceptanceDiagnosticState(diagnostic_state)
        semantic_conforms = (
            semantic_state is BrowserAcceptanceSemanticState.PASSED
            if expected_state is BrowserAcceptanceState.PASSED
            else semantic_state
            in {
                BrowserAcceptanceSemanticState.PASSED,
                BrowserAcceptanceSemanticState.NOT_APPLICABLE,
            }
        )
        material["conformance"] = (
            BrowserAcceptanceConformance.INCOMPLETE
            if infrastructure_state is BrowserAcceptanceInfrastructureState.UNAVAILABLE
            or completion_state is BrowserAcceptanceCompletionState.INCOMPLETE
            or diagnostic_state is BrowserAcceptanceDiagnosticState.UNAVAILABLE
            else BrowserAcceptanceConformance.PASSED
            if observed_state is expected_state and semantic_conforms
            else BrowserAcceptanceConformance.FAILED
        )
        return _build_content_bound(
            cls,
            material,
            "browser acceptance trial",
        )


class _BrowserAcceptanceTrialIntentV1(BaseModel):
    """Durable proof that one exact trial may already have dispatched."""

    model_config = _MODEL_CONFIG

    record_type: Literal["cayu.browser-acceptance-trial-intent"] = (
        "cayu.browser-acceptance-trial-intent"
    )
    schema_version: Literal[1] = BROWSER_ACCEPTANCE_SCHEMA_VERSION
    revision: StrictStr
    row_id: StrictStr
    run_identity_revision: StrictStr
    case_id: StrictStr
    case_revision: StrictStr
    trial_number: StrictInt = Field(ge=1, le=BROWSER_ACCEPTANCE_MAX_TRIALS)
    attempt_number: StrictInt = Field(ge=1, le=1_000)
    expected_state: BrowserAcceptanceState
    started_at: datetime

    @field_validator("revision", "row_id", "run_identity_revision", "case_revision")
    @classmethod
    def validate_revision(cls, value: str, info) -> str:
        return _revision(value, info.field_name)

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str, info) -> str:
        return _clean(value, info.field_name, maximum=128)

    @field_validator("started_at")
    @classmethod
    def validate_timestamp(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value

    @model_validator(mode="after")
    def validate_intent(self) -> Self:
        expected_row_id = _browser_acceptance_row_id(
            run_identity_revision=self.run_identity_revision,
            case_revision=self.case_revision,
            trial_number=self.trial_number,
            attempt_number=self.attempt_number,
        )
        if self.row_id != expected_row_id:
            raise ValueError("Browser acceptance trial intent row identity is invalid.")
        material = self.model_dump(mode="json", exclude={"revision"})
        if self.revision != _content_revision(material, "browser acceptance trial intent"):
            raise ValueError("Browser acceptance trial intent revision is invalid.")
        return self

    @classmethod
    def build(cls, **values: Any) -> _BrowserAcceptanceTrialIntentV1:
        material = dict(values)
        material["row_id"] = _browser_acceptance_row_id(
            run_identity_revision=material["run_identity_revision"],
            case_revision=material["case_revision"],
            trial_number=material["trial_number"],
            attempt_number=material["attempt_number"],
        )
        return _build_content_bound(
            cls,
            material,
            "browser acceptance trial intent",
        )


class BrowserAcceptanceCaseAggregateV1(BaseModel):
    model_config = _MODEL_CONFIG

    case_id: StrictStr
    case_revision: StrictStr
    variability: BrowserAcceptanceVariabilityState
    total_trials: StrictInt = Field(ge=0)
    conforming_trials: StrictInt = Field(ge=0)
    incomplete_trials: StrictInt = Field(ge=0)

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str, info) -> str:
        return _clean(value, info.field_name, maximum=128)

    @field_validator("case_revision")
    @classmethod
    def validate_case_revision(cls, value: str, info) -> str:
        return _revision(value, info.field_name)


class BrowserAcceptanceAggregateV1(BaseModel):
    model_config = _MODEL_CONFIG

    overall_status: BrowserAcceptanceConformance
    total_rows: StrictInt = Field(ge=1)
    conforming_rows: StrictInt = Field(ge=0)
    incomplete_rows: StrictInt = Field(ge=0)
    state_counts: dict[BrowserAcceptanceState, StrictInt]
    semantic_counts: dict[BrowserAcceptanceSemanticState, StrictInt]
    access_state_counts: dict[BrowserAcceptanceAccessState, StrictInt]
    error_counts: dict[StrictStr, StrictInt]
    truncation_counts: dict[StrictStr, StrictInt]
    total_model_steps: StrictInt | None = Field(default=None, ge=0)
    total_browser_operations: StrictInt | None = Field(default=None, ge=0)
    total_input_tokens: StrictInt | None = Field(default=None, ge=0)
    total_output_tokens: StrictInt | None = Field(default=None, ge=0)
    total_tokens: StrictInt | None = Field(default=None, ge=0)
    estimated_costs: dict[str, StrictStr] = Field(default_factory=dict)
    unpriced_model_steps: dict[str, StrictInt] = Field(default_factory=dict)
    total_artifact_bytes: StrictInt | None = Field(default=None, ge=0)
    total_elapsed_ms: StrictInt = Field(ge=0)
    limit_violations: tuple[StrictStr, ...] = Field(
        default=(),
        max_length=16,
    )

    @field_validator(
        "state_counts",
        "semantic_counts",
        "access_state_counts",
        "error_counts",
        "truncation_counts",
    )
    @classmethod
    def validate_counts(cls, value: dict[Any, int], info) -> dict[Any, int]:
        if any(type(item) is not int or item < 0 for item in value.values()):
            raise ValueError(f"{info.field_name} values must be nonnegative strict integers.")
        return value

    @field_validator("estimated_costs", mode="before")
    @classmethod
    def copy_costs(cls, value: object) -> dict[str, Any]:
        copied = copy_durable_json_object(value, "estimated_costs")
        if len(copied) > 32:
            raise ValueError("estimated_costs contains too many currencies.")
        return _validate_cost_evidence(copied, "estimated_costs")

    @field_validator("unpriced_model_steps", mode="before")
    @classmethod
    def copy_unpriced_steps(cls, value: object) -> dict[str, Any]:
        copied = copy_durable_json_object(value, "unpriced_model_steps")
        if len(copied) > 32 or any(
            type(currency) is not str
            or not currency.isalpha()
            or not currency.isupper()
            or len(currency.encode("utf-8")) > 16
            or type(count) is not int
            or count < 0
            for currency, count in copied.items()
        ):
            raise ValueError("unpriced_model_steps contains malformed evidence.")
        return dict(sorted(copied.items()))

    @field_validator("limit_violations")
    @classmethod
    def validate_limit_violations(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        cleaned = tuple(_clean(item, info.field_name, maximum=128) for item in value)
        if cleaned != tuple(sorted(set(cleaned))):
            raise ValueError("limit_violations must be unique and sorted.")
        return cleaned


class BrowserAcceptanceReportV1(BaseModel):
    model_config = _MODEL_CONFIG

    record_type: Literal["cayu.browser-acceptance-report"] = "cayu.browser-acceptance-report"
    schema_version: Literal[1] = BROWSER_ACCEPTANCE_SCHEMA_VERSION
    revision: StrictStr
    run_identity_revision: StrictStr
    manifest: BrowserAcceptanceManifestV1
    runtime_identity: BrowserAcceptanceRuntimeIdentityV1
    started_at: datetime
    completed_at: datetime
    rows: tuple[BrowserAcceptanceTrialReceiptV1, ...] = Field(
        min_length=1,
        max_length=BROWSER_ACCEPTANCE_MAX_ROWS,
    )
    prior_rows: tuple[BrowserAcceptanceTrialReceiptV1, ...] = Field(
        default=(),
        max_length=BROWSER_ACCEPTANCE_MAX_ROWS,
    )
    cases: tuple[BrowserAcceptanceCaseAggregateV1, ...]
    aggregate: BrowserAcceptanceAggregateV1
    source_report_revision: StrictStr | None = None

    @field_validator("revision", "run_identity_revision", "source_report_revision")
    @classmethod
    def validate_revision(cls, value: str | None, info) -> str | None:
        return None if value is None else _revision(value, info.field_name)

    @field_validator("started_at", "completed_at")
    @classmethod
    def validate_timestamp(cls, value: datetime, info) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{info.field_name} must be timezone-aware.")
        return value

    @field_validator("manifest", mode="before")
    @classmethod
    def copy_manifest(cls, value: object) -> object:
        return _copy_model(value, BrowserAcceptanceManifestV1)

    @field_validator("runtime_identity", mode="before")
    @classmethod
    def copy_runtime_identity(cls, value: object) -> object:
        return _copy_model(value, BrowserAcceptanceRuntimeIdentityV1)

    @field_validator("rows", mode="before")
    @classmethod
    def copy_rows(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            return value
        return tuple(_copy_model(item, BrowserAcceptanceTrialReceiptV1) for item in value)

    @field_validator("prior_rows", mode="before")
    @classmethod
    def copy_prior_rows(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            return value
        return tuple(_copy_model(item, BrowserAcceptanceTrialReceiptV1) for item in value)

    @field_validator("cases", mode="before")
    @classmethod
    def copy_case_aggregates(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            return value
        return tuple(_copy_model(item, BrowserAcceptanceCaseAggregateV1) for item in value)

    @field_validator("aggregate", mode="before")
    @classmethod
    def copy_aggregate(cls, value: object) -> object:
        return _copy_model(value, BrowserAcceptanceAggregateV1)

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if self.completed_at < self.started_at:
            raise ValueError("Browser acceptance report completed_at precedes started_at.")
        expected_identity = _run_identity_revision(self.manifest, self.runtime_identity)
        if self.run_identity_revision != expected_identity:
            raise ValueError("Browser acceptance run identity is invalid.")
        expected_row_keys = tuple(
            (case.case_id, case.revision, trial)
            for case in self.manifest.cases
            for trial in range(1, self.manifest.trial_count + 1)
        )
        actual_row_keys = tuple(
            (row.case_id, row.case_revision, row.trial_number) for row in self.rows
        )
        if actual_row_keys != expected_row_keys:
            raise ValueError("Browser acceptance report omits or reorders its trial matrix.")
        if (self.source_report_revision is None) != (not self.prior_rows):
            raise ValueError(
                "Browser acceptance retry reports require both source identity and prior rows."
            )
        history = self.prior_rows + self.rows
        history_keys = [
            (row.case_id, row.case_revision, row.trial_number, row.attempt_number)
            for row in history
        ]
        if len(set(history_keys)) != len(history_keys):
            raise ValueError("Browser acceptance report contains duplicate trial attempts.")
        for case_id, case_revision, trial_number in expected_row_keys:
            attempts = sorted(
                row.attempt_number
                for row in history
                if (
                    row.case_id,
                    row.case_revision,
                    row.trial_number,
                )
                == (case_id, case_revision, trial_number)
            )
            if attempts != list(range(1, max(attempts, default=0) + 1)):
                raise ValueError("Browser acceptance retry history is incomplete or conflicting.")
            latest = next(
                row
                for row in self.rows
                if row.case_id == case_id and row.trial_number == trial_number
            )
            if latest.attempt_number != attempts[-1]:
                raise ValueError(
                    "Browser acceptance rows must contain each trial's latest attempt."
                )
        manifest_cases = {case.case_id: case for case in self.manifest.cases}
        for row in history:
            case = manifest_cases.get(row.case_id)
            if (
                case is None
                or row.case_revision != case.revision
                or row.expected_state is not case.expected_state
                or row.run_identity_revision != self.run_identity_revision
            ):
                raise ValueError("Browser acceptance trial conflicts with its manifest authority.")
            allowed_currencies = set(self.runtime_identity.cost_currencies)
            if (
                set(row.usage.estimated_costs) - allowed_currencies
                or set(row.usage.unpriced_model_steps) - allowed_currencies
            ):
                raise ValueError(
                    "Browser acceptance trial cost evidence conflicts with pricing authority."
                )
        expected_cases = _case_aggregates(self.manifest, self.rows)
        if self.cases != expected_cases:
            raise ValueError("Browser acceptance case aggregates conflict with trial evidence.")
        expected_aggregate = _aggregate(self.manifest, self.rows)
        if self.aggregate != expected_aggregate:
            raise ValueError("Browser acceptance aggregate conflicts with trial evidence.")
        if (
            self.manifest.enabled
            and self.aggregate.overall_status is BrowserAcceptanceConformance.PASSED
            and self.runtime_identity.chromium_identity is None
        ):
            raise ValueError(
                "A conforming enabled browser report requires observed Chromium identity."
            )
        material = self.model_dump(mode="json", exclude={"revision"})
        if self.revision != _content_revision(material, "browser acceptance report"):
            raise ValueError("Browser acceptance report revision does not match its content.")
        if (
            compact_json_utf8_size(self.model_dump(mode="json"))
            > BROWSER_ACCEPTANCE_REPORT_MAX_BYTES
        ):
            raise ValueError("Browser acceptance report exceeds its byte bound.")
        return self


@dataclass(frozen=True, slots=True)
class BrowserAcceptanceScenarioExecutionV1:
    app: Any
    trial: EvalTrialResult
    fault: BrowserAcceptanceFaultEvidenceV1

    def __post_init__(self) -> None:
        from cayu.runtime.app import CayuApp

        if not isinstance(self.app, CayuApp):
            raise TypeError("scenario execution app must be a CayuApp.")
        object.__setattr__(self, "trial", EvalTrialResult.model_validate(self.trial))
        object.__setattr__(
            self,
            "fault",
            BrowserAcceptanceFaultEvidenceV1.model_validate(self.fault),
        )


@dataclass(frozen=True, slots=True)
class BrowserAcceptancePlanV1:
    """Trusted application plan loaded by the repository acceptance command."""

    manifest: BrowserAcceptanceManifestV1
    eval_plan: EvalPlan
    bridge: WebBridge
    pricing: PriceBook | None = None
    cost_currencies: tuple[str, ...] = ()
    scenario_executor_revision: str | None = None
    scenario_executor: (
        Callable[
            [BrowserAcceptanceCaseV1, int, int, float],
            Awaitable[BrowserAcceptanceScenarioExecutionV1],
        ]
        | None
    ) = None

    def __post_init__(self) -> None:
        if type(self.manifest) is not BrowserAcceptanceManifestV1:
            raise TypeError("manifest must be an exact BrowserAcceptanceManifestV1.")
        if type(self.eval_plan) is not EvalPlan:
            raise TypeError("eval_plan must be an exact EvalPlan.")
        if type(self.bridge) is not WebBridge:
            raise TypeError("bridge must be an exact WebBridge.")
        if self.bridge.kind is not WebBridgeProfileKind.SANDBOXED_BROWSER:
            raise ValueError("Browser acceptance requires a sandboxed WebBridge.")
        if self.bridge.browser_protocol is None or self.bridge.browser_worker_version is None:
            raise ValueError("Browser acceptance requires interactive browser identity.")
        if self.eval_plan.app is None or self.eval_plan.suite is None:
            raise ValueError("Browser acceptance requires a direct application EvalPlan.")
        if self.pricing is not None and type(self.pricing) is not PriceBook:
            raise TypeError("pricing must be an exact PriceBook or None.")
        cleaned_currencies = tuple(
            _clean(item, "cost_currencies", maximum=16) for item in self.cost_currencies
        )
        if cleaned_currencies != tuple(sorted(set(cleaned_currencies))) or any(
            not item.isalpha() or not item.isupper() for item in cleaned_currencies
        ):
            raise ValueError("cost_currencies must be unique sorted uppercase identifiers.")
        if bool(self.pricing) != bool(cleaned_currencies):
            raise ValueError("Pricing and cost currencies must be configured together.")
        if self.manifest.mode is BrowserAcceptanceMode.LIVE_PUBLIC and self.pricing is None:
            raise ValueError("Live browser acceptance requires exact pricing evidence.")
        if (self.scenario_executor is None) != (self.scenario_executor_revision is None):
            raise ValueError("Scenario execution and its revision must be configured together.")
        if self.scenario_executor_revision is not None:
            _revision(self.scenario_executor_revision, "scenario_executor_revision")
        if any(case.fault_scenario is not None for case in self.manifest.cases) and (
            self.scenario_executor is None
        ):
            raise ValueError("Executable fault cases require a scenario executor.")


def _registered_browser_acceptance_tool(plan: BrowserAcceptancePlanV1) -> BrowserSessionTool:
    app = plan.eval_plan.app
    suite = plan.eval_plan.suite
    if app is None or suite is None:  # narrowed by BrowserAcceptancePlanV1
        raise RuntimeError("Browser acceptance plan lost its executable suite.")
    if len(plan.bridge.tools) != 1 or type(plan.bridge.tools[0]) is not BrowserSessionTool:
        raise ValueError("Browser acceptance requires the closed browser_session tool.")
    bridge_tool = plan.bridge.tools[0]
    checked_agents: set[str] = set()
    for eval_case in suite.cases:
        agent_name = eval_case.request.agent_name
        if agent_name in checked_agents:
            continue
        registered_agent = app.get_agent(agent_name)
        if set(registered_agent.tools) != {"browser_session"} or registered_agent.hosted_tools:
            raise ValueError(
                "Browser acceptance requires one closed registered browser_session surface."
            )
        registered_tool = registered_agent.tools["browser_session"].tool
        if registered_tool is not bridge_tool:
            raise ValueError(
                "Browser acceptance plan WebBridge is not the registered browser_session tool."
            )
        checked_agents.add(agent_name)
    return bridge_tool


def _browser_public_operations(browser_tool: BrowserSessionTool) -> frozenset[str]:
    schema = browser_tool.schema
    properties = schema.get("properties")
    operation = properties.get("operation") if isinstance(properties, dict) else None
    values = operation.get("enum") if isinstance(operation, dict) else None
    if not isinstance(values, list) or any(type(item) is not str for item in values):
        raise ValueError("Browser acceptance could not inspect the public operation schema.")
    return frozenset(item for item in values if type(item) is str)


def _portable_execution_value(value: object, field_name: str) -> object:
    if value is None or type(value) in {str, int, bool}:
        return value
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", warnings="error")
    if isinstance(value, list | tuple):
        return [_portable_execution_value(item, field_name) for item in value]
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise ValueError(f"{field_name} contains a non-string key.")
        return {
            key: _portable_execution_value(item, field_name) for key, item in sorted(value.items())
        }
    raise ValueError(f"{field_name} contains unsupported execution material.")


def _browser_acceptance_execution_suite_fingerprint(plan: BrowserAcceptancePlanV1) -> str:
    app = plan.eval_plan.app
    suite = plan.eval_plan.suite
    if app is None or suite is None:  # pragma: no cover - plan validation owns this
        raise RuntimeError("Browser acceptance plan lost its executable suite.")
    cases: list[dict[str, object]] = []
    providers: dict[str, object] = {}
    for eval_case in suite.cases:
        target = app.resolve_run_model_target(eval_case.request)
        provider = app.get_provider(target.provider_name)
        assertions = [
            {
                "type": f"{type(assertion).__module__}.{type(assertion).__qualname__}",
                "revision": assertion.assertion_revision,
                "configuration": _portable_execution_value(
                    vars(assertion), "browser acceptance assertion"
                ),
            }
            for assertion in eval_case.assertions
        ]
        cases.append(
            {
                "case_id": eval_case.id,
                "request": eval_case.request.model_dump(mode="json", warnings="error"),
                "assertions": assertions,
                "metadata": eval_case.metadata,
            }
        )
        provider_material: dict[str, object] = {
            "type": f"{type(provider).__module__}.{type(provider).__qualname__}",
            "name": target.provider_name,
            "model": target.model,
        }
        if type(provider) is ScriptedModelProvider:
            provider_material["script"] = [
                [event.model_dump(mode="json", warnings="error") for event in batch]
                for batch in provider._batches
            ]
        else:
            execution_revision = getattr(provider, "execution_revision", None)
            if type(execution_revision) is str:
                provider_material["execution_revision"] = _revision(
                    execution_revision,
                    "browser acceptance provider execution revision",
                )
        providers[target.provider_name] = provider_material
    return _identity_fingerprint(
        {
            "suite_id": suite.id,
            "suite_metadata": suite.metadata,
            "cases": cases,
            "providers": providers,
            "scenario_executor_revision": plan.scenario_executor_revision,
        },
        "browser acceptance executable suite",
    )


def _manifest_origin_hosts(origins: tuple[str, ...]) -> tuple[str, ...]:
    hosts: list[str] = []
    for origin in origins:
        malformed = False
        try:
            split = urlsplit(origin)
            port = split.port
        except ValueError:
            malformed = True
            split = urlsplit("https://invalid.example")
            port = None
        if (
            malformed
            or split.scheme != "https"
            or split.hostname is None
            or split.username is not None
            or split.password is not None
            or port not in {None, 443}
            or split.path not in {"", "/"}
            or split.query
            or split.fragment
        ):
            raise ValueError("Browser acceptance origins must be canonical HTTPS origins.")
        canonical = f"https://{split.hostname.lower()}"
        if origin != canonical:
            raise ValueError("Browser acceptance origins must be canonical HTTPS origins.")
        hosts.append(split.hostname.lower())
    if len(hosts) != len(set(hosts)):
        raise ValueError("Browser acceptance origins must resolve to unique hosts.")
    return tuple(hosts)


def _require_browser_acceptance_egress_authority(
    manifest: BrowserAcceptanceManifestV1,
    authority: EgressAuthorityIdentity | None,
) -> EgressAuthorityIdentity:
    if type(authority) is not EgressAuthorityIdentity:
        raise ValueError("Browser acceptance requires factory-backed virtual-egress authority.")
    owned = EgressAuthorityIdentity.model_validate(authority)
    hosts = _manifest_origin_hosts(manifest.allowed_origins)
    if owned.authority_scope != "session":
        raise ValueError("Browser acceptance egress authority must be session-scoped.")
    if (
        len(owned.bindings) != len(hosts)
        or tuple(binding.destination for binding in owned.bindings) != hosts
        or any(binding.credential_kind != "credentialless" for binding in owned.bindings)
    ):
        raise ValueError(
            "Browser acceptance egress authority conflicts with the manifest allowlist."
        )
    policies = {policy.name: policy for policy in owned.policies}
    if any(
        policies[binding.policy_name].kind != "browser"
        or binding.destination not in policies[binding.policy_name].allowed_destinations
        for binding in owned.bindings
    ):
        raise ValueError("Browser acceptance requires browser-policy egress bindings.")
    return owned


def _require_browser_acceptance_execution_limits(plan: BrowserAcceptancePlanV1) -> None:
    suite = plan.eval_plan.suite
    if suite is None:  # narrowed by BrowserAcceptancePlanV1
        raise RuntimeError("Browser acceptance plan lost its EvalSuite.")
    limits = plan.manifest.limits
    browser_tool = _registered_browser_acceptance_tool(plan)
    artifact_limit = getattr(browser_tool, "max_artifact_bytes", None)
    operation_limit = getattr(browser_tool, "max_operations", None)
    if (
        type(artifact_limit) is not int
        or artifact_limit > limits.max_artifact_bytes
        or type(operation_limit) is not int
        or operation_limit > limits.max_browser_operations
    ):
        raise ValueError("Browser acceptance tool limits exceed the manifest ceilings.")
    maximum_model_steps = 0
    maximum_browser_operations = 0
    maximum_input_tokens = 0
    maximum_output_tokens = 0
    pricing_fingerprint = (
        None if plan.pricing is None else pricing_profile_identity(plan.pricing).fingerprint
    )
    common_live_app_budget: dict[str, Any] | None = None
    for eval_case in suite.cases:
        request = eval_case.request
        run_limits = request.limits
        maximum_model_steps += request.max_steps * plan.manifest.trial_count
        if request.max_steps > limits.max_model_steps:
            raise ValueError("Browser acceptance request exceeds the model-step ceiling.")
        if (
            run_limits.max_tool_calls is None
            or run_limits.max_tool_calls > limits.max_browser_operations
        ):
            raise ValueError("Browser acceptance request lacks its browser-operation ceiling.")
        maximum_browser_operations += run_limits.max_tool_calls * plan.manifest.trial_count
        maximum_seconds = max(1, (limits.max_wall_time_ms + 999) // 1_000)
        if (
            run_limits.max_elapsed_seconds is None
            or run_limits.max_elapsed_seconds > maximum_seconds
        ):
            raise ValueError("Browser acceptance request lacks its runtime wall-time ceiling.")
        if plan.manifest.mode is not BrowserAcceptanceMode.LIVE_PUBLIC:
            continue
        if (
            limits.max_input_tokens is None
            or limits.max_output_tokens is None
            or limits.max_estimated_cost is None
        ):  # pragma: no cover - manifest validator owns this invariant
            raise RuntimeError("Live browser acceptance lost its budget declaration.")
        if (
            run_limits.max_input_tokens is None
            or run_limits.max_input_tokens > limits.max_input_tokens
            or run_limits.max_output_tokens is None
            or run_limits.max_output_tokens > limits.max_output_tokens
        ):
            raise ValueError("Live browser acceptance request lacks its token ceilings.")
        maximum_input_tokens += run_limits.max_input_tokens * plan.manifest.trial_count
        maximum_output_tokens += run_limits.max_output_tokens * plan.manifest.trial_count
        amount, currency = _estimated_cost(limits.max_estimated_cost)
        if any(
            pricing_profile_identity(budget.pricing).fingerprint != pricing_fingerprint
            for budget in request.budget_limits
        ):
            raise ValueError(
                "Live browser acceptance budget conflicts with the report pricing authority."
            )
        app_budgets = tuple(budget for budget in request.budget_limits if budget.scope == "app")
        if (
            len(app_budgets) != 1
            or app_budgets[0].currency != currency
            or app_budgets[0].max_estimated_cost != amount
            or app_budgets[0].allow_unpriced
            or app_budgets[0].reservation is None
        ):
            raise ValueError(
                "Live browser acceptance requires one exact reserving app-wide cost ceiling."
            )
        app_budget_authority = app_budgets[0].model_dump(mode="json", warnings="error")
        if common_live_app_budget is None:
            common_live_app_budget = app_budget_authority
        elif app_budget_authority != common_live_app_budget:
            raise ValueError(
                "Live browser acceptance cases must share one exact app-budget authority."
            )
    if maximum_model_steps > limits.max_model_steps:
        raise ValueError("Browser acceptance suite exceeds its aggregate model-step ceiling.")
    if maximum_browser_operations > limits.max_browser_operations:
        raise ValueError("Browser acceptance suite exceeds its aggregate operation ceiling.")
    if artifact_limit * maximum_browser_operations > limits.max_artifact_bytes:
        raise ValueError("Browser acceptance suite exceeds its aggregate artifact ceiling.")
    if plan.manifest.mode is BrowserAcceptanceMode.LIVE_PUBLIC and (
        limits.max_input_tokens is None
        or limits.max_output_tokens is None
        or maximum_input_tokens > limits.max_input_tokens
        or maximum_output_tokens > limits.max_output_tokens
    ):
        raise ValueError("Live browser acceptance suite exceeds its aggregate token ceilings.")


def _unsupported_trial_receipt(
    *,
    case: BrowserAcceptanceCaseV1,
    run_identity_revision: str,
    public_operations: frozenset[str],
    observed_at: datetime,
    trial_number: int,
) -> BrowserAcceptanceTrialReceiptV1:
    observed = _public_schema_state(case, public_operations)
    return BrowserAcceptanceTrialReceiptV1.build(
        run_identity_revision=run_identity_revision,
        case_id=case.case_id,
        case_revision=case.revision,
        trial_number=trial_number,
        expected_state=case.expected_state,
        observed_state=observed,
        semantic_state=BrowserAcceptanceSemanticState.NOT_APPLICABLE,
        infrastructure_state=BrowserAcceptanceInfrastructureState.AVAILABLE,
        completion_state=BrowserAcceptanceCompletionState.COMPLETE,
        agent_report_state=BrowserAcceptanceAgentReportState.ABSENT,
        started_at=observed_at,
        completed_at=observed_at,
        elapsed_ms=0,
        usage=BrowserAcceptanceUsageV1(
            model_steps=0,
            browser_operations=0,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
        ),
        error_categories={},
        truncation_categories={},
        diagnostic=BrowserAcceptanceDiagnosticV1(
            state=BrowserAcceptanceDiagnosticState.NOT_REQUESTED
        ),
    )


def _trial_usage(
    trial: EvalTrialResult,
    *,
    browser_operations: int | None = None,
    estimated_costs: Mapping[str, str] | None = None,
    unpriced_model_steps: Mapping[str, int] | None = None,
) -> BrowserAcceptanceUsageV1:
    """Project exact retained usage independently of optional diagnostics."""

    if trial.usage_summary is None:
        return BrowserAcceptanceUsageV1(
            browser_operations=browser_operations,
            estimated_costs={} if estimated_costs is None else dict(estimated_costs),
            unpriced_model_steps=(
                {} if unpriced_model_steps is None else dict(unpriced_model_steps)
            ),
        )
    summary = SessionUsageSummary.model_validate(trial.usage_summary)
    return BrowserAcceptanceUsageV1(
        model_steps=summary.model_steps,
        browser_operations=browser_operations,
        input_tokens=int(summary.usage.input_tokens),
        output_tokens=int(summary.usage.output_tokens),
        total_tokens=int(summary.usage.total_tokens),
        estimated_costs={} if estimated_costs is None else dict(estimated_costs),
        unpriced_model_steps=({} if unpriced_model_steps is None else dict(unpriced_model_steps)),
    )


def _unavailable_trial_receipt(
    *,
    case: BrowserAcceptanceCaseV1,
    run_identity_revision: str,
    trial: EvalTrialResult,
    agent_report_state: BrowserAcceptanceAgentReportState,
    error_code: str,
    attempt_number: int = 1,
    observed_state: BrowserAcceptanceState | None = None,
) -> BrowserAcceptanceTrialReceiptV1:
    """Retain one row when bounded evidence projection cannot be completed."""

    owned_trial = EvalTrialResult.model_validate(trial)
    usage = _trial_usage(owned_trial)
    if observed_state is None:
        observed_state = {
            EvalStatus.PASSED: BrowserAcceptanceState.PASSED,
            EvalStatus.FAILED: BrowserAcceptanceState.FAILED,
            EvalStatus.ERROR: BrowserAcceptanceState.UNAVAILABLE,
            EvalStatus.UNAVAILABLE: BrowserAcceptanceState.UNAVAILABLE,
            EvalStatus.SKIPPED: BrowserAcceptanceState.UNAVAILABLE,
        }[owned_trial.status]
    return BrowserAcceptanceTrialReceiptV1.build(
        run_identity_revision=run_identity_revision,
        case_id=case.case_id,
        case_revision=case.revision,
        trial_number=owned_trial.trial_number,
        attempt_number=attempt_number,
        expected_state=case.expected_state,
        observed_state=observed_state,
        semantic_state=BrowserAcceptanceSemanticState.UNAVAILABLE,
        infrastructure_state=(
            BrowserAcceptanceInfrastructureState.UNAVAILABLE
            if owned_trial.status in {EvalStatus.ERROR, EvalStatus.UNAVAILABLE}
            else BrowserAcceptanceInfrastructureState.AVAILABLE
        ),
        completion_state=BrowserAcceptanceCompletionState.INCOMPLETE,
        agent_report_state=agent_report_state,
        started_at=owned_trial.started_at,
        completed_at=owned_trial.completed_at,
        elapsed_ms=owned_trial.duration_ms,
        usage=usage,
        error_categories={},
        truncation_categories={},
        diagnostic=BrowserAcceptanceDiagnosticV1(
            state=BrowserAcceptanceDiagnosticState.UNAVAILABLE,
            error_code=error_code,
        ),
    )


def _uninitialized_trial_receipt(
    *,
    case: BrowserAcceptanceCaseV1,
    run_identity_revision: str,
    trial_number: int,
    attempt_number: int,
    observed_at: datetime,
    error_code: str,
) -> BrowserAcceptanceTrialReceiptV1:
    """Retain one missing trial without fabricating execution or usage evidence."""

    return BrowserAcceptanceTrialReceiptV1.build(
        run_identity_revision=run_identity_revision,
        case_id=case.case_id,
        case_revision=case.revision,
        trial_number=trial_number,
        attempt_number=attempt_number,
        expected_state=case.expected_state,
        observed_state=BrowserAcceptanceState.UNAVAILABLE,
        semantic_state=BrowserAcceptanceSemanticState.UNAVAILABLE,
        infrastructure_state=BrowserAcceptanceInfrastructureState.UNAVAILABLE,
        completion_state=BrowserAcceptanceCompletionState.INCOMPLETE,
        agent_report_state=BrowserAcceptanceAgentReportState.UNAVAILABLE,
        started_at=observed_at,
        completed_at=observed_at,
        elapsed_ms=0,
        usage=BrowserAcceptanceUsageV1(),
        error_categories={},
        truncation_categories={},
        diagnostic=BrowserAcceptanceDiagnosticV1(
            state=BrowserAcceptanceDiagnosticState.UNAVAILABLE,
            error_code=error_code,
        ),
    )


def _agent_report_state(final_output: str) -> BrowserAcceptanceAgentReportState:
    if not final_output:
        return BrowserAcceptanceAgentReportState.ABSENT
    normalized = final_output.strip().lower()
    if normalized == "browser_acceptance:success":
        return BrowserAcceptanceAgentReportState.CLAIMED_SUCCESS
    if normalized == "browser_acceptance:failure":
        return BrowserAcceptanceAgentReportState.CLAIMED_FAILURE
    return BrowserAcceptanceAgentReportState.UNAVAILABLE


def _browser_identity_from_evidence(
    evidence_views: tuple[AssertionEvidenceView, ...],
    *,
    bridge: WebBridge,
) -> str | None:
    identities: set[str] = set()
    for evidence in evidence_views:
        for call in evidence.tool_calls:
            if call.tool_name != "browser_session" or call.result.state != "available":
                continue
            result = call.result.value
            if not isinstance(result, dict):
                continue
            structured = result.get("structured")
            if not isinstance(structured, dict):
                structured = result
            backend = structured.get("backend_identity")
            if not isinstance(backend, dict):
                continue
            if (
                backend.get("backend") != "playwright"
                or backend.get("browser") != "chromium"
                or backend.get("backend_version") != bridge.playwright_version
                or backend.get("worker_protocol") != bridge.browser_protocol
                or backend.get("worker_version") != bridge.browser_worker_version
            ):
                raise ValueError(
                    "Browser acceptance observed backend identity that conflicts with its "
                    "registered WebBridge."
                )
            browser_version = backend.get("browser_version")
            if type(browser_version) is not str:
                raise ValueError("Browser acceptance observed malformed Chromium identity.")
            identities.add(_clean(browser_version, "chromium_identity", maximum=256))
    if len(identities) > 1:
        raise ValueError("Browser acceptance observed more than one Chromium identity.")
    return next(iter(identities), None)


async def inspect_browser_acceptance_runtime_identity(
    plan: BrowserAcceptancePlanV1,
    *,
    evidence_views: tuple[AssertionEvidenceView, ...] = (),
) -> BrowserAcceptanceRuntimeIdentityV1:
    """Derive the report identity from the exact registered public execution path."""

    if type(plan) is not BrowserAcceptancePlanV1:
        raise TypeError("plan must be an exact BrowserAcceptancePlanV1.")
    app = plan.eval_plan.app
    suite = plan.eval_plan.suite
    if app is None or suite is None:  # narrowed by BrowserAcceptancePlanV1
        raise RuntimeError("Browser acceptance plan lost its direct eval application.")
    _registered_browser_acceptance_tool(plan)

    profile_fingerprints: set[str] = set()
    model_targets: set[tuple[str, str]] = set()
    environment_materials: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for eval_case in suite.cases:
        manifest_case = next(
            (case for case in plan.manifest.cases if case.case_id == eval_case.id),
            None,
        )
        if (
            manifest_case is None
            or eval_case.metadata.get("browser_acceptance_case_revision") != manifest_case.revision
        ):
            raise ValueError(
                "Browser acceptance executable case is not bound to its manifest revision."
            )
        profile_fingerprints.add(await app.inspect_run_execution_profile(eval_case.request))
        target = app.resolve_run_model_target(eval_case.request)
        if plan.manifest.mode is BrowserAcceptanceMode.DETERMINISTIC:
            from cayu.evals.internal.browser_acceptance import (
                BrowserAcceptanceDeterministicProvider,
            )

            provider = app.get_provider(target.provider_name)
            if type(provider) not in {
                ScriptedModelProvider,
                BrowserAcceptanceDeterministicProvider,
            }:
                raise ValueError(
                    "Deterministic browser acceptance requires Cayu's exact scripted provider."
                )
        model_targets.add((target.provider_name, target.model))
        try:
            factory = app.get_environment_factory(eval_case.request.environment_name)
        except RuntimeError:
            raise ValueError(
                "Browser acceptance requires a factory-backed execution environment."
            ) from None
        candidate = factory.construction_admission_candidate()
        authority = factory.execution_environment_authority()
        egress = _require_browser_acceptance_egress_authority(
            plan.manifest,
            factory.egress_authority_identity,
        )
        if candidate is None or authority is None:
            raise ValueError("Browser acceptance runtime identity is unavailable.")
        runner_material = {
            "candidate": candidate.model_dump(mode="json"),
            "environment_authority": authority.model_dump(mode="json"),
        }
        egress_material = {"egress_authority": egress.model_dump(mode="json")}
        environment_materials.append((runner_material, egress_material))

    if len(profile_fingerprints) != 1:
        raise ValueError("Browser acceptance cases do not share one execution profile.")
    if len(model_targets) != 1:
        raise ValueError("Browser acceptance cases do not share one provider/model target.")
    runner_fingerprints = {
        _identity_fingerprint(runner, "browser acceptance runner identity")
        for runner, _ in environment_materials
    }
    egress_fingerprints = {
        _identity_fingerprint(egress, "browser acceptance egress identity")
        for _, egress in environment_materials
    }
    if len(runner_fingerprints) != 1 or len(egress_fingerprints) != 1:
        raise ValueError("Browser acceptance cases do not share one execution environment.")
    provider_name, model = next(iter(model_targets))
    workload_fingerprint = _identity_fingerprint(
        {
            "name": PINNED_BROWSER_SESSION_WORKLOAD.name,
            "image": PINNED_BROWSER_SESSION_WORKLOAD.image,
            "command": list(PINNED_BROWSER_SESSION_WORKLOAD.command),
            "protocol_version": PINNED_BROWSER_SESSION_WORKLOAD.protocol_version,
            "worker_version": PINNED_BROWSER_SESSION_WORKLOAD.worker_version,
            "component_versions": [
                list(item) for item in PINNED_BROWSER_SESSION_WORKLOAD.component_versions
            ],
        },
        "browser acceptance workload identity",
    )
    artifact_store_id = plan.bridge.artifact_store_id
    if artifact_store_id is None:
        raise ValueError("Browser acceptance WebBridge has no artifact-store identity.")
    artifact_store_fingerprint = _identity_fingerprint(
        {"store_id": artifact_store_id},
        "browser acceptance artifact-store identity",
    )
    if (
        plan.bridge.browser_protocol is None
        or plan.bridge.browser_worker_version is None
        or plan.bridge.playwright_version is None
    ):
        raise ValueError("Browser acceptance WebBridge has incomplete worker identity.")
    pricing_identity: PricingProfileIdentityV1 | None = (
        None if plan.pricing is None else pricing_profile_identity(plan.pricing)
    )
    return BrowserAcceptanceRuntimeIdentityV1.build(
        runtime_build_provenance=current_runtime_build_provenance(),
        browser_protocol=plan.bridge.browser_protocol,
        browser_worker_version=plan.bridge.browser_worker_version,
        playwright_version=plan.bridge.playwright_version,
        chromium_identity=_browser_identity_from_evidence(evidence_views, bridge=plan.bridge),
        runner_fingerprint=next(iter(runner_fingerprints)),
        workload_fingerprint=workload_fingerprint,
        egress_fingerprint=next(iter(egress_fingerprints)),
        artifact_store_fingerprint=artifact_store_fingerprint,
        execution_profile_fingerprint=next(iter(profile_fingerprints)),
        execution_suite_fingerprint=_browser_acceptance_execution_suite_fingerprint(plan),
        provider_name=provider_name,
        model=model,
        pricing_profile_fingerprint=(
            None
            if pricing_identity is None
            else pricing_identity.fingerprint.removeprefix("sha256:")
        ),
        cost_currencies=plan.cost_currencies,
        platform_system=platform.system() or "unknown",
        platform_machine=platform.machine() or "unknown",
        python_version=platform.python_version(),
    )


def _persist_trial_receipt(
    receipt: BrowserAcceptanceTrialReceiptV1,
    receipt_directory: str | Path | None,
) -> None:
    """Publish one immutable trial receipt without replacing conflicting evidence."""

    if receipt_directory is None:
        return
    directory = Path(receipt_directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = _trial_receipt_destination(directory, receipt)
    encoded = (
        json.dumps(
            receipt.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".staging",
        dir=directory,
    )
    staging = Path(staging_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(staging, destination)
        except FileExistsError:
            existing = _load_trial_receipt_file(destination)
            if existing != receipt:
                raise ValueError(
                    "Browser acceptance trial receipt conflicts with durable evidence."
                ) from None
        _fsync_directory(directory)
        published = True
    finally:
        if published:
            staging.unlink(missing_ok=True)
            _trial_intent_destination(directory, receipt.row_id).unlink(missing_ok=True)
            _fsync_directory(directory)


def _trial_receipt_destination(
    directory: Path,
    receipt: BrowserAcceptanceTrialReceiptV1,
) -> Path:
    return directory / f"{receipt.row_id.removeprefix('sha256:')}.trial.json"


def _trial_intent_destination(directory: Path, row_id: str) -> Path:
    return directory / f"{row_id.removeprefix('sha256:')}.intent.json"


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    directory_descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _persist_trial_intent(
    intent: _BrowserAcceptanceTrialIntentV1,
    receipt_directory: str | Path | None,
) -> None:
    """Publish exact pre-dispatch authority before browser/model work can begin."""

    if receipt_directory is None:
        return
    directory = Path(receipt_directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = _trial_intent_destination(directory, intent.row_id)
    encoded = (
        json.dumps(
            intent.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".staging",
        dir=directory,
    )
    staging = Path(staging_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(staging, destination)
        except FileExistsError:
            existing = _load_trial_intent_file(destination)
            if existing != intent:
                raise ValueError(
                    "Browser acceptance trial intent conflicts with durable evidence."
                ) from None
        _fsync_directory(directory)
        published = True
    finally:
        if published:
            staging.unlink(missing_ok=True)
            _fsync_directory(directory)


def _load_trial_receipt_file(path: Path) -> BrowserAcceptanceTrialReceiptV1:
    with path.open("rb") as handle:
        raw = handle.read(BROWSER_ACCEPTANCE_REPORT_MAX_BYTES + 1)
    if len(raw) > BROWSER_ACCEPTANCE_REPORT_MAX_BYTES:
        raise ValueError("Browser acceptance trial receipt exceeds its byte bound.")
    return BrowserAcceptanceTrialReceiptV1.model_validate_json(raw)


def _load_trial_intent_file(path: Path) -> _BrowserAcceptanceTrialIntentV1:
    with path.open("rb") as handle:
        raw = handle.read(BROWSER_ACCEPTANCE_REPORT_MAX_BYTES + 1)
    if len(raw) > BROWSER_ACCEPTANCE_REPORT_MAX_BYTES:
        raise ValueError("Browser acceptance trial intent exceeds its byte bound.")
    return _BrowserAcceptanceTrialIntentV1.model_validate_json(raw)


def _staged_journal_destination(
    directory: Path,
    staging: Path,
    *,
    kind: Literal["trial", "intent"],
) -> Path:
    marker = f".{kind}.json."
    name = staging.name
    if not name.startswith(".") or not name.endswith(".staging") or marker not in name:
        raise ValueError("Browser acceptance journal staging path is malformed.")
    digest, separator, suffix = name[1 : -len(".staging")].partition(marker)
    if (
        not separator
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or not suffix
    ):
        raise ValueError("Browser acceptance journal staging path is malformed.")
    return directory / f"{digest}.{kind}.json"


def _require_trial_journal_file_bound(directory: Path) -> None:
    count = 0
    for pattern in (
        "*.trial.json",
        ".*.trial.json.*.staging",
        "*.intent.json",
        ".*.intent.json.*.staging",
    ):
        for _ in directory.glob(pattern):
            count += 1
            if count > BROWSER_ACCEPTANCE_MAX_ROWS * 4:
                raise ValueError("Browser acceptance receipt journal exceeds its file bound.")


def _reconcile_staged_trial_receipts(directory: Path) -> None:
    _require_trial_journal_file_bound(directory)
    staged_paths = tuple(sorted(directory.glob(".*.trial.json.*.staging")))
    directory_changed = False
    for staging in staged_paths:
        destination = _staged_journal_destination(directory, staging, kind="trial")
        try:
            receipt = _load_trial_receipt_file(staging)
        except FileNotFoundError:
            continue
        except ValueError:
            staging.unlink(missing_ok=True)
            directory_changed = True
            continue
        if destination != _trial_receipt_destination(directory, receipt):
            raise ValueError(
                "Browser acceptance staged receipt conflicts with its content identity."
            )
        try:
            os.link(staging, destination)
            directory_changed = True
        except FileExistsError:
            if _load_trial_receipt_file(destination) != receipt:
                raise ValueError(
                    "Browser acceptance trial receipt conflicts with durable evidence."
                ) from None
        except FileNotFoundError:
            if _load_trial_receipt_file(destination) != receipt:
                raise ValueError(
                    "Browser acceptance staged receipt disappeared without exact publication."
                ) from None
        staging.unlink(missing_ok=True)
        directory_changed = True
    if directory_changed:
        _fsync_directory(directory)


def _reconcile_staged_trial_intents(directory: Path) -> None:
    _require_trial_journal_file_bound(directory)
    staged_paths = tuple(sorted(directory.glob(".*.intent.json.*.staging")))
    directory_changed = False
    for staging in staged_paths:
        destination = _staged_journal_destination(directory, staging, kind="intent")
        try:
            intent = _load_trial_intent_file(staging)
        except FileNotFoundError:
            continue
        except ValueError:
            staging.unlink(missing_ok=True)
            directory_changed = True
            continue
        if destination != _trial_intent_destination(directory, intent.row_id):
            raise ValueError(
                "Browser acceptance staged intent conflicts with its content identity."
            )
        try:
            os.link(staging, destination)
            directory_changed = True
        except FileExistsError:
            if _load_trial_intent_file(destination) != intent:
                raise ValueError(
                    "Browser acceptance trial intent conflicts with durable evidence."
                ) from None
        except FileNotFoundError:
            if _load_trial_intent_file(destination) != intent:
                raise ValueError(
                    "Browser acceptance staged intent disappeared without exact publication."
                ) from None
        staging.unlink(missing_ok=True)
        directory_changed = True
    if directory_changed:
        _fsync_directory(directory)


def _require_receipt_matches_intent(
    receipt: BrowserAcceptanceTrialReceiptV1,
    intent: _BrowserAcceptanceTrialIntentV1,
) -> None:
    if (
        receipt.row_id != intent.row_id
        or receipt.run_identity_revision != intent.run_identity_revision
        or receipt.case_id != intent.case_id
        or receipt.case_revision != intent.case_revision
        or receipt.trial_number != intent.trial_number
        or receipt.attempt_number != intent.attempt_number
        or receipt.expected_state is not intent.expected_state
    ):
        raise ValueError("Browser acceptance receipt conflicts with its prepared trial intent.")


def _reconcile_trial_intents(
    directory: Path,
    *,
    run_identity_revision: str,
    manifest: BrowserAcceptanceManifestV1,
) -> None:
    cases = {case.case_id: case for case in manifest.cases}
    _require_trial_journal_file_bound(directory)
    intent_paths = tuple(sorted(directory.glob("*.intent.json")))
    for path in intent_paths:
        intent = _load_trial_intent_file(path)
        if path != _trial_intent_destination(directory, intent.row_id):
            raise ValueError("Browser acceptance trial intent conflicts with its content identity.")
        if intent.run_identity_revision != run_identity_revision:
            continue
        case = cases.get(intent.case_id)
        if (
            case is None
            or intent.case_revision != case.revision
            or intent.expected_state is not case.expected_state
        ):
            raise ValueError("Browser acceptance trial intent conflicts with its manifest.")
        receipt_path = directory / f"{intent.row_id.removeprefix('sha256:')}.trial.json"
        try:
            receipt = _load_trial_receipt_file(receipt_path)
        except FileNotFoundError:
            observed_at = max(datetime.now(UTC), intent.started_at)
            receipt = _uninitialized_trial_receipt(
                case=case,
                run_identity_revision=intent.run_identity_revision,
                trial_number=intent.trial_number,
                attempt_number=intent.attempt_number,
                observed_at=observed_at,
                error_code="trial_execution_interrupted",
            )
            _persist_trial_receipt(receipt, directory)
        else:
            _require_receipt_matches_intent(receipt, intent)
            path.unlink(missing_ok=True)
            _fsync_directory(directory)


def _load_trial_receipt_attempts(
    receipt_directory: str | Path | None,
    *,
    run_identity_revision: str,
    manifest: BrowserAcceptanceManifestV1,
) -> tuple[BrowserAcceptanceTrialReceiptV1, ...]:
    if receipt_directory is None:
        return ()
    directory = Path(receipt_directory)
    if not directory.exists():
        return ()
    _reconcile_staged_trial_receipts(directory)
    _reconcile_staged_trial_intents(directory)
    _reconcile_trial_intents(
        directory,
        run_identity_revision=run_identity_revision,
        manifest=manifest,
    )
    _require_trial_journal_file_bound(directory)
    paths = tuple(sorted(directory.glob("*.trial.json")))
    attempts: dict[tuple[str, int, int], BrowserAcceptanceTrialReceiptV1] = {}
    for path in paths:
        receipt = _load_trial_receipt_file(path)
        if path != _trial_receipt_destination(directory, receipt):
            raise ValueError(
                "Browser acceptance trial receipt conflicts with its content identity."
            )
        if receipt.run_identity_revision != run_identity_revision:
            continue
        attempt_key = (receipt.case_id, receipt.trial_number, receipt.attempt_number)
        existing = attempts.get(attempt_key)
        if existing is not None and existing != receipt:
            raise ValueError("Browser acceptance receipt journal contains conflicting attempts.")
        attempts[attempt_key] = receipt
    return tuple(attempts[key] for key in sorted(attempts))


def _latest_trial_receipts(
    attempts: tuple[BrowserAcceptanceTrialReceiptV1, ...],
) -> tuple[BrowserAcceptanceTrialReceiptV1, ...]:
    retained: dict[tuple[str, int], BrowserAcceptanceTrialReceiptV1] = {}
    for receipt in attempts:
        key = (receipt.case_id, receipt.trial_number)
        if key not in retained or receipt.attempt_number > retained[key].attempt_number:
            retained[key] = receipt
    return tuple(retained[key] for key in sorted(retained))


@asynccontextmanager
async def _exclusive_trial_journal(
    receipt_directory: str | Path | None,
    *,
    deadline: float,
) -> AsyncIterator[None]:
    """Serialize recovery and dispatch for one process-shared receipt journal."""

    if receipt_directory is None:
        yield
        return
    directory = Path(receipt_directory)
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(directory / ".browser-acceptance.lock", os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    operation_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        try:
            while not acquired:
                try:
                    if os.name == "nt":
                        import msvcrt

                        if os.fstat(descriptor).st_size == 0:
                            os.write(descriptor, b"\0")
                        os.lseek(descriptor, 0, os.SEEK_SET)
                        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except OSError as error:
                    if error.errno not in {errno.EACCES, errno.EAGAIN}:
                        raise
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise TimeoutError(
                            "Browser acceptance campaign deadline expired while waiting for "
                            "trial journal ownership."
                        ) from None
                    await asyncio.sleep(min(0.05, remaining))
            yield
        except BaseException as error:
            operation_error = error
    finally:
        # Closing the descriptor is the cross-platform final lock-release mechanism.
        try:
            os.close(descriptor)
        except BaseException as error:
            cleanup_error = error
    if operation_error is not None:
        if cleanup_error is not None:
            if cleanup_error.__context__ is operation_error:
                cleanup_error.__context__ = None
            raise operation_error from cleanup_error
        raise operation_error
    if cleanup_error is not None:
        raise cleanup_error


def _runtime_identity_with_chromium(
    identity: BrowserAcceptanceRuntimeIdentityV1,
    chromium_identity: str | None,
) -> BrowserAcceptanceRuntimeIdentityV1:
    if chromium_identity is None or identity.chromium_identity == chromium_identity:
        return identity
    if identity.chromium_identity is not None:
        raise ValueError("Browser acceptance Chromium identity changed during execution.")
    values = {
        field_name: getattr(identity, field_name)
        for field_name in type(identity).model_fields
        if field_name not in {"revision", "chromium_identity"}
    }
    return BrowserAcceptanceRuntimeIdentityV1.build(
        **values,
        chromium_identity=chromium_identity,
    )


async def run_browser_acceptance(
    plan: BrowserAcceptancePlanV1,
    *,
    deterministic_fixture: BrowserAcceptanceFixtureV1 | None = None,
    receipt_directory: str | Path | None = None,
    previous_report: BrowserAcceptanceReportV1 | None = None,
    retry_trials: tuple[tuple[str, int], ...] = (),
) -> BrowserAcceptanceReportV1:
    """Run one browser manifest through the public eval/application surface."""

    if type(plan) is not BrowserAcceptancePlanV1:
        raise TypeError("plan must be an exact BrowserAcceptancePlanV1.")
    loop = asyncio.get_running_loop()
    campaign_deadline = loop.time() + (plan.manifest.limits.max_wall_time_ms / 1_000)
    async with _exclusive_trial_journal(receipt_directory, deadline=campaign_deadline):
        return await _run_browser_acceptance_locked(
            plan,
            campaign_deadline=campaign_deadline,
            deterministic_fixture=deterministic_fixture,
            receipt_directory=receipt_directory,
            previous_report=previous_report,
            retry_trials=retry_trials,
        )


async def _run_browser_acceptance_locked(
    plan: BrowserAcceptancePlanV1,
    *,
    campaign_deadline: float,
    deterministic_fixture: BrowserAcceptanceFixtureV1 | None,
    receipt_directory: str | Path | None,
    previous_report: BrowserAcceptanceReportV1 | None,
    retry_trials: tuple[tuple[str, int], ...],
) -> BrowserAcceptanceReportV1:
    """Run after acquiring exclusive recovery and dispatch ownership."""

    loop = asyncio.get_running_loop()
    if plan.manifest.mode is BrowserAcceptanceMode.DETERMINISTIC:
        if type(deterministic_fixture) is not BrowserAcceptanceFixtureV1:
            raise ValueError("Deterministic browser acceptance requires its owned fixture.")
        # Prove the exact fixture is live before any application work begins.
        _ = deterministic_fixture.upstream_origin
    elif deterministic_fixture is not None:
        raise ValueError("Live browser acceptance cannot use the deterministic fixture.")
    app = plan.eval_plan.app
    suite = plan.eval_plan.suite
    if app is None or suite is None:  # narrowed by BrowserAcceptancePlanV1
        raise RuntimeError("Browser acceptance plan lost its direct eval application.")
    public_operations = _browser_public_operations(_registered_browser_acceptance_tool(plan))
    unsupported = tuple(
        case
        for case in plan.manifest.cases
        if case.semantic_oracle is BrowserAcceptanceSemanticOracle.PUBLIC_SCHEMA_UNSUPPORTED
    )
    executable = tuple(case for case in plan.manifest.cases if case not in unsupported)
    suite_case_ids = tuple(case.id for case in suite.cases)
    expected_case_ids = tuple(case.case_id for case in executable)
    if suite_case_ids != expected_case_ids:
        raise ValueError(
            "Browser acceptance EvalSuite must contain every executable manifest case in order."
        )
    _require_browser_acceptance_execution_limits(plan)
    preflight_identity = await inspect_browser_acceptance_runtime_identity(plan)
    run_identity_revision = _run_identity_revision(plan.manifest, preflight_identity)
    owned_previous = (
        None
        if previous_report is None
        else BrowserAcceptanceReportV1.model_validate(previous_report)
    )
    selected = set(retry_trials)
    if len(selected) != len(retry_trials):
        raise ValueError("Browser acceptance retry selection contains duplicates.")
    if owned_previous is None and selected:
        raise ValueError("retry_trials requires an exact previous_report.")
    if owned_previous is not None:
        if (
            owned_previous.manifest != plan.manifest
            or owned_previous.run_identity_revision != run_identity_revision
        ):
            changed_fields = tuple(
                sorted(
                    key
                    for key, value in preflight_identity.model_dump(
                        mode="json", exclude={"revision", "chromium_identity"}
                    ).items()
                    if value
                    != owned_previous.runtime_identity.model_dump(
                        mode="json", exclude={"revision", "chromium_identity"}
                    ).get(key)
                )
            )
            suffix = ",".join(changed_fields) or "manifest"
            raise ValueError(
                f"Browser acceptance retry runtime or manifest identity changed ({suffix})."
            )
        allowed = {
            (case.case_id, trial_number)
            for case in plan.manifest.cases
            for trial_number in range(1, plan.manifest.trial_count + 1)
        }
        if not selected or not selected.issubset(allowed):
            raise ValueError("Browser acceptance retry selection is empty or unknown.")
    journal_attempts = _load_trial_receipt_attempts(
        receipt_directory,
        run_identity_revision=run_identity_revision,
        manifest=plan.manifest,
    )
    recovered_rows = () if owned_previous is not None else _latest_trial_receipts(journal_attempts)
    manifest_cases = {case.case_id: case for case in plan.manifest.cases}
    for recovered in recovered_rows:
        case = manifest_cases.get(recovered.case_id)
        if (
            case is None
            or recovered.case_revision != case.revision
            or recovered.expected_state is not case.expected_state
            or recovered.attempt_number != 1
        ):
            raise ValueError("Browser acceptance receipt journal conflicts with its manifest.")
    policy = EvaluationEvidencePolicySpec.create(
        include_tool_arguments=True,
        include_tool_results=True,
    )
    suite_cases = {case.id: case for case in suite.cases}
    rows: list[BrowserAcceptanceTrialReceiptV1] = list(recovered_rows)
    evidence_views: list[AssertionEvidenceView] = []
    started_at: datetime | None = min((row.started_at for row in recovered_rows), default=None)
    completed_at: datetime | None = max((row.completed_at for row in recovered_rows), default=None)
    recovered_keys = {(row.case_id, row.trial_number) for row in recovered_rows}
    previous_rows: dict[tuple[str, int], BrowserAcceptanceTrialReceiptV1] = (
        {}
        if owned_previous is None
        else {(row.case_id, row.trial_number): row for row in owned_previous.rows}
    )
    if owned_previous is not None:
        history = {
            (row.case_id, row.trial_number, row.attempt_number): row
            for row in owned_previous.prior_rows + owned_previous.rows
        }
        journal_by_trial: dict[tuple[str, int], list[BrowserAcceptanceTrialReceiptV1]] = {}
        for receipt in journal_attempts:
            key = (receipt.case_id, receipt.trial_number)
            if key not in selected:
                continue
            case = manifest_cases.get(receipt.case_id)
            if (
                case is None
                or receipt.case_revision != case.revision
                or receipt.expected_state is not case.expected_state
            ):
                raise ValueError("Browser acceptance retry receipt conflicts with its manifest.")
            historical = history.get(
                (receipt.case_id, receipt.trial_number, receipt.attempt_number)
            )
            if historical is not None and historical != receipt:
                raise ValueError("Browser acceptance retry receipt conflicts with source history.")
            journal_by_trial.setdefault(key, []).append(receipt)
        recovered_retry_rows: list[BrowserAcceptanceTrialReceiptV1] = []
        for key in selected:
            previous_row = previous_rows[key]
            expected_attempt = previous_row.attempt_number + 1
            attempts = journal_by_trial.get(key, [])
            future = tuple(row for row in attempts if row.attempt_number > expected_attempt)
            if future:
                raise ValueError("Browser acceptance retry journal contains a future attempt.")
            exact = tuple(row for row in attempts if row.attempt_number == expected_attempt)
            if len(exact) > 1:
                raise ValueError("Browser acceptance retry journal contains conflicting attempts.")
            if exact:
                recovered_retry_rows.append(exact[0])
        rows.extend(recovered_retry_rows)
        recovered_keys.update((row.case_id, row.trial_number) for row in recovered_retry_rows)
    campaign_aborted = False
    campaign_abort_code = "campaign_aborted"
    for case in executable:
        eval_case = suite_cases[case.case_id]
        for trial_number in range(1, plan.manifest.trial_count + 1):
            key = (case.case_id, trial_number)
            if owned_previous is not None and key not in selected:
                continue
            if key in recovered_keys:
                continue
            previous_row = previous_rows.get(key)
            attempt_number = 1 if previous_row is None else previous_row.attempt_number + 1
            if not campaign_aborted and loop.time() >= campaign_deadline:
                campaign_aborted = True
                campaign_abort_code = "campaign_deadline_exceeded"
            if campaign_aborted:
                observed_at = datetime.now(UTC)
                projected = _uninitialized_trial_receipt(
                    case=case,
                    run_identity_revision=run_identity_revision,
                    trial_number=trial_number,
                    attempt_number=attempt_number,
                    observed_at=observed_at,
                    error_code=campaign_abort_code,
                )
                rows.append(projected)
                _persist_trial_receipt(projected, receipt_directory)
                continue
            _persist_trial_intent(
                _BrowserAcceptanceTrialIntentV1.build(
                    run_identity_revision=run_identity_revision,
                    case_id=case.case_id,
                    case_revision=case.revision,
                    trial_number=trial_number,
                    attempt_number=attempt_number,
                    expected_state=case.expected_state,
                    started_at=datetime.now(UTC),
                ),
                receipt_directory,
            )
            fixture_before = (
                {} if deterministic_fixture is None else deterministic_fixture.request_counts()
            )
            fault_evidence: BrowserAcceptanceFaultEvidenceV1 | None = None
            execution_app = app
            try:
                if case.fault_scenario is None:
                    run = await run_eval_suite(
                        app,
                        EvalSuite(id=suite.id, cases=[eval_case], metadata=suite.metadata),
                        retain_trajectory=True,
                        max_concurrency=1,
                        case_timeout_seconds=max(campaign_deadline - loop.time(), 0.001),
                        trials=1,
                    )
                    trial = run.cases[0].trials[0].model_copy(update={"trial_number": trial_number})
                    trial_started_at = run.started_at
                    trial_completed_at = run.completed_at
                else:
                    executor = plan.scenario_executor
                    if executor is None:  # pragma: no cover - plan validation owns this
                        raise RuntimeError("Browser acceptance fault executor is unavailable.")
                    scenario = await executor(
                        case,
                        trial_number,
                        attempt_number,
                        max(campaign_deadline - loop.time(), 0.001),
                    )
                    if type(scenario) is not BrowserAcceptanceScenarioExecutionV1:
                        raise TypeError(
                            "Browser acceptance scenario executor returned an invalid result."
                        )
                    execution_app = scenario.app
                    trial = scenario.trial
                    fault_evidence = scenario.fault
                    trial_started_at = trial.started_at
                    trial_completed_at = trial.completed_at
            except Exception:
                observed_at = datetime.now(UTC)
                projected = _uninitialized_trial_receipt(
                    case=case,
                    run_identity_revision=run_identity_revision,
                    trial_number=trial_number,
                    attempt_number=attempt_number,
                    observed_at=observed_at,
                    error_code="trial_execution_unavailable",
                )
                rows.append(projected)
                _persist_trial_receipt(projected, receipt_directory)
                # The dependency failure may not prove external work quiescent.
                # Preserve every remaining row without dispatching another trial.
                campaign_aborted = True
                continue
            started_at = (
                trial_started_at if started_at is None else min(started_at, trial_started_at)
            )
            completed_at = (
                trial_completed_at
                if completed_at is None
                else max(completed_at, trial_completed_at)
            )
            fixture_after = (
                {} if deterministic_fixture is None else deterministic_fixture.request_counts()
            )
            fixture_delta = {
                route: fixture_after.get(route, 0) - fixture_before.get(route, 0)
                for route in set(fixture_before) | set(fixture_after)
                if fixture_after.get(route, 0) > fixture_before.get(route, 0)
            }
            route_count = (
                fixture_delta.get(case.fixture_route, 0)
                if case.fixture_route is not None and case.fixture_route.startswith("/")
                else None
            )
            fixture_effects = {
                route.removeprefix("/effect/"): count
                for route, count in fixture_delta.items()
                if route.startswith("/effect/")
            }
            agent_state = _agent_report_state(trial.final_output)
            if trial.trajectory is None:
                projected = _unavailable_trial_receipt(
                    case=case,
                    run_identity_revision=run_identity_revision,
                    trial=trial,
                    agent_report_state=agent_state,
                    error_code="trajectory_unavailable",
                    attempt_number=attempt_number,
                )
                rows.append(projected)
                _persist_trial_receipt(projected, receipt_directory)
                continue
            fallback_observed_state: BrowserAcceptanceState | None = None
            try:
                evidence = project_assertion_evidence_view(
                    execution_app,
                    trial.trajectory,
                    evidence_policy=policy,
                    pricing=plan.pricing,
                    cost_currencies=plan.cost_currencies,
                )
                evidence_views.append(evidence)
                fallback_observed_state = _observed_state_from_evidence(
                    case,
                    trial,
                    evidence,
                    public_operations,
                )
                request_summaries, request_summaries_truncated = _request_summaries_from_trajectory(
                    trial
                )
                projected = project_browser_acceptance_trial(
                    case=case,
                    run_identity_revision=run_identity_revision,
                    trial=trial,
                    evidence=evidence,
                    fixture_route_observed=(None if route_count is None else route_count > 0),
                    fixture_route_request_count=route_count,
                    fixture_effects=fixture_effects,
                    public_operations=public_operations,
                    request_summaries=request_summaries,
                    request_summaries_truncated=request_summaries_truncated,
                    agent_report_state=agent_state,
                    fault=fault_evidence,
                    attempt_number=attempt_number,
                    required_cost_currencies=plan.cost_currencies,
                )
            except Exception:
                projected = _unavailable_trial_receipt(
                    case=case,
                    run_identity_revision=run_identity_revision,
                    trial=trial,
                    agent_report_state=agent_state,
                    error_code="diagnostic_projection_failed",
                    attempt_number=attempt_number,
                    observed_state=fallback_observed_state,
                )
            rows.append(projected)
            _persist_trial_receipt(projected, receipt_directory)
    owned_identity = await inspect_browser_acceptance_runtime_identity(
        plan,
        evidence_views=tuple(evidence_views),
    )
    chromium_identities = {
        row.diagnostic.chromium_identity
        for row in rows
        if row.diagnostic.chromium_identity is not None
    }
    if owned_identity.chromium_identity is not None:
        chromium_identities.add(owned_identity.chromium_identity)
    if len(chromium_identities) > 1:
        raise RuntimeError("Browser acceptance Chromium identity changed between trials.")
    owned_identity = _runtime_identity_with_chromium(
        owned_identity,
        next(iter(chromium_identities), None),
    )
    if preflight_identity.model_dump(
        mode="json",
        exclude={"revision", "chromium_identity"},
    ) != owned_identity.model_dump(
        mode="json",
        exclude={"revision", "chromium_identity"},
    ):
        raise RuntimeError("Browser acceptance runtime identity changed during execution.")
    observed_at = completed_at or datetime.now(UTC)
    for case in unsupported:
        for trial_number in range(1, plan.manifest.trial_count + 1):
            key = (case.case_id, trial_number)
            if owned_previous is not None and key not in selected:
                continue
            if key in recovered_keys:
                continue
            projected = _unsupported_trial_receipt(
                case=case,
                run_identity_revision=run_identity_revision,
                public_operations=public_operations,
                observed_at=observed_at,
                trial_number=trial_number,
            )
            if key in previous_rows:
                projected = BrowserAcceptanceTrialReceiptV1.build(
                    **projected.model_dump(
                        mode="python",
                        exclude={"revision", "row_id", "conformance", "attempt_number"},
                    ),
                    attempt_number=previous_rows[key].attempt_number + 1,
                )
            rows.append(projected)
            _persist_trial_receipt(projected, receipt_directory)
    if owned_previous is not None:
        previous_chromium = owned_previous.runtime_identity.chromium_identity
        observed_chromium = owned_identity.chromium_identity
        if (
            previous_chromium is not None
            and observed_chromium is not None
            and previous_chromium != observed_chromium
        ):
            raise ValueError("Browser acceptance retry observed a different Chromium identity.")
        retry_identity = (
            owned_previous.runtime_identity if observed_chromium is None else owned_identity
        )
        return build_browser_acceptance_retry_report(
            owned_previous,
            retry_rows=tuple(rows),
            completed_at=completed_at or observed_at,
            runtime_identity=retry_identity,
        )
    ordered_rows = tuple(
        row
        for case in plan.manifest.cases
        for trial_number in range(1, plan.manifest.trial_count + 1)
        for row in rows
        if row.case_id == case.case_id and row.trial_number == trial_number
    )
    report_started_at = min(row.started_at for row in ordered_rows)
    report_completed_at = max(row.completed_at for row in ordered_rows)
    return build_browser_acceptance_report(
        manifest=plan.manifest,
        runtime_identity=owned_identity,
        started_at=report_started_at,
        completed_at=report_completed_at,
        rows=ordered_rows,
    )


def _artifact_evidence(
    operation_revision: str,
    artifacts: object,
) -> tuple[BrowserAcceptanceArtifactEvidenceV1, ...]:
    if not isinstance(artifacts, list | tuple):
        return ()
    retained: list[BrowserAcceptanceArtifactEvidenceV1] = []
    for index, artifact in enumerate(artifacts[:BROWSER_ACCEPTANCE_MAX_ARTIFACTS_PER_ROW], 1):
        if not isinstance(artifact, dict):
            continue
        kind = artifact.get("kind")
        artifact_id = artifact.get("artifact_id")
        content_type = artifact.get("content_type")
        size_bytes = artifact.get("size_bytes")
        if (
            type(artifact_id) is not str
            or type(kind) is not str
            or type(content_type) is not str
            or type(size_bytes) is not int
        ):
            continue
        retained.append(
            BrowserAcceptanceArtifactEvidenceV1(
                artifact_id=artifact_id,
                artifact_revision=_content_revision(
                    {
                        "operation_revision": operation_revision,
                        "index": index,
                        "artifact_id": artifact_id,
                        "kind": kind,
                        "content_type": content_type,
                        "size_bytes": size_bytes,
                    },
                    "browser acceptance artifact",
                ),
                kind=kind,
                content_type=content_type,
                size_bytes=size_bytes,
            )
        )
    return tuple(retained)


def _operation_state(
    structured: dict[str, Any],
) -> tuple[BrowserAcceptanceOperationState, bool]:
    error = structured.get("error")
    execution = structured.get("execution")
    if not isinstance(execution, dict):
        return BrowserAcceptanceOperationState.INTENT, False
    admission = execution.get("admission")
    dispatch = execution.get("dispatch")
    observation = execution.get("observation")
    terminal = execution.get("terminal")
    valid = (
        (admission, dispatch, observation, terminal)
        == ("rejected", "not_started", "not_published", "settled")
        or (
            admission == "admitted"
            and dispatch == "completed"
            and observation in {"not_published", "published", "not_applicable"}
            and terminal == "settled"
        )
        or (
            (admission, dispatch, observation, terminal)
            == ("admitted", "acknowledgement_lost", "not_published", "outcome_ambiguous")
        )
    )
    if not valid:
        return BrowserAcceptanceOperationState.INTENT, False
    if error == "outcome_ambiguous":
        return BrowserAcceptanceOperationState.OUTCOME_AMBIGUOUS, True
    if dispatch == "not_started":
        return BrowserAcceptanceOperationState.OPERATION_NOT_DISPATCHED, True
    if terminal == "settled":
        return BrowserAcceptanceOperationState.TERMINAL, True
    if dispatch in {"completed", "acknowledgement_lost"}:
        return BrowserAcceptanceOperationState.DISPATCHED, True
    return BrowserAcceptanceOperationState.INTENT, False


def project_browser_acceptance_diagnostic(
    evidence: AssertionEvidenceView,
    *,
    fixture_route_observed: bool | None = None,
    fixture_route_request_count: int | None = None,
    browser_dispatches: int | None = None,
    fixture_effects: Mapping[str, int] | None = None,
    request_summaries: tuple[BrowserAcceptanceRequestSummaryV1, ...] = (),
    request_summaries_truncated: bool = False,
    capture_error_code: str | None = None,
    fault: BrowserAcceptanceFaultEvidenceV1 | None = None,
) -> BrowserAcceptanceDiagnosticV1:
    """Project the already-bounded eval evidence into browser-only diagnostics."""

    if type(evidence) is not AssertionEvidenceView:
        raise TypeError("evidence must be an exact AssertionEvidenceView.")
    if capture_error_code is not None:
        return BrowserAcceptanceDiagnosticV1(
            state=BrowserAcceptanceDiagnosticState.UNAVAILABLE,
            error_code=_clean(capture_error_code, "capture_error_code", maximum=128),
        )
    operations: list[BrowserAcceptanceOperationEvidenceV1] = []
    chromium_identities: set[str] = set()
    diagnostic_truncation: set[str] = set()
    if request_summaries_truncated:
        diagnostic_truncation.add("requests")
    browser_calls = tuple(
        call for call in evidence.tool_calls if call.tool_name == "browser_session"
    )
    if len(browser_calls) > BROWSER_ACCEPTANCE_MAX_OPERATIONS_PER_ROW:
        browser_calls = browser_calls[:BROWSER_ACCEPTANCE_MAX_OPERATIONS_PER_ROW]
        diagnostic_truncation.add("operations")
    if evidence.tool_call_evidence_state == "limit_exceeded":
        diagnostic_truncation.add("tool_calls")
    for sequence, call in enumerate(browser_calls, 1):
        arguments = call.arguments.value if call.arguments.state == "available" else None
        result = call.result.value if call.result.state == "available" else None
        if not isinstance(arguments, dict) or not isinstance(result, dict):
            diagnostic_truncation.add("operation_values")
            continue
        operation = arguments.get("operation")
        if type(operation) is not str:
            diagnostic_truncation.add("operation_values")
            continue
        structured = result.get("structured")
        if not isinstance(structured, dict):
            structured = result
        backend_identity = structured.get("backend_identity")
        if isinstance(backend_identity, dict):
            browser_identity = backend_identity.get("browser_version")
            if type(browser_identity) is str:
                chromium_identities.add(_clean(browser_identity, "chromium_identity", maximum=256))
        error = structured.get("error")
        error_category = error if type(error) is str else None
        disposition = structured.get("allocation_disposition")
        allocation_disposition = (
            BrowserAllocationDisposition(disposition)
            if type(disposition) is str and disposition in {"live", "retired", "uncertain"}
            else BrowserAllocationDisposition.UNAVAILABLE
        )
        operation_state, execution_complete = _operation_state(structured)
        if not execution_complete:
            diagnostic_truncation.add("execution_evidence")
        if (
            operation_state is not BrowserAcceptanceOperationState.OPERATION_NOT_DISPATCHED
            and allocation_disposition is BrowserAllocationDisposition.UNAVAILABLE
        ):
            diagnostic_truncation.add("allocation_disposition")
        observation_revision = structured.get("revision")
        refs = structured.get("refs")
        snapshot = structured.get("snapshot")
        ref_count = structured.get("ref_count")
        snapshot_bytes = structured.get("snapshot_bytes")
        load_state = structured.get("load_state")
        access_state = structured.get("access_state")
        if access_state is not None and (
            type(access_state) is not str
            or access_state not in {state.value for state in BrowserAcceptanceAccessState}
        ):
            diagnostic_truncation.add("access_state")
        truncation = structured.get("truncation_reasons")
        normalized_truncation = (
            tuple(
                sorted(
                    {
                        item
                        for item in truncation
                        if type(item) is str and len(item.encode("utf-8")) <= 128
                    }
                )
            )
            if isinstance(truncation, list | tuple)
            else ()
        )
        operations.append(
            BrowserAcceptanceOperationEvidenceV1(
                sequence=sequence,
                invocation_revision=call.invocation_revision,
                operation=operation,
                state=operation_state,
                error_category=error_category,
                allocation_disposition=allocation_disposition,
                target_revision=(
                    _content_revision(
                        {"url": arguments["url"]},
                        "browser acceptance operation target",
                    )
                    if type(arguments.get("url")) is str
                    else None
                ),
                observed_target_revision=(
                    _content_revision(
                        {"url": structured["url"]},
                        "browser acceptance observed target",
                    )
                    if type(structured.get("url")) is str
                    else None
                ),
                observation_revision=(
                    observation_revision if type(observation_revision) is str else None
                ),
                ref_count=(
                    len(refs)
                    if isinstance(refs, list | tuple)
                    else ref_count
                    if type(ref_count) is int and ref_count >= 0
                    else None
                ),
                snapshot_bytes=(
                    len(snapshot.encode("utf-8"))
                    if type(snapshot) is str
                    else snapshot_bytes
                    if type(snapshot_bytes) is int and snapshot_bytes >= 0
                    else None
                ),
                load_state=load_state if type(load_state) is str else None,
                access_state=(
                    BrowserAcceptanceAccessState(access_state)
                    if type(access_state) is str
                    and access_state in {state.value for state in BrowserAcceptanceAccessState}
                    else None
                ),
                truncation=normalized_truncation,
                artifacts=_artifact_evidence(call.invocation_revision, structured.get("artifacts")),
            )
        )
    owned_requests = tuple(
        BrowserAcceptanceRequestSummaryV1.model_validate(item) for item in request_summaries
    )
    if len(owned_requests) > BROWSER_ACCEPTANCE_MAX_REQUEST_SUMMARIES_PER_ROW:
        owned_requests = owned_requests[:BROWSER_ACCEPTANCE_MAX_REQUEST_SUMMARIES_PER_ROW]
        diagnostic_truncation.add("requests")
    if len(chromium_identities) > 1:
        raise ValueError("Browser acceptance trial observed multiple Chromium identities.")
    return BrowserAcceptanceDiagnosticV1(
        state=BrowserAcceptanceDiagnosticState.CAPTURED,
        fixture_route_observed=fixture_route_observed,
        fixture_route_request_count=fixture_route_request_count,
        browser_dispatches=browser_dispatches,
        fixture_effects={} if fixture_effects is None else dict(fixture_effects),
        chromium_identity=next(iter(chromium_identities), None),
        fault=fault,
        operations=tuple(operations),
        requests=owned_requests,
        truncated_categories=tuple(sorted(diagnostic_truncation)),
    )


def _request_summaries_from_trajectory(
    trial: EvalTrialResult,
) -> tuple[tuple[BrowserAcceptanceRequestSummaryV1, ...], bool]:
    trajectory = trial.trajectory
    if trajectory is None:
        return (), False
    request_events = tuple(
        event
        for event in trajectory.events
        if event.type
        in {
            EventType.EGRESS_REQUEST_AUTHORIZED,
            EventType.EGRESS_REQUEST_DENIED,
        }
    )
    retained = request_events[:BROWSER_ACCEPTANCE_MAX_REQUEST_SUMMARIES_PER_ROW]
    summaries: list[BrowserAcceptanceRequestSummaryV1] = []
    invalid_event = False
    for event in retained:
        method = event.payload.get("method")
        destination = event.payload.get("destination")
        path = event.payload.get("path")
        status_code = event.payload.get("status_code")
        if (
            type(method) is not str
            or type(destination) is not str
            or type(path) is not str
            or (status_code is not None and type(status_code) is not int)
        ):
            invalid_event = True
            continue
        summaries.append(
            BrowserAcceptanceRequestSummaryV1(
                sequence=len(summaries) + 1,
                method=method,
                destination_revision=_content_revision(
                    {"destination": destination},
                    "browser acceptance request destination",
                ),
                route_revision=_content_revision(
                    {"path": path},
                    "browser acceptance request route",
                ),
                outcome=(
                    "authorized" if event.type == EventType.EGRESS_REQUEST_AUTHORIZED else "denied"
                ),
                status_code=status_code,
            )
        )
    return (
        tuple(summaries),
        invalid_event or len(request_events) > BROWSER_ACCEPTANCE_MAX_REQUEST_SUMMARIES_PER_ROW,
    )


def _browser_dispatches_from_trial(trial: EvalTrialResult) -> int:
    """Count browser dispatches at the runtime-owned runner execution boundary."""

    trajectory = trial.trajectory
    if trajectory is None:
        return 0
    return sum(
        event.type is EventType.RUNNER_EXEC_STARTED and event.tool_name == "browser_session"
        for event in trajectory.events
    )


def _public_schema_state(
    case: BrowserAcceptanceCaseV1,
    public_operations: frozenset[str],
) -> BrowserAcceptanceState:
    operation = case.oracle_parameters.get("operation")
    if type(operation) is not str:
        return BrowserAcceptanceState.UNAVAILABLE
    return (
        BrowserAcceptanceState.SUPPORTED
        if operation in public_operations
        else BrowserAcceptanceState.UNSUPPORTED
    )


def _observed_state_from_evidence(
    case: BrowserAcceptanceCaseV1,
    trial: EvalTrialResult,
    evidence: AssertionEvidenceView,
    public_operations: frozenset[str],
) -> BrowserAcceptanceState:
    """Retain the browser outcome independently of optional diagnostic projection."""

    if trial.status in {EvalStatus.ERROR, EvalStatus.UNAVAILABLE}:
        return BrowserAcceptanceState.UNAVAILABLE
    if case.semantic_oracle is BrowserAcceptanceSemanticOracle.PUBLIC_SCHEMA_UNSUPPORTED:
        return _public_schema_state(case, public_operations)
    errors: list[str] = []
    for call in evidence.tool_calls:
        if call.tool_name != "browser_session" or call.result.state != "available":
            continue
        result = call.result.value
        if not isinstance(result, dict):
            continue
        structured = result.get("structured")
        if not isinstance(structured, dict):
            structured = result
        error = structured.get("error")
        if type(error) is str:
            errors.append(error)
    if "outcome_ambiguous" in errors:
        return BrowserAcceptanceState.AMBIGUOUS
    if errors and all(error in _REFUSAL_ERRORS for error in errors):
        return BrowserAcceptanceState.REFUSED
    if errors:
        return BrowserAcceptanceState.FAILED
    return BrowserAcceptanceState.PASSED


def _observed_state(
    case: BrowserAcceptanceCaseV1,
    trial: EvalTrialResult,
    diagnostic: BrowserAcceptanceDiagnosticV1,
    public_operations: frozenset[str],
) -> BrowserAcceptanceState:
    if trial.status in {EvalStatus.ERROR, EvalStatus.UNAVAILABLE}:
        return BrowserAcceptanceState.UNAVAILABLE
    if case.semantic_oracle is BrowserAcceptanceSemanticOracle.PUBLIC_SCHEMA_UNSUPPORTED:
        return _public_schema_state(case, public_operations)
    errors = tuple(
        operation.error_category
        for operation in diagnostic.operations
        if operation.error_category is not None
    )
    if "outcome_ambiguous" in errors:
        return BrowserAcceptanceState.AMBIGUOUS
    if errors and all(error in _REFUSAL_ERRORS for error in errors):
        return BrowserAcceptanceState.REFUSED
    if errors:
        return BrowserAcceptanceState.FAILED
    return BrowserAcceptanceState.PASSED


def _semantic_state(
    case: BrowserAcceptanceCaseV1,
    diagnostic: BrowserAcceptanceDiagnosticV1,
    *,
    public_operations: frozenset[str],
) -> BrowserAcceptanceSemanticState:
    del public_operations
    if case.semantic_oracle is BrowserAcceptanceSemanticOracle.PUBLIC_SCHEMA_UNSUPPORTED:
        return BrowserAcceptanceSemanticState.NOT_APPLICABLE
    if diagnostic.state is not BrowserAcceptanceDiagnosticState.CAPTURED:
        return BrowserAcceptanceSemanticState.UNAVAILABLE
    for operation in diagnostic.operations:
        if operation.state is BrowserAcceptanceOperationState.OPERATION_NOT_DISPATCHED:
            if operation.error_category is None:
                return BrowserAcceptanceSemanticState.FAILED
            continue
        if operation.allocation_disposition is BrowserAllocationDisposition.UNAVAILABLE:
            return BrowserAcceptanceSemanticState.FAILED
        if operation.error_category is None and (
            operation.state is not BrowserAcceptanceOperationState.TERMINAL
        ):
            return BrowserAcceptanceSemanticState.FAILED
    if case.fault_scenario is not None:
        fault = diagnostic.fault
        expected_dispatches = case.oracle_parameters.get("expected_browser_dispatches")
        if (
            fault is None
            or fault.scenario is not case.fault_scenario
            or not fault.boundary_observed
            or type(expected_dispatches) is not int
            or expected_dispatches < 0
            or fault.browser_dispatches != expected_dispatches
            or diagnostic.browser_dispatches != fault.browser_dispatches
            or (
                case.fault_scenario.value.startswith("process_")
                and not fault.recovered_in_fresh_app
            )
        ):
            return BrowserAcceptanceSemanticState.FAILED
    expected_target = case.fixture_route
    expects_fixture_route = expected_target is not None and expected_target.startswith("/")
    if expected_target is not None:
        if expects_fixture_route and diagnostic.fixture_route_observed is not True:
            return BrowserAcceptanceSemanticState.FAILED
        if expects_fixture_route:
            expected_target = f"https://docs.browser.test{expected_target}"
        expected_target_revision = _content_revision(
            {"url": expected_target},
            "browser acceptance operation target",
        )
        if not any(
            operation.operation == "navigate"
            and operation.target_revision == expected_target_revision
            for operation in diagnostic.operations
        ):
            return BrowserAcceptanceSemanticState.FAILED
    parameters = case.oracle_parameters
    expected_observed_target = parameters.get("expected_observed_target")
    if expected_observed_target is not None:
        if type(expected_observed_target) is not str:
            return BrowserAcceptanceSemanticState.FAILED
        expected_observed_revision = _content_revision(
            {"url": expected_observed_target},
            "browser acceptance observed target",
        )
        if not any(
            operation.operation == "navigate"
            and operation.observed_target_revision == expected_observed_revision
            for operation in diagnostic.operations
        ):
            return BrowserAcceptanceSemanticState.FAILED
    actual_operations = tuple(item.operation for item in diagnostic.operations)
    required_operations = parameters.get("required_operations", list(case.operations))
    if (
        not isinstance(required_operations, list | tuple)
        or any(type(item) is not str for item in required_operations)
        or actual_operations != tuple(required_operations)
    ):
        return BrowserAcceptanceSemanticState.FAILED
    forbidden_operations = parameters.get("forbidden_operations", [])
    if (
        not isinstance(forbidden_operations, list | tuple)
        or any(type(item) is not str for item in forbidden_operations)
        or set(actual_operations).intersection(forbidden_operations)
    ):
        return BrowserAcceptanceSemanticState.FAILED
    required_disposition = parameters.get("allocation_disposition")
    if required_disposition is not None and (
        type(required_disposition) is not str
        or not diagnostic.operations
        or diagnostic.operations[-1].allocation_disposition.value != required_disposition
    ):
        return BrowserAcceptanceSemanticState.FAILED
    expected_browser_dispatches = parameters.get("expected_browser_dispatches")
    if expected_browser_dispatches is not None and (
        type(expected_browser_dispatches) is not int
        or expected_browser_dispatches < 0
        or diagnostic.browser_dispatches != expected_browser_dispatches
    ):
        return BrowserAcceptanceSemanticState.FAILED
    required_access_state = parameters.get("access_state")
    if required_access_state is not None and (
        type(required_access_state) is not str
        or required_access_state not in {state.value for state in BrowserAcceptanceAccessState}
        or not diagnostic.operations
        or diagnostic.operations[-1].access_state is None
        or diagnostic.operations[-1].access_state.value != required_access_state
    ):
        return BrowserAcceptanceSemanticState.FAILED
    expected_route_requests = parameters.get("expected_route_requests")
    if expects_fixture_route and expected_route_requests is not None:
        if (
            type(expected_route_requests) is not int
            or expected_route_requests < 1
            or diagnostic.fixture_route_request_count != expected_route_requests
        ):
            return BrowserAcceptanceSemanticState.FAILED
    elif expects_fixture_route:
        minimum_route_requests = parameters.get("minimum_route_requests", 1)
        if (
            type(minimum_route_requests) is not int
            or minimum_route_requests < 1
            or diagnostic.fixture_route_request_count is None
            or diagnostic.fixture_route_request_count < minimum_route_requests
        ):
            return BrowserAcceptanceSemanticState.FAILED
    if case.semantic_oracle is BrowserAcceptanceSemanticOracle.FIXTURE_EFFECT:
        expected = parameters.get("expected_effects")
        return (
            BrowserAcceptanceSemanticState.PASSED
            if isinstance(expected, dict)
            and all(type(key) is str and type(value) is int for key, value in expected.items())
            and diagnostic.fixture_effects == expected
            else BrowserAcceptanceSemanticState.FAILED
        )
    if case.semantic_oracle is BrowserAcceptanceSemanticOracle.ARTIFACT:
        expected_kind = parameters.get("kind")
        artifacts = tuple(
            artifact for operation in diagnostic.operations for artifact in operation.artifacts
        )
        return (
            BrowserAcceptanceSemanticState.PASSED
            if type(expected_kind) is str and any(item.kind == expected_kind for item in artifacts)
            else BrowserAcceptanceSemanticState.FAILED
        )
    if case.semantic_oracle in {
        BrowserAcceptanceSemanticOracle.OBSERVATION,
        BrowserAcceptanceSemanticOracle.RECOVERY_STATE,
    }:
        required_truncation = parameters.get("required_truncation", [])
        observed_truncation = {
            reason for operation in diagnostic.operations for reason in operation.truncation
        }
        expected_error = parameters.get("error")
        return (
            BrowserAcceptanceSemanticState.PASSED
            if isinstance(required_truncation, list | tuple)
            and all(type(item) is str for item in required_truncation)
            and set(required_truncation).issubset(observed_truncation)
            and (
                expected_error is None
                or (
                    type(expected_error) is str
                    and any(
                        operation.error_category == expected_error
                        for operation in diagnostic.operations
                    )
                )
            )
            else BrowserAcceptanceSemanticState.FAILED
        )
    if case.semantic_oracle is BrowserAcceptanceSemanticOracle.STABLE_ERROR:
        expected_error = parameters.get("error")
        errors = tuple(
            item.error_category for item in diagnostic.operations if item.error_category is not None
        )
        return (
            BrowserAcceptanceSemanticState.PASSED
            if type(expected_error) is str and errors == (expected_error,)
            else BrowserAcceptanceSemanticState.FAILED
        )
    return BrowserAcceptanceSemanticState.UNAVAILABLE


def project_browser_acceptance_trial(
    *,
    case: BrowserAcceptanceCaseV1,
    run_identity_revision: str,
    trial: EvalTrialResult,
    evidence: AssertionEvidenceView,
    fixture_route_observed: bool | None = None,
    fixture_route_request_count: int | None = None,
    fixture_effects: Mapping[str, int] | None = None,
    public_operations: frozenset[str] = frozenset(),
    request_summaries: tuple[BrowserAcceptanceRequestSummaryV1, ...] = (),
    request_summaries_truncated: bool = False,
    agent_report_state: BrowserAcceptanceAgentReportState = (
        BrowserAcceptanceAgentReportState.UNAVAILABLE
    ),
    diagnostic_capture_error: str | None = None,
    fault: BrowserAcceptanceFaultEvidenceV1 | None = None,
    attempt_number: int = 1,
    required_cost_currencies: tuple[str, ...] = (),
) -> BrowserAcceptanceTrialReceiptV1:
    """Create one immutable browser receipt from generic eval evidence."""

    owned_case = BrowserAcceptanceCaseV1.model_validate(case)
    owned_trial = EvalTrialResult.model_validate(trial)
    owned_evidence = AssertionEvidenceView.model_validate(evidence)
    required_cost_currencies = tuple(
        _clean(item, "required_cost_currencies", maximum=16) for item in required_cost_currencies
    )
    if required_cost_currencies != tuple(sorted(set(required_cost_currencies))) or any(
        not item.isalpha() or not item.isupper() for item in required_cost_currencies
    ):
        raise ValueError("required_cost_currencies must be unique sorted uppercase identifiers.")
    diagnostic = project_browser_acceptance_diagnostic(
        owned_evidence,
        fixture_route_observed=fixture_route_observed,
        fixture_route_request_count=fixture_route_request_count,
        browser_dispatches=_browser_dispatches_from_trial(owned_trial),
        fixture_effects=fixture_effects,
        request_summaries=request_summaries,
        request_summaries_truncated=request_summaries_truncated,
        capture_error_code=diagnostic_capture_error,
        fault=fault,
    )
    infrastructure_state = (
        BrowserAcceptanceInfrastructureState.UNAVAILABLE
        if owned_trial.status in {EvalStatus.ERROR, EvalStatus.UNAVAILABLE}
        else BrowserAcceptanceInfrastructureState.AVAILABLE
    )
    completion_state = (
        BrowserAcceptanceCompletionState.COMPLETE
        if owned_trial.evidence_complete
        and diagnostic.state is not BrowserAcceptanceDiagnosticState.UNAVAILABLE
        and not diagnostic.truncated_categories
        and (
            not required_cost_currencies
            or (
                tuple(item.currency for item in owned_evidence.costs) == required_cost_currencies
                and all(item.unpriced_model_steps == 0 for item in owned_evidence.costs)
            )
        )
        else BrowserAcceptanceCompletionState.INCOMPLETE
    )
    observed_state = _observed_state(
        owned_case,
        owned_trial,
        diagnostic,
        public_operations,
    )
    semantic_state = _semantic_state(
        owned_case,
        diagnostic,
        public_operations=public_operations,
    )
    errors = Counter(
        operation.error_category
        for operation in diagnostic.operations
        if operation.error_category is not None
    )
    truncations = Counter(
        item for operation in diagnostic.operations for item in operation.truncation
    )
    truncations.update(diagnostic.truncated_categories)
    costs = {item.currency: item.total_cost for item in owned_evidence.costs}
    unpriced_model_steps = {
        item.currency: item.unpriced_model_steps for item in owned_evidence.costs
    }
    usage = _trial_usage(
        owned_trial,
        browser_operations=len(diagnostic.operations),
        estimated_costs=costs,
        unpriced_model_steps=unpriced_model_steps,
    )
    if (
        usage.total_tokens is not None
        and owned_evidence.total_tokens is not None
        and usage.total_tokens != int(owned_evidence.total_tokens)
    ):
        raise ValueError("Browser acceptance usage conflicts with projected evidence.")
    return BrowserAcceptanceTrialReceiptV1.build(
        run_identity_revision=_revision(run_identity_revision, "run_identity_revision"),
        case_id=owned_case.case_id,
        case_revision=owned_case.revision,
        trial_number=owned_trial.trial_number,
        attempt_number=attempt_number,
        expected_state=owned_case.expected_state,
        observed_state=observed_state,
        semantic_state=semantic_state,
        infrastructure_state=infrastructure_state,
        completion_state=completion_state,
        agent_report_state=agent_report_state,
        started_at=owned_trial.started_at,
        completed_at=owned_trial.completed_at,
        elapsed_ms=owned_trial.duration_ms,
        usage=usage,
        error_categories=dict(sorted(errors.items())),
        truncation_categories=dict(sorted(truncations.items())),
        diagnostic=diagnostic,
    )


def _case_aggregates(
    manifest: BrowserAcceptanceManifestV1,
    rows: tuple[BrowserAcceptanceTrialReceiptV1, ...],
) -> tuple[BrowserAcceptanceCaseAggregateV1, ...]:
    results: list[BrowserAcceptanceCaseAggregateV1] = []
    for case in manifest.cases:
        case_rows = tuple(row for row in rows if row.case_id == case.case_id)
        outcomes = {(row.observed_state, row.semantic_state, row.conformance) for row in case_rows}
        variability = (
            BrowserAcceptanceVariabilityState.NOT_APPLICABLE
            if manifest.mode is BrowserAcceptanceMode.DETERMINISTIC
            else BrowserAcceptanceVariabilityState.INCOMPLETE
            if len(case_rows) != manifest.trial_count
            or any(row.conformance is BrowserAcceptanceConformance.INCOMPLETE for row in case_rows)
            else BrowserAcceptanceVariabilityState.STABLE
            if len(outcomes) == 1
            else BrowserAcceptanceVariabilityState.VARIABLE
        )
        results.append(
            BrowserAcceptanceCaseAggregateV1(
                case_id=case.case_id,
                case_revision=case.revision,
                variability=variability,
                total_trials=len(case_rows),
                conforming_trials=sum(
                    row.conformance is BrowserAcceptanceConformance.PASSED for row in case_rows
                ),
                incomplete_trials=sum(
                    row.conformance is BrowserAcceptanceConformance.INCOMPLETE for row in case_rows
                ),
            )
        )
    return tuple(results)


def _aggregate(
    manifest: BrowserAcceptanceManifestV1,
    rows: tuple[BrowserAcceptanceTrialReceiptV1, ...],
) -> BrowserAcceptanceAggregateV1:
    required_ids = {case.case_id for case in manifest.cases if case.required}
    required_rows = tuple(row for row in rows if row.case_id in required_ids)
    overall = (
        BrowserAcceptanceConformance.INCOMPLETE
        if any(row.conformance is BrowserAcceptanceConformance.INCOMPLETE for row in required_rows)
        else BrowserAcceptanceConformance.PASSED
        if required_rows
        and all(row.conformance is BrowserAcceptanceConformance.PASSED for row in required_rows)
        else BrowserAcceptanceConformance.FAILED
    )
    state_counts = Counter(row.observed_state for row in rows)
    semantic_counts = Counter(row.semantic_state for row in rows)
    access_state_counts = Counter(
        operation.access_state
        for row in rows
        for operation in row.diagnostic.operations
        if operation.access_state is not None
    )
    error_counts: Counter[str] = Counter()
    truncation_counts: Counter[str] = Counter()
    for row in rows:
        error_counts.update(row.error_categories)
        if row.diagnostic.error_code is not None:
            error_counts[row.diagnostic.error_code] += 1
        truncation_counts.update(row.truncation_categories)
    total_model_steps = (
        None
        if any(row.usage.model_steps is None for row in rows)
        else sum(row.usage.model_steps or 0 for row in rows)
    )
    total_browser_operations = (
        None
        if any(row.usage.browser_operations is None for row in rows)
        else sum(row.usage.browser_operations or 0 for row in rows)
    )
    total_tokens = (
        None
        if any(row.usage.total_tokens is None for row in rows)
        else sum(row.usage.total_tokens or 0 for row in rows)
    )
    total_input_tokens = (
        None
        if any(row.usage.input_tokens is None for row in rows)
        else sum(row.usage.input_tokens or 0 for row in rows)
    )
    total_output_tokens = (
        None
        if any(row.usage.output_tokens is None for row in rows)
        else sum(row.usage.output_tokens or 0 for row in rows)
    )
    estimated_costs: dict[str, Decimal] = {}
    unpriced_model_steps: Counter[str] = Counter()
    for row in rows:
        for currency, amount in row.usage.estimated_costs.items():
            try:
                parsed = Decimal(amount)
            except InvalidOperation as exc:  # pragma: no cover - source evidence is validated
                raise ValueError("Browser acceptance cost evidence is malformed.") from exc
            if not parsed.is_finite() or parsed < 0:
                raise ValueError("Browser acceptance cost evidence is malformed.")
            estimated_costs[currency] = estimated_costs.get(currency, Decimal(0)) + parsed
        unpriced_model_steps.update(row.usage.unpriced_model_steps)
    total_artifact_bytes = (
        None
        if any(row.diagnostic.state is BrowserAcceptanceDiagnosticState.UNAVAILABLE for row in rows)
        else sum(
            artifact.size_bytes
            for row in rows
            for operation in row.diagnostic.operations
            for artifact in operation.artifacts
        )
    )
    total_elapsed_ms = sum(row.elapsed_ms for row in rows)
    violations: set[str] = set()
    for value, ceiling, label in (
        (total_model_steps, manifest.limits.max_model_steps, "model_steps"),
        (
            total_browser_operations,
            manifest.limits.max_browser_operations,
            "browser_operations",
        ),
        (total_artifact_bytes, manifest.limits.max_artifact_bytes, "artifact_bytes"),
        (total_elapsed_ms, manifest.limits.max_wall_time_ms, "wall_time"),
    ):
        if value is not None and value > ceiling:
            violations.add(label)
    if (
        manifest.limits.max_input_tokens is not None
        and total_input_tokens is not None
        and total_input_tokens > manifest.limits.max_input_tokens
    ):
        violations.add("input_tokens")
    if (
        manifest.limits.max_output_tokens is not None
        and total_output_tokens is not None
        and total_output_tokens > manifest.limits.max_output_tokens
    ):
        violations.add("output_tokens")
    if manifest.limits.max_estimated_cost is not None:
        cost_ceiling, cost_currency = _estimated_cost(manifest.limits.max_estimated_cost)
        if estimated_costs.get(cost_currency, Decimal(0)) > cost_ceiling:
            violations.add("estimated_cost")
    if violations and overall is BrowserAcceptanceConformance.PASSED:
        overall = BrowserAcceptanceConformance.FAILED
    return BrowserAcceptanceAggregateV1(
        overall_status=overall,
        total_rows=len(rows),
        conforming_rows=sum(row.conformance is BrowserAcceptanceConformance.PASSED for row in rows),
        incomplete_rows=sum(
            row.conformance is BrowserAcceptanceConformance.INCOMPLETE for row in rows
        ),
        state_counts={state: state_counts[state] for state in BrowserAcceptanceState},
        semantic_counts={state: semantic_counts[state] for state in BrowserAcceptanceSemanticState},
        access_state_counts={
            state: access_state_counts[state] for state in BrowserAcceptanceAccessState
        },
        error_counts=dict(sorted(error_counts.items())),
        truncation_counts=dict(sorted(truncation_counts.items())),
        total_model_steps=total_model_steps,
        total_browser_operations=total_browser_operations,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_tokens=total_tokens,
        estimated_costs={
            currency: format(amount, "f") for currency, amount in sorted(estimated_costs.items())
        },
        unpriced_model_steps=dict(sorted(unpriced_model_steps.items())),
        total_artifact_bytes=total_artifact_bytes,
        total_elapsed_ms=total_elapsed_ms,
        limit_violations=tuple(sorted(violations)),
    )


def build_browser_acceptance_report(
    *,
    manifest: BrowserAcceptanceManifestV1,
    runtime_identity: BrowserAcceptanceRuntimeIdentityV1,
    started_at: datetime,
    completed_at: datetime,
    rows: tuple[BrowserAcceptanceTrialReceiptV1, ...],
    prior_rows: tuple[BrowserAcceptanceTrialReceiptV1, ...] = (),
    source_report_revision: str | None = None,
) -> BrowserAcceptanceReportV1:
    owned_manifest = BrowserAcceptanceManifestV1.model_validate(manifest)
    owned_identity = BrowserAcceptanceRuntimeIdentityV1.model_validate(runtime_identity)
    run_identity_revision = _run_identity_revision(owned_manifest, owned_identity)
    owned_rows = tuple(BrowserAcceptanceTrialReceiptV1.model_validate(row) for row in rows)
    material: dict[str, Any] = {
        "run_identity_revision": run_identity_revision,
        "manifest": owned_manifest,
        "runtime_identity": owned_identity,
        "started_at": started_at,
        "completed_at": completed_at,
        "rows": owned_rows,
        "prior_rows": tuple(
            BrowserAcceptanceTrialReceiptV1.model_validate(row) for row in prior_rows
        ),
        "cases": _case_aggregates(owned_manifest, owned_rows),
        "aggregate": _aggregate(owned_manifest, owned_rows),
        "source_report_revision": source_report_revision,
    }
    return _build_content_bound(
        BrowserAcceptanceReportV1,
        material,
        "browser acceptance report",
    )


def build_browser_acceptance_retry_report(
    previous: BrowserAcceptanceReportV1,
    *,
    retry_rows: tuple[BrowserAcceptanceTrialReceiptV1, ...],
    completed_at: datetime,
    runtime_identity: BrowserAcceptanceRuntimeIdentityV1 | None = None,
) -> BrowserAcceptanceReportV1:
    """Replace selected latest rows while retaining every immutable prior attempt."""

    owned_previous = BrowserAcceptanceReportV1.model_validate(previous)
    owned_runtime_identity = (
        owned_previous.runtime_identity
        if runtime_identity is None
        else BrowserAcceptanceRuntimeIdentityV1.model_validate(runtime_identity)
    )
    if (
        _run_identity_revision(owned_previous.manifest, owned_runtime_identity)
        != owned_previous.run_identity_revision
    ):
        raise ValueError("Browser acceptance retry runtime identity conflicts with its source.")
    previous_chromium = owned_previous.runtime_identity.chromium_identity
    retry_chromium = owned_runtime_identity.chromium_identity
    if previous_chromium is not None and retry_chromium != previous_chromium:
        raise ValueError("Browser acceptance retry changed known Chromium identity.")
    replacements = {
        (row.case_id, row.trial_number): BrowserAcceptanceTrialReceiptV1.model_validate(row)
        for row in retry_rows
    }
    if len(replacements) != len(retry_rows):
        raise ValueError("Browser acceptance retry rows contain duplicate trial identities.")
    current = {(row.case_id, row.trial_number): row for row in owned_previous.rows}
    unknown = set(replacements) - set(current)
    if unknown:
        raise ValueError("Browser acceptance retry rows do not belong to the source report.")
    for key, replacement in replacements.items():
        prior = current[key]
        if (
            replacement.run_identity_revision != owned_previous.run_identity_revision
            or replacement.case_revision != prior.case_revision
            or replacement.expected_state is not prior.expected_state
            or replacement.attempt_number != prior.attempt_number + 1
        ):
            raise ValueError("Browser acceptance retry row conflicts with source authority.")
    rows = tuple(
        replacements.get((row.case_id, row.trial_number), row) for row in owned_previous.rows
    )
    retired = tuple(
        row for row in owned_previous.rows if (row.case_id, row.trial_number) in replacements
    )
    return build_browser_acceptance_report(
        manifest=owned_previous.manifest,
        runtime_identity=owned_runtime_identity,
        started_at=owned_previous.started_at,
        completed_at=completed_at,
        rows=rows,
        prior_rows=owned_previous.prior_rows + retired,
        source_report_revision=owned_previous.revision,
    )


def browser_acceptance_report_to_json(report: BrowserAcceptanceReportV1) -> str:
    if type(report) is not BrowserAcceptanceReportV1:
        raise TypeError("report must be an exact BrowserAcceptanceReportV1.")
    validated = BrowserAcceptanceReportV1.model_validate(report)
    encoded = json.dumps(
        validated.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if len(encoded.encode("utf-8")) > BROWSER_ACCEPTANCE_REPORT_MAX_BYTES:
        raise ValueError("Browser acceptance report exceeds its JSON byte bound.")
    return encoded


def browser_acceptance_report_from_json(source: str | bytes) -> BrowserAcceptanceReportV1:
    raw = source.encode("utf-8") if isinstance(source, str) else source
    if type(raw) is not bytes:
        raise TypeError("source must be str or bytes.")
    if len(raw) > BROWSER_ACCEPTANCE_REPORT_MAX_BYTES:
        raise ValueError("Browser acceptance report exceeds its JSON byte bound.")
    return BrowserAcceptanceReportV1.model_validate_json(raw)


def render_browser_acceptance_html(report: BrowserAcceptanceReportV1) -> str:
    validated = BrowserAcceptanceReportV1.model_validate(report)
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(row.case_id)}</td>"
        f"<td>{row.trial_number}</td>"
        f"<td>{html.escape(row.expected_state.value)}</td>"
        f"<td>{html.escape(row.observed_state.value)}</td>"
        f"<td>{html.escape(row.semantic_state.value)}</td>"
        f"<td>{html.escape(row.conformance.value)}</td>"
        f"<td>{row.usage.browser_operations}</td>"
        f"<td>{row.elapsed_ms}</td>"
        "</tr>"
        for row in validated.rows
    )
    document = (
        '<!doctype html><meta charset="utf-8"><title>Browser acceptance</title>'
        "<h1>Browser acceptance</h1>"
        f"<p>Report <code>{html.escape(validated.revision)}</code></p>"
        f"<p>Overall: <strong>{html.escape(validated.aggregate.overall_status.value)}</strong></p>"
        "<table><thead><tr><th>Case</th><th>Trial</th><th>Expected</th>"
        "<th>Observed</th><th>Semantic</th><th>Conformance</th>"
        "<th>Browser operations</th><th>Elapsed ms</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )
    if len(document.encode("utf-8")) > BROWSER_ACCEPTANCE_HTML_MAX_BYTES:
        raise ValueError("Browser acceptance HTML exceeds its byte bound.")
    return document


def _publish_browser_acceptance_report_file(
    destination: Path,
    content: bytes,
    *,
    maximum_bytes: int,
) -> None:
    """Publish or exactly reconcile one immutable report representation."""

    if len(content) > maximum_bytes:
        raise ValueError("Browser acceptance report representation exceeds its byte bound.")
    directory = destination.parent
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".staging",
        dir=directory,
    )
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(staging, destination)
        except FileExistsError:
            try:
                size = destination.stat().st_size
                if size > maximum_bytes or destination.read_bytes() != content:
                    raise ValueError(
                        "Browser acceptance report conflicts with immutable durable output."
                    ) from None
            except FileNotFoundError:
                # A concurrent publisher can disappear only through external mutation;
                # retrying the immutable link remains safe.
                try:
                    os.link(staging, destination)
                except FileExistsError:
                    if (
                        destination.stat().st_size > maximum_bytes
                        or destination.read_bytes() != content
                    ):
                        raise ValueError(
                            "Browser acceptance report conflicts with immutable durable output."
                        ) from None
        if os.name != "nt":
            directory_descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        staging.unlink(missing_ok=True)


def write_browser_acceptance_report(
    report: BrowserAcceptanceReportV1,
    output_directory: str | Path,
) -> tuple[Path, Path]:
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = report.revision.replace(":", "-")
    json_path = directory / f"{stem}.json"
    html_path = directory / f"{stem}.html"
    json_content = (browser_acceptance_report_to_json(report) + "\n").encode("utf-8")
    html_content = (render_browser_acceptance_html(report) + "\n").encode("utf-8")
    _publish_browser_acceptance_report_file(
        json_path,
        json_content,
        maximum_bytes=BROWSER_ACCEPTANCE_REPORT_MAX_BYTES + 1,
    )
    _publish_browser_acceptance_report_file(
        html_path,
        html_content,
        maximum_bytes=BROWSER_ACCEPTANCE_HTML_MAX_BYTES + 1,
    )
    return json_path, html_path


__all__ = [
    "BROWSER_ACCEPTANCE_HTML_MAX_BYTES",
    "BROWSER_ACCEPTANCE_MANIFEST_MAX_BYTES",
    "BROWSER_ACCEPTANCE_REPORT_MAX_BYTES",
    "BROWSER_ACCEPTANCE_SCHEMA_VERSION",
    "BrowserAcceptanceAccessState",
    "BrowserAcceptanceAgentReportState",
    "BrowserAcceptanceAggregateV1",
    "BrowserAcceptanceArtifactEvidenceV1",
    "BrowserAcceptanceCaseAggregateV1",
    "BrowserAcceptanceCaseCategory",
    "BrowserAcceptanceCaseV1",
    "BrowserAcceptanceCompletionState",
    "BrowserAcceptanceConformance",
    "BrowserAcceptanceDiagnosticState",
    "BrowserAcceptanceDiagnosticV1",
    "BrowserAcceptanceFaultEvidenceV1",
    "BrowserAcceptanceFaultScenario",
    "BrowserAcceptanceInfrastructureState",
    "BrowserAcceptanceLimitsV1",
    "BrowserAcceptanceManifestV1",
    "BrowserAcceptanceMode",
    "BrowserAcceptanceOperationEvidenceV1",
    "BrowserAcceptanceOperationState",
    "BrowserAcceptancePlanV1",
    "BrowserAcceptanceReportV1",
    "BrowserAcceptanceRequestSummaryV1",
    "BrowserAcceptanceRuntimeIdentityV1",
    "BrowserAcceptanceScenarioExecutionV1",
    "BrowserAcceptanceSemanticOracle",
    "BrowserAcceptanceSemanticState",
    "BrowserAcceptanceState",
    "BrowserAcceptanceTrialReceiptV1",
    "BrowserAcceptanceUsageV1",
    "BrowserAcceptanceVariabilityState",
    "BrowserAllocationDisposition",
    "browser_acceptance_report_from_json",
    "browser_acceptance_report_to_json",
    "build_browser_acceptance_report",
    "build_browser_acceptance_retry_report",
    "inspect_browser_acceptance_runtime_identity",
    "project_browser_acceptance_diagnostic",
    "project_browser_acceptance_trial",
    "render_browser_acceptance_html",
    "run_browser_acceptance",
    "write_browser_acceptance_report",
]
