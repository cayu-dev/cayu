from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Annotated, Any, Literal, TypeAlias, cast

from pydantic import (
    BaseModel,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    compact_json_utf8_size,
    json_utf8_size_within_limit,
)
from cayu.evals._structural_paths import _validate_portable_structural_workspace_path
from cayu.evals.capture_policy import SessionTrajectoryBounds, WorkflowCaptureDiagnostic
from cayu.evals.corpus import (
    _CURRENCY_PATTERN,
    _MODEL_JUDGE_RESULT_METADATA_KEY,
    _STRUCTURED_MODEL_JUDGE_RESULT_METADATA_KEY,
    EVAL_ARTIFACT_EVIDENCE_MAX_ARTIFACTS,
    EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE,
    EVAL_CORPUS_MAX_CASES,
    EVAL_CORPUS_MAX_JUDGE_CRITERIA,
    EVAL_CORPUS_MAX_JUDGE_EXPLANATION_CHARS,
    EVAL_CORPUS_MAX_JUDGE_RUBRIC_CHARS,
    EVAL_CORPUS_MAX_JUDGE_RUBRIC_VERSION_CHARS,
    EVAL_CORPUS_MAX_PROCESS_EVENTS,
    EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS,
    EVAL_CORPUS_MAX_TOOL_NAMES,
    EVAL_CORPUS_MAX_TRIALS,
    EVAL_CORPUS_MAX_WORKSPACE_PATH_CHARS,
    EVAL_MEMORY_ATTRIBUTION_MAX_ADMITTED_ITEMS,
    EVAL_MEMORY_ATTRIBUTION_MAX_PROVIDER_EXPOSURES,
    EVAL_PROCESS_EVENT_EVIDENCE_MAX_EVENTS,
    EVAL_TOOL_CALL_EVIDENCE_MAX_CALLS,
    EVIDENCE_MAX_CHILD_SESSIONS,
    EVIDENCE_MAX_MODEL_STEPS,
    EVIDENCE_MAX_TOOL_CALLS,
    EVIDENCE_MAX_TOTAL_TOKENS,
    ArtifactAssertionSpec,
    AssertionSpec,
    ChildStatusAssertionSpec,
    EvalCaseSpec,
    EvalCorpusDocument,
    EvalJudgeEvidenceSelectionV1,
    EvalProcessEventKind,
    FinalOutputContainsAssertionSpec,
    FinalOutputEqualsAssertionSpec,
    JudgeProfileIdentityV1,
    MaxEstimatedCostAssertionSpec,
    MaxModelStepsAssertionSpec,
    MaxToolCallsAssertionSpec,
    MaxTotalTokensAssertionSpec,
    MemoryAttributionAssertionSpec,
    ModelJudgeAssertionSpec,
    PrivateJudgeReferenceV1,
    ProcessEventAssertionSpec,
    ProcessEventsInOrderAssertionSpec,
    PublicJudgeReferenceV1,
    RootStatusAssertionSpec,
    StructuredModelJudgeAssertionSpec,
    ToolArgumentsContainAssertionSpec,
    ToolCalledAssertionSpec,
    ToolResultContainsAssertionSpec,
    ToolsCalledInOrderAssertionSpec,
    UsageRecordedAssertionSpec,
    WorkspaceFileAssertionSpec,
    _bounded_durable_text,
    _canonical_decimal_text,
    _content_revision,
    _eval_run_contract_for_validated_corpus,
    _exact_decimal_sum,
    _exact_weighted_decimal,
    _model_content_revision,
    _ordered_sequence_input,
    _portable_id,
    _PortableModel,
    _sha256_revision,
    assertion_spec_revision,
    eval_run_contract_for_corpus,
)
from cayu.evals.evidence import _canonical_decimal
from cayu.evals.json_subset import (
    JsonSubsetOutcome,
    compare_json_subset,
    copy_eval_tool_json_object,
)
from cayu.evals.memory_attribution import (
    EVAL_MEMORY_ATTRIBUTION_RESULT_BUDGET_BYTES,
    EvalMemoryAttributionEvidenceV1,
    EvalMemoryEvidenceCompleteness,
    EvalMemoryEvidenceLimitation,
    eval_memory_attribution_counts,
    eval_memory_attribution_limitations,
)
from cayu.evals.models import (
    EvalAssertionResult,
    EvalRun,
    EvalTrialResult,
    _model_instance_python_input,
)
from cayu.evals.result_contract import (
    PUBLISHED_EVAL_OUTPUT_PREVIEW_BUDGET_BYTES,
    EvalTrialDiagnosticCode,
    EvalTrialOutputPreviewV1,
    _EvalTrialPublicData,
)
from cayu.evals.revisions import eval_trial_result_revision
from cayu.evals.trial_policy import (
    EvalCaseReliabilityV1,
    EvalSuiteRunExposureV1,
    EvalSuiteTrialPolicyV1,
)
from cayu.runtime.usage import AggregateCount, aggregate_usage_metrics_from_durable_payload

PUBLISHED_EVAL_SCHEMA_VERSION = 10
PUBLISHED_EVAL_MAX_BYTES = 32 << 20
PUBLISHED_EVAL_MAX_DURATION_MS = 2**63 - 1

PublishedStatus = Literal["passed", "failed", "unavailable", "error"]
PublishedOutcome = Literal["passed", "failed", "unavailable", "error"]
PublishedStructuralObservationState = Literal[
    "available",
    "unavailable",
    "limit_exceeded",
    "unsupported",
    "truncated",
    "redacted",
    "malformed",
]
PublishedModelJudgeDiagnostic = Literal[
    "judgment_recorded",
    "evaluator_error",
    "evidence_unavailable",
]

_ASSERTION_MESSAGE = {
    "passed": "Assertion passed.",
    "failed": "Assertion failed.",
    "unavailable": "Required evidence was unavailable.",
    "error": "Assertion evaluation failed.",
}
_TRIAL_MESSAGE = {
    EvalTrialDiagnosticCode.PASSED: "Trial passed.",
    EvalTrialDiagnosticCode.ASSERTION_FAILED: "One or more assertions failed.",
    EvalTrialDiagnosticCode.ASSERTION_EVIDENCE_UNAVAILABLE: (
        "Required assertion evidence was unavailable."
    ),
    EvalTrialDiagnosticCode.TERMINAL_EVIDENCE_UNAVAILABLE: (
        "Exact terminal evidence was unavailable."
    ),
    EvalTrialDiagnosticCode.INTERRUPTED_EVIDENCE_UNAVAILABLE: (
        "Exact interrupted-session evidence was unavailable."
    ),
    EvalTrialDiagnosticCode.CHILD_EVIDENCE_UNAVAILABLE: (
        "Complete child-session evidence was unavailable."
    ),
    EvalTrialDiagnosticCode.EXTERNAL_TARGET_UNAVAILABLE: ("The external target was unavailable."),
    EvalTrialDiagnosticCode.EXTERNAL_TARGET_CANCELLED: (
        "The external target execution was cancelled."
    ),
    EvalTrialDiagnosticCode.EXTERNAL_TARGET_UNKNOWN: (
        "The external target outcome could not be reconciled."
    ),
    EvalTrialDiagnosticCode.EXTERNAL_TARGET_INCOMPLETE: (
        "The external target evidence was incomplete."
    ),
    EvalTrialDiagnosticCode.EXTERNAL_TARGET_IDENTITY_MISMATCH: (
        "The external target execution identity could not be proven."
    ),
    EvalTrialDiagnosticCode.EXTERNAL_TARGET_FAILED: "The external target execution failed.",
    EvalTrialDiagnosticCode.WORKFLOW_TARGET_FAILED: "The workflow target construction failed.",
    EvalTrialDiagnosticCode.WORKFLOW_EXECUTION_FAILED: "The workflow execution failed.",
    EvalTrialDiagnosticCode.WORKFLOW_COMPLETION_MISSING: (
        "The workflow completion evidence was missing."
    ),
    EvalTrialDiagnosticCode.WORKFLOW_COMPLETION_CONFLICT: (
        "The workflow completion evidence was conflicting."
    ),
    EvalTrialDiagnosticCode.WORKFLOW_ATTEMPT_SUPERSEDED: (
        "The workflow attempt was superseded during evaluation."
    ),
    EvalTrialDiagnosticCode.WORKFLOW_PROJECTOR_FAILED: ("The workflow result projector failed."),
    EvalTrialDiagnosticCode.WORKFLOW_OUTPUT_INVALID: (
        "The workflow result projector returned invalid output."
    ),
    EvalTrialDiagnosticCode.WORKFLOW_CAPTURE_FAILED: "Workflow completed; evidence capture failed and scoring is unavailable.",
    EvalTrialDiagnosticCode.WORKFLOW_QUIESCENCE_FAILED: (
        "The workflow target did not close cleanly."
    ),
    EvalTrialDiagnosticCode.EXECUTION_FAILED: "Trial execution failed.",
    EvalTrialDiagnosticCode.SESSION_FAILED: "The trial session failed.",
    EvalTrialDiagnosticCode.TERMINAL_EVIDENCE_FAILED: "Terminal evidence capture failed.",
    EvalTrialDiagnosticCode.EVIDENCE_PREPARATION_FAILED: ("Assertion evidence preparation failed."),
    EvalTrialDiagnosticCode.ASSERTION_EVALUATION_FAILED: "Assertion evaluation failed.",
    EvalTrialDiagnosticCode.CASE_TIMEOUT: "The trial exceeded its configured timeout.",
}

_DEFAULT_TRIAL_CODE = {
    "passed": EvalTrialDiagnosticCode.PASSED,
    "failed": EvalTrialDiagnosticCode.ASSERTION_FAILED,
    "unavailable": EvalTrialDiagnosticCode.ASSERTION_EVIDENCE_UNAVAILABLE,
    "error": EvalTrialDiagnosticCode.EXECUTION_FAILED,
}

_TRIAL_CODES_BY_STATUS = {
    "passed": {EvalTrialDiagnosticCode.PASSED},
    "failed": {EvalTrialDiagnosticCode.ASSERTION_FAILED},
    "unavailable": {
        EvalTrialDiagnosticCode.WORKFLOW_CAPTURE_FAILED,
        EvalTrialDiagnosticCode.ASSERTION_EVIDENCE_UNAVAILABLE,
        EvalTrialDiagnosticCode.TERMINAL_EVIDENCE_UNAVAILABLE,
        EvalTrialDiagnosticCode.INTERRUPTED_EVIDENCE_UNAVAILABLE,
        EvalTrialDiagnosticCode.CHILD_EVIDENCE_UNAVAILABLE,
        EvalTrialDiagnosticCode.EXTERNAL_TARGET_UNAVAILABLE,
        EvalTrialDiagnosticCode.EXTERNAL_TARGET_CANCELLED,
        EvalTrialDiagnosticCode.EXTERNAL_TARGET_UNKNOWN,
        EvalTrialDiagnosticCode.EXTERNAL_TARGET_INCOMPLETE,
        EvalTrialDiagnosticCode.EXTERNAL_TARGET_IDENTITY_MISMATCH,
    },
    "error": {
        EvalTrialDiagnosticCode.EXTERNAL_TARGET_FAILED,
        EvalTrialDiagnosticCode.WORKFLOW_TARGET_FAILED,
        EvalTrialDiagnosticCode.WORKFLOW_EXECUTION_FAILED,
        EvalTrialDiagnosticCode.WORKFLOW_COMPLETION_MISSING,
        EvalTrialDiagnosticCode.WORKFLOW_COMPLETION_CONFLICT,
        EvalTrialDiagnosticCode.WORKFLOW_ATTEMPT_SUPERSEDED,
        EvalTrialDiagnosticCode.WORKFLOW_PROJECTOR_FAILED,
        EvalTrialDiagnosticCode.WORKFLOW_OUTPUT_INVALID,
        EvalTrialDiagnosticCode.WORKFLOW_QUIESCENCE_FAILED,
        EvalTrialDiagnosticCode.EXECUTION_FAILED,
        EvalTrialDiagnosticCode.SESSION_FAILED,
        EvalTrialDiagnosticCode.TERMINAL_EVIDENCE_FAILED,
        EvalTrialDiagnosticCode.EVIDENCE_PREPARATION_FAILED,
        EvalTrialDiagnosticCode.ASSERTION_EVALUATION_FAILED,
        EvalTrialDiagnosticCode.CASE_TIMEOUT,
    },
}


def _model_judge_diagnostic(outcome: PublishedOutcome) -> PublishedModelJudgeDiagnostic:
    if outcome in {"passed", "failed"}:
        return "judgment_recorded"
    if outcome == "unavailable":
        return "evidence_unavailable"
    return "evaluator_error"


class _PublishedAssertionDetail(_PortableModel):
    kind: StrictStr


class PublishedRootStatusDetail(_PublishedAssertionDetail):
    kind: Literal["root_status"] = "root_status"
    expected: Literal["completed", "failed"]
    actual: Literal["completed", "failed", "interrupted"] | None = None


class PublishedChildStatusDetail(_PublishedAssertionDetail):
    kind: Literal["child_status"] = "child_status"
    expected: Literal["completed", "failed", "interrupted"]
    min_count: StrictInt = Field(ge=0, le=EVIDENCE_MAX_CHILD_SESSIONS)
    max_count: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_CHILD_SESSIONS)
    matching_count: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_CHILD_SESSIONS)

    @model_validator(mode="after")
    def validate_count_range(self) -> PublishedChildStatusDetail:
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError("max_count must be greater than or equal to min_count.")
        return self


class PublishedFinalOutputEqualsDetail(_PublishedAssertionDetail):
    kind: Literal["final_output_equals"] = "final_output_equals"
    matched: StrictBool | None = None


class PublishedFinalOutputContainsDetail(_PublishedAssertionDetail):
    kind: Literal["final_output_contains"] = "final_output_contains"
    matched: StrictBool | None = None


class PublishedToolCalledDetail(_PublishedAssertionDetail):
    kind: Literal["tool_called"] = "tool_called"
    tool_name: StrictStr
    min_count: StrictInt = Field(ge=0, le=EVIDENCE_MAX_TOOL_CALLS)
    max_count: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_TOOL_CALLS)
    matching_count: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_TOOL_CALLS)

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )

    @model_validator(mode="after")
    def validate_count_range(self) -> PublishedToolCalledDetail:
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError("max_count must be greater than or equal to min_count.")
        return self


ToolJsonObservationState = Literal[
    "available",
    "absent",
    "unavailable",
    "unsupported",
    "malformed",
    "incompatible",
    "limit_exceeded",
    "truncated",
    "redacted",
]


class _PublishedToolJsonSubsetDetail(_PublishedAssertionDetail):
    tool_name: StrictStr
    occurrence: StrictInt = Field(ge=1, le=EVAL_TOOL_CALL_EVIDENCE_MAX_CALLS)
    expected_subset: dict[str, Any]
    observation_state: ToolJsonObservationState
    invocation_index: StrictInt | None = Field(
        default=None,
        ge=1,
        le=EVAL_TOOL_CALL_EVIDENCE_MAX_CALLS,
    )
    invocation_revision: StrictStr | None = None
    actual: dict[str, Any] | None = None
    matched: StrictBool | None = None

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=256,
            nonblank=True,
            clean=True,
        )

    @field_validator("expected_subset", "actual", mode="before")
    @classmethod
    def validate_json_object(cls, value: object, info) -> dict[str, Any] | None:
        if value is None:
            return None
        return copy_eval_tool_json_object(value, info.field_name)

    @field_validator("invocation_revision")
    @classmethod
    def validate_invocation_revision(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_revision(value, info.field_name)

    @model_validator(mode="after")
    def validate_observation(self) -> _PublishedToolJsonSubsetDetail:
        has_identity = self.invocation_index is not None and self.invocation_revision is not None
        if (self.invocation_index is None) != (self.invocation_revision is None):
            raise ValueError("Tool JSON observations require complete invocation identity.")
        if self.observation_state == "available":
            if not has_identity or self.actual is None or self.matched is None:
                raise ValueError("Available tool JSON evidence requires identity and comparison.")
            comparison = compare_json_subset(self.expected_subset, self.actual)
            if comparison is JsonSubsetOutcome.REDACTED:
                raise ValueError(
                    "Available tool JSON evidence cannot be redacted on an expected path."
                )
            if self.matched is not (comparison is JsonSubsetOutcome.MATCHED):
                raise ValueError(
                    "Available tool JSON comparison contradicts its retained evidence."
                )
        elif self.observation_state == "absent":
            if has_identity or self.actual is not None or self.matched is not False:
                raise ValueError("Absent tool JSON evidence requires one conclusive mismatch.")
        elif self.actual is not None or self.matched is not None:
            raise ValueError("Incomplete tool JSON evidence cannot carry a comparison.")
        return self


class PublishedToolArgumentsContainDetail(_PublishedToolJsonSubsetDetail):
    kind: Literal["tool_arguments_contain"] = "tool_arguments_contain"


class PublishedToolResultContainsDetail(_PublishedToolJsonSubsetDetail):
    kind: Literal["tool_result_contains"] = "tool_result_contains"


class PublishedToolsCalledInOrderDetail(_PublishedAssertionDetail):
    kind: Literal["tools_called_in_order"] = "tools_called_in_order"
    expected_count: StrictInt = Field(ge=0, le=EVAL_CORPUS_MAX_TOOL_NAMES)
    actual_count: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_TOOL_CALLS)
    matched: StrictBool | None = None

    @model_validator(mode="after")
    def validate_observation(self) -> PublishedToolsCalledInOrderDetail:
        if (self.actual_count is None) != (self.matched is None):
            raise ValueError("Published tool-order observations must be present together.")
        if self.matched and self.actual_count != self.expected_count:
            raise ValueError("A matching tool order must have the expected call count.")
        if self.matched is False and self.expected_count == self.actual_count == 0:
            raise ValueError("Empty expected and actual tool orders must match.")
        return self


class PublishedProcessEventDetail(_PublishedAssertionDetail):
    kind: Literal["process_event"] = "process_event"
    event: EvalProcessEventKind
    min_count: StrictInt = Field(ge=0, le=EVAL_PROCESS_EVENT_EVIDENCE_MAX_EVENTS)
    max_count: StrictInt | None = Field(
        default=None,
        ge=0,
        le=EVAL_PROCESS_EVENT_EVIDENCE_MAX_EVENTS,
    )
    matching_count: StrictInt | None = Field(
        default=None,
        ge=0,
        le=EVAL_PROCESS_EVENT_EVIDENCE_MAX_EVENTS,
    )

    @model_validator(mode="after")
    def validate_count_range(self) -> PublishedProcessEventDetail:
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError("max_count must be greater than or equal to min_count.")
        return self


class PublishedProcessEventsInOrderDetail(_PublishedAssertionDetail):
    kind: Literal["process_events_in_order"] = "process_events_in_order"
    expected: tuple[EvalProcessEventKind, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_PROCESS_EVENTS,
    )
    actual_count: StrictInt | None = Field(
        default=None,
        ge=0,
        le=EVAL_PROCESS_EVENT_EVIDENCE_MAX_EVENTS,
    )
    matched: StrictBool | None = None

    @field_validator("expected", mode="before")
    @classmethod
    def validate_expected_is_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @model_validator(mode="after")
    def validate_observation(self) -> PublishedProcessEventsInOrderDetail:
        if (self.actual_count is None) != (self.matched is None):
            raise ValueError("Published process-order observations must be present together.")
        if self.matched and self.actual_count != len(self.expected):
            raise ValueError("A matching process order must have the expected event count.")
        return self


class PublishedWorkspaceFileDetail(_PublishedAssertionDetail):
    kind: Literal["workspace_file"] = "workspace_file"
    path: StrictStr
    expected_present: StrictBool
    minimum_bytes: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_TOTAL_TOKENS)
    maximum_bytes: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_TOTAL_TOKENS)
    digest_required: StrictBool
    observation_state: PublishedStructuralObservationState
    actual_present: StrictBool | None = None
    actual_size_bytes: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_TOTAL_TOKENS)
    digest_matched: StrictBool | None = None

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str, info) -> str:
        path = _bounded_durable_text(
            value,
            info.field_name,
            max_chars=EVAL_CORPUS_MAX_WORKSPACE_PATH_CHARS,
            nonblank=True,
            clean=True,
        )
        return _validate_portable_structural_workspace_path(path)

    @model_validator(mode="after")
    def validate_observation(self) -> PublishedWorkspaceFileDetail:
        if self.maximum_bytes is not None and (
            self.minimum_bytes is not None and self.maximum_bytes < self.minimum_bytes
        ):
            raise ValueError("maximum_bytes must be greater than or equal to minimum_bytes.")
        if not self.expected_present and any(
            value is not None for value in (self.minimum_bytes, self.maximum_bytes)
        ):
            raise ValueError("Absent workspace expectations cannot carry size bounds.")
        if not self.expected_present and self.digest_required:
            raise ValueError("Absent workspace expectations cannot require a digest.")
        if self.actual_present is not True and (
            self.actual_size_bytes is not None or self.digest_matched is not None
        ):
            raise ValueError("Only present workspace observations can carry structure details.")
        if not self.digest_required and self.digest_matched is not None:
            raise ValueError("Digest comparison requires a digest expectation.")
        if (self.observation_state == "available") != (self.actual_present is not None):
            raise ValueError(
                "Available workspace observations require exactly one presence observation."
            )
        if self.observation_state in {"unsupported", "truncated", "redacted", "malformed"}:
            raise ValueError("Workspace observations cannot use artifact-content states.")
        return self


class PublishedArtifactDetail(_PublishedAssertionDetail):
    kind: Literal["artifact"] = "artifact"
    scope: Literal["session", "environment"]
    filename: StrictStr | None = None
    content_type: StrictStr | None = None
    minimum_bytes: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_TOTAL_TOKENS)
    maximum_bytes: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_TOTAL_TOKENS)
    digest_required: StrictBool
    text_required: StrictBool
    observation_state: PublishedStructuralObservationState
    min_count: StrictInt = Field(ge=0, le=EVAL_ARTIFACT_EVIDENCE_MAX_ARTIFACTS)
    max_count: StrictInt | None = Field(
        default=None,
        ge=0,
        le=EVAL_ARTIFACT_EVIDENCE_MAX_ARTIFACTS,
    )
    matching_count: StrictInt | None = Field(
        default=None,
        ge=0,
        le=EVAL_ARTIFACT_EVIDENCE_MAX_ARTIFACTS,
    )

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=1_024,
            nonblank=True,
            clean=False,
        )

    @field_validator("content_type")
    @classmethod
    def validate_content_type(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        content_type = _bounded_durable_text(
            value,
            info.field_name,
            max_chars=1_024,
            nonblank=True,
            clean=True,
        )
        if any(ord(character) < 0x20 or ord(character) > 0x7E for character in content_type):
            raise ValueError("content_type must contain printable ASCII characters only.")
        return content_type

    @model_validator(mode="after")
    def validate_contract(self) -> PublishedArtifactDetail:
        if self.maximum_bytes is not None and (
            self.minimum_bytes is not None and self.maximum_bytes < self.minimum_bytes
        ):
            raise ValueError("maximum_bytes must be greater than or equal to minimum_bytes.")
        if self.max_count is not None and self.max_count < self.min_count:
            raise ValueError("max_count must be greater than or equal to min_count.")
        if (self.observation_state == "available") != (self.matching_count is not None):
            raise ValueError("Available artifact observations require exactly one matching count.")
        return self


MemoryAttributionObservationState: TypeAlias = Literal[
    "complete",
    "truncated",
    "unavailable",
    "indeterminate",
]


class PublishedMemoryAttributionDetail(_PublishedAssertionDetail):
    """Bounded structural memory expectation and its conclusive counts, when available."""

    kind: Literal["memory_attribution"] = "memory_attribution"
    min_admitted_items: StrictInt = Field(
        ge=0,
        le=EVAL_MEMORY_ATTRIBUTION_MAX_ADMITTED_ITEMS,
    )
    max_admitted_items: StrictInt | None = Field(
        default=None,
        ge=0,
        le=EVAL_MEMORY_ATTRIBUTION_MAX_ADMITTED_ITEMS,
    )
    min_provider_exposures: StrictInt = Field(
        ge=0,
        le=EVAL_MEMORY_ATTRIBUTION_MAX_PROVIDER_EXPOSURES,
    )
    max_provider_exposures: StrictInt | None = Field(
        default=None,
        ge=0,
        le=EVAL_MEMORY_ATTRIBUTION_MAX_PROVIDER_EXPOSURES,
    )
    observation_state: MemoryAttributionObservationState
    evidence_revision: StrictStr
    limitations: tuple[EvalMemoryEvidenceLimitation, ...] = ()
    admitted_item_count: StrictInt | None = Field(
        default=None,
        ge=0,
        le=EVAL_MEMORY_ATTRIBUTION_MAX_ADMITTED_ITEMS,
    )
    provider_exposure_count: StrictInt | None = Field(
        default=None,
        ge=0,
        le=EVAL_MEMORY_ATTRIBUTION_MAX_PROVIDER_EXPOSURES,
    )

    @field_validator("evidence_revision")
    @classmethod
    def validate_evidence_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("limitations", mode="before")
    @classmethod
    def validate_limitations_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @model_validator(mode="after")
    def validate_contract(self) -> PublishedMemoryAttributionDetail:
        if (
            self.max_admitted_items is not None
            and self.max_admitted_items < self.min_admitted_items
        ):
            raise ValueError("Maximum admitted items cannot be below the minimum.")
        if (
            self.max_provider_exposures is not None
            and self.max_provider_exposures < self.min_provider_exposures
        ):
            raise ValueError("Maximum provider exposures cannot be below the minimum.")
        limitation_values = tuple(item.value for item in self.limitations)
        if limitation_values != tuple(sorted(set(limitation_values))):
            raise ValueError("Memory-attribution limitations must be unique and sorted.")
        observed = self.admitted_item_count is not None
        if observed != (self.provider_exposure_count is not None):
            raise ValueError("Memory-attribution counts must be present together.")
        if observed != (self.observation_state == "complete"):
            raise ValueError("Only complete memory attribution can carry conclusive counts.")
        if self.observation_state == "complete" and self.limitations:
            raise ValueError("Complete memory attribution cannot carry limitations.")
        return self


class PublishedMaxToolCallsDetail(_PublishedAssertionDetail):
    kind: Literal["max_tool_calls"] = "max_tool_calls"
    maximum: StrictInt = Field(ge=0, le=EVIDENCE_MAX_TOOL_CALLS)
    actual: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_TOOL_CALLS)


class PublishedMaxModelStepsDetail(_PublishedAssertionDetail):
    kind: Literal["max_model_steps"] = "max_model_steps"
    maximum: StrictInt = Field(ge=0, le=EVIDENCE_MAX_MODEL_STEPS)
    actual: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_MODEL_STEPS)


class PublishedUsageRecordedDetail(_PublishedAssertionDetail):
    kind: Literal["usage_recorded"] = "usage_recorded"
    minimum: StrictInt = Field(ge=0, le=EVIDENCE_MAX_TOTAL_TOKENS)
    actual: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_TOTAL_TOKENS)


class PublishedMaxTotalTokensDetail(_PublishedAssertionDetail):
    kind: Literal["max_total_tokens"] = "max_total_tokens"
    maximum: StrictInt = Field(ge=0, le=EVIDENCE_MAX_TOTAL_TOKENS)
    actual: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_TOTAL_TOKENS)


class PublishedMaxEstimatedCostDetail(_PublishedAssertionDetail):
    kind: Literal["max_estimated_cost"] = "max_estimated_cost"
    maximum: StrictStr
    currency: StrictStr
    estimated_cost: StrictStr | None = None
    priced_model_steps: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_MODEL_STEPS)
    unpriced_model_steps: StrictInt | None = Field(default=None, ge=0, le=EVIDENCE_MAX_MODEL_STEPS)

    @field_validator("maximum", "estimated_cost")
    @classmethod
    def validate_decimal_text(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _canonical_decimal_text(
            value,
            info.field_name,
            max_chars=64 if info.field_name == "maximum" else 128,
        )

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str, info) -> str:
        value = _bounded_durable_text(
            value,
            info.field_name,
            max_chars=16,
            nonblank=True,
            clean=True,
        )
        if _CURRENCY_PATTERN.fullmatch(value) is None:
            raise ValueError("currency must be a portable uppercase identifier.")
        return value

    @model_validator(mode="after")
    def validate_cost_counts(self) -> PublishedMaxEstimatedCostDetail:
        observations = (
            self.estimated_cost,
            self.priced_model_steps,
            self.unpriced_model_steps,
        )
        if any(item is None for item in observations) and any(
            item is not None for item in observations
        ):
            raise ValueError("Published cost observations must be present together.")
        if (
            self.priced_model_steps is not None
            and self.unpriced_model_steps is not None
            and self.priced_model_steps + self.unpriced_model_steps > EVIDENCE_MAX_MODEL_STEPS
        ):
            raise ValueError("Published cost observations exceed the model-step evidence bound.")
        return self


class _PublishedJudgeUsageV1(_PortableModel):
    model_steps: StrictInt = Field(ge=1, le=EVIDENCE_MAX_MODEL_STEPS)
    input_tokens: AggregateCount = Field(ge=0)
    output_tokens: AggregateCount = Field(ge=0)
    total_tokens: AggregateCount = Field(ge=0)

    @model_validator(mode="after")
    def validate_token_totals(self) -> _PublishedJudgeUsageV1:
        if self.total_tokens < max(self.input_tokens, self.output_tokens):
            raise ValueError("Published judge total tokens contradict component usage.")
        return self


class _PublishedJudgeCostV1(_PortableModel):
    availability: Literal["priced", "unavailable"]
    currency: StrictStr | None = None
    estimated_cost: StrictStr | None = None
    priced_model_steps: StrictInt | None = Field(default=None, ge=0)
    unpriced_model_steps: StrictInt | None = Field(default=None, ge=0)

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        value = _bounded_durable_text(
            value,
            info.field_name,
            max_chars=16,
            nonblank=True,
            clean=True,
        )
        if _CURRENCY_PATTERN.fullmatch(value) is None:
            raise ValueError("Judge cost currency must be a portable uppercase identifier.")
        return value

    @field_validator("estimated_cost")
    @classmethod
    def validate_estimated_cost(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _canonical_decimal_text(value, info.field_name, max_chars=64)

    @model_validator(mode="after")
    def validate_availability(self) -> _PublishedJudgeCostV1:
        observations = (
            self.currency,
            self.estimated_cost,
            self.priced_model_steps,
            self.unpriced_model_steps,
        )
        if self.availability == "unavailable":
            if any(item is not None for item in observations):
                raise ValueError("Unavailable judge cost cannot carry priced observations.")
            return self
        if any(item is None for item in observations):
            raise ValueError("Priced judge cost requires complete observations.")
        if self.unpriced_model_steps != 0:
            raise ValueError("Priced judge cost cannot contain unpriced model steps.")
        return self


class PublishedModelJudgeUsageV1(_PublishedJudgeUsageV1):
    """Observed usage for one successfully recorded rubric-string judgment."""


class PublishedModelJudgeCostV1(_PublishedJudgeCostV1):
    """Observed priced cost, or an explicit unpriced state, for one judgment."""


class PublishedModelJudgeDetail(_PublishedAssertionDetail):
    """Bounded public contract and safe outcome evidence for one model judgment."""

    kind: Literal["model_judge"] = "model_judge"
    evaluator_key: StrictStr
    evaluator_implementation_revision: StrictStr
    rubric: StrictStr
    rubric_version: StrictStr
    threshold: StrictFloat = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    include_transcript: StrictBool
    diagnostic: PublishedModelJudgeDiagnostic
    judge_profile: JudgeProfileIdentityV1
    candidate_route_relation: Literal["independent_model", "same_model", "unknown"]
    usage: PublishedModelJudgeUsageV1 | None = None
    cost: PublishedModelJudgeCostV1 | None = None

    @field_validator("evaluator_key")
    @classmethod
    def validate_evaluator_key(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("evaluator_implementation_revision")
    @classmethod
    def validate_evaluator_implementation_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("rubric")
    @classmethod
    def validate_rubric(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=EVAL_CORPUS_MAX_JUDGE_RUBRIC_CHARS,
            nonblank=True,
            clean=False,
        )

    @field_validator("rubric_version")
    @classmethod
    def validate_rubric_version(cls, value: str, info) -> str:
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=EVAL_CORPUS_MAX_JUDGE_RUBRIC_VERSION_CHARS,
            nonblank=True,
            clean=True,
        )

    @model_validator(mode="after")
    def validate_judgment(self) -> PublishedModelJudgeDetail:
        if (
            self.evaluator_key != self.judge_profile.key
            or self.evaluator_implementation_revision != self.judge_profile.implementation_revision
        ):
            raise ValueError("Published model-judge identity contradicts its profile.")
        if self.diagnostic == "judgment_recorded" and (
            self.candidate_route_relation == "unknown"
            or (
                self.candidate_route_relation == "same_model"
                and self.judge_profile.same_model_use != "allowed_and_labeled"
            )
        ):
            raise ValueError("Published model judgment used a forbidden same-model route.")
        if self.include_transcript and "transcript" not in self.judge_profile.allowed_evidence:
            raise ValueError("Published model judgment used disallowed transcript evidence.")
        recorded = self.diagnostic == "judgment_recorded"
        if recorded != (self.usage is not None) or recorded != (self.cost is not None):
            raise ValueError("Recorded model judgments require judge usage and cost state.")
        if not recorded:
            return self
        if self.usage is None or self.cost is None:
            raise RuntimeError("Recorded model judgment lost its observations.")
        if (
            self.usage.input_tokens > self.judge_profile.max_input_tokens
            or self.usage.output_tokens > self.judge_profile.max_output_tokens
            or self.usage.total_tokens > self.judge_profile.max_total_tokens
        ):
            raise ValueError("Published judge usage exceeds its profile ceilings.")
        priced_profile = self.judge_profile.pricing_profile_fingerprint is not None
        if priced_profile != (self.cost.availability == "priced"):
            raise ValueError("Published judge cost contradicts its profile pricing identity.")
        if self.cost.availability == "priced" and (
            self.cost.currency != self.judge_profile.cost_currency
            or self.cost.priced_model_steps != self.usage.model_steps
        ):
            raise ValueError("Published judge cost does not match its profile or usage.")
        if (
            self.cost.estimated_cost is not None
            and self.judge_profile.max_estimated_cost is not None
            and Decimal(self.cost.estimated_cost) > Decimal(self.judge_profile.max_estimated_cost)
        ):
            raise ValueError("Published judge cost exceeds its profile ceiling.")
        return self


class PublishedJudgeReferenceIdentityV1(_PortableModel):
    """Safe reference identity; private evaluator truth is never represented."""

    kind: Literal["public_reference", "private_reference"]
    key: StrictStr
    revision: StrictStr
    availability: Literal["available"] = "available"
    privacy_policy_key: StrictStr | None = None
    privacy_policy_revision: StrictStr | None = None

    @field_validator("key", "privacy_policy_key")
    @classmethod
    def validate_ids(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _portable_id(value, info.field_name)

    @field_validator("revision", "privacy_policy_revision")
    @classmethod
    def validate_revisions(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_revision(value, info.field_name)

    @model_validator(mode="after")
    def validate_privacy_identity(self) -> PublishedJudgeReferenceIdentityV1:
        privacy = (self.privacy_policy_key, self.privacy_policy_revision)
        if self.kind == "public_reference" and any(value is not None for value in privacy):
            raise ValueError("Public reference identity cannot carry a private policy.")
        if self.kind == "private_reference" and any(value is None for value in privacy):
            raise ValueError("Private reference identity requires its privacy policy.")
        return self


class PublishedStructuredJudgeCriterionV1(_PortableModel):
    criterion_id: StrictStr
    weight: StrictStr
    score: StrictStr
    explanation: StrictStr | None
    explanation_state: Literal["available", "redacted", "unavailable"]

    @field_validator("criterion_id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("weight", "score")
    @classmethod
    def validate_unit_decimal(cls, value: str, info) -> str:
        value = _canonical_decimal_text(value, info.field_name, max_chars=20)
        if Decimal(value) > 1:
            raise ValueError(f"{info.field_name} must be between 0 and 1.")
        return value

    @field_validator("explanation")
    @classmethod
    def validate_explanation(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _bounded_durable_text(
            value,
            info.field_name,
            max_chars=EVAL_CORPUS_MAX_JUDGE_EXPLANATION_CHARS,
            nonblank=True,
            clean=False,
        )

    @model_validator(mode="after")
    def validate_explanation_state(self) -> PublishedStructuredJudgeCriterionV1:
        if (self.explanation is None) != (self.explanation_state == "unavailable"):
            raise ValueError("Explanation availability must match its publication state.")
        return self


class PublishedStructuredJudgeUsageV1(_PublishedJudgeUsageV1):
    """Observed usage for one successfully recorded structured judgment."""


class PublishedStructuredJudgeCostV1(_PublishedJudgeCostV1):
    """Observed priced cost, or an explicit unpriced state, for one judgment."""


class PublishedStructuredModelJudgeDetail(_PublishedAssertionDetail):
    """Typed public judgment with no prompt, credentials, options, or private truth."""

    kind: Literal["structured_model_judge"] = "structured_model_judge"
    judge_profile: JudgeProfileIdentityV1
    candidate_route_relation: Literal["independent_model", "same_model", "unknown"]
    rubric_id: StrictStr
    rubric_revision: StrictStr
    reference: PublishedJudgeReferenceIdentityV1 | None = None
    threshold: StrictStr
    evidence: EvalJudgeEvidenceSelectionV1
    diagnostic: PublishedModelJudgeDiagnostic
    criteria: tuple[PublishedStructuredJudgeCriterionV1, ...] = Field(
        default=(),
        max_length=EVAL_CORPUS_MAX_JUDGE_CRITERIA,
    )
    aggregate_score: StrictStr | None = None
    usage: PublishedStructuredJudgeUsageV1 | None = None
    cost: PublishedStructuredJudgeCostV1 | None = None

    @field_validator("rubric_id")
    @classmethod
    def validate_rubric_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("rubric_revision")
    @classmethod
    def validate_rubric_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, value: str, info) -> str:
        value = _canonical_decimal_text(value, info.field_name, max_chars=20)
        if Decimal(value) > 1:
            raise ValueError(f"{info.field_name} must be between 0 and 1.")
        return value

    @field_validator("aggregate_score")
    @classmethod
    def validate_aggregate_score(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        value = _canonical_decimal_text(value, info.field_name, max_chars=64)
        if Decimal(value) > 1:
            raise ValueError("aggregate_score must be between 0 and 1.")
        return value

    @field_validator("criteria", mode="before")
    @classmethod
    def validate_criteria_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @model_validator(mode="after")
    def validate_judgment(self) -> PublishedStructuredModelJudgeDetail:
        if self.diagnostic == "judgment_recorded" and (
            self.candidate_route_relation == "unknown"
            or (
                self.candidate_route_relation == "same_model"
                and self.judge_profile.same_model_use != "allowed_and_labeled"
            )
        ):
            raise ValueError("Published structured judgment used a forbidden same-model route.")
        if (
            self.evidence.include_transcript
            and "transcript" not in self.judge_profile.allowed_evidence
        ):
            raise ValueError("Published structured judgment used disallowed transcript evidence.")
        if self.reference is not None:
            evidence_kind = (
                "public_reference"
                if self.reference.kind == "public_reference"
                else "private_reference"
            )
            if evidence_kind not in self.judge_profile.allowed_evidence:
                raise ValueError("Published structured judgment used a disallowed reference.")
            if self.reference.kind == "private_reference" and (
                self.reference.privacy_policy_key,
                self.reference.privacy_policy_revision,
            ) != (
                self.judge_profile.privacy_policy_key,
                self.judge_profile.privacy_policy_revision,
            ):
                raise ValueError(
                    "Published private reference does not match the judge privacy policy."
                )
        recorded = self.diagnostic == "judgment_recorded"
        if recorded != bool(self.criteria) or recorded != (self.aggregate_score is not None):
            raise ValueError("Recorded structured judgments require complete criterion evidence.")
        if recorded != (self.usage is not None) or recorded != (self.cost is not None):
            raise ValueError("Recorded structured judgments require judge usage and cost state.")
        if not recorded:
            return self
        if self.usage is None or self.cost is None:
            raise RuntimeError("Recorded structured judgment lost its observations.")
        if (
            self.usage.input_tokens > self.judge_profile.max_input_tokens
            or self.usage.output_tokens > self.judge_profile.max_output_tokens
            or self.usage.total_tokens > self.judge_profile.max_total_tokens
        ):
            raise ValueError("Published judge usage exceeds its profile ceilings.")
        priced_profile = self.judge_profile.pricing_profile_fingerprint is not None
        if priced_profile != (self.cost.availability == "priced"):
            raise ValueError("Published judge cost contradicts its profile pricing identity.")
        if self.cost.availability == "priced" and (
            self.cost.currency != self.judge_profile.cost_currency
            or self.cost.priced_model_steps != self.usage.model_steps
        ):
            raise ValueError("Published judge cost does not match its profile or usage.")
        if (
            self.cost.estimated_cost is not None
            and self.judge_profile.max_estimated_cost is not None
            and Decimal(self.cost.estimated_cost) > Decimal(self.judge_profile.max_estimated_cost)
        ):
            raise ValueError("Published judge cost exceeds its profile ceiling.")
        ids = tuple(item.criterion_id for item in self.criteria)
        if len(ids) != len(set(ids)):
            raise ValueError("Published structured criterion IDs must be unique.")
        if _exact_decimal_sum(Decimal(item.weight) for item in self.criteria) != 1:
            raise ValueError("Published structured criterion weights must sum exactly to 1.")
        expected = _exact_weighted_decimal(
            (item.weight, Decimal(item.score)) for item in self.criteria
        )
        if Decimal(self.aggregate_score or "0") != expected:
            raise ValueError("Published structured aggregate does not match its criteria.")
        if (
            self.reference is not None
            and self.reference.kind == "private_reference"
            and any(item.explanation_state != "unavailable" for item in self.criteria)
        ):
            raise ValueError("Private-reference judgments cannot publish explanations.")
        return self


PublishedAssertionDetail: TypeAlias = Annotated[
    PublishedRootStatusDetail
    | PublishedChildStatusDetail
    | PublishedFinalOutputEqualsDetail
    | PublishedFinalOutputContainsDetail
    | PublishedToolCalledDetail
    | PublishedToolArgumentsContainDetail
    | PublishedToolResultContainsDetail
    | PublishedToolsCalledInOrderDetail
    | PublishedProcessEventDetail
    | PublishedProcessEventsInOrderDetail
    | PublishedWorkspaceFileDetail
    | PublishedArtifactDetail
    | PublishedMemoryAttributionDetail
    | PublishedMaxToolCallsDetail
    | PublishedMaxModelStepsDetail
    | PublishedUsageRecordedDetail
    | PublishedMaxTotalTokensDetail
    | PublishedMaxEstimatedCostDetail
    | PublishedModelJudgeDetail
    | PublishedStructuredModelJudgeDetail,
    Field(discriminator="kind"),
]


class PublishedAssertionResult(_PortableModel):
    assertion_id: StrictStr
    assertion_revision: StrictStr
    outcome: PublishedOutcome
    score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    code: PublishedOutcome
    message: StrictStr
    detail: PublishedAssertionDetail

    @field_validator("assertion_id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("assertion_revision")
    @classmethod
    def validate_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @model_validator(mode="after")
    def validate_contract(self) -> PublishedAssertionResult:
        if self.outcome in {"unavailable", "error"}:
            if self.score is not None:
                raise ValueError("Unavailable/error published assertions cannot have a score.")
        elif isinstance(
            self.detail,
            (PublishedModelJudgeDetail, PublishedStructuredModelJudgeDetail),
        ):
            if self.score is None:
                raise ValueError("Scored published model judges require a score.")
            threshold = (
                self.detail.threshold
                if type(self.detail) is PublishedModelJudgeDetail
                else float(Decimal(self.detail.threshold))
            )
            expected_outcome = "passed" if self.score >= threshold else "failed"
            if self.outcome != expected_outcome:
                raise ValueError("Published model-judge score is inconsistent.")
            if type(self.detail) is PublishedStructuredModelJudgeDetail and self.score != float(
                Decimal(self.detail.aggregate_score or "0")
            ):
                raise ValueError("Published structured score does not match its aggregate.")
        elif self.score != (1.0 if self.outcome == "passed" else 0.0):
            raise ValueError("Published deterministic assertion score is inconsistent.")
        if self.code != self.outcome or self.message != _ASSERTION_MESSAGE[self.outcome]:
            raise ValueError("Published assertion diagnostics do not match the outcome.")
        if self.outcome in {"passed", "failed"} and not _detail_has_observation(self.detail):
            raise ValueError("Scored published assertions require their observed evidence.")
        if self.outcome == "error" and _detail_has_observation(self.detail):
            raise ValueError("Errored published assertions cannot carry observed evidence.")
        if self.outcome == "unavailable" and _detail_has_observation(self.detail):
            if not isinstance(self.detail, PublishedMaxEstimatedCostDetail):
                raise ValueError(
                    "Unavailable published assertions cannot carry conclusive observations."
                )
            if not self.detail.unpriced_model_steps:
                raise ValueError(
                    "Unavailable published cost observations require unpriced model steps."
                )
        if isinstance(
            self.detail,
            (PublishedModelJudgeDetail, PublishedStructuredModelJudgeDetail),
        ):
            expected_diagnostic = _model_judge_diagnostic(self.outcome)
            if self.detail.diagnostic != expected_diagnostic:
                raise ValueError("Published model-judge diagnostic contradicts the outcome.")
        if (
            self.outcome in {"passed", "failed"}
            and isinstance(self.detail, PublishedMaxEstimatedCostDetail)
            and self.detail.unpriced_model_steps != 0
        ):
            raise ValueError("Scored published cost assertions require fully priced evidence.")
        expected_passed = _detail_expected_pass(self.detail)
        if self.outcome in {"passed", "failed"} and expected_passed is not None:
            expected_outcome = "passed" if expected_passed else "failed"
            if self.outcome != expected_outcome:
                raise ValueError("Published assertion outcome contradicts its detail evidence.")
        return self


class PublishedUsageSummaryV1(_PortableModel):
    model_steps: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    tool_calls: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    total_tokens: AggregateCount = Field(ge=0)


def _validate_memory_assertions_for_evidence(
    assertions: tuple[PublishedAssertionResult, ...],
    evidence: EvalMemoryAttributionEvidenceV1,
) -> None:
    """Bind every published memory assertion to its retained trial evidence."""

    expected_state: MemoryAttributionObservationState = (
        "indeterminate"
        if evidence.completeness is EvalMemoryEvidenceCompleteness.COMPLETE
        and evidence.has_indeterminate_exposure
        else cast(
            "MemoryAttributionObservationState",
            evidence.completeness.value,
        )
    )
    expected_limitations = eval_memory_attribution_limitations(evidence)
    expected_counts = eval_memory_attribution_counts(evidence)
    for assertion in assertions:
        detail = assertion.detail
        if type(detail) is not PublishedMemoryAttributionDetail:
            continue
        if (
            detail.evidence_revision != evidence.revision
            or detail.observation_state != expected_state
            or detail.limitations != expected_limitations
        ):
            raise ValueError("Published memory assertion does not match retained memory evidence.")
        observed_counts = (
            detail.admitted_item_count,
            detail.provider_exposure_count,
        )
        conclusive_counts = expected_counts if expected_state == "complete" else (None, None)
        if observed_counts != conclusive_counts:
            raise ValueError(
                "Published memory assertion counts do not match retained memory evidence."
            )


class PublishedEvalTrialResult(_PortableModel):
    execution_status: Literal["completed"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    capture_bounds: SessionTrajectoryBounds | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    capture_diagnostic: WorkflowCaptureDiagnostic | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    trial_number: StrictInt = Field(ge=1, le=EVAL_CORPUS_MAX_TRIALS)
    source_trial_revision: StrictStr = Field(min_length=64, max_length=64)
    status: PublishedStatus
    score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    assertions: tuple[PublishedAssertionResult, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE,
    )
    evidence_complete: StrictBool
    duration_ms: StrictInt = Field(ge=0, le=PUBLISHED_EVAL_MAX_DURATION_MS)
    usage: PublishedUsageSummaryV1 | None = None
    output: EvalTrialOutputPreviewV1
    memory_attribution: EvalMemoryAttributionEvidenceV1
    code: EvalTrialDiagnosticCode
    message: StrictStr

    @field_validator("assertions", mode="before")
    @classmethod
    def validate_assertions_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @field_validator("source_trial_revision")
    @classmethod
    def validate_source_trial_revision(cls, value: str) -> str:
        if any(character not in "0123456789abcdef" for character in value):
            raise ValueError("source_trial_revision must be lowercase SHA-256 hex.")
        return value

    @field_validator("output", mode="before")
    @classmethod
    def copy_output(cls, value: object) -> object:
        if type(value) is EvalTrialOutputPreviewV1:
            return EvalTrialOutputPreviewV1.model_validate(value.model_dump(mode="python"))
        if isinstance(value, BaseModel):
            raise TypeError("output must be an exact EvalTrialOutputPreviewV1 or JSON object.")
        return value

    @field_validator("memory_attribution", mode="before")
    @classmethod
    def copy_memory_attribution(cls, value: object) -> object:
        if type(value) is EvalMemoryAttributionEvidenceV1:
            return EvalMemoryAttributionEvidenceV1.model_validate(value.model_dump(mode="python"))
        if isinstance(value, BaseModel):
            raise TypeError(
                "memory_attribution must be exact eval memory evidence or a JSON object."
            )
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> PublishedEvalTrialResult:
        assertion_ids = tuple(assertion.assertion_id for assertion in self.assertions)
        if len(assertion_ids) != len(set(assertion_ids)):
            raise ValueError("Published assertion IDs must be unique within a trial.")
        expected_status = _published_status_from_outcomes(
            assertion.outcome for assertion in self.assertions
        )
        expected_score = _published_score(assertion.score for assertion in self.assertions)
        if self.status != expected_status or self.score != expected_score:
            raise ValueError("Published trial aggregates do not match its assertions.")
        if self.status in {"passed", "failed"} and not self.evidence_complete:
            raise ValueError("Scored published trials require complete evidence.")
        if self.evidence_complete and self.usage is None:
            raise ValueError("Complete published trials require exact usage.")
        if self.code not in _TRIAL_CODES_BY_STATUS[self.status]:
            raise ValueError("Published trial diagnostic code contradicts the status.")
        if self.message != _TRIAL_MESSAGE[self.code]:
            raise ValueError("Published trial diagnostics do not match the status.")
        _validate_trial_observations(
            self.assertions,
            self.usage,
            evidence_complete=self.evidence_complete,
        )
        _validate_memory_assertions_for_evidence(
            self.assertions,
            self.memory_attribution,
        )
        return self


class PublishedEvalCaseResult(_PortableModel):
    case_id: StrictStr
    case_revision: StrictStr
    status: PublishedStatus
    score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    reliability: EvalCaseReliabilityV1
    trials: tuple[PublishedEvalTrialResult, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_TRIALS,
    )
    duration_ms: StrictInt = Field(ge=0, le=PUBLISHED_EVAL_MAX_DURATION_MS)

    @field_validator("trials", mode="before")
    @classmethod
    def validate_trials_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @field_validator("case_id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @field_validator("case_revision")
    @classmethod
    def validate_revision(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @model_validator(mode="after")
    def validate_contract(self) -> PublishedEvalCaseResult:
        if tuple(trial.trial_number for trial in self.trials) != tuple(
            range(1, len(self.trials) + 1)
        ):
            raise ValueError("Published trial numbers must be contiguous and ordered.")
        expected_status = self.reliability.outcome
        expected_score = _published_score(trial.score for trial in self.trials)
        if self.status != expected_status or self.score != expected_score:
            raise ValueError("Published case aggregates do not match its trials.")
        if self.duration_ms != sum(trial.duration_ms for trial in self.trials):
            raise ValueError("Published case duration must equal its trial durations.")
        contracts = tuple(
            tuple(_assertion_contract(item) for item in trial.assertions) for trial in self.trials
        )
        if any(contract != contracts[0] for contract in contracts[1:]):
            raise ValueError("Published trials must share one ordered assertion contract.")
        return self


class PublishedEvalRun(_PortableModel):
    schema_version: Literal[10]
    revision: StrictStr
    corpus_revision: StrictStr
    target_key: StrictStr
    suite_id: StrictStr
    suite_revision: StrictStr
    evidence_policy_revision: StrictStr
    pricing_profile_fingerprint: StrictStr | None = None
    trial_policy: EvalSuiteTrialPolicyV1
    accepted_exposure: EvalSuiteRunExposureV1 | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    status: PublishedStatus
    score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    cases: tuple[PublishedEvalCaseResult, ...] = Field(
        min_length=1,
        max_length=EVAL_CORPUS_MAX_CASES,
    )
    duration_ms: StrictInt = Field(ge=0, le=PUBLISHED_EVAL_MAX_DURATION_MS)

    @field_validator("cases", mode="before")
    @classmethod
    def validate_cases_are_ordered(cls, value: object, info) -> object:
        return _ordered_sequence_input(value, info.field_name)

    @model_validator(mode="before")
    @classmethod
    def validate_expanded_result_limit(cls, value):
        """Reject oversized raw graphs before constructing their nested models."""

        if isinstance(value, Mapping):
            if "schema_version" not in value:
                raise ValueError("Published eval run schema_version is required.")
            schema_version = value["schema_version"]
            if type(schema_version) is not int or (schema_version != PUBLISHED_EVAL_SCHEMA_VERSION):
                raise ValueError(
                    "Published eval run schema_version must be the integer "
                    f"{PUBLISHED_EVAL_SCHEMA_VERSION}; other versions are unsupported."
                )

        def raw_sequence(container, field_name, path, model_type):
            if isinstance(container, Mapping):
                if field_name not in container:
                    raise ValueError(f"{path}.{field_name} is required.")
                sequence = container[field_name]
            elif isinstance(container, model_type):
                sequence = getattr(container, field_name)
            else:
                raise ValueError(f"{path} must be an object.")
            if not isinstance(sequence, list | tuple):
                raise ValueError(f"{path}.{field_name} must be an array.")
            return sequence

        if isinstance(value, Mapping):
            if "cases" not in value:
                raise ValueError("Published eval run cases are required.")
            cases = value["cases"]
        elif isinstance(value, cls):
            cases = value.cases
        else:
            raise ValueError("Published eval run input must be an object.")
        if not isinstance(cases, list | tuple):
            raise ValueError("Published eval run cases must be an array.")
        if not 1 <= len(cases) <= EVAL_CORPUS_MAX_CASES:
            raise ValueError(
                "Published eval run cases must contain between 1 and "
                f"{EVAL_CORPUS_MAX_CASES} items."
            )
        published_assertion_results = 0
        output_preview_bytes = 0
        for case_index, case in enumerate(cases):
            trials = raw_sequence(
                case,
                "trials",
                f"Published eval run cases[{case_index}]",
                PublishedEvalCaseResult,
            )
            if not 1 <= len(trials) <= EVAL_CORPUS_MAX_TRIALS:
                raise ValueError(
                    f"Published eval run cases[{case_index}].trials must contain between 1 and "
                    f"{EVAL_CORPUS_MAX_TRIALS} items."
                )
            for trial_index, trial in enumerate(trials):
                if isinstance(trial, Mapping):
                    if "output" not in trial:
                        raise ValueError(
                            "Published eval run "
                            f"cases[{case_index}].trials[{trial_index}].output is required."
                        )
                    raw_output = trial["output"]
                elif isinstance(trial, PublishedEvalTrialResult):
                    raw_output = trial.output
                else:
                    raise ValueError(
                        "Published eval run "
                        f"cases[{case_index}].trials[{trial_index}] must be an object."
                    )
                if isinstance(raw_output, Mapping):
                    raw_text = raw_output.get("text")
                elif isinstance(raw_output, EvalTrialOutputPreviewV1):
                    raw_text = raw_output.text
                else:
                    raw_text = None
                if type(raw_text) is str:
                    remaining = PUBLISHED_EVAL_OUTPUT_PREVIEW_BUDGET_BYTES - output_preview_bytes
                    # Every Unicode character occupies at least one UTF-8 byte. Avoid
                    # encoding a Python-supplied string once it already exceeds the
                    # aggregate public-output budget.
                    if len(raw_text) > remaining:
                        raise ValueError(
                            "Published eval run exceeds its aggregate output-preview "
                            f"limit of {PUBLISHED_EVAL_OUTPUT_PREVIEW_BUDGET_BYTES} UTF-8 bytes."
                        )
                    try:
                        encoded_text_bytes = len(raw_text.encode("utf-8"))
                    except UnicodeEncodeError:
                        # The nested durable-text validator reports malformed Unicode.
                        # This preflight only accounts well-formed preview strings.
                        encoded_text_bytes = 0
                    output_preview_bytes += encoded_text_bytes
                    if output_preview_bytes > PUBLISHED_EVAL_OUTPUT_PREVIEW_BUDGET_BYTES:
                        raise ValueError(
                            "Published eval run exceeds its aggregate output-preview "
                            f"limit of {PUBLISHED_EVAL_OUTPUT_PREVIEW_BUDGET_BYTES} UTF-8 bytes."
                        )
                assertions = raw_sequence(
                    trial,
                    "assertions",
                    f"Published eval run cases[{case_index}].trials[{trial_index}]",
                    PublishedEvalTrialResult,
                )
                if not 1 <= len(assertions) <= EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE:
                    raise ValueError(
                        "Published eval run "
                        f"cases[{case_index}].trials[{trial_index}].assertions must contain "
                        f"between 1 and {EVAL_CORPUS_MAX_ASSERTIONS_PER_CASE} items."
                    )
                published_assertion_results += len(assertions)
                if published_assertion_results > EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS:
                    raise ValueError(
                        "Published eval run exceeds the corpus expanded assertion-result "
                        f"limit of {EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS}."
                    )
        return value

    @field_validator(
        "revision",
        "corpus_revision",
        "suite_revision",
        "evidence_policy_revision",
        "pricing_profile_fingerprint",
    )
    @classmethod
    def validate_revision_fields(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _sha256_revision(value, info.field_name)

    @field_validator("target_key", "suite_id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return _portable_id(value, info.field_name)

    @model_validator(mode="after")
    def validate_contract(self) -> PublishedEvalRun:
        case_ids = tuple(case.case_id for case in self.cases)
        if case_ids != tuple(sorted(set(case_ids))):
            raise ValueError("Published cases must be unique and sorted by id.")
        expected_status = _published_status_from_statuses(case.status for case in self.cases)
        expected_score = _published_score(case.score for case in self.cases)
        if self.status != expected_status or self.score != expected_score:
            raise ValueError("Published run aggregates do not match its cases.")
        if self.duration_ms != sum(case.duration_ms for case in self.cases):
            raise ValueError("Published run duration must equal its case durations.")
        trial_counts = {len(case.trials) for case in self.cases}
        if len(trial_counts) != 1:
            raise ValueError("Published cases must share one suite-wide trial count.")
        if trial_counts != {self.trial_policy.trial_count}:
            raise ValueError("Published trial counts must match the suite trial policy.")
        if self.accepted_exposure is not None and (
            self.accepted_exposure.trial_policy_revision != self.trial_policy.revision
            or self.accepted_exposure.candidate_trials
            < len(self.cases) * self.trial_policy.trial_count
        ):
            raise ValueError("Published accepted exposure contradicts its result graph.")
        for case in self.cases:
            uses_model_judge = any(
                assertion.detail.kind in {"model_judge", "structured_model_judge"}
                for trial in case.trials
                for assertion in trial.assertions
            )
            expected_reliability = EvalCaseReliabilityV1.create(
                policy=self.trial_policy,
                trials=((trial.status, trial.score, trial.code) for trial in case.trials),
                uses_model_judge=uses_model_judge,
            )
            if case.reliability != expected_reliability:
                raise ValueError("Published case reliability does not match retained trials.")
        published_assertion_results = sum(
            len(trial.assertions) for case in self.cases for trial in case.trials
        )
        if published_assertion_results > EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS:
            raise ValueError(
                "Published eval run exceeds the corpus expanded assertion-result limit of "
                f"{EVAL_CORPUS_MAX_PUBLISHED_ASSERTION_RESULTS}."
            )
        output_preview_bytes = sum(
            len(trial.output.text.encode("utf-8")) for case in self.cases for trial in case.trials
        )
        if output_preview_bytes > PUBLISHED_EVAL_OUTPUT_PREVIEW_BUDGET_BYTES:
            raise ValueError(
                "Published eval run exceeds its aggregate output-preview limit of "
                f"{PUBLISHED_EVAL_OUTPUT_PREVIEW_BUDGET_BYTES} UTF-8 bytes."
            )
        memory_attribution_bytes = sum(
            compact_json_utf8_size(trial.memory_attribution.model_dump(mode="json"))
            for case in self.cases
            for trial in case.trials
        )
        if memory_attribution_bytes > EVAL_MEMORY_ATTRIBUTION_RESULT_BUDGET_BYTES:
            raise ValueError(
                "Published eval run exceeds its aggregate memory-attribution limit of "
                f"{EVAL_MEMORY_ATTRIBUTION_RESULT_BUDGET_BYTES} UTF-8 bytes."
            )
        has_cost_assertion = any(
            assertion.detail.kind == "max_estimated_cost"
            for case in self.cases
            for trial in case.trials
            for assertion in trial.assertions
        )
        if has_cost_assertion and self.pricing_profile_fingerprint is None:
            raise ValueError("Published cost assertions require a pricing profile fingerprint.")
        if not json_utf8_size_within_limit(self, PUBLISHED_EVAL_MAX_BYTES):
            raise ValueError(
                f"Published eval run exceeds {PUBLISHED_EVAL_MAX_BYTES} canonical JSON bytes."
            )
        if self.revision != _model_content_revision(self, "published eval run"):
            raise ValueError("Published eval run revision does not match its content.")
        return self


def _published_status_from_outcomes(outcomes) -> PublishedStatus:
    values = tuple(outcomes)
    for outcome in ("error", "unavailable", "failed"):
        if outcome in values:
            return outcome
    return "passed"


def _detail_has_observation(detail: PublishedAssertionDetail) -> bool:
    if isinstance(detail, (PublishedFinalOutputEqualsDetail, PublishedFinalOutputContainsDetail)):
        return detail.matched is not None
    if isinstance(detail, PublishedRootStatusDetail):
        return detail.actual is not None
    if isinstance(detail, (PublishedChildStatusDetail, PublishedToolCalledDetail)):
        return detail.matching_count is not None
    if isinstance(detail, PublishedToolsCalledInOrderDetail):
        return detail.actual_count is not None and detail.matched is not None
    if isinstance(detail, PublishedProcessEventDetail):
        return detail.matching_count is not None
    if isinstance(detail, PublishedProcessEventsInOrderDetail):
        return detail.actual_count is not None and detail.matched is not None
    if isinstance(detail, PublishedWorkspaceFileDetail):
        return detail.observation_state == "available"
    if isinstance(detail, PublishedArtifactDetail):
        return detail.observation_state == "available"
    if isinstance(detail, PublishedMemoryAttributionDetail):
        return detail.observation_state == "complete"
    if isinstance(
        detail,
        (PublishedToolArgumentsContainDetail, PublishedToolResultContainsDetail),
    ):
        return detail.observation_state in {"available", "absent"}
    if isinstance(
        detail,
        (
            PublishedMaxToolCallsDetail,
            PublishedMaxModelStepsDetail,
            PublishedMaxTotalTokensDetail,
        ),
    ):
        return detail.actual is not None
    if isinstance(detail, PublishedUsageRecordedDetail):
        return detail.actual is not None
    if isinstance(detail, PublishedMaxEstimatedCostDetail):
        return detail.estimated_cost is not None
    if isinstance(detail, PublishedModelJudgeDetail):
        return detail.diagnostic == "judgment_recorded"
    if isinstance(detail, PublishedStructuredModelJudgeDetail):
        return detail.diagnostic == "judgment_recorded"
    return True


def _detail_expected_pass(detail: PublishedAssertionDetail) -> bool | None:
    if isinstance(detail, (PublishedFinalOutputEqualsDetail, PublishedFinalOutputContainsDetail)):
        return detail.matched
    if isinstance(detail, PublishedRootStatusDetail):
        return None if detail.actual is None else detail.actual == detail.expected
    if isinstance(detail, (PublishedChildStatusDetail, PublishedToolCalledDetail)):
        if detail.matching_count is None:
            return None
        return detail.matching_count >= detail.min_count and (
            detail.max_count is None or detail.matching_count <= detail.max_count
        )
    if isinstance(
        detail,
        (PublishedToolArgumentsContainDetail, PublishedToolResultContainsDetail),
    ):
        return detail.matched
    if isinstance(detail, PublishedMaxToolCallsDetail):
        return None if detail.actual is None else detail.actual <= detail.maximum
    if isinstance(detail, PublishedMaxModelStepsDetail):
        return None if detail.actual is None else detail.actual <= detail.maximum
    if isinstance(detail, PublishedUsageRecordedDetail):
        return None if detail.actual is None else detail.actual >= detail.minimum
    if isinstance(detail, PublishedMaxTotalTokensDetail):
        return None if detail.actual is None else detail.actual <= detail.maximum
    if isinstance(detail, PublishedMaxEstimatedCostDetail):
        if detail.estimated_cost is None:
            return None
        if detail.unpriced_model_steps:
            return None
        return Decimal(detail.estimated_cost) <= Decimal(detail.maximum)
    if isinstance(detail, (PublishedModelJudgeDetail, PublishedStructuredModelJudgeDetail)):
        return None
    if isinstance(detail, PublishedToolsCalledInOrderDetail):
        return detail.matched
    if isinstance(detail, PublishedProcessEventDetail):
        if detail.matching_count is None:
            return None
        return detail.matching_count >= detail.min_count and (
            detail.max_count is None or detail.matching_count <= detail.max_count
        )
    if isinstance(detail, PublishedProcessEventsInOrderDetail):
        return detail.matched
    if isinstance(detail, PublishedWorkspaceFileDetail):
        if detail.actual_present is None:
            return None
        if detail.actual_present != detail.expected_present:
            return False
        if not detail.actual_present:
            return True
        if detail.actual_size_bytes is None:
            return None
        if (
            detail.minimum_bytes is not None and detail.actual_size_bytes < detail.minimum_bytes
        ) or (detail.maximum_bytes is not None and detail.actual_size_bytes > detail.maximum_bytes):
            return False
        return detail.digest_matched if detail.digest_required else True
    if isinstance(detail, PublishedArtifactDetail):
        if detail.matching_count is None:
            return None
        return detail.matching_count >= detail.min_count and (
            detail.max_count is None or detail.matching_count <= detail.max_count
        )
    if isinstance(detail, PublishedMemoryAttributionDetail):
        if detail.admitted_item_count is None or detail.provider_exposure_count is None:
            return None
        return (
            detail.admitted_item_count >= detail.min_admitted_items
            and (
                detail.max_admitted_items is None
                or detail.admitted_item_count <= detail.max_admitted_items
            )
            and detail.provider_exposure_count >= detail.min_provider_exposures
            and (
                detail.max_provider_exposures is None
                or detail.provider_exposure_count <= detail.max_provider_exposures
            )
        )
    return None


def _detail_evidence_area(detail: PublishedAssertionDetail) -> str:
    if isinstance(detail, PublishedRootStatusDetail):
        return "root-status"
    if isinstance(detail, PublishedChildStatusDetail):
        return "child-status"
    if isinstance(detail, (PublishedFinalOutputEqualsDetail, PublishedFinalOutputContainsDetail)):
        return "final-output"
    if isinstance(
        detail,
        (
            PublishedToolCalledDetail,
            PublishedToolsCalledInOrderDetail,
            PublishedMaxToolCallsDetail,
        ),
    ):
        return "tool"
    if isinstance(detail, PublishedToolArgumentsContainDetail):
        return "tool-arguments"
    if isinstance(detail, PublishedToolResultContainsDetail):
        return "tool-result"
    if isinstance(
        detail,
        (PublishedProcessEventDetail, PublishedProcessEventsInOrderDetail),
    ):
        return "process-event"
    if isinstance(detail, PublishedWorkspaceFileDetail):
        return f"workspace:{detail.path}:digest={detail.digest_required}"
    if isinstance(detail, PublishedArtifactDetail):
        return (
            f"artifact:{detail.scope}:{detail.filename!r}:{detail.content_type!r}:"
            f"{detail.minimum_bytes}:{detail.maximum_bytes}:"
            f"digest={detail.digest_required}:text={detail.text_required}:"
            f"{detail.min_count}:{detail.max_count}"
        )
    if isinstance(detail, PublishedMemoryAttributionDetail):
        return "memory-attribution"
    if isinstance(detail, PublishedMaxModelStepsDetail):
        return "model-step"
    if isinstance(detail, (PublishedUsageRecordedDetail, PublishedMaxTotalTokensDetail)):
        return "usage"
    if isinstance(detail, PublishedMaxEstimatedCostDetail):
        return "cost"
    if isinstance(detail, (PublishedModelJudgeDetail, PublishedStructuredModelJudgeDetail)):
        return "model-judge"
    raise AssertionError("Unreachable published assertion detail type.")


def _published_reference_contract(
    reference: PublishedJudgeReferenceIdentityV1 | None,
) -> list[object] | None:
    if reference is None:
        return None
    return [
        reference.kind,
        reference.key,
        reference.revision,
        reference.availability,
        reference.privacy_policy_key,
        reference.privacy_policy_revision,
    ]


def _spec_reference_contract(reference) -> list[object] | None:
    if reference is None:
        return None
    if type(reference) is PublicJudgeReferenceV1:
        return [reference.kind, reference.id, reference.revision, "available", None, None]
    if type(reference) is PrivateJudgeReferenceV1:
        return [
            reference.kind,
            reference.key,
            reference.revision,
            "available",
            reference.privacy_policy_key,
            reference.privacy_policy_revision,
        ]
    raise AssertionError("Unreachable judge reference type.")


def _assertion_contract(assertion: PublishedAssertionResult) -> tuple[object, ...]:
    detail = assertion.detail
    if isinstance(detail, PublishedRootStatusDetail):
        static_detail: tuple[object, ...] = (detail.kind, detail.expected)
    elif isinstance(detail, PublishedChildStatusDetail):
        static_detail = (detail.kind, detail.expected, detail.min_count, detail.max_count)
    elif isinstance(detail, (PublishedFinalOutputEqualsDetail, PublishedFinalOutputContainsDetail)):
        static_detail = (detail.kind,)
    elif isinstance(detail, PublishedToolCalledDetail):
        static_detail = (detail.kind, detail.tool_name, detail.min_count, detail.max_count)
    elif isinstance(
        detail,
        (PublishedToolArgumentsContainDetail, PublishedToolResultContainsDetail),
    ):
        static_detail = (
            detail.kind,
            detail.tool_name,
            detail.occurrence,
            detail.expected_subset,
        )
    elif isinstance(detail, PublishedToolsCalledInOrderDetail):
        static_detail = (detail.kind, detail.expected_count)
    elif isinstance(detail, PublishedProcessEventDetail):
        static_detail = (detail.kind, detail.event, detail.min_count, detail.max_count)
    elif isinstance(detail, PublishedProcessEventsInOrderDetail):
        static_detail = (detail.kind, *detail.expected)
    elif isinstance(detail, PublishedWorkspaceFileDetail):
        static_detail = (
            detail.kind,
            detail.path,
            detail.expected_present,
            detail.minimum_bytes,
            detail.maximum_bytes,
            detail.digest_required,
        )
    elif isinstance(detail, PublishedArtifactDetail):
        static_detail = (
            detail.kind,
            detail.scope,
            detail.filename,
            detail.content_type,
            detail.minimum_bytes,
            detail.maximum_bytes,
            detail.digest_required,
            detail.text_required,
            detail.min_count,
            detail.max_count,
        )
    elif isinstance(detail, PublishedMemoryAttributionDetail):
        static_detail = (
            detail.kind,
            detail.min_admitted_items,
            detail.max_admitted_items,
            detail.min_provider_exposures,
            detail.max_provider_exposures,
        )
    elif isinstance(detail, (PublishedMaxToolCallsDetail, PublishedMaxModelStepsDetail)):
        static_detail = (detail.kind, detail.maximum)
    elif isinstance(detail, PublishedUsageRecordedDetail):
        static_detail = (detail.kind, detail.minimum)
    elif isinstance(detail, PublishedMaxTotalTokensDetail):
        static_detail = (detail.kind, detail.maximum)
    elif isinstance(detail, PublishedMaxEstimatedCostDetail):
        static_detail = (detail.kind, detail.maximum, detail.currency)
    elif isinstance(detail, PublishedModelJudgeDetail):
        static_detail = (
            detail.kind,
            detail.evaluator_key,
            detail.evaluator_implementation_revision,
            detail.judge_profile.revision,
            detail.candidate_route_relation,
            detail.rubric,
            detail.rubric_version,
            detail.threshold,
            detail.include_transcript,
        )
    elif isinstance(detail, PublishedStructuredModelJudgeDetail):
        static_detail = (
            detail.kind,
            detail.judge_profile.key,
            detail.judge_profile.revision,
            detail.candidate_route_relation,
            detail.rubric_id,
            detail.rubric_revision,
            _published_reference_contract(detail.reference),
            detail.threshold,
            detail.evidence.include_final_output,
            detail.evidence.include_transcript,
        )
    else:
        raise AssertionError("Unreachable published assertion detail type.")
    return assertion.assertion_id, assertion.assertion_revision, *static_detail


def _assertion_spec_contract(spec: AssertionSpec) -> tuple[object, ...]:
    base: tuple[object, ...] = (
        spec.id,
        _model_content_revision(spec, "assertion spec"),
        spec.kind,
    )
    if type(spec) is RootStatusAssertionSpec:
        return (*base, spec.expected)
    if type(spec) is ChildStatusAssertionSpec:
        return (*base, spec.expected, spec.min_count, spec.max_count)
    if type(spec) in {FinalOutputEqualsAssertionSpec, FinalOutputContainsAssertionSpec}:
        return base
    if type(spec) is ToolCalledAssertionSpec:
        return (*base, spec.tool_name, spec.min_count, spec.max_count)
    if type(spec) in {ToolArgumentsContainAssertionSpec, ToolResultContainsAssertionSpec}:
        tool_json_spec = cast(
            "ToolArgumentsContainAssertionSpec | ToolResultContainsAssertionSpec",
            spec,
        )
        return (
            *base,
            tool_json_spec.tool_name,
            tool_json_spec.occurrence,
            tool_json_spec.expected_subset,
        )
    if type(spec) is ToolsCalledInOrderAssertionSpec:
        return (*base, len(spec.tool_names))
    if type(spec) is ProcessEventAssertionSpec:
        return (*base, spec.event, spec.min_count, spec.max_count)
    if type(spec) is ProcessEventsInOrderAssertionSpec:
        return (*base, *spec.events)
    if type(spec) is WorkspaceFileAssertionSpec:
        return (
            *base,
            spec.path,
            spec.present,
            spec.minimum_bytes,
            spec.maximum_bytes,
            spec.sha256 is not None,
        )
    if type(spec) is ArtifactAssertionSpec:
        return (
            *base,
            spec.scope,
            spec.filename,
            spec.content_type,
            spec.minimum_bytes,
            spec.maximum_bytes,
            spec.sha256 is not None,
            spec.text_contains is not None,
            spec.min_count,
            spec.max_count,
        )
    if type(spec) is MemoryAttributionAssertionSpec:
        return (
            *base,
            spec.min_admitted_items,
            spec.max_admitted_items,
            spec.min_provider_exposures,
            spec.max_provider_exposures,
        )
    if isinstance(spec, (MaxToolCallsAssertionSpec, MaxModelStepsAssertionSpec)):
        return (*base, spec.maximum)
    if type(spec) is UsageRecordedAssertionSpec:
        return (*base, spec.min_total_tokens)
    if type(spec) is MaxTotalTokensAssertionSpec:
        return (*base, spec.maximum)
    if type(spec) is MaxEstimatedCostAssertionSpec:
        return (*base, spec.maximum, spec.currency)
    if type(spec) is ModelJudgeAssertionSpec:
        return (
            *base,
            spec.evaluator_key,
            spec.rubric,
            spec.rubric_version,
            spec.threshold,
            spec.include_transcript,
        )
    if type(spec) is StructuredModelJudgeAssertionSpec:
        return (
            *base,
            spec.judge_profile_key,
            spec.judge_profile_revision,
            spec.rubric.id,
            spec.rubric.revision,
            _spec_reference_contract(spec.reference),
            spec.threshold,
            spec.evidence.include_final_output,
            spec.evidence.include_transcript,
        )
    raise AssertionError("Unreachable portable assertion specification type.")


def _published_assertion_matches_spec(
    assertion: PublishedAssertionResult,
    spec: AssertionSpec,
) -> bool:
    """Check that published assertion settings retain the corpus's portable contract.

    A model-judge implementation revision is resolved from the execution target and
    deliberately retained in published results for comparison.  It is not part of
    the portable corpus assertion, so it must not prevent that result from being
    bound to the corpus that requested the evaluator key and rubric.
    """

    if type(spec) not in {ModelJudgeAssertionSpec, StructuredModelJudgeAssertionSpec}:
        return _assertion_contract(assertion) == _assertion_spec_contract(spec)
    if type(spec) is StructuredModelJudgeAssertionSpec:
        detail = assertion.detail
        if type(detail) is not PublishedStructuredModelJudgeDetail:
            return False
        return (
            assertion.assertion_id,
            assertion.assertion_revision,
            detail.kind,
            detail.judge_profile.key,
            detail.judge_profile.revision,
            detail.rubric_id,
            detail.rubric_revision,
            _published_reference_contract(detail.reference),
            detail.threshold,
            detail.evidence.include_final_output,
            detail.evidence.include_transcript,
        ) == _assertion_spec_contract(spec)
    detail = assertion.detail
    if type(detail) is not PublishedModelJudgeDetail:
        return False
    return (
        assertion.assertion_id,
        assertion.assertion_revision,
        detail.kind,
        detail.evaluator_key,
        detail.rubric,
        detail.rubric_version,
        detail.threshold,
        detail.include_transcript,
    ) == _assertion_spec_contract(spec)


def _validate_published_eval_run_for_corpus(
    corpus: EvalCorpusDocument,
    run: PublishedEvalRun,
) -> PublishedEvalRun:
    """Bind a self-consistent public result to one complete immutable corpus suite."""

    if type(corpus) is not EvalCorpusDocument:
        raise TypeError("corpus must be an exact EvalCorpusDocument.")
    if type(run) is not PublishedEvalRun:
        raise TypeError("run must be an exact PublishedEvalRun.")
    expected = _eval_run_contract_for_validated_corpus(corpus, run.suite_id)
    if (
        run.corpus_revision,
        run.target_key,
        run.suite_id,
        run.suite_revision,
        run.evidence_policy_revision,
        run.pricing_profile_fingerprint,
    ) != (
        expected.corpus_revision,
        expected.target_key,
        expected.suite_id,
        expected.suite_revision,
        expected.evidence_policy_revision,
        expected.pricing_profile_fingerprint,
    ):
        raise ValueError("Published eval run identity does not match its immutable corpus suite.")

    case_specs = tuple(case for case in corpus.cases if case.suite_id == run.suite_id)
    if tuple((case.case_id, case.case_revision) for case in run.cases) != tuple(
        (case.id, case.revision) for case in case_specs
    ):
        raise ValueError("Published eval run cases do not match its immutable corpus suite.")
    for case_spec, published_case in zip(case_specs, run.cases, strict=True):
        if len(published_case.trials) != expected.trials:
            raise ValueError("Published eval run trial counts do not match its corpus suite.")
        if any(
            len(trial.assertions) != len(case_spec.assertions)
            or any(
                not _published_assertion_matches_spec(assertion, spec)
                for assertion, spec in zip(trial.assertions, case_spec.assertions, strict=True)
            )
            for trial in published_case.trials
        ):
            raise ValueError(
                "Published eval run assertions do not match its immutable corpus cases."
            )
    return run


def _validate_trial_observations(
    assertions: tuple[PublishedAssertionResult, ...],
    usage: PublishedUsageSummaryV1 | None,
    *,
    evidence_complete: bool,
) -> None:
    metric_values: dict[str, list[int]] = {
        "model_steps": [],
        "tool_calls": [],
        "total_tokens": [],
    }
    root_statuses: set[str] = set()
    child_counts: dict[str, set[int]] = {}
    tool_counts: dict[str, set[int]] = {}
    tool_order_counts: set[int] = set()
    process_event_counts: dict[str, set[int]] = {}
    cost_observations: dict[str, set[tuple[str, int, int]]] = {}
    availability_by_area: dict[str, set[bool]] = {}
    for assertion in assertions:
        detail = assertion.detail
        if assertion.outcome != "error" and not isinstance(
            detail,
            (PublishedModelJudgeDetail, PublishedStructuredModelJudgeDetail),
        ):
            area = _detail_evidence_area(detail)
            availability_by_area.setdefault(area, set()).add(_detail_has_observation(detail))
        if isinstance(detail, PublishedRootStatusDetail) and detail.actual is not None:
            root_statuses.add(detail.actual)
        elif isinstance(detail, PublishedChildStatusDetail) and detail.matching_count is not None:
            child_counts.setdefault(detail.expected, set()).add(detail.matching_count)
        elif isinstance(detail, PublishedToolCalledDetail) and detail.matching_count is not None:
            tool_counts.setdefault(detail.tool_name, set()).add(detail.matching_count)
        elif (
            isinstance(detail, PublishedToolsCalledInOrderDetail)
            and detail.actual_count is not None
        ):
            tool_order_counts.add(detail.actual_count)
        elif isinstance(detail, PublishedProcessEventDetail) and detail.matching_count is not None:
            process_event_counts.setdefault(detail.event, set()).add(detail.matching_count)
        elif isinstance(detail, PublishedMaxModelStepsDetail) and detail.actual is not None:
            metric_values["model_steps"].append(detail.actual)
        elif isinstance(detail, PublishedMaxToolCallsDetail) and detail.actual is not None:
            metric_values["tool_calls"].append(detail.actual)
        elif (
            isinstance(
                detail,
                (PublishedUsageRecordedDetail, PublishedMaxTotalTokensDetail),
            )
            and detail.actual is not None
        ):
            metric_values["total_tokens"].append(detail.actual)
        elif (
            isinstance(detail, PublishedMaxEstimatedCostDetail)
            and detail.estimated_cost is not None
            and detail.priced_model_steps is not None
            and detail.unpriced_model_steps is not None
        ):
            cost_observations.setdefault(detail.currency, set()).add(
                (
                    detail.estimated_cost,
                    detail.priced_model_steps,
                    detail.unpriced_model_steps,
                )
            )
            metric_values["model_steps"].append(
                detail.priced_model_steps + detail.unpriced_model_steps
            )
    inconsistent_areas = sorted(
        area for area, availability in availability_by_area.items() if len(availability) > 1
    )
    if inconsistent_areas:
        raise ValueError(
            "Published assertions must agree on evidence availability within each trial area: "
            + ", ".join(inconsistent_areas)
            + "."
        )
    if evidence_complete and availability_by_area.get("root-status") == {False}:
        raise ValueError("Complete published trial evidence requires a root-status observation.")
    if len(root_statuses) > 1:
        raise ValueError("Published root-status observations must agree within a trial.")
    if any(len(values) > 1 for values in child_counts.values()):
        raise ValueError(
            "Published child-status observations for the same status must agree within a trial."
        )
    observed_child_counts = [count for values in child_counts.values() for count in values]
    if sum(observed_child_counts) > EVIDENCE_MAX_CHILD_SESSIONS:
        raise ValueError(
            "Published child-status observations cannot exceed retained child evidence."
        )
    if any(len(values) > 1 for values in tool_counts.values()):
        raise ValueError(
            "Published tool-call observations for the same tool must agree within a trial."
        )
    if len(tool_order_counts) > 1:
        raise ValueError("Published tool-order observations must agree within a trial.")
    if any(len(values) > 1 for values in process_event_counts.values()):
        raise ValueError(
            "Published process-event observations for the same event must agree within a trial."
        )
    if (
        sum(count for values in process_event_counts.values() for count in values)
        > EVAL_PROCESS_EVENT_EVIDENCE_MAX_EVENTS
    ):
        raise ValueError(
            "Published process-event observations cannot exceed retained process evidence."
        )
    if any(len(values) > 1 for values in cost_observations.values()):
        raise ValueError(
            "Published cost observations for the same currency must agree within a trial."
        )
    observed_tool_counts = [count for values in tool_counts.values() for count in values]
    if usage is None and observed_tool_counts:
        raise ValueError("Published tool-call observations require trial usage.")
    if usage is not None and sum(observed_tool_counts) > usage.tool_calls:
        raise ValueError("Published tool-call observations cannot exceed trial tool calls.")
    for metric, values in metric_values.items():
        if len(set(values)) > 1:
            raise ValueError(f"Published {metric} observations must agree within a trial.")
        if usage is None and values:
            raise ValueError(f"Published {metric} observations require trial usage.")
        if usage is not None and values and values[0] != getattr(usage, metric):
            raise ValueError(f"Published {metric} observations must match trial usage.")


def _published_status_from_statuses(statuses) -> PublishedStatus:
    values = tuple(statuses)
    for status in ("error", "unavailable", "failed"):
        if status in values:
            return status
    return "passed"


def _published_score(scores) -> float | None:
    values = tuple(scores)
    if any(value is None for value in values):
        return None
    return sum(values) / len(values)


def _safe_metadata_int(result: EvalAssertionResult, key: str) -> int | None:
    value = result.metadata.get(key)
    return value if type(value) is int and 0 <= value <= MAX_DURABLE_JSON_INTEGER else None


def _safe_metadata_bool(result: EvalAssertionResult, key: str) -> bool | None:
    value = result.metadata.get(key)
    return value if type(value) is bool else None


def _safe_metadata_text(result: EvalAssertionResult, key: str, *, max_chars: int) -> str | None:
    value = result.metadata.get(key)
    if type(value) is not str:
        return None
    try:
        return _bounded_durable_text(
            value,
            key,
            max_chars=max_chars,
            nonblank=True,
            clean=True,
        )
    except ValueError:
        return None


def _safe_metadata_decimal(result: EvalAssertionResult, key: str) -> str | None:
    value = _safe_metadata_text(result, key, max_chars=128)
    if value is None:
        return None
    try:
        return _canonical_decimal_text(value, key, max_chars=128)
    except ValueError:
        return None


def _safe_memory_limitations(
    result: EvalAssertionResult,
) -> tuple[EvalMemoryEvidenceLimitation, ...]:
    value = result.metadata.get("limitations")
    if type(value) is not list:
        return ()
    try:
        limitations = tuple(EvalMemoryEvidenceLimitation(item) for item in value)
    except (TypeError, ValueError):
        return ()
    if limitations != tuple(sorted(set(limitations), key=str)):
        return ()
    return limitations


def _safe_metadata_json_object(result: EvalAssertionResult, key: str) -> dict[str, Any] | None:
    value = result.metadata.get(key)
    if value is None:
        return None
    try:
        return copy_eval_tool_json_object(value, key)
    except (TypeError, ValueError):
        return None


def _structural_observation_state(
    result: EvalAssertionResult,
    *,
    observed: bool,
) -> PublishedStructuralObservationState:
    if observed:
        return "available"
    value = result.metadata.get("evidence_state")
    if value in {
        "unavailable",
        "limit_exceeded",
        "unsupported",
        "truncated",
        "redacted",
        "malformed",
    }:
        return cast("PublishedStructuralObservationState", value)
    return "unavailable"


def _published_detail(
    spec: AssertionSpec,
    result: EvalAssertionResult,
) -> PublishedAssertionDetail:
    if type(spec) is RootStatusAssertionSpec:
        actual = result.metadata.get("actual")
        return PublishedRootStatusDetail(
            expected=spec.expected,
            actual=(
                actual
                if type(actual) is str and actual in {"completed", "failed", "interrupted"}
                else None
            ),
        )
    if type(spec) is ChildStatusAssertionSpec:
        return PublishedChildStatusDetail(
            expected=spec.expected,
            min_count=spec.min_count,
            max_count=spec.max_count,
            matching_count=_safe_metadata_int(result, "count"),
        )
    if type(spec) is FinalOutputEqualsAssertionSpec:
        return PublishedFinalOutputEqualsDetail(matched=_safe_metadata_bool(result, "matched"))
    if type(spec) is FinalOutputContainsAssertionSpec:
        return PublishedFinalOutputContainsDetail(matched=_safe_metadata_bool(result, "matched"))
    if type(spec) is ToolCalledAssertionSpec:
        return PublishedToolCalledDetail(
            tool_name=spec.tool_name,
            min_count=spec.min_count,
            max_count=spec.max_count,
            matching_count=_safe_metadata_int(result, "count"),
        )
    if type(spec) in {ToolArgumentsContainAssertionSpec, ToolResultContainsAssertionSpec}:
        tool_json_spec = cast(
            "ToolArgumentsContainAssertionSpec | ToolResultContainsAssertionSpec",
            spec,
        )
        raw_state = result.metadata.get("observation_state")
        observation_state: ToolJsonObservationState = (
            raw_state
            if type(raw_state) is str
            and raw_state
            in {
                "available",
                "absent",
                "unavailable",
                "unsupported",
                "malformed",
                "incompatible",
                "limit_exceeded",
                "truncated",
                "redacted",
            }
            else "unavailable"
        )
        detail_type = (
            PublishedToolArgumentsContainDetail
            if type(tool_json_spec) is ToolArgumentsContainAssertionSpec
            else PublishedToolResultContainsDetail
        )
        return detail_type(
            tool_name=tool_json_spec.tool_name,
            occurrence=tool_json_spec.occurrence,
            expected_subset=tool_json_spec.expected_subset,
            observation_state=observation_state,
            invocation_index=_safe_metadata_int(result, "invocation_index"),
            invocation_revision=_safe_metadata_text(
                result,
                "invocation_revision",
                max_chars=71,
            ),
            actual=_safe_metadata_json_object(result, "actual"),
            matched=_safe_metadata_bool(result, "matched"),
        )
    if type(spec) is ToolsCalledInOrderAssertionSpec:
        actual = result.metadata.get("actual")
        valid_actual = (
            type(actual) is list
            and len(actual) <= EVIDENCE_MAX_TOOL_CALLS
            and all(type(item) is str for item in actual)
        )
        return PublishedToolsCalledInOrderDetail(
            expected_count=len(spec.tool_names),
            actual_count=len(actual) if valid_actual else None,
            matched=(actual == list(spec.tool_names) if valid_actual else None),
        )
    if type(spec) is ProcessEventAssertionSpec:
        return PublishedProcessEventDetail(
            event=spec.event,
            min_count=spec.min_count,
            max_count=spec.max_count,
            matching_count=_safe_metadata_int(result, "count"),
        )
    if type(spec) is ProcessEventsInOrderAssertionSpec:
        actual = result.metadata.get("actual")
        valid_actual = (
            type(actual) is list
            and len(actual) <= EVAL_PROCESS_EVENT_EVIDENCE_MAX_EVENTS
            and all(type(item) is str for item in actual)
        )
        return PublishedProcessEventsInOrderDetail(
            expected=spec.events,
            actual_count=len(actual) if valid_actual else None,
            matched=(actual == list(spec.events) if valid_actual else None),
        )
    if type(spec) is WorkspaceFileAssertionSpec:
        actual_present = _safe_metadata_bool(result, "present")
        return PublishedWorkspaceFileDetail(
            path=spec.path,
            expected_present=spec.present,
            minimum_bytes=spec.minimum_bytes,
            maximum_bytes=spec.maximum_bytes,
            digest_required=spec.sha256 is not None,
            observation_state=_structural_observation_state(
                result,
                observed=actual_present is not None,
            ),
            actual_present=actual_present,
            actual_size_bytes=_safe_metadata_int(result, "size_bytes"),
            digest_matched=_safe_metadata_bool(result, "digest_matched"),
        )
    if type(spec) is ArtifactAssertionSpec:
        matching_count = _safe_metadata_int(result, "matching_count")
        return PublishedArtifactDetail(
            scope=spec.scope,
            filename=spec.filename,
            content_type=spec.content_type,
            minimum_bytes=spec.minimum_bytes,
            maximum_bytes=spec.maximum_bytes,
            digest_required=spec.sha256 is not None,
            text_required=spec.text_contains is not None,
            observation_state=_structural_observation_state(
                result,
                observed=matching_count is not None,
            ),
            min_count=spec.min_count,
            max_count=spec.max_count,
            matching_count=matching_count,
        )
    if type(spec) is MemoryAttributionAssertionSpec:
        raw_state = result.metadata.get("evidence_state")
        evidence_revision = _safe_metadata_text(
            result,
            "evidence_revision",
            max_chars=71,
        )
        if evidence_revision is None:
            raise ValueError(
                "Internal memory-attribution result did not retain its evidence revision."
            )
        observation_state: MemoryAttributionObservationState = (
            raw_state
            if type(raw_state) is str
            and raw_state in {"complete", "truncated", "unavailable", "indeterminate"}
            else "unavailable"
        )
        return PublishedMemoryAttributionDetail(
            min_admitted_items=spec.min_admitted_items,
            max_admitted_items=spec.max_admitted_items,
            min_provider_exposures=spec.min_provider_exposures,
            max_provider_exposures=spec.max_provider_exposures,
            observation_state=observation_state,
            evidence_revision=evidence_revision,
            limitations=_safe_memory_limitations(result),
            admitted_item_count=_safe_metadata_int(result, "admitted_item_count"),
            provider_exposure_count=_safe_metadata_int(result, "provider_exposure_count"),
        )
    if type(spec) is MaxToolCallsAssertionSpec:
        return PublishedMaxToolCallsDetail(
            maximum=spec.maximum,
            actual=_safe_metadata_int(result, "actual"),
        )
    if type(spec) is MaxModelStepsAssertionSpec:
        return PublishedMaxModelStepsDetail(
            maximum=spec.maximum,
            actual=_safe_metadata_int(result, "actual"),
        )
    if type(spec) is UsageRecordedAssertionSpec:
        return PublishedUsageRecordedDetail(
            minimum=spec.min_total_tokens,
            actual=_safe_metadata_int(result, "total_tokens"),
        )
    if type(spec) is MaxTotalTokensAssertionSpec:
        return PublishedMaxTotalTokensDetail(
            maximum=spec.maximum,
            actual=_safe_metadata_int(result, "actual"),
        )
    if type(spec) is MaxEstimatedCostAssertionSpec:
        estimated_cost = _safe_metadata_decimal(result, "estimated_cost")
        priced_steps = _safe_metadata_int(result, "priced_model_steps")
        unpriced_steps = _safe_metadata_int(result, "unpriced_model_steps")
        if any(item is None for item in (estimated_cost, priced_steps, unpriced_steps)):
            estimated_cost = None
            priced_steps = None
            unpriced_steps = None
        else:
            metadata_currency = _safe_metadata_text(result, "currency", max_chars=16)
            if metadata_currency != spec.currency:
                raise ValueError("Internal cost metadata currency does not match the corpus.")
        summary = result.cost_summary
        if summary is not None:
            summary_observation = (
                _canonical_decimal(summary.total_cost),
                summary.priced_model_steps,
                summary.unpriced_model_steps,
            )
            if summary.currency != spec.currency:
                raise ValueError("Internal cost summary currency does not match the corpus.")
            if summary.priced_model_steps + summary.unpriced_model_steps != summary.model_steps:
                raise ValueError(
                    "Internal cost summary step counts do not form an exact partition."
                )
            if (estimated_cost, priced_steps, unpriced_steps) != summary_observation:
                raise ValueError("Internal cost metadata does not match its exact cost summary.")
            estimated_cost, priced_steps, unpriced_steps = summary_observation
        return PublishedMaxEstimatedCostDetail(
            maximum=spec.maximum,
            currency=spec.currency,
            estimated_cost=estimated_cost,
            priced_model_steps=priced_steps,
            unpriced_model_steps=unpriced_steps,
        )
    if type(spec) is ModelJudgeAssertionSpec:
        raw = result.metadata.get(_MODEL_JUDGE_RESULT_METADATA_KEY)
        if type(raw) is not dict:
            raise ValueError("Internal model-judge result did not record its public contract.")
        try:
            profile = JudgeProfileIdentityV1.model_validate(raw.get("judge_profile"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Internal model-judge profile is invalid.") from exc
        if profile.key != spec.evaluator_key:
            raise ValueError("Internal model-judge profile does not match the corpus.")
        route_relation = raw.get("candidate_route_relation")
        if route_relation not in {"independent_model", "same_model", "unknown"}:
            raise ValueError("Internal model-judge route relation is invalid.")
        if result.outcome.value in {"passed", "failed"} and (
            route_relation == "unknown"
            or (route_relation == "same_model" and profile.same_model_use != "allowed_and_labeled")
        ):
            raise ValueError("Internal model judge used a forbidden same-model route.")
        usage = None
        cost = None
        if result.outcome.value in {"passed", "failed"}:
            try:
                usage = PublishedModelJudgeUsageV1.model_validate(raw.get("usage"))
                cost = PublishedModelJudgeCostV1.model_validate(raw.get("cost"))
            except (TypeError, ValueError) as exc:
                raise ValueError("Internal model-judge accounting is invalid.") from exc
        return PublishedModelJudgeDetail(
            evaluator_key=spec.evaluator_key,
            evaluator_implementation_revision=profile.implementation_revision,
            rubric=spec.rubric,
            rubric_version=spec.rubric_version,
            threshold=spec.threshold,
            include_transcript=spec.include_transcript,
            diagnostic=_model_judge_diagnostic(result.outcome.value),
            judge_profile=profile,
            candidate_route_relation=route_relation,
            usage=usage,
            cost=cost,
        )
    if type(spec) is StructuredModelJudgeAssertionSpec:
        raw = result.metadata.get(_STRUCTURED_MODEL_JUDGE_RESULT_METADATA_KEY)
        if type(raw) is not dict:
            raise ValueError("Internal structured judgment did not record its public contract.")
        try:
            profile = JudgeProfileIdentityV1.model_validate(raw.get("judge_profile"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Internal structured judgment profile is invalid.") from exc
        if (profile.key, profile.revision) != (
            spec.judge_profile_key,
            spec.judge_profile_revision,
        ):
            raise ValueError("Internal structured judgment profile does not match the corpus.")
        route_relation = raw.get("candidate_route_relation")
        if route_relation not in {"independent_model", "same_model", "unknown"}:
            raise ValueError("Internal structured judgment route relation is invalid.")
        if result.outcome.value in {"passed", "failed"} and (
            route_relation == "unknown"
            or (route_relation == "same_model" and profile.same_model_use != "allowed_and_labeled")
        ):
            raise ValueError("Internal structured judgment used a forbidden same-model route.")
        if (raw.get("rubric_id"), raw.get("rubric_revision")) != (
            spec.rubric.id,
            spec.rubric.revision,
        ):
            raise ValueError("Internal structured judgment rubric does not match the corpus.")
        expected_reference = _spec_reference_contract(spec.reference)
        raw_reference = raw.get("reference")
        reference = None
        if expected_reference is None:
            if raw_reference is not None:
                raise ValueError("Internal structured judgment added an unexpected reference.")
        else:
            if type(raw_reference) is not dict:
                raise ValueError("Internal structured judgment reference identity is invalid.")
            reference = PublishedJudgeReferenceIdentityV1(
                kind=raw_reference.get("kind"),
                key=raw_reference.get("id", raw_reference.get("key")),
                revision=raw_reference.get("revision"),
                privacy_policy_key=raw_reference.get("privacy_policy_key"),
                privacy_policy_revision=raw_reference.get("privacy_policy_revision"),
            )
            if _published_reference_contract(reference) != expected_reference:
                raise ValueError(
                    "Internal structured judgment reference does not match the corpus."
                )
        scored = result.outcome.value in {"passed", "failed"}
        criteria: tuple[PublishedStructuredJudgeCriterionV1, ...] = ()
        aggregate_score = None
        usage = None
        cost = None
        if scored:
            raw_criteria = raw.get("criteria")
            if type(raw_criteria) is not list or len(raw_criteria) != len(spec.rubric.criteria):
                raise ValueError("Internal structured judgment criterion evidence is incomplete.")
            published_criteria: list[PublishedStructuredJudgeCriterionV1] = []
            for raw_item, criterion in zip(
                raw_criteria,
                spec.rubric.criteria,
                strict=True,
            ):
                if type(raw_item) is not dict:
                    raise ValueError("Internal structured criterion evidence is invalid.")
                raw_criterion = cast("dict[str, Any]", raw_item)
                criterion_id = raw_criterion.get("criterion_id")
                score = raw_criterion.get("score")
                explanation = raw_criterion.get("explanation")
                explanation_state = raw_criterion.get("explanation_state")
                if (
                    type(criterion_id) is not str
                    or type(score) is not str
                    or (explanation is not None and type(explanation) is not str)
                    or explanation_state not in {"available", "redacted", "unavailable"}
                ):
                    raise ValueError("Internal structured criterion evidence is invalid.")
                published = PublishedStructuredJudgeCriterionV1(
                    criterion_id=criterion_id,
                    weight=criterion.weight,
                    score=score,
                    explanation=cast("str | None", explanation),
                    explanation_state=cast(
                        'Literal["available", "redacted", "unavailable"]',
                        explanation_state,
                    ),
                )
                if published.criterion_id != criterion.id:
                    raise ValueError("Internal structured criterion order is invalid.")
                published_criteria.append(published)
            criteria = tuple(published_criteria)
            raw_aggregate = raw.get("aggregate_score")
            if type(raw_aggregate) is not str:
                raise ValueError("Internal structured judgment aggregate is invalid.")
            aggregate_score = raw_aggregate
            try:
                usage = PublishedStructuredJudgeUsageV1.model_validate(raw.get("usage"))
                cost = PublishedStructuredJudgeCostV1.model_validate(raw.get("cost"))
            except (TypeError, ValueError) as exc:
                raise ValueError("Internal structured judge observations are invalid.") from exc
        elif any(key in raw for key in ("criteria", "aggregate_score", "usage", "cost")):
            raise ValueError("Unscored structured judgments cannot retain judge observations.")
        return PublishedStructuredModelJudgeDetail(
            judge_profile=profile,
            candidate_route_relation=route_relation,
            rubric_id=spec.rubric.id,
            rubric_revision=spec.rubric.revision,
            reference=reference,
            threshold=spec.threshold,
            evidence=spec.evidence,
            diagnostic=_model_judge_diagnostic(result.outcome.value),
            criteria=criteria,
            aggregate_score=aggregate_score,
            usage=usage,
            cost=cost,
        )
    raise AssertionError("Unreachable portable assertion detail type.")


def _published_assertion(
    spec: AssertionSpec,
    result: EvalAssertionResult,
) -> PublishedAssertionResult:
    if result.name != spec.id:
        raise ValueError("Internal assertion results do not match the corpus assertion IDs.")
    expected_revision = assertion_spec_revision(spec)
    if result.assertion_revision != expected_revision:
        raise ValueError(
            "Internal assertion result revision does not match the corpus assertion contract."
        )
    if result.score is not None:
        if type(spec) is ModelJudgeAssertionSpec and result.threshold != spec.threshold:
            raise ValueError("Internal model-judge threshold does not match the corpus contract.")
        if type(spec) is StructuredModelJudgeAssertionSpec and result.threshold != float(
            Decimal(spec.threshold)
        ):
            raise ValueError("Internal model-judge threshold does not match the corpus contract.")
    outcome = result.outcome.value
    return PublishedAssertionResult(
        assertion_id=spec.id,
        assertion_revision=expected_revision,
        outcome=outcome,
        score=result.score,
        code=outcome,
        message=_ASSERTION_MESSAGE[outcome],
        detail=_published_detail(spec, result),
    )


def _published_usage(value: Mapping[str, Any] | None) -> PublishedUsageSummaryV1 | None:
    if value is None:
        return None
    usage = value.get("usage")
    if type(usage) is not dict:
        return None
    model_steps = value.get("model_steps")
    tool_calls = value.get("tool_calls")
    if not all(type(item) is int and item >= 0 for item in (model_steps, tool_calls)):
        return None
    try:
        aggregate_usage = aggregate_usage_metrics_from_durable_payload(usage)
    except (TypeError, ValueError):
        return None
    return PublishedUsageSummaryV1(
        model_steps=cast("int", model_steps),
        tool_calls=cast("int", tool_calls),
        total_tokens=aggregate_usage.total_tokens,
    )


def _published_case(
    case: EvalCaseSpec,
    result,
    *,
    trial_policy: EvalSuiteTrialPolicyV1,
    trial_public_data: tuple[_EvalTrialPublicData, ...] | None,
) -> PublishedEvalCaseResult:
    trials: list[PublishedEvalTrialResult] = []
    for index, trial in enumerate(result.trials):
        if len(trial.assertions) != len(case.assertions):
            raise ValueError("Internal trial assertions do not match the corpus contract.")
        assertions = tuple(
            _published_assertion(spec, assertion)
            for spec, assertion in zip(case.assertions, trial.assertions, strict=True)
        )
        status = trial.status.value
        if status == "skipped":
            raise ValueError("Portable corpus trials cannot be skipped.")
        if trial_public_data is None:
            public_data = _EvalTrialPublicData(
                diagnostic_code=_DEFAULT_TRIAL_CODE[status],
                output=EvalTrialOutputPreviewV1.unavailable(),
            )
        else:
            public_data = trial_public_data[index]
        trials.append(
            PublishedEvalTrialResult(
                execution_status=trial.execution_status,
                capture_bounds=trial.capture_bounds,
                capture_diagnostic=trial.capture_diagnostic,
                trial_number=trial.trial_number,
                source_trial_revision=eval_trial_result_revision(
                    EvalTrialResult.model_validate(
                        trial.model_dump(mode="python", round_trip=True, warnings="none")
                    )
                ),
                status=status,
                score=trial.score,
                assertions=assertions,
                evidence_complete=trial.evidence_complete,
                duration_ms=trial.duration_ms,
                usage=_published_usage(trial.usage_summary),
                output=public_data.output,
                memory_attribution=trial.memory_attribution,
                code=public_data.diagnostic_code,
                message=_TRIAL_MESSAGE[public_data.diagnostic_code],
            )
        )
    uses_model_judge = any(
        type(assertion) in {ModelJudgeAssertionSpec, StructuredModelJudgeAssertionSpec}
        for assertion in case.assertions
    )
    reliability = EvalCaseReliabilityV1.create(
        policy=trial_policy,
        trials=((trial.status, trial.score, trial.code) for trial in trials),
        uses_model_judge=uses_model_judge,
    )
    return PublishedEvalCaseResult(
        case_id=case.id,
        case_revision=case.revision,
        status=reliability.outcome,
        score=_published_score(trial.score for trial in trials),
        reliability=reliability,
        trials=tuple(trials),
        duration_ms=sum(trial.duration_ms for trial in trials),
    )


def publish_eval_run(corpus: EvalCorpusDocument, run: EvalRun) -> PublishedEvalRun:
    """Project one lossless internal suite run into the public corpus result graph."""

    return _publish_eval_run_with_trial_public_data(
        corpus,
        run,
        trial_public_data_by_case=None,
    )


def _publish_eval_run_with_trial_public_data(
    corpus: EvalCorpusDocument,
    run: EvalRun,
    *,
    trial_public_data_by_case: dict[str, tuple[_EvalTrialPublicData, ...]] | None,
    accepted_exposure: EvalSuiteRunExposureV1 | None = None,
) -> PublishedEvalRun:
    """Internal execution projection with separately supplied redacted trial data."""

    if type(corpus) is not EvalCorpusDocument:
        raise TypeError("corpus must be an exact EvalCorpusDocument.")
    if type(run) is not EvalRun:
        raise TypeError("run must be an exact EvalRun.")
    corpus = EvalCorpusDocument.model_validate(_model_instance_python_input(corpus))
    run = EvalRun.model_validate(_model_instance_python_input(run))
    suite = next((item for item in corpus.suites if item.id == run.suite_id), None)
    if suite is None:
        raise ValueError("Internal eval run references a suite absent from the corpus.")
    expected_run_contract = eval_run_contract_for_corpus(corpus, suite.id)
    if run.run_contract is None:
        raise ValueError("Portable publication requires an execution-time run contract.")
    if run.run_contract != expected_run_contract:
        raise ValueError("Internal eval run contract does not match the corpus contract.")
    case_specs = tuple(case for case in corpus.cases if case.suite_id == suite.id)
    result_by_id = {case.case_id: case for case in run.cases}
    if set(result_by_id) != {case.id for case in case_specs}:
        raise ValueError("Internal eval run cases do not match the complete corpus suite.")
    if any(len(result_by_id[case.id].trials) != suite.trial_request.trials for case in case_specs):
        raise ValueError("Internal eval run trial counts do not match the corpus suite.")
    validated_public_data: dict[str, tuple[_EvalTrialPublicData, ...]] | None = None
    if trial_public_data_by_case is not None:
        if type(trial_public_data_by_case) is not dict:
            raise TypeError("trial_public_data_by_case must be an exact dict.")
        if set(trial_public_data_by_case) != {case.id for case in case_specs}:
            raise ValueError("Trial public data must match the complete corpus suite.")
        validated_public_data = {}
        for case in case_specs:
            values = trial_public_data_by_case[case.id]
            if type(values) is not tuple or len(values) != len(result_by_id[case.id].trials):
                raise ValueError("Trial public data counts must match the corpus suite.")
            copied: list[_EvalTrialPublicData] = []
            for value in values:
                if type(value) is not _EvalTrialPublicData:
                    raise TypeError(
                        "Trial public data must contain exact runner-owned projection values."
                    )
                copied.append(_EvalTrialPublicData.model_validate(value.model_dump(mode="python")))
            validated_public_data[case.id] = tuple(copied)
    cases = tuple(
        _published_case(
            case,
            result_by_id[case.id],
            trial_policy=expected_run_contract.trial_policy,
            trial_public_data=(
                None if validated_public_data is None else validated_public_data[case.id]
            ),
        )
        for case in case_specs
    )
    status = _published_status_from_statuses(case.status for case in cases)
    document: dict[str, Any] = {
        "schema_version": PUBLISHED_EVAL_SCHEMA_VERSION,
        "corpus_revision": corpus.revision,
        "target_key": corpus.target_key,
        "suite_id": suite.id,
        "suite_revision": suite.revision,
        "evidence_policy_revision": corpus.evidence_policy.revision,
        "pricing_profile_fingerprint": expected_run_contract.pricing_profile_fingerprint,
        "trial_policy": expected_run_contract.trial_policy.model_dump(mode="json"),
        "status": status,
        "score": _published_score(case.score for case in cases),
        "cases": cases,
        "duration_ms": sum(case.duration_ms for case in cases),
    }
    if accepted_exposure is not None:
        document["accepted_exposure"] = accepted_exposure.model_dump(mode="json")
    revision_document = {
        **document,
        "cases": [case.model_dump(mode="json") for case in cases],
    }
    return PublishedEvalRun(
        revision=_content_revision(revision_document, "published eval run"),
        **document,
    )
