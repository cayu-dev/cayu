from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Literal

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

from cayu._validation import require_durable_text

EVAL_TRIAL_OUTPUT_MAX_RETAINED_CHARS = 65_536
EVAL_TRIAL_OUTPUT_MAX_RETAINED_BYTES = EVAL_TRIAL_OUTPUT_MAX_RETAINED_CHARS * 4
EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES = 16 << 10
PUBLISHED_EVAL_OUTPUT_PREVIEW_BUDGET_BYTES = 2 << 20

EvalTrialOutputEvidenceState = Literal["complete", "unavailable", "limit_exceeded"]


class EvalTrialDiagnosticCode(StrEnum):
    """Stable, non-secret reason for one fresh trial's terminal outcome."""

    PASSED = "passed"
    ASSERTION_FAILED = "assertion_failed"
    ASSERTION_EVIDENCE_UNAVAILABLE = "assertion_evidence_unavailable"
    TERMINAL_EVIDENCE_UNAVAILABLE = "terminal_evidence_unavailable"
    INTERRUPTED_EVIDENCE_UNAVAILABLE = "interrupted_evidence_unavailable"
    CHILD_EVIDENCE_UNAVAILABLE = "child_evidence_unavailable"
    EXTERNAL_TARGET_UNAVAILABLE = "external_target_unavailable"
    EXTERNAL_TARGET_CANCELLED = "external_target_cancelled"
    EXTERNAL_TARGET_UNKNOWN = "external_target_unknown"
    EXTERNAL_TARGET_INCOMPLETE = "external_target_incomplete"
    EXTERNAL_TARGET_IDENTITY_MISMATCH = "external_target_identity_mismatch"
    EXTERNAL_TARGET_FAILED = "external_target_failed"
    WORKFLOW_TARGET_FAILED = "workflow_target_failed"
    WORKFLOW_EXECUTION_FAILED = "workflow_execution_failed"
    WORKFLOW_COMPLETION_MISSING = "workflow_completion_missing"
    WORKFLOW_COMPLETION_CONFLICT = "workflow_completion_conflict"
    WORKFLOW_ATTEMPT_SUPERSEDED = "workflow_attempt_superseded"
    WORKFLOW_PROJECTOR_FAILED = "workflow_projector_failed"
    WORKFLOW_OUTPUT_INVALID = "workflow_output_invalid"
    WORKFLOW_QUIESCENCE_FAILED = "workflow_quiescence_failed"
    EXECUTION_FAILED = "execution_failed"
    SESSION_FAILED = "session_failed"
    TERMINAL_EVIDENCE_FAILED = "terminal_evidence_failed"
    EVIDENCE_PREPARATION_FAILED = "evidence_preparation_failed"
    ASSERTION_EVALUATION_FAILED = "assertion_evaluation_failed"
    CASE_TIMEOUT = "case_timeout"


class EvalTrialOutputPreviewV1(BaseModel):
    """Bounded redacted output evidence retained for safe result inspection."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    schema_version: Literal[1] = 1
    text: StrictStr = ""
    evidence_state: EvalTrialOutputEvidenceState
    preview_truncated: StrictBool = False
    retained_chars: StrictInt = Field(ge=0, le=EVAL_TRIAL_OUTPUT_MAX_RETAINED_CHARS)
    retained_bytes: StrictInt = Field(ge=0, le=EVAL_TRIAL_OUTPUT_MAX_RETAINED_BYTES)
    retained_sha256: StrictStr | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if len(value) > EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES:
            raise ValueError(
                "Eval trial output preview exceeds "
                f"{EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES} UTF-8 bytes."
            )
        value = require_durable_text(value, "text")
        if len(value.encode("utf-8")) > EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES:
            raise ValueError(
                "Eval trial output preview exceeds "
                f"{EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES} UTF-8 bytes."
            )
        return value

    @field_validator("retained_sha256")
    @classmethod
    def validate_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("retained_sha256 must be a lowercase SHA-256 digest.")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> EvalTrialOutputPreviewV1:
        text_bytes = self.text.encode("utf-8")
        if self.evidence_state == "unavailable":
            if (
                self.text
                or self.preview_truncated
                or self.retained_chars
                or self.retained_bytes
                or self.retained_sha256 is not None
            ):
                raise ValueError("Unavailable output evidence cannot carry retained output data.")
            return self
        if self.retained_sha256 is None:
            raise ValueError("Available output evidence requires a retained SHA-256 digest.")
        if not self.retained_chars <= self.retained_bytes <= self.retained_chars * 4:
            raise ValueError("Retained output character and UTF-8 byte counts are inconsistent.")
        if self.retained_chars < len(self.text) or self.retained_bytes < len(text_bytes):
            raise ValueError("Output preview cannot exceed its retained evidence size.")
        if self.preview_truncated:
            if len(self.text) >= self.retained_chars:
                raise ValueError("A truncated output preview must omit retained characters.")
            if len(text_bytes) >= self.retained_bytes:
                raise ValueError("A truncated output preview must omit retained UTF-8 bytes.")
        else:
            if self.retained_chars != len(self.text) or self.retained_bytes != len(text_bytes):
                raise ValueError("A complete output preview must equal its retained evidence.")
            if self.retained_sha256 != hashlib.sha256(text_bytes).hexdigest():
                raise ValueError("Output preview digest does not match its complete text.")
        if (
            self.evidence_state == "limit_exceeded"
            and self.retained_chars != EVAL_TRIAL_OUTPUT_MAX_RETAINED_CHARS
        ):
            raise ValueError("Limited output evidence must retain its complete bounded prefix.")
        return self

    @classmethod
    def unavailable(cls) -> EvalTrialOutputPreviewV1:
        return cls(
            evidence_state="unavailable",
            retained_chars=0,
            retained_bytes=0,
        )

    @classmethod
    def from_retained_evidence(
        cls,
        text: str,
        evidence_state: EvalTrialOutputEvidenceState,
        *,
        max_preview_bytes: int,
    ) -> EvalTrialOutputPreviewV1:
        if type(text) is not str:
            raise TypeError("text must be a str.")
        if type(evidence_state) is not str or evidence_state not in {
            "complete",
            "unavailable",
            "limit_exceeded",
        }:
            raise ValueError("evidence_state is not supported.")
        if type(max_preview_bytes) is not int:
            raise TypeError("max_preview_bytes must be an int.")
        if not 1 <= max_preview_bytes <= EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES:
            raise ValueError(
                f"max_preview_bytes must be between 1 and {EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES}."
            )
        if evidence_state == "unavailable":
            if text:
                raise ValueError("Unavailable output evidence cannot carry text.")
            return cls.unavailable()
        if type(text) is str and len(text) > EVAL_TRIAL_OUTPUT_MAX_RETAINED_CHARS:
            raise ValueError("Output evidence exceeds its retained character limit.")
        retained = require_durable_text(text, "output evidence")
        raw = retained.encode("utf-8")
        if len(raw) <= max_preview_bytes:
            preview = retained
            preview_truncated = False
        else:
            preview = raw[:max_preview_bytes].decode("utf-8", errors="ignore")
            preview_truncated = True
        return cls(
            text=preview,
            evidence_state=evidence_state,
            preview_truncated=preview_truncated,
            retained_chars=len(retained),
            retained_bytes=len(raw),
            retained_sha256=hashlib.sha256(raw).hexdigest(),
        )


class _EvalTrialPublicData(BaseModel):
    """Non-secret trial data carried privately until corpus publication."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    diagnostic_code: EvalTrialDiagnosticCode
    output: EvalTrialOutputPreviewV1

    @field_validator("output", mode="before")
    @classmethod
    def copy_output(cls, value: object) -> object:
        if type(value) is EvalTrialOutputPreviewV1:
            return EvalTrialOutputPreviewV1.model_validate(value.model_dump(mode="python"))
        if isinstance(value, BaseModel):
            raise TypeError("output must be an exact EvalTrialOutputPreviewV1 or JSON object.")
        return value
