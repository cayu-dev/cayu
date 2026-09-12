"""Finite evidence retention policy shared by live and saved workflow evaluations."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from cayu.memory_attribution import MemoryAttributionBounds
from cayu.runtime.sessions import (
    TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_EVENTS,
    TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_RECORD_BYTES,
    TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_TOTAL_BYTES,
    TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_TRANSCRIPT_RECORDS,
    TERMINAL_SESSION_EVIDENCE_HARD_MAX_EVENTS,
    TERMINAL_SESSION_EVIDENCE_HARD_MAX_RECORD_BYTES,
    TERMINAL_SESSION_EVIDENCE_HARD_MAX_TOTAL_BYTES,
    TERMINAL_SESSION_EVIDENCE_HARD_MAX_TRANSCRIPT_RECORDS,
    TerminalSessionEvidenceErrorCode,
)

_SESSION_TRAJECTORY_DEFAULT_MAX_SESSIONS = 100
_SESSION_TRAJECTORY_HARD_MAX_SESSIONS = 500
_SESSION_TRAJECTORY_DEFAULT_MAX_DEPTH = 32
_SESSION_TRAJECTORY_HARD_MAX_DEPTH = 32
_SESSION_TRAJECTORY_HARD_MAX_LINEAGE_CANDIDATES = 500


class SessionTrajectoryBounds(BaseModel):
    """Global retained-evidence limits for one production-session trajectory."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    max_sessions: StrictInt = Field(
        default=_SESSION_TRAJECTORY_DEFAULT_MAX_SESSIONS,
        ge=1,
        le=_SESSION_TRAJECTORY_HARD_MAX_SESSIONS,
    )
    max_depth: StrictInt = Field(
        default=_SESSION_TRAJECTORY_DEFAULT_MAX_DEPTH,
        ge=1,
        le=_SESSION_TRAJECTORY_HARD_MAX_DEPTH,
    )
    max_events: StrictInt = Field(
        default=TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_EVENTS,
        ge=1,
        le=TERMINAL_SESSION_EVIDENCE_HARD_MAX_EVENTS,
    )
    max_transcript_records: StrictInt = Field(
        default=TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_TRANSCRIPT_RECORDS,
        ge=0,
        le=TERMINAL_SESSION_EVIDENCE_HARD_MAX_TRANSCRIPT_RECORDS,
    )
    max_record_bytes: StrictInt = Field(
        default=TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_RECORD_BYTES,
        ge=1,
        le=TERMINAL_SESSION_EVIDENCE_HARD_MAX_RECORD_BYTES,
    )
    max_total_bytes: StrictInt = Field(
        default=TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_TOTAL_BYTES,
        ge=1,
        le=TERMINAL_SESSION_EVIDENCE_HARD_MAX_TOTAL_BYTES,
    )
    memory_attribution_bounds: MemoryAttributionBounds = Field(
        default_factory=MemoryAttributionBounds
    )


class SessionTrajectoryErrorCode(StrEnum):
    """Stable reason a durable session tree cannot become exact eval evidence."""

    DEADLINE_EXCEEDED = "deadline_exceeded"
    STORE_UNSUPPORTED = "store_unsupported"
    EVIDENCE_READ_FAILED = "evidence_read_failed"
    TERMINAL_EVIDENCE_REJECTED = "terminal_evidence_rejected"
    DESCENDANT_ENUMERATION_FAILED = "descendant_enumeration_failed"
    ORIGIN_EVIDENCE_REJECTED = "origin_evidence_rejected"
    PARENT_CONTRADICTION = "parent_contradiction"
    CYCLE_DETECTED = "cycle_detected"
    SESSION_LIMIT_EXCEEDED = "session_limit_exceeded"
    DEPTH_LIMIT_EXCEEDED = "depth_limit_exceeded"
    CLOSURE_CHANGED = "closure_changed"
    EVIDENCE_INCONSISTENT = "evidence_inconsistent"


WorkflowCaptureStage = Literal[
    "execution",
    "terminal_load",
    "result_projection",
    "child_capture",
    "probe_capture",
    "capture_revalidation",
    "assertion",
    "post_scoring_revalidation",
]


class WorkflowCaptureDiagnostic(BaseModel):
    """Payload-free bounded-read rejection; observed is a witness, never a total."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    stage: WorkflowCaptureStage = "child_capture"
    code: SessionTrajectoryErrorCode
    session_id: str
    terminal_code: TerminalSessionEvidenceErrorCode | None = None
    limit: StrictInt | None = Field(default=None, ge=0)
    observed_lower_bound: StrictInt | None = Field(default=None, ge=0)
    consumed_events: StrictInt = Field(default=0, ge=0)
    consumed_transcript_records: StrictInt = Field(default=0, ge=0)
    consumed_bytes: StrictInt = Field(default=0, ge=0)
    bounds: SessionTrajectoryBounds


class WorkflowAttemptAnchor(BaseModel):
    """Original completed execution identity, independent of subsequent capture/scoring."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    origin: Literal["execution", "saved_store_import"] = "execution"
    source_report_sha256: str | None = None
    run_id: str
    suite_id: str
    case_id: str
    trial_number: int
    session_id: str
    target_revision: str
    projector_revision: str
    input_messages_sha256: str
    attempt_id: str
    completion_event_id: str
    completion_sequence: int
    root_sha256: str
    final_output_sha256: str
    structured_output_sha256: str


class WorkflowFailureRecordReference(BaseModel):
    """Exact retained-store record range, without copying its event payloads."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    session_id: str = Field(min_length=1, max_length=2048)
    run_epoch: StrictInt = Field(ge=0)
    first_sequence: StrictInt = Field(ge=1)
    last_sequence: StrictInt = Field(ge=1)
    first_event_id: str = Field(min_length=1, max_length=512)
    last_event_id: str = Field(min_length=1, max_length=512)
    record_count: StrictInt = Field(ge=1, le=100_000)
    records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_range(self):
        if self.first_sequence > self.last_sequence:
            raise ValueError("Failure evidence record range must be ordered.")
        return self


class WorkflowFailureCapture(BaseModel):
    """Partial observations of failed execution; never successful scoring evidence.

    Counts cover only the validated record ranges listed here. Usage is a lower
    bound from observed usage-bearing model completions, not provider billing.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    session_id: str = Field(min_length=1, max_length=2048)
    attempt_id: str | None = Field(default=None, max_length=512)
    started_event_id: str | None = Field(default=None, max_length=512)
    state: Literal["partial", "unavailable"] = "unavailable"
    events_count_basis: Literal["validated_durable_records"] = "validated_durable_records"
    usage_basis: Literal["observed_usage_records", "unavailable"] = "unavailable"
    records: tuple[WorkflowFailureRecordReference, ...] = Field(default=(), max_length=500)
    model_calls: StrictInt = Field(default=0, ge=0)
    tool_calls: StrictInt = Field(default=0, ge=0)
    model_calls_with_usage: StrictInt = Field(default=0, ge=0)
    diagnostics: tuple[WorkflowCaptureDiagnostic, ...] = Field(default=(), max_length=16)
    diagnostics_truncated: bool = False

    @property
    def events_count(self) -> int:
        return sum(reference.record_count for reference in self.records)

    @model_validator(mode="after")
    def validate_observations(self):
        if self.state == "unavailable" and self.records:
            raise ValueError("Unavailable failure capture cannot claim retained records.")
        if self.state == "partial" and (
            not self.records or self.attempt_id is None or self.started_event_id is None
        ):
            raise ValueError("Partial failure capture requires exact workflow-attempt identity.")
        if self.records and self.records[0].session_id != self.session_id:
            raise ValueError("Failed-workflow records must begin with the workflow root.")
        if len({record.session_id for record in self.records}) != len(self.records):
            raise ValueError("Failure capture cannot count a session twice.")
        if (self.usage_basis == "observed_usage_records") != (self.model_calls_with_usage > 0):
            raise ValueError("Failure usage basis must match observed usage records.")
        return self
