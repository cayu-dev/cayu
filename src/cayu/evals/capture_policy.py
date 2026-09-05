"""Finite evidence retention policy shared by live and saved workflow evaluations."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

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


class WorkflowCaptureDiagnostic(BaseModel):
    """Payload-free bounded-read rejection; observed is a witness, never a total."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    stage: Literal["child_capture", "capture_revalidation", "post_scoring_revalidation"] = (
        "child_capture"
    )
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
