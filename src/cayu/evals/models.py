from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.artifacts import ArtifactMetadata, ArtifactScope
from cayu.core.events import Event
from cayu.core.messages import Message
from cayu.runtime.sessions import Session
from cayu.runtime.usage import SessionUsageSummary

# Version of the persisted EvalRun JSON shape. Bump this by hand whenever the
# saved structure changes incompatibly so load_eval_run can detect a baseline
# written by a newer cayu instead of silently misreading it. (Migration of old
# baselines is a follow-up; today the guard only rejects newer-than-supported.)
EVAL_SCHEMA_VERSION = 1

# Cap on the bytes copied out of a probed workspace file into the serialized trajectory. A file
# larger than this is captured truncated — with its true size and a content hash still recorded —
# so a multi-GB workspace file can never balloon the trajectory JSON (which base64-encodes bytes)
# or the in-memory result. Assertions still see the leading window of the file.
WORKSPACE_PROBE_MAX_BYTES = 1 << 20  # 1 MiB


class EvalStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    # A case with zero assertions is not scored as a pass (that would fail open); it is
    # recorded as SKIPPED so an author sees the case ran but asserted nothing.
    SKIPPED = "skipped"


class EvalAssertionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    # `score` is the primitive (0..1); `passed` is the threshold view. Deterministic
    # checks emit 0.0/1.0, so a graded LLM-judge later is an additive change, not a
    # breaking one. A boolean check that omits `score` gets it derived from `passed`.
    score: StrictFloat = Field(default=0.0, ge=0.0, le=1.0)
    threshold: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    passed: StrictBool
    message: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def default_score_from_passed(cls, data: Any) -> Any:
        if isinstance(data, dict) and "score" not in data and "passed" in data:
            return {**data, "score": 1.0 if data["passed"] else 0.0}
        return data

    @model_validator(mode="after")
    def check_score_passed_consistency(self) -> EvalAssertionResult:
        bar = self.threshold if self.threshold is not None else 1.0
        if self.passed != (self.score >= bar):
            raise ValueError(
                f"EvalAssertionResult passed={self.passed} disagrees with "
                f"score={self.score} at threshold {bar}."
            )
        return self

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")


class EvalCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    status: EvalStatus
    session_id: str | None = None
    score: StrictFloat = Field(default=0.0, ge=0.0, le=1.0)
    final_output: str = ""
    assertions: tuple[EvalAssertionResult, ...] = Field(default_factory=tuple)
    error: str | None = None
    events_count: StrictInt = Field(default=0, ge=0)
    usage_summary: dict[str, Any] | None = None
    started_at: datetime
    completed_at: datetime
    duration_ms: StrictInt = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # The full run record, populated only when run_eval_case(..., retain_trajectory=True). It is
    # excluded from the persisted score-first JSON (exclude=True) and from repr; it is an in-memory
    # handle for export/replay (write_trajectory_json), never part of the saved baseline.
    trajectory: Trajectory | None = Field(default=None, exclude=True, repr=False)

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("error", mode="before")
    @classmethod
    def normalize_error(cls, value: object) -> str | None:
        # `error` is a captured diagnostic (often a raw exception string), not an
        # identifier; normalize whitespace instead of rejecting it so an exception
        # message ending in a newline can never crash result construction.
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("usage_summary", "metadata", mode="before")
    @classmethod
    def copy_json_data(cls, value, info):
        return copy_json_value(value, info.field_name)


class EvalRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: StrictInt = Field(default=EVAL_SCHEMA_VERSION, ge=1)
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    suite_id: str
    status: EvalStatus
    score: StrictFloat = Field(default=0.0, ge=0.0, le=1.0)
    cases: tuple[EvalCaseResult, ...] = Field(default_factory=tuple)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    duration_ms: StrictInt = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("run_id", "suite_id")
    @classmethod
    def validate_ids(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")


@dataclass(frozen=True)
class ProbeRequirements:
    """What an assertion needs captured from the live environment to evaluate offline.

    The runner unions these across a case's assertions and snapshots the result into the
    Trajectory while the environment is still live, so assertions never touch the app.
    """

    workspace_paths: frozenset[str] = frozenset()
    artifact_scopes: frozenset[ArtifactScope] = frozenset()

    def merged_with(self, other: ProbeRequirements) -> ProbeRequirements:
        return ProbeRequirements(
            workspace_paths=self.workspace_paths | other.workspace_paths,
            artifact_scopes=self.artifact_scopes | other.artifact_scopes,
        )


class WorkspaceFileProbe(BaseModel):
    """Stat + integrity record for a probed workspace file.

    Companion to ``TrajectoryProbes.workspace_files`` (the captured, possibly-truncated bytes):
    records the file's true size on disk, whether the captured window was truncated at
    ``WORKSPACE_PROBE_MAX_BYTES``, and a sha256 of the captured bytes. This lets a large file's
    size and identity survive in the trajectory without base64-ing its whole content into JSON.
    """

    model_config = ConfigDict(extra="forbid")

    total_bytes: StrictInt = Field(ge=0)
    truncated: StrictBool = False
    sha256: str

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)


class TrajectoryProbes(BaseModel):
    """Serializable snapshot of the live-environment data assertions need.

    Captured by the runner before the environment is torn down. The ``*_available`` flags
    distinguish "no workspace / artifact store configured" from "configured but the file /
    artifact is absent", which the workspace and artifact assertions report differently.
    """

    model_config = ConfigDict(extra="forbid", ser_json_bytes="base64", val_json_bytes="base64")

    workspace_available: StrictBool = False
    # path -> file bytes (capped at WORKSPACE_PROBE_MAX_BYTES), or None when the file is
    # absent/unreadable. A declared path is always a key (so "missing key" means it was never
    # probed), distinct from a None value.
    workspace_files: dict[str, bytes | None] = Field(default_factory=dict)
    # path -> stat/hash for each present, readable probed file. A path is absent here when the
    # file was missing/unreadable (its workspace_files value is None); consult total_bytes /
    # truncated to tell a fully-captured file from one whose bytes were truncated to the cap.
    workspace_file_stats: dict[str, WorkspaceFileProbe] = Field(default_factory=dict)
    artifacts_available: StrictBool = False
    artifacts: tuple[ArtifactMetadata, ...] = Field(default_factory=tuple)


class Trajectory(BaseModel):
    """The serializable **record** of one completed run — and the eval assertion substrate.

    Composes already-serializable cayu types (events / transcript / usage / session) with a
    captured probe snapshot and the recursive sub-agent trajectories, so a whole run — its
    sub-agent tree included — can be persisted, reloaded, and replayed against assertions
    without a live runtime. It is the single object that flows through the lifecycle: a run
    produces it (`EvalCaseResult.trajectory`), `write_trajectory_json` / `load_trajectory`
    move it to and from disk, and `evaluate_assertions` re-checks it (the replay path).
    Distinct from `EvalContext`, which is the assertion's *view* of a Trajectory.
    """

    model_config = ConfigDict(extra="forbid")

    session: Session | None = None
    events: tuple[Event, ...] = Field(default_factory=tuple)
    transcript: tuple[Message, ...] = Field(default_factory=tuple)
    usage_summary: SessionUsageSummary | None = None
    final_output: str = ""
    probes: TrajectoryProbes = Field(default_factory=TrajectoryProbes)
    children: tuple[Trajectory, ...] = Field(default_factory=tuple)
    # True when the sub-agent walk stopped before enumerating every child — a store error mid-walk
    # or hitting the page cap — so a partial `children` capture is never mistaken for "no more".
    children_incomplete: StrictBool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class EvalContext:
    """The assertion's **view** of a run: the `Trajectory` under test + case identity.

    Where `Trajectory` is the serializable run *record*, `EvalContext` is what an assertion's
    `evaluate()` receives — it exposes the trajectory's data (`session` / `events` /
    `transcript` / `usage_summary` / `final_output` / `probes`) plus the `suite_id` / `case_id`
    / `metadata`, and carries no live `app` handle (assertions run offline). It is also the
    intended home for future dataset *expectations* (a `reference` of expected values an
    assertion compares the trajectory against).
    """

    trajectory: Trajectory
    suite_id: str
    case_id: str
    metadata: dict[str, Any]

    @property
    def session(self) -> Session | None:
        return self.trajectory.session

    @property
    def events(self) -> tuple[Event, ...]:
        return self.trajectory.events

    @property
    def transcript(self) -> tuple[Message, ...]:
        return self.trajectory.transcript

    @property
    def usage_summary(self) -> SessionUsageSummary | None:
        return self.trajectory.usage_summary

    @property
    def final_output(self) -> str:
        return self.trajectory.final_output

    @property
    def probes(self) -> TrajectoryProbes:
        return self.trajectory.probes


# EvalCaseResult.trajectory forward-references Trajectory, which is defined later in this module;
# rebuild now that it exists so Pydantic can resolve the annotation.
EvalCaseResult.model_rebuild()
