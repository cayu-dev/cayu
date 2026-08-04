from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
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
from cayu.runtime.costs import SessionCostSummary
from cayu.runtime.sessions import Session
from cayu.runtime.usage import (
    SessionUsageSummary,
    aggregate_usage_metrics_from_durable_payload,
    session_usage_summary_payload,
)

# Version of the persisted EvalRun JSON shape. Bump this by hand whenever the
# saved structure changes incompatibly so load_eval_run can reject a baseline
# written for a different contract instead of silently misreading it. Version 4
# preserves every trial and represents unavailable/error outcomes without fake
# zero scores. Earlier prerelease formats are intentionally not migrated.
EVAL_SCHEMA_VERSION = 4

# Version of the standalone trajectory JSON document written by
# write_trajectory_json. Trajectories were an unversioned preview before v1;
# load_trajectory intentionally does not guess or migrate those shapes.
TRAJECTORY_SCHEMA_VERSION = 1

# Cap on the bytes copied out of a probed workspace file into the serialized trajectory. A file
# larger than this is captured truncated — with its true size and a content hash still recorded —
# so a multi-GB workspace file can never balloon the trajectory JSON (which base64-encodes bytes)
# or the in-memory result. Assertions still see the leading window of the file.
WORKSPACE_PROBE_MAX_BYTES = 1 << 20  # 1 MiB


class EvalStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    # A case with zero assertions is not scored as a pass (that would fail open); it is
    # recorded as SKIPPED so an author sees the case ran but asserted nothing.
    SKIPPED = "skipped"


class EvalOutcome(StrEnum):
    """One evaluated assertion outcome.

    ``unavailable`` means the assertion could not obtain complete evidence;
    ``error`` means execution or evaluation failed. Neither has a numeric score.
    """

    PASSED = "passed"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class EvalAssertionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    outcome: EvalOutcome
    # Deterministic checks emit 0.0/1.0 and graded checks retain their continuous
    # score. Unavailable/error outcomes deliberately carry no score.
    score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    threshold: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    message: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Cost is assertion-specific because pricing is injected by MaxEstimatedCost.
    # Keeping the exact summary beside that trial's assertion avoids inventing one
    # case-wide price book when a case contains multiple cost checks.
    cost_summary: SessionCostSummary | None = None

    @model_validator(mode="after")
    def check_outcome_score_consistency(self) -> EvalAssertionResult:
        if self.outcome in (EvalOutcome.UNAVAILABLE, EvalOutcome.ERROR):
            if self.score is not None:
                raise ValueError(f"{self.outcome.value} assertions cannot have a score.")
            return self
        if self.score is None:
            raise ValueError(f"{self.outcome.value} assertions require a score.")
        bar = self.threshold if self.threshold is not None else 1.0
        expected = EvalOutcome.PASSED if self.score >= bar else EvalOutcome.FAILED
        if self.outcome != expected:
            raise ValueError(
                f"EvalAssertionResult outcome={self.outcome.value} disagrees with "
                f"score={self.score} at threshold {bar}."
            )
        return self

    @property
    def passed(self) -> bool:
        """Return the boolean view used by assertion consumers."""

        return self.outcome == EvalOutcome.PASSED

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value):
        return copy_json_value(value, "metadata")

    @field_validator("cost_summary")
    @classmethod
    def copy_cost_summary(
        cls,
        value: SessionCostSummary | None,
    ) -> SessionCostSummary | None:
        return None if value is None else value.model_copy(deep=True)


class EvalTrialResult(BaseModel):
    """Lossless result for one concrete, independently executed case trial."""

    model_config = ConfigDict(extra="forbid")

    trial_number: StrictInt = Field(ge=1)
    status: EvalStatus
    session_id: str | None = None
    score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    final_output: str = ""
    assertions: tuple[EvalAssertionResult, ...] = Field(default_factory=tuple)
    error: str | None = None
    unavailable_reason: str | None = None
    # Whether the exact run snapshot and complete child tree were captured. Assertion
    # inputs such as a price book can still be unavailable when this is true.
    evidence_complete: StrictBool = False
    events_count: StrictInt = Field(default=0, ge=0)
    usage_summary: dict[str, Any] | None = None
    started_at: datetime
    completed_at: datetime
    duration_ms: StrictInt = Field(default=0, ge=0)
    # Opt-in in-memory export/replay handle. Persisted EvalRun JSON remains score-first.
    trajectory: Trajectory | None = Field(default=None, exclude=True, repr=False)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("error", "unavailable_reason", mode="before")
    @classmethod
    def normalize_diagnostic(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("usage_summary", mode="before")
    @classmethod
    def copy_usage_summary(cls, value):
        if value is None:
            return None
        copied = copy_json_value(value, "usage_summary")
        if type(copied) is not dict:
            raise ValueError("usage_summary must be a session usage summary object.")
        projected = dict(copied)
        projected["usage"] = aggregate_usage_metrics_from_durable_payload(projected.get("usage"))
        return session_usage_summary_payload(SessionUsageSummary.model_validate(projected))

    @model_validator(mode="after")
    def validate_result_contract(self) -> EvalTrialResult:
        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at.")
        expected_duration = max(
            int((self.completed_at - self.started_at).total_seconds() * 1000),
            0,
        )
        if self.duration_ms != expected_duration:
            raise ValueError("duration_ms must match started_at and completed_at.")
        if self.status in (EvalStatus.ERROR, EvalStatus.UNAVAILABLE):
            if self.score is not None:
                raise ValueError(f"{self.status.value} trials cannot have a score.")
        elif self.score is None:
            raise ValueError(f"{self.status.value} trials require a score.")
        if self.status == EvalStatus.ERROR:
            if self.error is None:
                raise ValueError("error trials require an error diagnostic.")
            if self.assertions and not any(
                assertion.outcome == EvalOutcome.ERROR for assertion in self.assertions
            ):
                raise ValueError("error trials must retain an error assertion outcome.")
        elif self.error is not None:
            raise ValueError("Only error trials can carry an error diagnostic.")
        if self.status == EvalStatus.UNAVAILABLE:
            if self.unavailable_reason is None:
                raise ValueError("unavailable trials require an unavailable_reason.")
            if self.assertions and not any(
                assertion.outcome == EvalOutcome.UNAVAILABLE for assertion in self.assertions
            ):
                raise ValueError("unavailable trials must retain an unavailable assertion outcome.")
        elif self.unavailable_reason is not None:
            raise ValueError("Only unavailable trials can carry an unavailable_reason.")
        if self.status in (EvalStatus.PASSED, EvalStatus.FAILED, EvalStatus.SKIPPED):
            if not self.evidence_complete:
                raise ValueError(f"{self.status.value} trials require complete evidence.")
            if self.session_id is None:
                raise ValueError(f"{self.status.value} trials require a concrete session_id.")
        if self.evidence_complete and self.session_id is None:
            raise ValueError("Complete trial evidence requires a concrete session_id.")
        if self.usage_summary is not None and (
            self.session_id is None or self.usage_summary["session_id"] != self.session_id
        ):
            raise ValueError("usage_summary must belong to the trial session_id.")
        if any(
            assertion.cost_summary is not None
            and (self.session_id is None or assertion.cost_summary.session_id != self.session_id)
            for assertion in self.assertions
        ):
            raise ValueError("Assertion cost summaries must belong to the trial session_id.")
        if self.trajectory is not None:
            if (
                self.session_id is None
                or self.trajectory.session is None
                or self.trajectory.session.id != self.session_id
            ):
                raise ValueError("trajectory must belong to the trial session_id.")
            if self.events_count != len(self.trajectory.events):
                raise ValueError("events_count must match the retained trajectory.")
            if self.final_output != self.trajectory.final_output:
                raise ValueError("final_output must match the retained trajectory.")
            trajectory_usage = (
                None
                if self.trajectory.usage_summary is None
                else session_usage_summary_payload(self.trajectory.usage_summary)
            )
            if self.usage_summary != trajectory_usage:
                raise ValueError("usage_summary must match the retained trajectory.")
        expected_status = _status_from_assertions(self.assertions)
        # Execution/evidence can fail before assertions exist. Once assertion
        # outcomes do exist, however, their fail-closed precedence is exact:
        # an ERROR outcome can never be represented as merely UNAVAILABLE.
        if self.assertions and self.status != expected_status:
            raise ValueError("Trial status does not match its assertion outcomes.")
        if not self.assertions and self.status not in (
            EvalStatus.ERROR,
            EvalStatus.UNAVAILABLE,
            EvalStatus.SKIPPED,
        ):
            raise ValueError("Trial status does not match its assertion outcomes.")
        if self.status not in (EvalStatus.ERROR, EvalStatus.UNAVAILABLE):
            expected_score = (
                0.0
                if not self.assertions
                else sum(
                    assertion.score for assertion in self.assertions if assertion.score is not None
                )
                / len(self.assertions)
            )
            if self.score != expected_score:
                raise ValueError("Trial score does not match its assertion scores.")
        return self


class EvalCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    status: EvalStatus
    # The caller-authored logical session identity, if one was supplied on the case request.
    # Eval execution never uses this as a concrete session ID.
    authored_session_id: str | None = None
    trials: tuple[EvalTrialResult, ...] = Field(min_length=1)
    score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    assertions: tuple[EvalAssertionResult, ...] = Field(default_factory=tuple)
    error: str | None = None
    unavailable_reason: str | None = None
    started_at: datetime
    completed_at: datetime
    duration_ms: StrictInt = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_trials(
        cls,
        *,
        case_id: str,
        trials: Iterable[EvalTrialResult],
        authored_session_id: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvalCaseResult:
        retained = tuple(trials)
        if not retained:
            raise ValueError("EvalCaseResult requires at least one trial.")
        case_started_at = started_at or retained[0].started_at
        case_completed_at = completed_at or retained[-1].completed_at
        aggregate = _aggregate_eval_case(retained)
        return cls(
            case_id=case_id,
            status=aggregate.status,
            authored_session_id=authored_session_id,
            trials=retained,
            score=aggregate.score,
            assertions=aggregate.assertions,
            error=aggregate.error,
            unavailable_reason=aggregate.unavailable_reason,
            started_at=case_started_at,
            completed_at=case_completed_at,
            duration_ms=max(
                int((case_completed_at - case_started_at).total_seconds() * 1000),
                0,
            ),
            metadata={} if metadata is None else metadata,
        )

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("authored_session_id")
    @classmethod
    def validate_session_id(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("error", "unavailable_reason", mode="before")
    @classmethod
    def normalize_error(cls, value: object) -> str | None:
        # `error` is a captured diagnostic (often a raw exception string), not an
        # identifier; normalize whitespace instead of rejecting it so an exception
        # message ending in a newline can never crash result construction.
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_json_data(cls, value, info):
        return copy_json_value(value, info.field_name)

    @model_validator(mode="after")
    def validate_aggregate_contract(self) -> EvalCaseResult:
        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at.")
        expected_duration = max(
            int((self.completed_at - self.started_at).total_seconds() * 1000),
            0,
        )
        if self.duration_ms != expected_duration:
            raise ValueError("duration_ms must match started_at and completed_at.")
        expected_numbers = tuple(range(1, len(self.trials) + 1))
        if tuple(trial.trial_number for trial in self.trials) != expected_numbers:
            raise ValueError("Trial numbers must be contiguous and match tuple order.")
        session_ids = tuple(
            trial.session_id for trial in self.trials if trial.session_id is not None
        )
        if len(session_ids) != len(set(session_ids)):
            raise ValueError("Trials must contain distinct concrete session IDs.")
        expected = _aggregate_eval_case(self.trials)
        if self.status != expected.status:
            raise ValueError("Case status does not match its retained trials.")
        if self.score != expected.score:
            raise ValueError("Case score does not match its retained trials.")
        if self.assertions != expected.assertions:
            raise ValueError("Case assertions do not match its retained trials.")
        if self.error != expected.error:
            raise ValueError("Case error does not match its retained trials.")
        if self.unavailable_reason != expected.unavailable_reason:
            raise ValueError("Case unavailable_reason does not match its retained trials.")
        return self


class EvalRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Type checkers require the literal token here rather than the exported
    # EVAL_SCHEMA_VERSION constant.
    schema_version: Literal[4] = EVAL_SCHEMA_VERSION
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    suite_id: str
    status: EvalStatus
    score: StrictFloat | None = Field(default=None, ge=0.0, le=1.0)
    # EvalSuite already requires at least one case. Keep the persisted result
    # equally fail-closed so an empty document can never claim a passing run.
    cases: tuple[EvalCaseResult, ...] = Field(min_length=1)
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

    @model_validator(mode="after")
    def validate_aggregate_contract(self) -> EvalRun:
        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at.")
        expected_duration = max(
            int((self.completed_at - self.started_at).total_seconds() * 1000),
            0,
        )
        if self.duration_ms != expected_duration:
            raise ValueError("duration_ms must match started_at and completed_at.")
        case_ids = tuple(case.case_id for case in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("EvalRun case IDs must be unique.")
        expected_status = aggregate_eval_status(case.status for case in self.cases)
        expected_score = aggregate_eval_score(case.score for case in self.cases)
        if self.status != expected_status:
            raise ValueError("Run status does not match its cases.")
        if self.score != expected_score:
            raise ValueError("Run score does not match its cases.")
        return self


def _status_from_assertions(assertions: Iterable[EvalAssertionResult]) -> EvalStatus:
    outcomes = tuple(assertion.outcome for assertion in assertions)
    if not outcomes:
        return EvalStatus.SKIPPED
    if EvalOutcome.ERROR in outcomes:
        return EvalStatus.ERROR
    if EvalOutcome.UNAVAILABLE in outcomes:
        return EvalStatus.UNAVAILABLE
    if EvalOutcome.FAILED in outcomes:
        return EvalStatus.FAILED
    return EvalStatus.PASSED


def aggregate_eval_status(statuses: Iterable[EvalStatus]) -> EvalStatus:
    """Apply the fail-closed result precedence shared by cases and runs."""

    values = tuple(statuses)
    if not values:
        return EvalStatus.SKIPPED
    for status in (
        EvalStatus.ERROR,
        EvalStatus.UNAVAILABLE,
        EvalStatus.FAILED,
        EvalStatus.SKIPPED,
    ):
        if status in values:
            return status
    return EvalStatus.PASSED


def aggregate_eval_score(scores: Iterable[float | None]) -> float | None:
    """Average a complete score set without dropping unscored results."""

    values = tuple(scores)
    if not values:
        raise ValueError("Cannot aggregate an empty eval score set.")
    if any(value is None for value in values):
        return None
    return sum(values) / len(values)


@dataclass(frozen=True)
class _AssertionAggregate:
    assertions: tuple[EvalAssertionResult, ...]
    error: str | None = None


@dataclass(frozen=True)
class _CaseAggregate:
    status: EvalStatus
    score: float | None
    assertions: tuple[EvalAssertionResult, ...]
    error: str | None
    unavailable_reason: str | None


def aggregate_eval_assertions(
    trials: tuple[EvalTrialResult, ...],
) -> tuple[EvalAssertionResult, ...]:
    """Reproduce case-level assertion aggregates from retained trials alone."""

    return _aggregate_eval_assertions(trials).assertions


def _aggregate_eval_assertions(
    trials: tuple[EvalTrialResult, ...],
) -> _AssertionAggregate:
    """Return a deterministic aggregate or an explicit contract diagnostic."""

    try:
        assertions = _aggregate_eval_assertions_strict(trials)
    except ValueError as exc:
        return _AssertionAggregate(
            assertions=(),
            error=f"Failed to aggregate trial assertions: {exc}",
        )
    return _AssertionAggregate(assertions=assertions)


def _aggregate_eval_assertions_strict(
    trials: tuple[EvalTrialResult, ...],
) -> tuple[EvalAssertionResult, ...]:
    """Aggregate compatible assertion contracts, raising on ambiguous projections."""

    if not trials:
        return ()
    if len(trials) == 1:
        return tuple(assertion.model_copy(deep=True) for assertion in trials[0].assertions)
    assertion_count = len(trials[0].assertions)
    if any(len(trial.assertions) != assertion_count for trial in trials):
        raise ValueError("Every trial must retain the same assertion count.")
    aggregated: list[EvalAssertionResult] = []
    for index in range(assertion_count):
        group = tuple(trial.assertions[index] for trial in trials)
        first = group[0]
        if any(assertion.name != first.name for assertion in group[1:]):
            raise ValueError("Every trial must retain the same ordered assertion contract.")
        scored_thresholds = {
            assertion.threshold for assertion in group if assertion.score is not None
        }
        if len(scored_thresholds) > 1:
            raise ValueError("Every scored trial must retain the same assertion threshold.")
        threshold = next(iter(scored_thresholds), first.threshold)
        outcomes = tuple(assertion.outcome for assertion in group)
        scores = tuple(assertion.score for assertion in group)
        metadata = {
            "trials": len(group),
            "trial_outcomes": [outcome.value for outcome in outcomes],
            "trial_scores": list(scores),
            "pass_count": sum(outcome == EvalOutcome.PASSED for outcome in outcomes),
        }
        if EvalOutcome.ERROR in outcomes:
            outcome = EvalOutcome.ERROR
            score = None
            message = f"error in {outcomes.count(EvalOutcome.ERROR)} of {len(group)} trials"
        elif EvalOutcome.UNAVAILABLE in outcomes:
            outcome = EvalOutcome.UNAVAILABLE
            score = None
            message = (
                f"unavailable in {outcomes.count(EvalOutcome.UNAVAILABLE)} of {len(group)} trials"
            )
        else:
            numeric_scores = tuple(score for score in scores if score is not None)
            score = sum(numeric_scores) / len(numeric_scores)
            bar = threshold if threshold is not None else 1.0
            outcome = EvalOutcome.PASSED if score >= bar else EvalOutcome.FAILED
            message = (
                f"mean score {score:.3f} over {len(group)} trials "
                f"({metadata['pass_count']}/{len(group)} passed)"
            )
        aggregated.append(
            EvalAssertionResult(
                name=first.name,
                outcome=outcome,
                score=score,
                threshold=threshold,
                message=message,
                metadata=metadata,
            )
        )
    return tuple(aggregated)


def _aggregate_eval_case(trials: tuple[EvalTrialResult, ...]) -> _CaseAggregate:
    """Derive every case-level result field from retained trials alone."""

    if not trials:
        raise ValueError("Cannot aggregate an empty eval trial set.")
    assertion_aggregate = _aggregate_eval_assertions(trials)
    trial_error = aggregate_trial_error(trials)
    if assertion_aggregate.error is not None:
        error = assertion_aggregate.error
        if trial_error is not None:
            error = f"{trial_error}; {error}"
        return _CaseAggregate(
            status=EvalStatus.ERROR,
            score=None,
            assertions=assertion_aggregate.assertions,
            error=error,
            unavailable_reason=None,
        )
    return _CaseAggregate(
        status=aggregate_eval_status(trial.status for trial in trials),
        score=aggregate_eval_score(trial.score for trial in trials),
        assertions=assertion_aggregate.assertions,
        error=trial_error,
        unavailable_reason=aggregate_trial_unavailable_reason(trials),
    )


def aggregate_trial_error(trials: tuple[EvalTrialResult, ...]) -> str | None:
    errored = tuple(trial for trial in trials if trial.status == EvalStatus.ERROR)
    if not errored:
        return None
    if len(trials) == 1:
        return errored[0].error
    prefix = (
        f"All {len(trials)} trials errored"
        if len(errored) == len(trials)
        else f"{len(errored)} of {len(trials)} trials errored"
    )
    return f"{prefix}; first: {errored[0].error}"


def aggregate_trial_unavailable_reason(
    trials: tuple[EvalTrialResult, ...],
) -> str | None:
    if any(trial.status == EvalStatus.ERROR for trial in trials):
        return None
    unavailable = tuple(trial for trial in trials if trial.status == EvalStatus.UNAVAILABLE)
    if not unavailable:
        return None
    if len(trials) == 1:
        return unavailable[0].unavailable_reason
    prefix = (
        f"All {len(trials)} trials unavailable"
        if len(unavailable) == len(trials)
        else f"{len(unavailable)} of {len(trials)} trials unavailable"
    )
    return f"{prefix}; first: {unavailable[0].unavailable_reason}"


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
    produces it (`EvalTrialResult.trajectory`), `write_trajectory_json` / `load_trajectory`
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


# EvalTrialResult.trajectory forward-references Trajectory, which is defined later in this module;
# rebuild now that it exists so Pydantic can resolve the annotation through the result graph.
EvalTrialResult.model_rebuild()
EvalCaseResult.model_rebuild()
