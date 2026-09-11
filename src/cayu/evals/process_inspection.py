"""Inspect and export native process-evaluation receipts without loading a target."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from cayu.build_provenance import RuntimeBuildProvenance
from cayu.evals._admission import LaunchScheduling, TrialAdmission
from cayu.evals._inspection_documents import ProcessDocuments, decode_document, read_regular_file
from cayu.evals.models import EvalRun, EvalStatus, aggregate_eval_score, aggregate_eval_status
from cayu.evals.session_inspection import EvalSessionInspectionV1, inspect_eval_sessions


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)


class EvalSessionReferenceV1(_Model):
    """An explicit local inspection locator, never execution or recovery authority."""

    session_id: str = Field(min_length=1, max_length=512)
    sqlite_path: str | None = Field(default=None, min_length=1, max_length=4096)

    @field_validator("sqlite_path")
    @classmethod
    def absolute_path(cls, value: str | None) -> str | None:
        if value is not None and not Path(value).is_absolute():
            raise ValueError("Session inspection paths must be absolute.")
        return value


class EvalProcessCaseInspectionV1(_Model):
    case_id: str
    worker_index: int
    observed_state: Literal["not_observed", "started", "finished", "interrupted"] = "not_observed"
    observed_trial_status: EvalStatus | None = None
    observed_error: str | None = Field(default=None, max_length=4096)
    observed_exception_type: str | None = Field(default=None, max_length=512)
    result_status: EvalStatus | None = None
    score: float | None = None
    error: str | None = None
    unavailable_reason: str | None = None
    assertion_failures: tuple[dict[str, Any], ...] = ()
    session: EvalSessionReferenceV1 | None = None
    session_inspection: EvalSessionInspectionV1 | None = None
    limitations: tuple[str, ...] = ()


class EvalProcessWorkerInspectionV1(_Model):
    index: int
    pid: int | None = None
    ready: bool = False
    assigned_case_ids: tuple[str, ...] = ()
    result_run_id: str | None = None
    progress_observed_at: datetime | None = None


class EvalProcessInspectionV1(_Model):
    schema_version: Literal[1] = 1
    launch_id: str
    observed_at: datetime
    phase: Literal["preparing", "admitted", "completed", "incomplete"]
    target: str
    started_at: datetime | None = None
    suite_id: str | None = None
    plan_fingerprint: str | None = None
    plan_metadata: dict[str, Any] = Field(default_factory=dict)
    max_concurrency: int
    launch_scheduling: LaunchScheduling | None = None
    case_timeout_seconds: float | None
    supervisor_pid: int | None = None
    python_version: str | None = None
    runtime_build_provenance: RuntimeBuildProvenance | None = None
    workers: tuple[EvalProcessWorkerInspectionV1, ...]
    cases: tuple[EvalProcessCaseInspectionV1, ...]
    result_status: EvalStatus | None = None
    score: float | None = None
    counts: dict[str, int]
    limitations: tuple[str, ...] = ()
    incomplete_exception_type: str | None = None
    owner_liveness: Literal["not_checked"] = "not_checked"
    automatic_replay: Literal[False] = False
    observation_note: str = (
        "Receipts and session reads are independent observations. Admission and PIDs do not "
        "prove a live owner. Trial observations are provisional until worker result admission. "
        "Only a validated completed marker and all admitted results establish run completion."
    )


class _Launch(_Model):
    schema_version: Literal[1, 2]
    launch_id: str = Field(min_length=1, max_length=512)
    target: str = Field(min_length=1, max_length=4096)
    processes: StrictInt = Field(ge=1, le=256)
    max_concurrency: StrictInt = Field(ge=1, le=100)
    stagger_seconds: float = Field(default=0, ge=0, allow_inf_nan=False)
    case_timeout_seconds: float | None
    startup_timeout_seconds: float
    shutdown_grace_seconds: float
    started_at: datetime | None = None
    supervisor_pid: StrictInt | None = Field(default=None, ge=1)
    automatic_replay: Literal[False] = False
    python_version: str | None = Field(default=None, max_length=64)
    runtime_build_provenance: RuntimeBuildProvenance | None = None

    @field_validator("case_timeout_seconds", "startup_timeout_seconds", "shutdown_grace_seconds")
    @classmethod
    def finite_timeout(cls, value: float | None) -> float | None:
        from math import isfinite

        if value is not None and (not isfinite(value) or value <= 0):
            raise ValueError("Process timeouts must be finite and positive.")
        return value

    @field_validator("started_at")
    @classmethod
    def aware_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("Process timestamps must be timezone-aware.")
        return value

    @model_validator(mode="after")
    def validate_worker_count(self) -> _Launch:
        if self.processes > self.max_concurrency:
            raise ValueError("Admitted worker count exceeds case concurrency.")
        if self.schema_version == 2 and (self.started_at is None or self.supervisor_pid is None):
            raise ValueError("Version 2 launches require supervisor identity and start time.")
        return self


class _Identity(_Model):
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    suite_id: str = Field(min_length=1, max_length=4096)
    case_ids: tuple[str, ...] = Field(min_length=1, max_length=10000)
    metadata: dict[str, Any]

    @field_validator("case_ids")
    @classmethod
    def distinct_cases(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value) or any(
            not item.strip() or len(item) > 4096 for item in value
        ):
            raise ValueError("Process case IDs must be bounded, nonblank, and unique.")
        return value


class _Ready(_Model):
    launch_id: str
    index: StrictInt = Field(ge=0, le=255)
    pid: StrictInt = Field(ge=1)
    identity: _Identity


class _Start(_Model):
    launch_id: str
    fingerprint: str
    assignments: tuple[tuple[str, ...], ...] = Field(max_length=256)


class _Terminal(_Model):
    launch_id: str
    exception_type: str | None = Field(default=None, max_length=512)
    automatic_replay: Literal[False] = False


class _Trial(_Model):
    case_id: str
    trial_number: Literal[1]
    state: Literal["started", "finished", "interrupted"]
    started_at: datetime
    completed_at: datetime | None = None
    status: EvalStatus | None = None
    score: float | None = None
    error: str | None = Field(default=None, max_length=4096)
    exception_type: str | None = Field(default=None, max_length=512)
    session: EvalSessionReferenceV1 | None = None

    @model_validator(mode="after")
    def validate_observation(self) -> _Trial:
        for timestamp in (self.started_at, self.completed_at):
            if timestamp is not None and (
                timestamp.tzinfo is None or timestamp.utcoffset() is None
            ):
                raise ValueError("Trial observation timestamps must be timezone-aware.")
        if self.state == "finished":
            if (
                self.status is None
                or self.completed_at is None
                or self.completed_at < self.started_at
            ):
                raise ValueError("Finished observations require status and ordered timestamps.")
        elif self.status is not None or self.completed_at is not None or self.score is not None:
            raise ValueError("Unfinished observations cannot claim a result.")
        if self.state == "interrupted" and self.exception_type is None:
            raise ValueError("Interrupted observations require an exception type.")
        return self


class _Progress(_Model):
    schema_version: Literal[1]
    launch_id: str
    index: StrictInt
    fingerprint: str
    observed_at: datetime
    trials: tuple[_Trial, ...] = Field(max_length=10000)

    @field_validator("observed_at")
    @classmethod
    def aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Progress timestamps must be timezone-aware.")
        return value


class _ExternalReference(EvalSessionReferenceV1):
    case_id: str


class _SessionEvidence(_Model):
    schema_version: Literal[1]
    launch_id: str
    cases: tuple[_ExternalReference, ...] = Field(max_length=10000)


def _same_json(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def _observe_documents(documents: ProcessDocuments) -> EvalProcessInspectionV1:
    launch = _Launch.model_validate(documents.read("launch.json", required=True))
    # Read terminal markers first. A marker appearing later cannot make a partial
    # read of earlier worker records appear to be a completed observation.
    completed = documents.read("completed.json")
    incomplete = documents.read("incomplete.json")
    if completed is not None and incomplete is not None:
        raise ValueError("Process receipts contain conflicting terminal markers.")
    terminal = None
    for value in (completed, incomplete):
        if value is not None:
            terminal = _Terminal.model_validate(value)
            if terminal.launch_id != launch.launch_id:
                raise ValueError("Process terminal marker belongs to a different launch.")
    if completed is not None and terminal is not None and terminal.exception_type is not None:
        raise ValueError("Completed process marker cannot contain an exception.")
    if incomplete is not None and terminal is not None and terminal.exception_type is None:
        raise ValueError("Incomplete process marker is missing its exception type.")
    admission = documents.read("start.json")
    start = None if admission is None else _Start.model_validate(admission)
    if start is not None and start.launch_id != launch.launch_id:
        raise ValueError("Process admission belongs to a different launch.")
    ready: dict[int, _Ready] = {}
    for index in range(launch.processes):
        value = documents.read(f"ready-{index}.json")
        if value is None:
            continue
        item = _Ready.model_validate(value)
        if item.launch_id != launch.launch_id or item.index != index:
            raise ValueError("Worker readiness belongs to a different launch or index.")
        ready[index] = item
    identity = next(iter(ready.values())).identity if ready else None
    limitations: set[str] = set()
    if identity is not None and any(
        not _same_json(item.identity.model_dump(mode="json"), identity.model_dump(mode="json"))
        for item in ready.values()
    ):
        if start is not None:
            raise ValueError("Admitted worker identities disagree.")
        limitations.add("worker_admission_identities_disagree")
        identity = None
    assignments: tuple[tuple[str, ...], ...] = tuple(() for _ in range(launch.processes))
    if start is not None:
        if identity is None or len(ready) != launch.processes:
            raise ValueError("Admission is missing exact worker readiness.")
        assignments = tuple(
            identity.case_ids[index :: launch.processes] for index in range(launch.processes)
        )
        if start.fingerprint != identity.fingerprint or start.assignments != assignments:
            raise ValueError("Process assignments do not match the admitted plan.")
        if len({item.pid for item in ready.values()}) != launch.processes:
            raise ValueError("Admitted worker PIDs are not distinct.")
    observed_admissions: list[TrialAdmission] = []
    workers: list[EvalProcessWorkerInspectionV1] = []
    cases: dict[str, EvalProcessCaseInspectionV1] = {}
    statuses: list[EvalStatus] = []
    scores: list[float | None] = []
    for index in range(launch.processes):
        expected = assignments[index]
        scheduling_document = documents.read(f"admissions-{index}.json")
        if scheduling_document is not None:
            scheduling = LaunchScheduling.model_validate(scheduling_document)
            if scheduling.stagger_seconds != launch.stagger_seconds or any(
                entry.case_id not in expected or entry.trial_number != 1
                for entry in scheduling.admissions
            ):
                raise ValueError("Worker scheduling evidence does not match its admission.")
            observed_admissions.extend(scheduling.admissions)
        item = ready.get(index)
        progress_document = documents.read(f"progress-{index}.json")
        progress = (
            None if progress_document is None else _Progress.model_validate(progress_document)
        )
        observations: dict[str, _Trial] = {}
        if progress is not None:
            if (
                start is None
                or progress.launch_id != launch.launch_id
                or progress.index != index
                or progress.fingerprint != start.fingerprint
            ):
                raise ValueError("Worker progress does not match its admission.")
            for trial in progress.trials:
                if trial.case_id not in expected or trial.case_id in observations:
                    raise ValueError("Worker progress contains an unassigned or duplicate case.")
                observations[trial.case_id] = trial
        result_document = documents.read(f"result-{index}.json")
        result = None if result_document is None else EvalRun.model_validate(result_document)
        result_cases = {}
        if result is not None:
            if (
                identity is None
                or start is None
                or not expected
                or tuple(case.case_id for case in result.cases) != expected
                or result.suite_id != identity.suite_id
                or not _same_json(result.metadata, identity.metadata)
                or result.run_contract is not None
                or any(
                    case.trial_policy.trial_count != 1
                    or case.trial_policy.max_concurrency != launch.max_concurrency
                    for case in result.cases
                )
            ):
                raise ValueError("Worker results do not match their admitted cases and plan.")
            result_cases = {case.case_id: case for case in result.cases}
        workers.append(
            EvalProcessWorkerInspectionV1(
                index=index,
                pid=None if item is None else item.pid,
                ready=item is not None,
                assigned_case_ids=expected,
                result_run_id=None if result is None else result.run_id,
                progress_observed_at=None if progress is None else progress.observed_at,
            )
        )
        for case_id in expected:
            observation = observations.get(case_id)
            case = result_cases.get(case_id)
            cases[case_id] = EvalProcessCaseInspectionV1(
                case_id=case_id,
                worker_index=index,
                observed_state="not_observed" if observation is None else observation.state,
                observed_trial_status=None if observation is None else observation.status,
                observed_error=None if observation is None else observation.error,
                observed_exception_type=None if observation is None else observation.exception_type,
                result_status=None if case is None else case.status,
                score=None if case is None else case.score,
                error=None if case is None or case.error is None else case.error[:4096],
                unavailable_reason=None
                if case is None or case.unavailable_reason is None
                else case.unavailable_reason[:4096],
                assertion_failures=()
                if case is None
                else tuple(
                    {
                        "name": assertion.name[:256],
                        "outcome": assertion.outcome.value,
                        "message": None if assertion.message is None else assertion.message[:4096],
                        "score": assertion.score,
                    }
                    for assertion in case.assertions
                    if assertion.outcome.value != "passed"
                )[:100],
                session=None if observation is None else observation.session,
                limitations=("trial_progress_unavailable",)
                if observation is None and case is None
                else (),
            )
            if case is not None:
                statuses.append(case.status)
                scores.append(case.score)
    if completed is not None and (start is None or len(statuses) != len(cases)):
        raise ValueError("Completed process receipt is missing admitted results.")
    phase = (
        "completed"
        if completed is not None
        else "incomplete"
        if incomplete is not None
        else "admitted"
        if start is not None
        else "preparing"
    )
    ordered = (
        tuple(cases[case_id] for case_id in identity.case_ids if case_id in cases)
        if identity is not None
        else ()
    )
    counts = Counter({status.value: 0 for status in EvalStatus})
    counts.update({"assigned": len(ordered), "results_recorded": len(statuses)})
    counts.update(status.value for status in statuses)
    counts["trials_observed_started"] = sum(
        case.observed_state != "not_observed" for case in ordered
    )
    counts["trials_observed_finished"] = sum(case.observed_state == "finished" for case in ordered)
    if launch.schema_version == 1:
        limitations.add("legacy_launch_has_no_supervisor_timestamp_or_automatic_session_links")
    return EvalProcessInspectionV1(
        launch_id=launch.launch_id,
        observed_at=datetime.now(UTC),
        phase=phase,
        target=launch.target,
        started_at=launch.started_at,
        suite_id=None if identity is None else identity.suite_id,
        plan_fingerprint=None if identity is None else identity.fingerprint,
        plan_metadata={} if identity is None else identity.metadata,
        max_concurrency=launch.max_concurrency,
        launch_scheduling=LaunchScheduling(
            stagger_seconds=launch.stagger_seconds,
            admissions=tuple(sorted(observed_admissions, key=lambda item: item.monotonic_seconds)),
        ),
        case_timeout_seconds=launch.case_timeout_seconds,
        supervisor_pid=launch.supervisor_pid,
        python_version=launch.python_version,
        runtime_build_provenance=launch.runtime_build_provenance,
        workers=tuple(workers),
        cases=ordered,
        result_status=aggregate_eval_status(statuses) if completed is not None else None,
        score=aggregate_eval_score(scores) if completed is not None else None,
        counts=dict(counts),
        limitations=tuple(sorted(limitations)),
        incomplete_exception_type=None
        if incomplete is None or terminal is None
        else terminal.exception_type,
    )


async def inspect_process_eval_run(
    directory: str | Path,
    *,
    include_sessions: bool = False,
    session_evidence: str | Path | None = None,
    max_sessions: int = 100,
    max_diagnostics: int = 20,
    case_id: str | None = None,
) -> EvalProcessInspectionV1:
    """Read exact native process receipts, optionally resolving explicit SQLite links.

    This never imports the target factory, starts work, tests remote PIDs, or
    infers application-specific evidence paths. Older launches can supply a
    launch- and case-bound session-evidence manifest explicitly.
    """
    if type(include_sessions) is not bool:
        raise TypeError("include_sessions must be a bool.")
    for value, name in ((max_sessions, "max_sessions"), (max_diagnostics, "max_diagnostics")):
        if type(value) is not int or not 1 <= value <= 1000:
            raise ValueError(f"{name} must be between 1 and 1000.")
    snapshot = _observe_documents(ProcessDocuments(Path(directory)))
    cases = list(snapshot.cases)
    if case_id is not None and not any(case.case_id == case_id for case in cases):
        raise ValueError("Requested case is not in the admitted process run.")
    if session_evidence is not None:
        data = read_regular_file(Path(session_evidence), max_bytes=8 * 1024 * 1024)
        if data is None:
            raise ValueError("Session-evidence manifest does not exist.")
        evidence = _SessionEvidence.model_validate(decode_document(data))
        if evidence.launch_id != snapshot.launch_id:
            raise ValueError("Session evidence belongs to a different evaluation launch.")
        by_id = {case.case_id: index for index, case in enumerate(cases)}
        seen: set[str] = set()
        for reference in evidence.cases:
            if reference.case_id not in by_id or reference.case_id in seen:
                raise ValueError("Session evidence contains an unassigned or duplicate case.")
            seen.add(reference.case_id)
            index = by_id[reference.case_id]
            binding = EvalSessionReferenceV1(
                session_id=reference.session_id, sqlite_path=reference.sqlite_path
            )
            if cases[index].session is not None and cases[index].session != binding:
                raise ValueError("External session evidence conflicts with the recorded trial.")
            cases[index] = cases[index].model_copy(update={"session": binding})
            cases[index] = cases[index].model_copy(
                update={
                    "limitations": (*cases[index].limitations, "operator_supplied_session_binding")
                }
            )
    if include_sessions:
        from cayu.runtime.public_authority import public_authority_alias_codec_from_environment
        from cayu.storage import SQLiteSessionStore
        from cayu.storage.migrations import SchemaMode

        remaining_sessions, remaining_diagnostics = max_sessions, max_diagnostics
        for index, case in enumerate(cases):
            if case_id is not None and case.case_id != case_id:
                continue
            reference = case.session
            limitations = list(case.limitations)
            observation = None
            if reference is None or reference.sqlite_path is None:
                limitations.append("portable_session_locator_unavailable")
            elif remaining_sessions <= 0:
                limitations.append("session_inspection_limit_reached")
            else:
                store = None
                try:
                    store = SQLiteSessionStore(
                        reference.sqlite_path,
                        schema_mode=SchemaMode.VALIDATE,
                        read_only=True,
                        public_authority_alias_codec=public_authority_alias_codec_from_environment(),
                    )
                    observation = await inspect_eval_sessions(
                        store,
                        reference.session_id,
                        max_sessions=remaining_sessions,
                        max_diagnostics=max(1, remaining_diagnostics),
                        now=snapshot.observed_at,
                    )
                    remaining_sessions -= max(1, len(observation.sessions))
                    if remaining_diagnostics == 0 and observation.diagnostics:
                        observation = observation.model_copy(
                            update={
                                "diagnostics": (),
                                "limitations": (*observation.limitations, "diagnostics_truncated"),
                            }
                        )
                    remaining_diagnostics -= len(observation.diagnostics)
                except Exception as exc:
                    remaining_sessions -= 1
                    limitations.append(f"session_inspection_unavailable:{type(exc).__name__}")
                finally:
                    if store is not None:
                        await store.close()
            cases[index] = case.model_copy(
                update={"session_inspection": observation, "limitations": tuple(limitations)}
            )
    return snapshot.model_copy(update={"cases": tuple(cases)})


def export_process_eval_run(directory: str | Path, output: str | Path) -> EvalProcessInspectionV1:
    """Export validated receipt/result bytes and hashes to a new private ZIP.

    Only fixed native receipt names are included. Logs, databases, attachments,
    pending writes, and application files are not swept into the archive.
    The retained observation can be incomplete; export does not promote it.
    """
    source, destination = Path(directory).resolve(), Path(output).absolute()
    if destination.resolve().is_relative_to(source):
        raise ValueError("Export output must be outside the process directory.")
    if destination.exists() or destination.is_symlink():
        raise ValueError("Export output must be a new file.")
    documents = ProcessDocuments(source)
    snapshot = _observe_documents(documents)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".cayu-eval-export-", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as stream:
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                manifest = {
                    "schema_version": 1,
                    "launch_id": snapshot.launch_id,
                    "phase": snapshot.phase,
                    "observed_at": snapshot.observed_at.isoformat(),
                    "files": [],
                    "scope": "native process receipts and worker results; no session-store snapshot",
                }
                for name, data in sorted(documents.files.items()):
                    archive.writestr(name, data)
                    manifest["files"].append(
                        {
                            "path": name,
                            "bytes": len(data),
                            "sha256": hashlib.sha256(data).hexdigest(),
                        }
                    )
                archive.writestr("inspection.json", snapshot.model_dump_json(indent=2))
                archive.writestr("export.json", json.dumps(manifest, sort_keys=True, indent=2))
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic, no-clobber publication also rejects a concurrently created target.
        os.link(temporary, destination)
        parent_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return snapshot
