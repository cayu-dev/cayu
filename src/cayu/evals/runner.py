from __future__ import annotations

import asyncio
import hashlib
import inspect
import traceback
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.artifacts import ArtifactMetadata
from cayu.core.events import Event, EventType
from cayu.core.messages import Message, MessageRole, TextPart
from cayu.evals.assertions import EvalAssertion, SessionStatusIs
from cayu.evals.models import (
    WORKSPACE_PROBE_MAX_BYTES,
    EvalAssertionResult,
    EvalCaseResult,
    EvalContext,
    EvalRun,
    EvalStatus,
    ProbeRequirements,
    Trajectory,
    TrajectoryProbes,
    WorkspaceFileProbe,
)
from cayu.runtime.app import CayuApp
from cayu.runtime.sessions import (
    RunRequest,
    Session,
    SessionQuery,
    SessionStatus,
    copy_run_request,
)
from cayu.runtime.usage import SessionUsageSummary

# Sessions per page when walking the sub-agent tree, and the max pages walked per node. The walk
# pages past the first 1000 children (rather than silently keeping only the first page) but stays
# bounded so a pathological fan-out can't stall the eval; hitting the cap sets children_incomplete.
_CHILD_TRAJECTORY_PAGE_SIZE = 1000
_CHILD_TRAJECTORY_MAX_PAGES = 100


class _IncompleteFlag:
    """Node-local, mutable "children were not fully enumerated" signal for the sub-agent walk."""

    __slots__ = ("value",)

    def __init__(self) -> None:
        self.value = False


def _format_exception(exc: BaseException) -> str:
    # Record the exception type name + traceback, not a bare str(exc): an empty-message error
    # (e.g. KeyError() or a re-raised cancellation) otherwise collapsed to a blank, untraceable
    # eval error string. format_exception's final line already carries "TypeName: message".
    formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
    return formatted or f"{type(exc).__name__}: {exc}"


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    id: str
    request: RunRequest
    assertions: list[EvalAssertion] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("request")
    @classmethod
    def copy_request(cls, value: RunRequest) -> RunRequest:
        return copy_run_request(value)

    @field_validator("assertions", mode="before")
    @classmethod
    def validate_assertions(cls, value) -> list[EvalAssertion]:
        if value is None:
            return []
        if isinstance(value, EvalAssertion):
            return [value]
        if isinstance(value, str | bytes):
            raise TypeError("EvalCase assertions must be EvalAssertion instances.")
        try:
            assertions = list(value)
        except TypeError as exc:
            raise TypeError("EvalCase assertions must be an iterable.") from exc
        for assertion in assertions:
            if not isinstance(assertion, EvalAssertion):
                raise TypeError("EvalCase assertions must contain EvalAssertion instances.")
        return assertions

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")


class EvalSuite(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    id: str
    cases: list[EvalCase]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("cases", mode="before")
    @classmethod
    def validate_cases(cls, value) -> list[EvalCase]:
        if isinstance(value, EvalCase):
            return [value]
        if isinstance(value, str | bytes):
            raise TypeError("EvalSuite cases must be EvalCase instances.")
        try:
            cases = list(value)
        except TypeError as exc:
            raise TypeError("EvalSuite cases must be an iterable.") from exc
        if not cases:
            raise ValueError("EvalSuite requires at least one case.")
        normalized = [
            case if type(case) is EvalCase else EvalCase.model_validate(case) for case in cases
        ]
        # Reject duplicate IDs at the root: compare_eval_runs indexes cases by id, so a
        # duplicate would run but be silently dropped from every baseline comparison.
        counts = Counter(case.id for case in normalized)
        duplicates = sorted(cid for cid, n in counts.items() if n > 1)
        if duplicates:
            raise ValueError(
                f"EvalSuite case IDs must be unique; duplicated: {', '.join(duplicates)}."
            )
        return normalized

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")


@dataclass(frozen=True)
class EvalPlan:
    app: CayuApp
    suite: EvalSuite


async def run_eval_suite(
    app: CayuApp,
    suite: EvalSuite,
    *,
    retain_trajectory: bool = False,
    max_concurrency: int = 1,
    case_timeout_seconds: float | None = None,
    trials: int = 1,
) -> EvalRun:
    """Run every case in the suite and aggregate the results.

    `max_concurrency` runs up to that many cases at once (default 1 = sequential).
    Results always keep suite order. Note: `ScriptedModelProvider` consumes batches
    by positional request index, so with concurrency > 1 interleaved cases may pull
    each other's batches — keep the default for scripted multi-case suites.

    `case_timeout_seconds` bounds each case's run; a case that exceeds it is
    cancelled and recorded as `EvalStatus.ERROR` instead of stalling the suite.

    `trials` runs each case N times and reports the mean per-assertion score, so a
    stochastic model whose score wobbles (e.g. 0.83 -> 0.82) settles to a stable
    average instead of flipping a baseline comparison on every run.
    """
    if not isinstance(app, CayuApp):
        raise TypeError("run_eval_suite requires a CayuApp.")
    if type(suite) is not EvalSuite:
        raise TypeError("run_eval_suite requires an EvalSuite.")
    if type(max_concurrency) is not int:
        raise TypeError("run_eval_suite max_concurrency must be an int.")
    if max_concurrency < 1:
        raise ValueError("run_eval_suite max_concurrency must be >= 1.")
    _validate_trials(trials, "run_eval_suite trials")
    _validate_timeout_seconds(case_timeout_seconds, "run_eval_suite case_timeout_seconds")

    run_id = str(uuid4())
    started_at = datetime.now(UTC)
    results = await _run_suite_cases(
        app,
        suite,
        retain_trajectory=retain_trajectory,
        max_concurrency=max_concurrency,
        case_timeout_seconds=case_timeout_seconds,
        trials=trials,
    )
    completed_at = datetime.now(UTC)
    status = _run_status(results)
    score = _average_score(result.score for result in results)
    return EvalRun(
        run_id=run_id,
        suite_id=suite.id,
        status=status,
        score=score,
        cases=tuple(results),
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=_duration_ms(started_at, completed_at),
        metadata=suite.metadata,
    )


async def run_eval_plan(
    plan: EvalPlan,
    *,
    retain_trajectory: bool = False,
    max_concurrency: int = 1,
    case_timeout_seconds: float | None = None,
    trials: int = 1,
) -> EvalRun:
    if type(plan) is not EvalPlan:
        raise TypeError("run_eval_plan requires an EvalPlan.")
    return await run_eval_suite(
        plan.app,
        plan.suite,
        retain_trajectory=retain_trajectory,
        max_concurrency=max_concurrency,
        case_timeout_seconds=case_timeout_seconds,
        trials=trials,
    )


async def _run_suite_cases(
    app: CayuApp,
    suite: EvalSuite,
    *,
    retain_trajectory: bool,
    max_concurrency: int,
    case_timeout_seconds: float | None,
    trials: int,
) -> list[EvalCaseResult]:
    if max_concurrency == 1:
        return [
            await run_eval_case(
                app,
                case,
                suite_id=suite.id,
                retain_trajectory=retain_trajectory,
                timeout_seconds=case_timeout_seconds,
                trials=trials,
            )
            for case in suite.cases
        ]

    # Fill a positional slot per case so results keep suite order regardless of
    # completion order. run_eval_case never raises for a failed run (it records an
    # ERROR result), so a TaskGroup abort only happens on a genuine programming error.
    slots: list[EvalCaseResult | None] = [None] * len(suite.cases)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _run_slot(index: int, case: EvalCase) -> None:
        async with semaphore:
            slots[index] = await run_eval_case(
                app,
                case,
                suite_id=suite.id,
                retain_trajectory=retain_trajectory,
                timeout_seconds=case_timeout_seconds,
                trials=trials,
            )

    async with asyncio.TaskGroup() as group:
        for index, case in enumerate(suite.cases):
            group.create_task(_run_slot(index, case))
    return [result for result in slots if result is not None]


async def run_eval_case(
    app: CayuApp,
    case: EvalCase,
    *,
    suite_id: str,
    retain_trajectory: bool = False,
    timeout_seconds: float | None = None,
    trials: int = 1,
) -> EvalCaseResult:
    """Run one case, optionally `trials` times, aggregating to the mean per-assertion score.

    `trials=1` (the default) returns the single run unchanged. `trials>1` runs the case N
    times and reports each assertion's mean score across trials, so a stochastic model whose
    score wobbles run-to-run settles to a stable average instead of a coin-flip pass/fail.
    """
    _validate_trials(trials, "run_eval_case trials")
    _validate_timeout_seconds(timeout_seconds, "run_eval_case timeout_seconds")
    if trials == 1:
        return await _run_case_once(
            app,
            case,
            suite_id=suite_id,
            retain_trajectory=retain_trajectory,
            timeout_seconds=timeout_seconds,
        )
    started_at = datetime.now(UTC)
    trial_results = [
        await _run_case_once(
            app,
            case,
            suite_id=suite_id,
            retain_trajectory=retain_trajectory,
            timeout_seconds=timeout_seconds,
        )
        for _ in range(trials)
    ]
    completed_at = datetime.now(UTC)
    return _aggregate_trials(
        case,
        trial_results,
        started_at=started_at,
        completed_at=completed_at,
        retain_trajectory=retain_trajectory,
    )


async def _run_case_once(
    app: CayuApp,
    case: EvalCase,
    *,
    suite_id: str,
    retain_trajectory: bool = False,
    timeout_seconds: float | None = None,
) -> EvalCaseResult:
    started_at = datetime.now(UTC)
    emitted_events: list[Event] = []
    session_id: str | None = None
    run_error: str | None = None

    try:
        # asyncio.timeout(None) never expires, so the unbounded default shares the path.
        async with asyncio.timeout(timeout_seconds) as deadline:
            async for event in app.run(case.request):
                emitted_events.append(event)
                if session_id is None:
                    session_id = event.session_id
    except TimeoutError as exc:
        if deadline.expired():
            run_error = f"Eval case timed out after {timeout_seconds} seconds."
        else:
            run_error = _format_exception(exc)
    except Exception as exc:
        run_error = _format_exception(exc)

    if session_id is None:
        session_id = case.request.session_id

    session: Session | None = None
    events: tuple[Event, ...] = tuple(emitted_events)
    transcript: tuple[Message, ...] = ()
    usage_summary: SessionUsageSummary | None = None
    if session_id is not None:
        try:
            session = await app.session_store.load(session_id)
            events, transcript, usage_summary = await _load_session_records(app, session_id)
        except Exception as exc:
            if run_error is None:
                run_error = f"Failed to load eval session state: {_format_exception(exc)}"

    # app.run() does not raise on a model/tool failure; it ends the session as
    # SESSION_FAILED and returns normally. Surface that as an eval ERROR so a
    # crashed run is never scored as PASSED — unless the case explicitly asserts
    # on session status, in which case the assertion owns the outcome.
    if (
        run_error is None
        and session is not None
        and session.status == SessionStatus.FAILED
        and not any(isinstance(assertion, SessionStatusIs) for assertion in case.assertions)
    ):
        run_error = _session_failure_reason(events)

    final_output = final_output_text(transcript)
    probe_requirements = _collect_probe_requirements(case.assertions)
    probes = await _capture_probes(app, session, probe_requirements)
    children_incomplete = _IncompleteFlag()
    children = await _build_child_trajectories(
        app,
        session_id,
        visited={session_id} if session_id is not None else set(),
        incomplete=children_incomplete,
    )
    trajectory = Trajectory(
        session=session,
        events=events,
        transcript=transcript,
        usage_summary=usage_summary,
        final_output=final_output,
        probes=probes,
        children=children,
        children_incomplete=children_incomplete.value,
        metadata=case.metadata,
    )
    context = EvalContext(
        trajectory=trajectory,
        suite_id=suite_id,
        case_id=case.id,
        metadata=case.metadata,
    )
    assertion_results = list(await _evaluate_assertions(case.assertions, context))
    completed_at = datetime.now(UTC)
    status = _case_status(run_error, assertion_results)
    return EvalCaseResult(
        case_id=case.id,
        status=status,
        session_id=session_id,
        score=_case_score(status, assertion_results),
        final_output=final_output,
        assertions=tuple(assertion_results),
        error=run_error,
        events_count=len(events),
        usage_summary=usage_summary.model_dump(mode="json") if usage_summary is not None else None,
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=_duration_ms(started_at, completed_at),
        metadata=case.metadata,
        # The probe-complete trajectory captured during this run, for export/replay. Opt-in so
        # the default run doesn't retain every case's trajectory (incl. file bytes) in memory.
        trajectory=trajectory if retain_trajectory else None,
    )


async def evaluate_assertions(
    trajectory: Trajectory,
    assertions: Iterable[EvalAssertion],
    *,
    suite_id: str = "replay",
    case_id: str = "replay",
) -> tuple[EvalAssertionResult, ...]:
    """Run assertions against a Trajectory — the replay entry point.

    Wraps the trajectory in an `EvalContext` and evaluates the assertions, returning their
    results. Pair with `load_trajectory` to re-check a saved run without a live runtime:
    ``evaluate_assertions(load_trajectory(path), assertions)``.
    """
    if type(trajectory) is not Trajectory:
        raise TypeError("evaluate_assertions requires a Trajectory.")
    context = EvalContext(
        trajectory=trajectory,
        suite_id=suite_id,
        case_id=case_id,
        metadata=dict(trajectory.metadata),
    )
    return await _evaluate_assertions(tuple(assertions), context)


async def _evaluate_assertions(
    assertions: Sequence[EvalAssertion],
    context: EvalContext,
) -> tuple[EvalAssertionResult, ...]:
    results: list[EvalAssertionResult] = []
    for assertion in assertions:
        try:
            result = assertion.evaluate(context)
            if inspect.isawaitable(result):
                result = await result
            if type(result) is not EvalAssertionResult:
                raise TypeError("EvalAssertion.evaluate must return EvalAssertionResult.")
            results.append(result)
        except Exception as exc:
            results.append(
                EvalAssertionResult(
                    name=assertion.name,
                    passed=False,
                    message=f"Assertion raised {type(exc).__name__}: {exc}",
                    metadata={"error_type": type(exc).__name__},
                )
            )
    return tuple(results)


async def _load_session_records(
    app: CayuApp, session_id: str
) -> tuple[tuple[Event, ...], tuple[Message, ...], SessionUsageSummary]:
    events = tuple(await app.session_store.load_events(session_id))
    transcript = tuple(await app.session_store.load_transcript(session_id))
    usage_summary = await app.get_session_usage(session_id)
    return events, transcript, usage_summary


def _collect_probe_requirements(assertions: Sequence[EvalAssertion]) -> ProbeRequirements:
    requirements = ProbeRequirements()
    for assertion in assertions:
        requirements = requirements.merged_with(assertion.required_probes())
    return requirements


async def _capture_probes(
    app: CayuApp,
    session: Session | None,
    requirements: ProbeRequirements,
) -> TrajectoryProbes:
    # Snapshot exactly what the case's assertions declared, while the environment is still
    # live, so assertions evaluate against the serializable trajectory instead of the app.
    if session is None or not (requirements.workspace_paths or requirements.artifact_scopes):
        return TrajectoryProbes()
    try:
        environment = app.get_environment(session.environment_name)
    except Exception:
        environment = None
    if environment is None:
        return TrajectoryProbes()
    env = environment.environment

    workspace_available = False
    workspace_files: dict[str, bytes | None] = {}
    workspace_file_stats: dict[str, WorkspaceFileProbe] = {}
    workspace = getattr(env, "workspace", None)
    if requirements.workspace_paths and workspace is not None:
        workspace_available = True
        for path in sorted(requirements.workspace_paths):
            try:
                # Cap the read so an oversized workspace file can't balloon the trajectory JSON
                # (bytes are base64-encoded there). The full size + a content hash are recorded
                # alongside so a truncated capture is still identifiable and stat-able.
                result = await workspace.read_bytes(path, max_bytes=WORKSPACE_PROBE_MAX_BYTES)
                content = result.content
                workspace_files[path] = content
                total_bytes = getattr(result, "total_bytes", len(content))
                truncated = getattr(result, "truncated", len(content) < total_bytes)
                workspace_file_stats[path] = WorkspaceFileProbe(
                    total_bytes=total_bytes,
                    truncated=truncated,
                    sha256=hashlib.sha256(content).hexdigest(),
                )
            except Exception:
                # Absent/unreadable: record the path as probed-but-missing (None), distinct
                # from "never probed" (key absent), which the assertion reports differently.
                workspace_files[path] = None

    artifacts_available = False
    artifacts: list[ArtifactMetadata] = []
    artifact_store = getattr(env, "artifact_store", None)
    if requirements.artifact_scopes and artifact_store is not None:
        artifacts_available = True
        seen_ids: set[str] = set()
        for scope in requirements.artifact_scopes:
            try:
                listed = await artifact_store.list(scope=scope)
            except Exception:
                # Degrade like the per-path workspace reads: a store error fails the
                # affected assertion (it sees no artifacts), never crashes the eval case.
                continue
            for artifact in listed.artifacts:
                if artifact.id not in seen_ids:
                    seen_ids.add(artifact.id)
                    artifacts.append(artifact)

    return TrajectoryProbes(
        workspace_available=workspace_available,
        workspace_files=workspace_files,
        workspace_file_stats=workspace_file_stats,
        artifacts_available=artifacts_available,
        artifacts=tuple(artifacts),
    )


async def _build_child_trajectories(
    app: CayuApp,
    parent_session_id: str | None,
    *,
    visited: set[str],
    incomplete: _IncompleteFlag | None = None,
) -> tuple[Trajectory, ...]:
    # Walk the sub-agent tree by parent linkage so a run's sub-agents are captured in the
    # trajectory. `visited` guards against cycles / re-visiting a shared id. The walk pages
    # through the keyset cursor so a parent with more than one page of children is captured in
    # full; a store error mid-walk or hitting the page cap sets `incomplete` (surfaced on the
    # node's Trajectory.children_incomplete) instead of being silently swallowed.
    if parent_session_id is None:
        return ()
    children: list[Trajectory] = []
    cursor: str | None = None
    for _ in range(_CHILD_TRAJECTORY_MAX_PAGES):
        try:
            result = await app.session_store.list_sessions(
                SessionQuery(
                    parent_session_id=parent_session_id,
                    limit=_CHILD_TRAJECTORY_PAGE_SIZE,
                    cursor=cursor,
                )
            )
        except Exception:
            if incomplete is not None:
                incomplete.value = True
            return tuple(children)
        for child_session in result.sessions:
            if child_session.id in visited:
                continue
            visited.add(child_session.id)
            child = await _load_child_trajectory(app, child_session, visited=visited)
            if child is not None:
                children.append(child)
        cursor = result.next_cursor
        if cursor is None:
            return tuple(children)
    # Fell out of the page loop with a cursor still pending: more children remain than the cap
    # allows us to walk. Mark the capture partial rather than dropping the rest silently.
    if incomplete is not None:
        incomplete.value = True
    return tuple(children)


async def _load_child_trajectory(
    app: CayuApp,
    session: Session,
    *,
    visited: set[str],
) -> Trajectory | None:
    try:
        events, transcript, usage_summary = await _load_session_records(app, session.id)
    except Exception:
        return None
    # Sub-agent nodes are captured for visibility/serialization; assertions in v1 evaluate
    # against the root case only, so child nodes carry no probe snapshot.
    grandchildren_incomplete = _IncompleteFlag()
    grandchildren = await _build_child_trajectories(
        app, session.id, visited=visited, incomplete=grandchildren_incomplete
    )
    return Trajectory(
        session=session,
        events=events,
        transcript=transcript,
        usage_summary=usage_summary,
        final_output=final_output_text(transcript),
        children=grandchildren,
        children_incomplete=grandchildren_incomplete.value,
    )


def _validate_trials(value: int, field_name: str) -> None:
    # bool is an int subclass; reject it so trials=True can't silently mean 1 trial.
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an int.")
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1.")


def _aggregate_trials(
    case: EvalCase,
    results: list[EvalCaseResult],
    *,
    started_at: datetime,
    completed_at: datetime,
    retain_trajectory: bool,
) -> EvalCaseResult:
    # Collapse N trial runs of one case into a single result whose per-assertion score is the
    # mean across trials. Any errored trial keeps the aggregate in ERROR so execution failures
    # cannot disappear behind assertion-list mismatch handling.
    n = len(results)
    errored = [result for result in results if result.status == EvalStatus.ERROR]
    all_error = all(result.status == EvalStatus.ERROR for result in results)
    combined_error: str | None = None
    if errored:
        errors = [result.error for result in errored if result.error]
        prefix = f"All {n} trials errored" if all_error else f"{len(errored)} of {n} trials errored"
        combined_error = f"{prefix}; first: {errors[0]}" if errors else f"{prefix}."
    aggregated = [] if errored else _aggregate_trial_assertions(results, n)
    status = _case_status(combined_error, aggregated)
    score = _case_score(status, aggregated)
    # The last trial supplies the representative session/output/trajectory for export.
    last = results[-1]
    metadata = {
        **case.metadata,
        "trials": n,
        "trial_scores": [result.score for result in results],
        "trial_statuses": [result.status.value for result in results],
    }
    return EvalCaseResult(
        case_id=case.id,
        status=status,
        session_id=last.session_id,
        score=score,
        final_output=last.final_output,
        assertions=tuple(aggregated),
        error=combined_error,
        events_count=last.events_count,
        usage_summary=last.usage_summary,
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=_duration_ms(started_at, completed_at),
        metadata=metadata,
        trajectory=last.trajectory if retain_trajectory else None,
    )


def _aggregate_trial_assertions(
    results: list[EvalCaseResult], trials: int
) -> list[EvalAssertionResult]:
    assertion_lists = [result.assertions for result in results]
    first = assertion_lists[0]
    # Same case, same assertion list, so every trial yields the same assertions in order. If a
    # trial diverges (defensive), skip aggregation so the case reports SKIPPED rather than
    # zipping mismatched assertions together.
    if not first or not all(len(a) == len(first) for a in assertion_lists):
        return []
    aggregated: list[EvalAssertionResult] = []
    for index in range(len(first)):
        group = [assertions[index] for assertions in assertion_lists]
        mean_score = sum(assertion.score for assertion in group) / trials
        threshold = group[0].threshold
        bar = threshold if threshold is not None else 1.0
        pass_count = sum(1 for assertion in group if assertion.passed)
        aggregated.append(
            EvalAssertionResult(
                name=group[0].name,
                score=mean_score,
                threshold=threshold,
                passed=mean_score >= bar,
                message=f"mean score {mean_score:.3f} over {trials} trials ({pass_count}/{trials} passed)",
                metadata={
                    "trials": trials,
                    "trial_scores": [assertion.score for assertion in group],
                    "pass_count": pass_count,
                },
            )
        )
    return aggregated


def _validate_timeout_seconds(value: float | None, field_name: str) -> None:
    if value is None:
        return
    if type(value) not in (int, float):
        raise TypeError(f"{field_name} must be a number or None.")
    if value != value or value <= 0:  # rejects NaN and non-positive values
        raise ValueError(f"{field_name} must be a positive number.")


def _session_failure_reason(events: Iterable[Event]) -> str:
    for event in events:
        if event.type == EventType.SESSION_FAILED:
            error = event.payload.get("error")
            if isinstance(error, str) and error.strip():
                return f"Session failed: {error}"
            return "Session failed."
    return "Session ended in a failed state."


def final_output_text(transcript: Iterable[Message]) -> str:
    """Return the text of the last assistant message in ``transcript``.

    Walks the transcript backwards and returns the concatenated text of the most
    recent assistant message that produced any text (``""`` if none did — e.g. the
    run ended on a tool call). Use this to pull an agent's final answer out of a
    loaded session transcript. ``run_to_completion`` instead reports the last
    completed model turn observed in the live event stream, which can be empty
    even when an earlier assistant transcript message had text.
    """
    for message in reversed(tuple(transcript)):
        if message.role != MessageRole.ASSISTANT:
            continue
        text_parts = [part.text for part in message.content if type(part) is TextPart]
        text = "".join(text_parts)
        if text:
            return text
    return ""


def _case_status(
    run_error: str | None,
    assertions: Sequence[EvalAssertionResult],
) -> EvalStatus:
    if run_error is not None:
        return EvalStatus.ERROR
    # A case that asserts nothing used to pass at score 1.0 — a fail-open default that let an
    # unfinished case masquerade as green. Record it as SKIPPED so it is visible, not counted.
    if not assertions:
        return EvalStatus.SKIPPED
    if all(assertion.passed for assertion in assertions):
        return EvalStatus.PASSED
    return EvalStatus.FAILED


def _case_score(status: EvalStatus, assertions: Sequence[EvalAssertionResult]) -> float:
    if status in (EvalStatus.ERROR, EvalStatus.SKIPPED):
        return 0.0
    if not assertions:
        return 0.0
    return sum(assertion.score for assertion in assertions) / len(assertions)


def _run_status(results: list[EvalCaseResult]) -> EvalStatus:
    if any(result.status == EvalStatus.ERROR for result in results):
        return EvalStatus.ERROR
    if any(result.status == EvalStatus.FAILED for result in results):
        return EvalStatus.FAILED
    # A skipped case (zero assertions) is not a pass; surface it at the run level so the
    # suite is not reported green while a case asserted nothing.
    if any(result.status == EvalStatus.SKIPPED for result in results):
        return EvalStatus.SKIPPED
    return EvalStatus.PASSED


def _average_score(scores: Iterable[float]) -> float:
    values = tuple(scores)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _duration_ms(started_at: datetime, completed_at: datetime) -> int:
    return max(int((completed_at - started_at).total_seconds() * 1000), 0)
