from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
import traceback
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import copy_durable_json_object, require_durable_clean_nonblank
from cayu.artifacts import ArtifactMetadata, ArtifactScope
from cayu.core.events import Event, EventType, event_durable_sequence
from cayu.core.messages import Message
from cayu.evals.assertions import EvalAssertion
from cayu.evals.models import (
    WORKSPACE_PROBE_MAX_BYTES,
    EvalAssertionResult,
    EvalCaseResult,
    EvalContext,
    EvalOutcome,
    EvalRun,
    EvalStatus,
    EvalTrialResult,
    ProbeRequirements,
    Trajectory,
    TrajectoryProbes,
    WorkspaceFileProbe,
    _model_instance_python_input,
    _validate_trajectory_record_contract,
    aggregate_eval_score,
    aggregate_eval_status,
)
from cayu.evals.result_contract import (
    EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES,
    EvalTrialDiagnosticCode,
    EvalTrialOutputPreviewV1,
    _EvalTrialPublicData,
)
from cayu.evals.trajectory import (
    SessionTrajectoryBounds,
    _build_child_trajectories,
    _CaptureState,
    _IncompleteFlag,
    _load_terminal_evidence,
    _project_terminal_evidence,
    _trajectory_from_terminal_evidence,
    final_output_text,
)
from cayu.runtime.app import CayuApp
from cayu.runtime.sessions import (
    TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_EVENTS,
    RunnerObservedEventIdentity,
    RunRequest,
    Session,
    SessionStatus,
    TerminalSessionEvidence,
    TerminalSessionEvidenceError,
    TerminalSessionEvidenceErrorCode,
    copy_run_request,
)
from cayu.runtime.usage import (
    SessionUsageSummary,
    session_usage_summary_payload,
)

if TYPE_CHECKING:
    from cayu.evals.corpus import EvalCorpusDocument
    from cayu.evals.evidence import AssertionEvidenceView
    from cayu.evals.execution import CorpusExecutionResult, CorpusTarget


class _FreshInterruptedEvidenceUnavailable(RuntimeError):
    """The runner-owned interrupted session could not be reconciled exactly."""


def _format_exception_summary(exc: BaseException) -> str:
    """Render an exception without trusting extension-owned ``__str__`` methods."""

    try:
        detail = str(exc)
    except Exception:
        detail = "<exception str() failed>"
    exception_type = type(exc).__name__
    return f"{exception_type}: {detail}" if detail else exception_type


def _format_exception(exc: BaseException) -> str:
    # Record the exception type name + traceback, not a bare str(exc): an empty-message error
    # (e.g. KeyError() or a re-raised cancellation) otherwise collapsed to a blank, untraceable
    # eval error string. format_exception's final line already carries "TypeName: message".
    try:
        formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
    except Exception:
        return _format_exception_summary(exc)
    return formatted or _format_exception_summary(exc)


class EvalCase(BaseModel):
    model_config = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True, hide_input_in_errors=True
    )

    id: str
    request: RunRequest
    assertions: list[EvalAssertion] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("request")
    @classmethod
    def copy_request(cls, value: RunRequest) -> RunRequest:
        request = copy_run_request(value)
        if request.task_id is not None:
            raise ValueError(
                "EvalCase request cannot set task_id or task_worker_id; eval trials "
                "create independent sessions and cannot share durable task identity."
            )
        return request

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
        return copy_durable_json_object(value, "metadata")


def _detach_eval_case(case: EvalCase) -> EvalCase:
    """Rebuild one case while preserving request-private runtime provenance."""

    return EvalCase(
        id=case.id,
        request=case.request,
        assertions=case.assertions,
        metadata=case.metadata,
    )


class EvalSuite(BaseModel):
    model_config = ConfigDict(
        extra="forbid", arbitrary_types_allowed=True, hide_input_in_errors=True
    )

    id: str
    cases: list[EvalCase]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("cases", mode="before")
    @classmethod
    def validate_cases(cls, value) -> list[EvalCase]:
        if isinstance(value, EvalCase):
            return [_detach_eval_case(value)]
        if isinstance(value, str | bytes):
            raise TypeError("EvalSuite cases must be EvalCase instances.")
        try:
            cases = list(value)
        except TypeError as exc:
            raise TypeError("EvalSuite cases must be an iterable.") from exc
        if not cases:
            raise ValueError("EvalSuite requires at least one case.")
        normalized = [
            _detach_eval_case(case) if isinstance(case, EvalCase) else EvalCase.model_validate(case)
            for case in cases
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
        return copy_durable_json_object(value, "metadata")


def _detach_eval_suite(suite: EvalSuite) -> EvalSuite:
    return EvalSuite(id=suite.id, cases=suite.cases, metadata=suite.metadata)


@dataclass(frozen=True)
class EvalPlan:
    """Trusted project eval entry point in exactly one execution mode."""

    app: CayuApp | None = None
    suite: EvalSuite | None = None
    corpus_target: CorpusTarget | None = None

    def __post_init__(self) -> None:
        from cayu.evals.execution import CorpusTarget

        direct_configured = self.app is not None or self.suite is not None
        corpus_configured = self.corpus_target is not None
        if direct_configured == corpus_configured:
            raise ValueError(
                "EvalPlan requires exactly one mode: app with suite, or corpus_target."
            )
        if direct_configured:
            if not isinstance(self.app, CayuApp):
                raise TypeError("EvalPlan app must be a CayuApp.")
            if type(self.suite) is not EvalSuite:
                raise TypeError("EvalPlan suite must be an exact EvalSuite.")
            object.__setattr__(
                self,
                "suite",
                _detach_eval_suite(self.suite),
            )
            return
        if type(self.corpus_target) is not CorpusTarget:
            raise TypeError("EvalPlan corpus_target must be an exact CorpusTarget.")


async def run_eval_suite(
    app: CayuApp,
    suite: EvalSuite,
    *,
    retain_trajectory: bool = False,
    retain_final_output: bool = True,
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

    `retain_final_output=False` discards raw trial output after assertions have
    evaluated. It cannot be combined with trajectory retention because a retained
    trajectory must remain lossless.
    """
    run, _ = await _run_eval_suite(
        app,
        suite,
        retain_trajectory=retain_trajectory,
        retain_final_output=retain_final_output,
        max_concurrency=max_concurrency,
        case_timeout_seconds=case_timeout_seconds,
        trials=trials,
        public_output_preview_bytes=None,
    )
    return run


async def _run_eval_suite_with_public_projection(
    app: CayuApp,
    suite: EvalSuite,
    *,
    max_concurrency: int,
    case_timeout_seconds: float | None,
    trials: int,
    output_preview_bytes: int,
) -> tuple[EvalRun, dict[str, tuple[_EvalTrialPublicData, ...]]]:
    """Run a corpus suite and return its separate runner-owned public sidecar."""

    run, public_data = await _run_eval_suite(
        app,
        suite,
        retain_trajectory=False,
        retain_final_output=False,
        max_concurrency=max_concurrency,
        case_timeout_seconds=case_timeout_seconds,
        trials=trials,
        public_output_preview_bytes=output_preview_bytes,
    )
    if public_data is None:
        raise RuntimeError("Corpus execution lost its runner-owned public projection.")
    return run, public_data


async def _run_eval_suite(
    app: CayuApp,
    suite: EvalSuite,
    *,
    retain_trajectory: bool,
    retain_final_output: bool,
    max_concurrency: int,
    case_timeout_seconds: float | None,
    trials: int,
    public_output_preview_bytes: int | None,
) -> tuple[EvalRun, dict[str, tuple[_EvalTrialPublicData, ...]] | None]:
    if not isinstance(app, CayuApp):
        raise TypeError("run_eval_suite requires a CayuApp.")
    if type(suite) is not EvalSuite:
        raise TypeError("run_eval_suite requires an EvalSuite.")
    suite = _detach_eval_suite(suite)
    if type(max_concurrency) is not int:
        raise TypeError("run_eval_suite max_concurrency must be an int.")
    if max_concurrency < 1:
        raise ValueError("run_eval_suite max_concurrency must be >= 1.")
    if type(retain_final_output) is not bool:
        raise TypeError("run_eval_suite retain_final_output must be a bool.")
    if not retain_final_output and retain_trajectory:
        raise ValueError("run_eval_suite cannot discard final output while retaining trajectories.")
    if public_output_preview_bytes is not None and retain_final_output:
        raise ValueError("run_eval_suite public output projection requires final-output disposal.")
    _validate_public_output_preview_bytes(
        public_output_preview_bytes,
        "run_eval_suite public_output_preview_bytes",
    )
    _validate_trials(trials, "run_eval_suite trials")
    _validate_timeout_seconds(case_timeout_seconds, "run_eval_suite case_timeout_seconds")
    run_id = str(uuid4())
    started_at = datetime.now(UTC)
    results, public_data_by_case = await _run_suite_cases(
        app,
        suite,
        retain_trajectory=retain_trajectory,
        retain_final_output=retain_final_output,
        max_concurrency=max_concurrency,
        case_timeout_seconds=case_timeout_seconds,
        trials=trials,
        public_output_preview_bytes=public_output_preview_bytes,
    )
    completed_at = datetime.now(UTC)
    status = aggregate_eval_status(result.status for result in results)
    score = aggregate_eval_score(result.score for result in results)
    return (
        EvalRun(
            run_id=run_id,
            suite_id=suite.id,
            status=status,
            score=score,
            cases=tuple(results),
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=_duration_ms(started_at, completed_at),
            metadata=suite.metadata,
            run_contract=None,
        ),
        public_data_by_case,
    )


async def run_eval_plan(
    plan: EvalPlan,
    *,
    corpus: EvalCorpusDocument | None = None,
    suite_id: str | None = None,
    retain_trajectory: bool = False,
    max_concurrency: int = 1,
    case_timeout_seconds: float | None = None,
    trials: int | None = None,
) -> EvalRun | CorpusExecutionResult:
    if type(plan) is not EvalPlan:
        raise TypeError("run_eval_plan requires an EvalPlan.")
    if plan.corpus_target is not None:
        from cayu.evals.corpus import EvalCorpusDocument
        from cayu.evals.execution import run_corpus_suite

        if type(corpus) is not EvalCorpusDocument:
            raise TypeError("Corpus EvalPlan execution requires an exact EvalCorpusDocument.")
        if suite_id is None:
            raise ValueError("Corpus EvalPlan execution requires suite_id.")
        if retain_trajectory:
            raise ValueError("Corpus EvalPlan execution never publishes raw trajectories.")
        if case_timeout_seconds is not None or trials is not None:
            raise ValueError("Corpus trial count and timeout come only from the corpus contract.")
        return await run_corpus_suite(
            plan.corpus_target,
            corpus,
            suite_id,
            max_concurrency=max_concurrency,
        )
    if corpus is not None or suite_id is not None:
        raise ValueError("Direct EvalPlan execution does not accept corpus or suite_id.")
    if not isinstance(plan.app, CayuApp) or type(plan.suite) is not EvalSuite:
        raise TypeError("Direct EvalPlan requires a CayuApp and exact EvalSuite.")
    return await run_eval_suite(
        plan.app,
        plan.suite,
        retain_trajectory=retain_trajectory,
        max_concurrency=max_concurrency,
        case_timeout_seconds=case_timeout_seconds,
        trials=1 if trials is None else trials,
    )


async def _run_suite_cases(
    app: CayuApp,
    suite: EvalSuite,
    *,
    retain_trajectory: bool,
    retain_final_output: bool,
    max_concurrency: int,
    case_timeout_seconds: float | None,
    trials: int,
    public_output_preview_bytes: int | None,
) -> tuple[
    list[EvalCaseResult],
    dict[str, tuple[_EvalTrialPublicData, ...]] | None,
]:
    if max_concurrency == 1:
        executions = [
            await _run_eval_case(
                app,
                case,
                suite_id=suite.id,
                retain_trajectory=retain_trajectory,
                retain_final_output=retain_final_output,
                timeout_seconds=case_timeout_seconds,
                trials=trials,
                public_output_preview_bytes=public_output_preview_bytes,
            )
            for case in suite.cases
        ]
        results = [result for result, _ in executions]
        if public_output_preview_bytes is None:
            return results, None
        public_data_by_case: dict[str, tuple[_EvalTrialPublicData, ...]] = {}
        for case, (_, public_data) in zip(suite.cases, executions, strict=True):
            if public_data is None:
                raise RuntimeError("Corpus case execution lost its public projection.")
            public_data_by_case[case.id] = public_data
        return results, public_data_by_case

    # Fill a positional slot per case so results keep suite order regardless of
    # completion order. run_eval_case never raises for a failed run (it records an
    # ERROR result), so a TaskGroup abort only happens on a genuine programming error.
    slots: list[tuple[EvalCaseResult, tuple[_EvalTrialPublicData, ...] | None] | None] = [
        None
    ] * len(suite.cases)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _run_slot(index: int, case: EvalCase) -> None:
        async with semaphore:
            slots[index] = await _run_eval_case(
                app,
                case,
                suite_id=suite.id,
                retain_trajectory=retain_trajectory,
                retain_final_output=retain_final_output,
                timeout_seconds=case_timeout_seconds,
                trials=trials,
                public_output_preview_bytes=public_output_preview_bytes,
            )

    async with asyncio.TaskGroup() as group:
        for index, case in enumerate(suite.cases):
            group.create_task(_run_slot(index, case))
    executions = [execution for execution in slots if execution is not None]
    if len(executions) != len(slots):
        raise RuntimeError("Concurrent eval execution lost a case result.")
    results = [result for result, _ in executions]
    if public_output_preview_bytes is None:
        return results, None
    public_data_by_case = {}
    for case, (_, public_data) in zip(suite.cases, executions, strict=True):
        if public_data is None:
            raise RuntimeError("Corpus case execution lost its public projection.")
        public_data_by_case[case.id] = public_data
    return results, public_data_by_case


async def run_eval_case(
    app: CayuApp,
    case: EvalCase,
    *,
    suite_id: str,
    retain_trajectory: bool = False,
    retain_final_output: bool = True,
    timeout_seconds: float | None = None,
    trials: int = 1,
) -> EvalCaseResult:
    """Run one case one or more times and retain every concrete trial.

    Every trial receives a fresh concrete session ID; an authored request session ID is never
    reused as concrete state, but remains result provenance and the causal-accounting fallback.
    Aggregate fields are deterministic projections of the ordered ``result.trials`` tuple.
    No trial is selected as a representative and no trial evidence is overwritten.
    """
    if type(case) is not EvalCase:
        raise TypeError("run_eval_case requires an EvalCase.")
    case = _detach_eval_case(case)
    result, _ = await _run_eval_case(
        app,
        case,
        suite_id=suite_id,
        retain_trajectory=retain_trajectory,
        retain_final_output=retain_final_output,
        timeout_seconds=timeout_seconds,
        trials=trials,
        public_output_preview_bytes=None,
    )
    return result


async def _run_eval_case(
    app: CayuApp,
    case: EvalCase,
    *,
    suite_id: str,
    retain_trajectory: bool,
    retain_final_output: bool,
    timeout_seconds: float | None,
    trials: int,
    public_output_preview_bytes: int | None,
) -> tuple[EvalCaseResult, tuple[_EvalTrialPublicData, ...] | None]:
    if type(retain_final_output) is not bool:
        raise TypeError("run_eval_case retain_final_output must be a bool.")
    if not retain_final_output and retain_trajectory:
        raise ValueError("run_eval_case cannot discard final output while retaining trajectories.")
    if public_output_preview_bytes is not None and retain_final_output:
        raise ValueError("run_eval_case public output projection requires final-output disposal.")
    _validate_public_output_preview_bytes(
        public_output_preview_bytes,
        "run_eval_case public_output_preview_bytes",
    )
    _validate_trials(trials, "run_eval_case trials")
    _validate_timeout_seconds(timeout_seconds, "run_eval_case timeout_seconds")
    started_at = datetime.now(UTC)
    trial_executions = [
        await _run_case_once_with_public_projection(
            app,
            case,
            trial_number=trial_number,
            suite_id=suite_id,
            retain_trajectory=retain_trajectory,
            retain_final_output=retain_final_output,
            timeout_seconds=timeout_seconds,
            public_output_preview_bytes=public_output_preview_bytes,
        )
        for trial_number in range(1, trials + 1)
    ]
    trial_results = [result for result, _ in trial_executions]
    trial_public_data: tuple[_EvalTrialPublicData, ...] | None = None
    if public_output_preview_bytes is not None:
        retained_public_data: list[_EvalTrialPublicData] = []
        for _, public_data in trial_executions:
            if public_data is None:
                raise RuntimeError("Corpus trial execution lost its public projection.")
            retained_public_data.append(public_data)
        trial_public_data = tuple(retained_public_data)
    completed_at = datetime.now(UTC)
    return (
        _aggregate_trials(
            case,
            trial_results,
            started_at=started_at,
            completed_at=completed_at,
        ),
        trial_public_data,
    )


async def _run_case_once(
    app: CayuApp,
    case: EvalCase,
    *,
    trial_number: int,
    suite_id: str,
    retain_trajectory: bool = False,
    retain_final_output: bool = True,
    timeout_seconds: float | None = None,
) -> EvalTrialResult:
    result, _ = await _run_case_once_with_public_projection(
        app,
        case,
        trial_number=trial_number,
        suite_id=suite_id,
        retain_trajectory=retain_trajectory,
        retain_final_output=retain_final_output,
        timeout_seconds=timeout_seconds,
        public_output_preview_bytes=None,
    )
    return result


async def _run_case_once_with_public_projection(
    app: CayuApp,
    case: EvalCase,
    *,
    trial_number: int,
    suite_id: str,
    retain_trajectory: bool = False,
    retain_final_output: bool = True,
    timeout_seconds: float | None = None,
    public_output_preview_bytes: int | None = None,
) -> tuple[EvalTrialResult, _EvalTrialPublicData | None]:
    started_at = datetime.now(UTC)
    trial_request = _isolated_trial_request(case.request)
    emitted_root_events: list[RunnerObservedEventIdentity] = []
    emitted_root_events_truncated = False
    observed_session_id: str | None = None
    run_drained = False
    session_id: str | None = None
    run_error: str | None = None
    unavailable_reason: str | None = None
    evidence_complete = False
    session: Session | None = None
    events: tuple[Event, ...] = ()
    transcript: tuple[Message, ...] = ()
    usage_summary: SessionUsageSummary | None = None
    final_output = ""
    trajectory: Trajectory | None = None
    terminal_evidence: TerminalSessionEvidence | None = None
    assertion_results: list[EvalAssertionResult] = []
    public_output = EvalTrialOutputPreviewV1.unavailable()
    diagnostic_code: EvalTrialDiagnosticCode | None = None
    deadline: asyncio.Timeout | None = None

    # asyncio.timeout(None) never expires, so the unbounded default shares the path.
    # Keep the deadline around the full case lifecycle: runtime execution, state/probe
    # capture, child traversal, and assertion evaluation all belong to one trial.
    try:
        async with asyncio.timeout(timeout_seconds) as deadline:
            try:
                async for event in app.run(trial_request):
                    # Anchor the trial to the first emitted session. If app.run() ever forwards
                    # child-session events, they must not replace the root trial identity.
                    observed_session_id = observed_session_id or event.session_id
                    if event.session_id == observed_session_id:
                        if len(emitted_root_events) < TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_EVENTS:
                            emitted_root_events.append(
                                RunnerObservedEventIdentity(
                                    session_id=event.session_id,
                                    sequence=event_durable_sequence(event),
                                    event_type=event.type,
                                )
                            )
                        else:
                            emitted_root_events_truncated = True
                run_drained = True
            except TimeoutError as exc:
                # A provider-originated TimeoutError is an eval run error, not the case
                # deadline. Deadline expiry arrives as cancellation and is handled below.
                run_error = _format_exception(exc)
                diagnostic_code = EvalTrialDiagnosticCode.EXECUTION_FAILED
            except Exception as exc:
                run_error = _format_exception(exc)
                diagnostic_code = EvalTrialDiagnosticCode.EXECUTION_FAILED

            candidate_session_id = observed_session_id or trial_request.session_id
            if candidate_session_id is not None:
                try:
                    terminal_evidence = await _load_terminal_evidence(app, candidate_session_id)
                    session_id = candidate_session_id
                    session, events, transcript, usage_summary = _project_terminal_evidence(
                        terminal_evidence
                    )
                    evidence_complete = True
                except TerminalSessionEvidenceError as exc:
                    if (
                        run_error is None
                        and run_drained
                        and observed_session_id == candidate_session_id
                        and exc.code == TerminalSessionEvidenceErrorCode.SESSION_INTERRUPTED
                    ):
                        try:
                            terminal_evidence = await _load_fresh_interrupted_evidence(
                                app,
                                candidate_session_id,
                                tuple(emitted_root_events),
                                emitted_events_truncated=emitted_root_events_truncated,
                            )
                            session, events, transcript, usage_summary = _project_terminal_evidence(
                                terminal_evidence
                            )
                            session_id = candidate_session_id
                            evidence_complete = True
                        except _FreshInterruptedEvidenceUnavailable as interrupted_exc:
                            unavailable_reason = (
                                f"Fresh interrupted evidence unavailable: {interrupted_exc}"
                            )
                            diagnostic_code = (
                                EvalTrialDiagnosticCode.INTERRUPTED_EVIDENCE_UNAVAILABLE
                            )
                        except Exception as interrupted_exc:
                            run_error = (
                                "Failed to load fresh interrupted eval evidence: "
                                f"{_format_exception(interrupted_exc)}"
                            )
                            diagnostic_code = EvalTrialDiagnosticCode.TERMINAL_EVIDENCE_FAILED
                    elif run_error is None:
                        unavailable_reason = (
                            f"Terminal evidence unavailable ({exc.code.value}): {exc}"
                        )
                        diagnostic_code = EvalTrialDiagnosticCode.TERMINAL_EVIDENCE_UNAVAILABLE
                except NotImplementedError:
                    if run_error is None:
                        unavailable_reason = (
                            "Terminal evidence unavailable: the configured session store does "
                            "not support exact terminal evidence reads."
                        )
                        diagnostic_code = EvalTrialDiagnosticCode.TERMINAL_EVIDENCE_UNAVAILABLE
                except Exception as exc:
                    if run_error is None:
                        run_error = (
                            f"Failed to load terminal eval evidence: {_format_exception(exc)}"
                        )
                        diagnostic_code = EvalTrialDiagnosticCode.TERMINAL_EVIDENCE_FAILED

            if evidence_complete:
                try:
                    # app.run() does not raise on a model/tool failure; it ends the session as
                    # SESSION_FAILED and returns normally. Surface that as an eval ERROR so a
                    # crashed run is never scored as PASSED — unless the case explicitly
                    # asserts on session status, in which case the assertion owns the outcome.
                    if (
                        run_error is None
                        and session is not None
                        and session.status == SessionStatus.FAILED
                    ):
                        failed_session_flags = tuple(
                            assertion.evaluates_failed_session for assertion in case.assertions
                        )
                        if any(type(flag) is not bool for flag in failed_session_flags):
                            raise TypeError(
                                "EvalAssertion.evaluates_failed_session must return bool."
                            )
                        if not any(failed_session_flags):
                            run_error = _session_failure_reason(events)
                            diagnostic_code = EvalTrialDiagnosticCode.SESSION_FAILED
                    final_output = final_output_text(transcript)
                    probe_requirements = _collect_probe_requirements(case.assertions)
                    probes = await _capture_probes(app, session, probe_requirements)
                    children_incomplete = _IncompleteFlag()
                    if terminal_evidence is None:
                        raise RuntimeError("Exact root evidence was lost before child capture.")
                    capture_state = _CaptureState(
                        bounds=SessionTrajectoryBounds(),
                        strict=False,
                    )
                    capture_state.retain(terminal_evidence)
                    children = await _build_child_trajectories(
                        app,
                        session_id,
                        visited={session_id} if session_id is not None else set(),
                        incomplete=children_incomplete,
                        parent_terminal_sequence=(
                            terminal_evidence.boundary.terminal_event_sequence
                        ),
                        state=capture_state,
                    )
                    trajectory = _trajectory_from_terminal_evidence(
                        terminal_evidence,
                        probes=probes,
                        children=children,
                        children_incomplete=children_incomplete.value,
                        metadata=case.metadata,
                    )
                    if children_incomplete.value:
                        evidence_complete = False
                        if run_error is None:
                            unavailable_reason = (
                                "Child-session evidence could not be captured completely."
                            )
                            diagnostic_code = EvalTrialDiagnosticCode.CHILD_EVIDENCE_UNAVAILABLE
                except Exception as exc:
                    # Probe declaration/capture and trajectory construction are part
                    # of assertion evidence preparation. Public assertion extensions
                    # must produce an explicit result instead of aborting the suite.
                    evidence_complete = False
                    trajectory = None
                    if run_error is None:
                        run_error = (
                            f"Failed to prepare eval assertion evidence: {_format_exception(exc)}"
                        )
                        diagnostic_code = EvalTrialDiagnosticCode.EVIDENCE_PREPARATION_FAILED

            prepared_portable_evidence = None
            portable_evidence_error: Exception | None = None
            prepared_context: EvalContext | None = None
            if trajectory is not None and (
                public_output_preview_bytes is not None
                or (run_error is None and unavailable_reason is None)
            ):
                prepared_context = EvalContext(
                    trajectory=trajectory,
                    suite_id=suite_id,
                    case_id=case.id,
                    metadata=case.metadata,
                    root_evidence_available=trajectory.session is not None,
                )
                prepared_portable_evidence, portable_evidence_error = _prepare_portable_evidence(
                    case.assertions,
                    prepared_context,
                    runtime_app=app,
                )
                if (
                    public_output_preview_bytes is not None
                    and prepared_portable_evidence is not None
                ):
                    public_output = EvalTrialOutputPreviewV1.from_retained_evidence(
                        prepared_portable_evidence.final_output,
                        prepared_portable_evidence.final_output_state,
                        max_preview_bytes=public_output_preview_bytes,
                    )

            if run_error is not None:
                assertion_results = list(
                    _blocked_assertion_results(case.assertions, EvalOutcome.ERROR, run_error)
                )
            elif unavailable_reason is not None:
                assertion_results = list(
                    _blocked_assertion_results(
                        case.assertions,
                        EvalOutcome.UNAVAILABLE,
                        unavailable_reason,
                    )
                )
                identity_error = _assertion_diagnostic(
                    assertion_results,
                    EvalOutcome.ERROR,
                    "Assertion identity failed",
                )
                if identity_error is not None:
                    run_error = identity_error
                    unavailable_reason = None
                    diagnostic_code = EvalTrialDiagnosticCode.ASSERTION_EVALUATION_FAILED
            elif prepared_context is not None:
                evaluated = await _evaluate_assertions_with_prepared_evidence(
                    case.assertions,
                    prepared_context,
                    portable_evidence=prepared_portable_evidence,
                    portable_evidence_error=portable_evidence_error,
                )
                assertion_results = list(evaluated)
                assertion_error = _assertion_diagnostic(
                    assertion_results,
                    EvalOutcome.ERROR,
                    "Assertion evaluation failed",
                )
                assertion_unavailable = _assertion_diagnostic(
                    assertion_results,
                    EvalOutcome.UNAVAILABLE,
                    "Assertion evidence was unavailable",
                )
                if assertion_error is not None:
                    run_error = assertion_error
                    diagnostic_code = (
                        EvalTrialDiagnosticCode.EVIDENCE_PREPARATION_FAILED
                        if portable_evidence_error is not None
                        else EvalTrialDiagnosticCode.ASSERTION_EVALUATION_FAILED
                    )
                elif assertion_unavailable is not None:
                    unavailable_reason = assertion_unavailable
                    diagnostic_code = EvalTrialDiagnosticCode.ASSERTION_EVIDENCE_UNAVAILABLE
    except TimeoutError as exc:
        if deadline is not None and deadline.expired():
            run_error = f"Eval case timed out after {timeout_seconds} seconds."
            diagnostic_code = EvalTrialDiagnosticCode.CASE_TIMEOUT
        else:
            run_error = _format_exception(exc)
            diagnostic_code = EvalTrialDiagnosticCode.EXECUTION_FAILED
        unavailable_reason = None
        # A timeout may happen after the exact terminal snapshot, probes, and
        # child tree were fully captured while an assertion was evaluating.
        # Preserve that completed evidence; the ERROR outcome already records
        # that evaluation itself did not finish. Earlier lifecycle timeouts have
        # no completed trajectory and remain evidence-incomplete.
        evidence_complete = trajectory is not None and not trajectory.children_incomplete
        # Never expose a partially evaluated assertion prefix after cancellation.
        assertion_results = list(
            _blocked_assertion_results(case.assertions, EvalOutcome.ERROR, run_error)
        )

    # A yielded runtime event proves the session was created before a deadline interrupted
    # evidence capture. Do not perform any store I/O after the deadline; without an event, the
    # request UUID remains only a plan and must not be published as a concrete trial ID.
    if session_id is None and observed_session_id is not None:
        session_id = observed_session_id
    completed_at = datetime.now(UTC)
    status = _trial_status(run_error, unavailable_reason, assertion_results)
    if diagnostic_code is None:
        diagnostic_code = {
            EvalStatus.PASSED: EvalTrialDiagnosticCode.PASSED,
            EvalStatus.FAILED: EvalTrialDiagnosticCode.ASSERTION_FAILED,
            EvalStatus.UNAVAILABLE: EvalTrialDiagnosticCode.ASSERTION_EVIDENCE_UNAVAILABLE,
            EvalStatus.ERROR: EvalTrialDiagnosticCode.EXECUTION_FAILED,
            EvalStatus.SKIPPED: None,
        }[status]
    public_data = (
        None
        if public_output_preview_bytes is None or diagnostic_code is None
        else _EvalTrialPublicData(
            diagnostic_code=diagnostic_code,
            output=public_output,
        )
    )
    return (
        EvalTrialResult(
            trial_number=trial_number,
            status=status,
            session_id=session_id,
            score=_trial_score(status, assertion_results),
            final_output=final_output if retain_final_output else "",
            assertions=tuple(assertion_results),
            error=run_error,
            unavailable_reason=unavailable_reason,
            evidence_complete=evidence_complete,
            events_count=len(events),
            usage_summary=session_usage_summary_payload(usage_summary)
            if usage_summary is not None
            else None,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=_duration_ms(started_at, completed_at),
            # The probe-complete trajectory captured during this trial, for export/replay. Opt-in
            # so the default run does not retain every trial's file bytes in memory.
            trajectory=trajectory if retain_trajectory else None,
        ),
        public_data,
    )


async def _load_fresh_interrupted_evidence(
    app: CayuApp,
    session_id: str,
    emitted_events: tuple[RunnerObservedEventIdentity, ...],
    *,
    emitted_events_truncated: bool,
) -> TerminalSessionEvidence:
    """Reconcile one runner-owned, fully drained interruption with durable state.

    Ordinary historical evidence remains completed/failed-only. This narrower path exists for
    direct Python evals: the runner created a unique session, drained its public event stream,
    and asks the store to prove the exact bounded snapshot before hydrating it.
    """

    if emitted_events_truncated:
        raise _FreshInterruptedEvidenceUnavailable(
            "the root event stream exceeds the fresh-evidence event limit."
        )
    if not emitted_events or any(event.session_id != session_id for event in emitted_events):
        raise _FreshInterruptedEvidenceUnavailable(
            "the drained runtime stream contained no root-session events."
        )
    if not app.session_store.supports_runner_owned_interrupted_evidence:
        raise _FreshInterruptedEvidenceUnavailable(
            "the configured session store does not support exact runner-owned interruptions."
        )
    try:
        evidence = await app.session_store.load_runner_owned_interrupted_evidence(
            session_id,
            observed_events=emitted_events,
        )
    except (NotImplementedError, TerminalSessionEvidenceError) as exc:
        detail = (
            exc.code.value
            if isinstance(exc, TerminalSessionEvidenceError)
            else str(exc).strip() or type(exc).__name__
        )
        raise _FreshInterruptedEvidenceUnavailable(detail) from exc
    return evidence


def _isolated_trial_request(request: RunRequest) -> RunRequest:
    copied = copy_run_request(request)
    trial_session_id = str(uuid4())
    # Session identity owns transcript/checkpoint/terminal state and is always isolated.
    # Causal identity intentionally groups related sessions for accounting, so preserve the
    # authored effective identity and only default it to the concrete trial when none exists.
    causal_budget_id = copied.causal_budget_id or copied.session_id or trial_session_id
    return copied.model_copy(
        update={
            "session_id": trial_session_id,
            "causal_budget_id": causal_budget_id,
        }
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
    # model_copy/model_construct can bypass field and nested-model validation.
    # Reconstruct the complete public input before applying derived record checks,
    # then evaluate only against that detached validated copy.
    trajectory = Trajectory.model_validate(_model_instance_python_input(trajectory))
    _validate_trajectory_record_contract(trajectory)
    context = EvalContext(
        trajectory=trajectory,
        suite_id=suite_id,
        case_id=case_id,
        metadata=dict(trajectory.metadata),
        root_evidence_available=trajectory.session is not None,
    )
    return await _evaluate_assertions(tuple(assertions), context)


async def _evaluate_assertions(
    assertions: Sequence[EvalAssertion],
    context: EvalContext,
    *,
    runtime_app: CayuApp | None = None,
) -> tuple[EvalAssertionResult, ...]:
    portable_evidence, portable_evidence_error = _prepare_portable_evidence(
        assertions,
        context,
        runtime_app=runtime_app,
    )
    return await _evaluate_assertions_with_prepared_evidence(
        assertions,
        context,
        portable_evidence=portable_evidence,
        portable_evidence_error=portable_evidence_error,
    )


def _prepare_portable_evidence(
    assertions: Sequence[EvalAssertion],
    context: EvalContext,
    *,
    runtime_app: CayuApp | None,
) -> tuple[AssertionEvidenceView | None, Exception | None]:
    # Import lazily to keep the public assertion base independent of the
    # optional portable-corpus adapter.
    from cayu.evals.portable_assertions import _prepare_portable_assertion_evidence

    try:
        return (
            _prepare_portable_assertion_evidence(
                assertions,
                context,
                runtime_app=runtime_app,
            ),
            None,
        )
    except Exception as exc:
        return None, exc


async def _evaluate_assertions_with_prepared_evidence(
    assertions: Sequence[EvalAssertion],
    context: EvalContext,
    *,
    portable_evidence: AssertionEvidenceView | None,
    portable_evidence_error: Exception | None,
) -> tuple[EvalAssertionResult, ...]:
    from cayu.evals.portable_assertions import (
        _CompiledModelJudgeAssertion,
        _CompiledPortableAssertion,
    )

    results: list[EvalAssertionResult] = []
    for assertion in assertions:
        assertion_revision: str | None = None
        try:
            assertion_revision = assertion.assertion_revision
            if type(assertion) is _CompiledPortableAssertion:
                if portable_evidence_error is not None:
                    raise portable_evidence_error
                if portable_evidence is None:
                    raise RuntimeError("Compiled assertion evidence was not prepared.")
                result = assertion.evaluate_evidence(portable_evidence)
            elif type(assertion) is _CompiledModelJudgeAssertion:
                if portable_evidence_error is not None:
                    raise portable_evidence_error
                if portable_evidence is None:
                    raise RuntimeError("Compiled assertion evidence was not prepared.")
                result = await assertion.evaluate_evidence(portable_evidence, context)
            else:
                result = assertion.evaluate(context)
                if inspect.isawaitable(result):
                    result = await result
            if type(result) is not EvalAssertionResult:
                raise TypeError("EvalAssertion.evaluate must return EvalAssertionResult.")
            # Assertion extensions can return a validator-bypassed model. Rebuild it
            # inside the protected boundary so replay never exposes an impossible
            # result and fresh runs do not fail later during trial construction.
            result = EvalAssertionResult.model_validate(_model_instance_python_input(result))
            if assertion_revision is not None and result.assertion_revision != assertion_revision:
                raise ValueError("Assertion result revision does not match its definition.")
            if result.cost_summary is not None and (
                context.session is None or result.cost_summary.session_id != context.session.id
            ):
                raise ValueError("Assertion cost summaries must belong to the trajectory session.")
            results.append(result)
        except Exception as exc:
            results.append(
                _assertion_error_result(
                    assertion,
                    assertion_revision=assertion_revision,
                    message=f"Assertion raised {_format_exception_summary(exc)}",
                    error_type=type(exc).__name__,
                )
            )
    return tuple(results)


def _assertion_error_result(
    assertion: EvalAssertion,
    *,
    assertion_revision: str | None,
    message: str,
    error_type: str,
) -> EvalAssertionResult:
    try:
        return EvalAssertionResult(
            name=assertion.name,
            assertion_revision=assertion_revision,
            outcome=EvalOutcome.ERROR,
            message=message,
            metadata={"error_type": error_type},
        )
    except Exception as identity_exc:
        return EvalAssertionResult(
            name="EvalAssertion",
            outcome=EvalOutcome.ERROR,
            message=(
                f"{message}; failed to resolve assertion identity: "
                f"{_format_exception_summary(identity_exc)}"
            ),
            metadata={
                "error_type": error_type,
                "identity_error": True,
                "identity_error_type": type(identity_exc).__name__,
            },
        )


def _blocked_assertion_results(
    assertions: Sequence[EvalAssertion],
    outcome: EvalOutcome,
    message: str,
) -> tuple[EvalAssertionResult, ...]:
    if outcome not in (EvalOutcome.ERROR, EvalOutcome.UNAVAILABLE):
        raise ValueError("Blocked assertions require an error or unavailable outcome.")
    results: list[EvalAssertionResult] = []
    for assertion in assertions:
        try:
            result = EvalAssertionResult(
                name=assertion.name,
                assertion_revision=assertion.assertion_revision,
                outcome=outcome,
                message=message,
            )
        except Exception as exc:
            result = _assertion_error_result(
                assertion,
                assertion_revision=None,
                message=(
                    "Failed to resolve blocked assertion identity: "
                    f"{_format_exception_summary(exc)}"
                ),
                error_type=type(exc).__name__,
            )
        results.append(result)
    return tuple(results)


def _assertion_diagnostic(
    assertions: Sequence[EvalAssertionResult],
    outcome: EvalOutcome,
    prefix: str,
) -> str | None:
    matching = tuple(assertion for assertion in assertions if assertion.outcome == outcome)
    if not matching:
        return None
    return f"{prefix} in {len(matching)} assertion(s); first: {matching[0].message}"


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
    workspace_unavailable_paths: list[str] = []
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
            except FileNotFoundError:
                # Confirmed absence is negative evidence. Other failures are retained
                # separately because they did not observe whether the file exists.
                workspace_files[path] = None
            except Exception:
                workspace_unavailable_paths.append(path)

    artifacts_available = False
    artifacts: list[ArtifactMetadata] = []
    artifact_scopes_captured: list[ArtifactScope] = []
    artifact_scopes_truncated: list[ArtifactScope] = []
    artifact_scopes_unavailable: list[ArtifactScope] = []
    artifact_store = getattr(env, "artifact_store", None)
    if requirements.artifact_scopes and artifact_store is not None:
        artifacts_available = True
        seen_ids: set[str] = set()
        for scope in sorted(requirements.artifact_scopes, key=str):
            try:
                listed = await artifact_store.list(scope=scope)
            except Exception:
                artifact_scopes_unavailable.append(scope)
                continue
            if listed.truncated:
                artifact_scopes_truncated.append(scope)
            else:
                artifact_scopes_captured.append(scope)
            for artifact in listed.artifacts:
                if artifact.id not in seen_ids:
                    seen_ids.add(artifact.id)
                    artifacts.append(artifact)

    return TrajectoryProbes(
        workspace_available=workspace_available,
        workspace_files=workspace_files,
        workspace_file_stats=workspace_file_stats,
        workspace_unavailable_paths=tuple(workspace_unavailable_paths),
        artifacts_available=artifacts_available,
        artifact_scopes_captured=tuple(artifact_scopes_captured),
        artifact_scopes_truncated=tuple(artifact_scopes_truncated),
        artifact_scopes_unavailable=tuple(artifact_scopes_unavailable),
        artifacts=tuple(artifacts),
    )


def _validate_public_output_preview_bytes(value: int | None, field_name: str) -> None:
    if value is None:
        return
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an int or None.")
    if not 1 <= value <= EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES:
        raise ValueError(
            f"{field_name} must be between 1 and {EVAL_TRIAL_OUTPUT_MAX_PREVIEW_BYTES}."
        )


def _validate_trials(value: int, field_name: str) -> None:
    # bool is an int subclass; reject it so trials=True can't silently mean 1 trial.
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an int.")
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1.")


def _aggregate_trials(
    case: EvalCase,
    results: list[EvalTrialResult],
    *,
    started_at: datetime,
    completed_at: datetime,
) -> EvalCaseResult:
    retained = tuple(results)
    return EvalCaseResult.from_trials(
        case_id=case.id,
        authored_session_id=case.request.session_id,
        trials=retained,
        started_at=started_at,
        completed_at=completed_at,
        metadata=case.metadata,
    )


def _validate_timeout_seconds(value: float | None, field_name: str) -> None:
    if value is None:
        return
    if type(value) not in (int, float):
        raise TypeError(f"{field_name} must be a number or None.")
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite positive number.")
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive number.")


def _session_failure_reason(events: Iterable[Event]) -> str:
    for event in events:
        if event.type == EventType.SESSION_FAILED:
            error = event.payload.get("error")
            if isinstance(error, str) and error.strip():
                return f"Session failed: {error}"
            return "Session failed."
    return "Session ended in a failed state."


def _trial_status(
    run_error: str | None,
    unavailable_reason: str | None,
    assertions: Sequence[EvalAssertionResult],
) -> EvalStatus:
    if run_error is not None:
        return EvalStatus.ERROR
    if unavailable_reason is not None:
        return EvalStatus.UNAVAILABLE
    if not assertions:
        return EvalStatus.SKIPPED
    outcomes = tuple(assertion.outcome for assertion in assertions)
    if EvalOutcome.ERROR in outcomes:
        return EvalStatus.ERROR
    if EvalOutcome.UNAVAILABLE in outcomes:
        return EvalStatus.UNAVAILABLE
    if EvalOutcome.FAILED in outcomes:
        return EvalStatus.FAILED
    return EvalStatus.PASSED


def _trial_score(
    status: EvalStatus,
    assertions: Sequence[EvalAssertionResult],
) -> float | None:
    if status in (EvalStatus.ERROR, EvalStatus.UNAVAILABLE):
        return None
    if status == EvalStatus.SKIPPED:
        return 0.0
    if not assertions:
        return 0.0
    scores = tuple(assertion.score for assertion in assertions)
    if any(score is None for score in scores):
        return None
    numeric = tuple(score for score in scores if score is not None)
    return sum(numeric) / len(numeric)


def _duration_ms(started_at: datetime, completed_at: datetime) -> int:
    return max(int((completed_at - started_at).total_seconds() * 1000), 0)
