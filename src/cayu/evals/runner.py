from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
import traceback
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import TYPE_CHECKING, Any
from uuid import uuid4
from weakref import WeakKeyDictionary

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._exception_groups import exception_tree_contains
from cayu._task_wait import capture_awaitable_outcome
from cayu._validation import (
    MAX_PORTABLE_JSON_INTEGER,
    copy_durable_json_object,
    require_durable_clean_nonblank,
    require_durable_text,
)
from cayu.artifacts import (
    ArtifactListResult,
    ArtifactMetadata,
    ArtifactScope,
    copy_artifact_read_result,
)
from cayu.core.events import Event, EventType, event_durable_sequence
from cayu.core.messages import Message
from cayu.evals._execution_profile_errors import EvalExecutionProfileChangedError
from cayu.evals._memory_attribution import (
    eval_memory_attribution_evidence_from_trajectory,
)
from cayu.evals.assertions import EvalAssertion
from cayu.evals.capacity import EVAL_MAX_CONCURRENCY, EvalExecutionCapacity
from cayu.evals.corpus import EVAL_ARTIFACT_EVIDENCE_MAX_ARTIFACTS
from cayu.evals.memory_attribution import (
    EVAL_MEMORY_ATTRIBUTION_MAX_BYTES,
    EVAL_MEMORY_ATTRIBUTION_MAX_SOURCES,
    EvalMemoryAttributionEvidenceV1,
    EvalMemoryEvidenceLimitation,
    eval_memory_attribution_bounds_for_trial_count,
    eval_memory_attribution_max_bytes_for_trial_count,
    eval_memory_attribution_source_limit_for_trial_count,
    standard_eval_memory_attribution_bounds,
)
from cayu.evals.models import (
    ARTIFACT_PROBE_MAX_BYTES,
    ARTIFACT_PUBLIC_TEXT_MAX_BYTES,
    WORKSPACE_PROBE_MAX_BYTES,
    ArtifactContentProbe,
    ArtifactProbeRequirement,
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
    WorkspaceStructuralProbe,
    _artifact_text_media_type_supported,
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
    SessionTrajectoryError,
    SessionTrajectoryErrorCode,
    _build_child_trajectories,
    _CaptureState,
    _IncompleteFlag,
    _load_terminal_evidence,
    _memory_attribution_snapshot,
    _project_terminal_evidence,
    _promote_memory_attribution,
    _revalidate_fresh_capture,
    _trajectory_from_terminal_evidence,
    _validated_terminal_session_evidence,
    _workflow_trajectory_from_session,
    final_output_text,
)
from cayu.evals.trial_policy import EvalSuiteTrialPolicyV1
from cayu.evals.workflow_target import (
    WorkflowEvalExecution,
    WorkflowEvalFailure,
    WorkflowEvalFailureCode,
    WorkflowEvalInvocation,
    WorkflowEvalOutputEvidenceV1,
    WorkflowEvalResult,
    WorkflowEvalTerminalEvidence,
    workflow_eval_input_messages_sha256,
    workflow_eval_output_sha256,
    workflow_eval_trial_session_id,
)
from cayu.memory_attribution import (
    MemoryAttribution,
    MemoryAttributionBounds,
    MemoryAttributionStatus,
    MemoryAttributionUnavailableReason,
)
from cayu.runtime._memory_evidence import memory_evidence_key
from cayu.runtime.app import CayuApp
from cayu.runtime.execution_profiles import ExecutionProfileMismatchError
from cayu.runtime.sessions import (
    TERMINAL_SESSION_EVIDENCE_DEFAULT_MAX_EVENTS,
    EventQuery,
    EventRecord,
    RunnerObservedEventIdentity,
    RunRequest,
    Session,
    SessionStatus,
    TerminalSessionEvidence,
    TerminalSessionEvidenceError,
    TerminalSessionEvidenceErrorCode,
    TerminalSessionEvidenceLimits,
    copy_run_request,
)
from cayu.runtime.usage import (
    SessionUsageSummary,
    session_usage_summary_payload,
)
from cayu.tools._operation_boundary import (
    BoundedInvocationOperationRegistry,
    InvocationOperationCapacityError,
    _RetainedInvocationOperationProbe,
    await_invocation_operation,
    retained_invocation_operation_outcome_if_done,
)
from cayu.workflows import (
    WORKFLOW_ATTEMPT_EVENT_TYPE,
    WORKFLOW_JOURNAL_PROVIDER,
    WorkflowSupersededError,
)
from cayu.workspaces import WorkspaceReadResult

TrialRequestTransform = Callable[[str, str, int, RunRequest], RunRequest]
TrialCompletionCallback = Callable[
    [str, EvalTrialResult, _EvalTrialPublicData],
    Awaitable[None],
]

if TYPE_CHECKING:
    from cayu.evals.corpus import EvalCorpusDocument
    from cayu.evals.evidence import AssertionEvidenceView
    from cayu.evals.execution import CorpusExecutionResult, CorpusTarget, WorkflowEvalTarget


class _FreshInterruptedEvidenceUnavailable(RuntimeError):
    """The runner-owned interrupted session could not be reconciled exactly."""


class _WorkflowInstanceTracker:
    """Enforce a target's declared shared/per-trial application ownership."""

    def __init__(self, scope: str) -> None:
        self._scope = scope
        self._lock = Lock()
        self._executions: list[WorkflowEvalExecution] = []

    def observe(self, execution: WorkflowEvalExecution) -> None:
        with self._lock:
            if self._executions:
                first = self._executions[0]
                reuses_instance = any(
                    execution.app is prior.app or execution.workflow is prior.workflow
                    for prior in self._executions
                )
                if self._scope == "per_trial" and reuses_instance:
                    raise WorkflowEvalFailure(
                        WorkflowEvalFailureCode.TARGET_FAILED,
                        "Per-trial workflow target reused application or workflow state.",
                    )
                if self._scope == "shared" and (
                    execution.app is not first.app or execution.workflow is not first.workflow
                ):
                    raise WorkflowEvalFailure(
                        WorkflowEvalFailureCode.TARGET_FAILED,
                        "Shared workflow target changed application or workflow state.",
                    )
            self._executions.append(execution)


class _FreshMemoryAttributionReadFailed(RuntimeError):
    """One fresh-eval attribution projection could not be read authoritatively."""


class _FreshMemoryAttributionReadStopped(BaseException):
    """A retained composite read reached its first post-abandonment dispatch seam."""


class _FreshEvalDeadlineExpired(BaseException):
    """The fresh-eval deadline expired before another evidence read was admitted."""


class _FreshMemoryAttributionReadStop:
    """Per-operation cooperative authority that forbids reads after abandonment."""

    __slots__ = ("_stopped",)

    def __init__(self) -> None:
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    def require_read_allowed(self) -> None:
        if self._stopped:
            raise _FreshMemoryAttributionReadStopped


_CONTRADICTORY_FRESH_CAPTURE_CODES = frozenset(
    {
        SessionTrajectoryErrorCode.ORIGIN_EVIDENCE_REJECTED,
        SessionTrajectoryErrorCode.PARENT_CONTRADICTION,
        SessionTrajectoryErrorCode.CYCLE_DETECTED,
        SessionTrajectoryErrorCode.EVIDENCE_INCONSISTENT,
    }
)
_INCOMPLETE_FRESH_CAPTURE_CODES = frozenset(
    {
        SessionTrajectoryErrorCode.SESSION_LIMIT_EXCEEDED,
        SessionTrajectoryErrorCode.DEPTH_LIMIT_EXCEEDED,
    }
)


def _fresh_capture_revalidation_limitation(
    error: SessionTrajectoryError,
) -> EvalMemoryEvidenceLimitation:
    """Map exact-capture failures onto the portable memory-evidence vocabulary."""

    if error.code is SessionTrajectoryErrorCode.CLOSURE_CHANGED:
        return EvalMemoryEvidenceLimitation.CLOSURE_CHANGED
    if error.code is SessionTrajectoryErrorCode.STORE_UNSUPPORTED:
        return EvalMemoryEvidenceLimitation.STORE_UNSUPPORTED
    if error.code in _CONTRADICTORY_FRESH_CAPTURE_CODES:
        return EvalMemoryEvidenceLimitation.CONTRADICTORY_LINEAGE
    if error.code in _INCOMPLETE_FRESH_CAPTURE_CODES:
        return EvalMemoryEvidenceLimitation.SOURCE_TREE_INCOMPLETE
    return EvalMemoryEvidenceLimitation.EVIDENCE_READ_FAILED


_MAX_RETAINED_FRESH_MEMORY_ATTRIBUTION_READS = 1_024


class _FreshMemoryAttributionReadSupervisor:
    """Retain bounded abandoned reads and deliver exact late control once."""

    def __init__(
        self,
        *,
        max_operations: int = _MAX_RETAINED_FRESH_MEMORY_ATTRIBUTION_READS,
    ) -> None:
        self.operation_registry = BoundedInvocationOperationRegistry(max_operations=max_operations)
        self._retained: list[_RetainedInvocationOperationProbe] = []

    def retain(self, retained: _RetainedInvocationOperationProbe) -> None:
        if type(retained) is not _RetainedInvocationOperationProbe:
            raise TypeError("retained must be a runtime-owned invocation-operation probe.")
        self._retained.append(retained)

    def claim_terminal_controls(self) -> tuple[BaseException, ...]:
        pending: list[_RetainedInvocationOperationProbe] = []
        controls: list[BaseException] = []
        for retained in self._retained:
            outcome = retained_invocation_operation_outcome_if_done(retained)
            if outcome is None:
                pending.append(retained)
                continue
            error = outcome.error
            if error is not None and exception_tree_contains(
                error,
                (GeneratorExit, KeyboardInterrupt, SystemExit),
            ):
                controls.append(error)
        self._retained = pending
        return tuple(_ordered_unique_failures(controls))


_FRESH_MEMORY_READ_SUPERVISORS: WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    _FreshMemoryAttributionReadSupervisor,
] = WeakKeyDictionary()
_FRESH_MEMORY_READ_SUPERVISORS_LOCK = Lock()


def _shared_fresh_memory_attribution_read_supervisor() -> _FreshMemoryAttributionReadSupervisor:
    """Return one loop-owned supervisor shared by sequential and overlapping runs."""

    loop = asyncio.get_running_loop()
    with _FRESH_MEMORY_READ_SUPERVISORS_LOCK:
        supervisor = _FRESH_MEMORY_READ_SUPERVISORS.get(loop)
        if supervisor is None:
            supervisor = _FreshMemoryAttributionReadSupervisor()
            _FRESH_MEMORY_READ_SUPERVISORS[loop] = supervisor
        return supervisor


class _FreshMemoryAttributionOperationRegistry:
    """Compose one eval-run limit with the loop-wide retained-read bound."""

    def __init__(
        self,
        *,
        max_operations: int,
        supervisor: _FreshMemoryAttributionReadSupervisor,
    ) -> None:
        self._local = BoundedInvocationOperationRegistry(max_operations=max_operations)
        self._shared = supervisor.operation_registry

    def reserve(self) -> bool:
        if not self._local.reserve():
            return False
        if self._shared.reserve():
            return True
        self._local.release_reservation()
        return False

    def track(self, operation: asyncio.Future[Any]) -> None:
        self._local.track(operation)
        self._shared.track(operation)

    def release_reservation(self) -> None:
        self._local.release_reservation()
        self._shared.release_reservation()

    def release(self, operation: asyncio.Future[Any]) -> None:
        self._local.release(operation)
        self._shared.release(operation)

    async def aclose(self, *, timeout_s: float) -> bool:
        del timeout_s
        self._local.seal()
        # The same wrapper task remains owned by the shared registry. Cancelling
        # it here can let a store such as SQLite detach physical off-thread work
        # and make the shared capacity slot appear settled too early.
        return len(self._local) == 0


def _ordered_unique_failures(failures: Iterable[BaseException]) -> list[BaseException]:
    ordered: list[BaseException] = []
    for failure in failures:
        if all(failure is not retained for retained in ordered):
            ordered.append(failure)
    return ordered


class _FreshMemoryAttributionReadLifecycle:
    """Own fresh-read capacity, bounded stop, and exact terminal control outcomes."""

    def __init__(self, *, max_operations: int) -> None:
        self._supervisor = _shared_fresh_memory_attribution_read_supervisor()
        self.operation_registry = _FreshMemoryAttributionOperationRegistry(
            max_operations=max_operations,
            supervisor=self._supervisor,
        )

    def retain_abandoned(self, retained: _RetainedInvocationOperationProbe) -> None:
        self._supervisor.retain(retained)

    async def _close(self) -> tuple[BaseException, ...]:
        await self.operation_registry.aclose(timeout_s=0.0)
        return self._supervisor.claim_terminal_controls()

    async def __aenter__(self) -> _FreshMemoryAttributionReadLifecycle:
        controls = self._supervisor.claim_terminal_controls()
        if len(controls) == 1:
            raise controls[0]
        if controls:
            raise BaseExceptionGroup(
                "Fresh eval memory attribution retained process control.",
                list(controls),
            ) from None
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        primary: BaseException | None,
        _traceback: object,
    ) -> bool:
        close_task = asyncio.create_task(capture_awaitable_outcome(self._close))
        concurrent: list[BaseException] = []
        while not close_task.done():
            try:
                await asyncio.shield(close_task)
            except BaseException as error:
                concurrent.append(error)
        try:
            await asyncio.sleep(0)
        except BaseException as error:
            concurrent.append(error)
        captured = close_task.result()
        if captured.error is not None:
            concurrent.append(captured.error)
        elif type(captured.result) is tuple:
            concurrent.extend(captured.result)
        else:
            concurrent.append(
                RuntimeError("Fresh eval memory-read cleanup returned invalid state.")
            )
        failures = _ordered_unique_failures(([primary] if primary is not None else []) + concurrent)
        if not concurrent:
            return False
        if len(failures) == 1:
            raise failures[0]
        raise BaseExceptionGroup(
            "Fresh eval memory attribution cleanup received concurrent control.",
            failures,
        ) from None


async def _owned_fresh_memory_attribution_projection(
    app: CayuApp,
    trajectory: Trajectory,
    *,
    bounds: MemoryAttributionBounds,
    lifecycle: _FreshMemoryAttributionReadLifecycle,
) -> Trajectory:
    """Read one projection without letting an opaque store defeat caller cancellation."""

    stop = _FreshMemoryAttributionReadStop()

    def retain_abandoned(retained: _RetainedInvocationOperationProbe) -> None:
        stop.stop()
        lifecycle.retain_abandoned(retained)

    outcome = await await_invocation_operation(
        lambda: _promote_memory_attribution(
            app,
            trajectory,
            bounds=bounds,
            before_store_read=stop.require_read_allowed,
        ),
        request_child_cancellation=False,
        abandon_on_caller_cancellation=True,
        operation_registry=lifecycle.operation_registry,
        on_abandoned_caller_operation=retain_abandoned,
        on_unsettled_supervisory_exit=retain_abandoned,
    )
    error = outcome.error
    if outcome.cancellation is not None:
        if error is not None and exception_tree_contains(
            error,
            (GeneratorExit, KeyboardInterrupt, SystemExit),
        ):
            raise BaseExceptionGroup(
                "Fresh eval memory attribution received concurrent process control and "
                "caller cancellation.",
                [outcome.cancellation, error],
            ) from None
        raise outcome.cancellation from None
    if error is not None and exception_tree_contains(
        error,
        (GeneratorExit, KeyboardInterrupt, SystemExit),
    ):
        raise error
    if (
        isinstance(error, (asyncio.CancelledError, BaseExceptionGroup))
        or type(error) is InvocationOperationCapacityError
    ):
        raise _FreshMemoryAttributionReadFailed(
            "Fresh eval memory attribution could not be read."
        ) from None
    if error is not None:
        raise error
    if type(outcome.result) is not Trajectory:
        raise _FreshMemoryAttributionReadFailed(
            "Fresh eval memory attribution returned an invalid projection."
        )
    return outcome.result


async def _owned_fresh_capture_revalidation(
    app: CayuApp,
    capture_state: _CaptureState,
    *,
    root_session_id: str,
    root_interrupted_observed_events: tuple[RunnerObservedEventIdentity, ...],
    lifecycle: _FreshMemoryAttributionReadLifecycle,
) -> None:
    """Revalidate one closure without letting opaque store reads defeat cancellation."""

    stop = _FreshMemoryAttributionReadStop()

    def retain_abandoned(retained: _RetainedInvocationOperationProbe) -> None:
        stop.stop()
        lifecycle.retain_abandoned(retained)

    outcome = await await_invocation_operation(
        lambda: _revalidate_fresh_capture(
            app,
            capture_state,
            root_session_id=root_session_id,
            root_interrupted_observed_events=root_interrupted_observed_events,
            before_store_read=stop.require_read_allowed,
        ),
        request_child_cancellation=False,
        abandon_on_caller_cancellation=True,
        operation_registry=lifecycle.operation_registry,
        on_abandoned_caller_operation=retain_abandoned,
        on_unsettled_supervisory_exit=retain_abandoned,
    )
    error = outcome.error
    if outcome.cancellation is not None:
        if error is not None and exception_tree_contains(
            error,
            (GeneratorExit, KeyboardInterrupt, SystemExit),
        ):
            raise BaseExceptionGroup(
                "Fresh eval closure revalidation received concurrent process control and "
                "caller cancellation.",
                [outcome.cancellation, error],
            ) from None
        raise outcome.cancellation from None
    if error is not None and exception_tree_contains(
        error,
        (GeneratorExit, KeyboardInterrupt, SystemExit),
    ):
        raise error
    if (
        isinstance(error, (asyncio.CancelledError, BaseExceptionGroup))
        or type(error) is InvocationOperationCapacityError
    ):
        raise _FreshMemoryAttributionReadFailed(
            "Fresh eval closure revalidation could not be read."
        ) from None
    if error is not None:
        raise error
    if outcome.result is not None:
        raise _FreshMemoryAttributionReadFailed(
            "Fresh eval closure revalidation returned an invalid result."
        )


def _memory_attribution_read_failed_trajectory(trajectory: Trajectory) -> Trajectory:
    """Fail closed for the complete retained tree after an untrusted read outcome."""

    attribution = (
        None
        if trajectory.session is None
        else MemoryAttribution(
            status=MemoryAttributionStatus.UNAVAILABLE,
            truncated=False,
            reason=MemoryAttributionUnavailableReason.EVIDENCE_READ_FAILED,
            observed_receipt_count=0,
            observed_exposure_count=0,
            observed_item_count=0,
            omitted_receipt_count_at_least=0,
            omitted_exposure_count_at_least=0,
            omitted_item_count_at_least=0,
        )
    )
    return trajectory.model_copy(
        update={
            "memory_attribution": attribution,
            "children": tuple(
                _memory_attribution_read_failed_trajectory(child) for child in trajectory.children
            ),
        }
    )


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
    workflow_target: WorkflowEvalTarget | None = None

    def __post_init__(self) -> None:
        from cayu.evals.execution import CorpusTarget, WorkflowEvalTarget

        direct_configured = self.app is not None
        corpus_configured = self.corpus_target is not None
        workflow_configured = self.workflow_target is not None
        if sum((direct_configured, corpus_configured, workflow_configured)) != 1:
            raise ValueError(
                "EvalPlan requires exactly one mode: app with suite, corpus_target, "
                "or workflow_target."
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
            if not workflow_configured:
                raise TypeError("EvalPlan corpus_target must be an exact CorpusTarget.")
        elif self.suite is not None:
            raise ValueError("Corpus EvalPlan execution does not accept a direct suite.")
        if workflow_configured:
            if type(self.workflow_target) is not WorkflowEvalTarget:
                raise TypeError("EvalPlan workflow_target must be an exact WorkflowEvalTarget.")
            if self.app is not None or self.corpus_target is not None:
                raise ValueError(
                    "Workflow EvalPlan execution cannot also configure another target."
                )
            if self.suite is not None:
                if type(self.suite) is not EvalSuite:
                    raise TypeError("EvalPlan suite must be an exact EvalSuite.")
                object.__setattr__(self, "suite", _detach_eval_suite(self.suite))


async def run_workflow_eval_suite(
    target: WorkflowEvalTarget,
    suite: EvalSuite,
    *,
    retain_trajectory: bool = False,
    retain_final_output: bool = True,
    max_concurrency: int = 1,
    case_timeout_seconds: float | None = None,
    trials: int = 1,
    execution_capacity: EvalExecutionCapacity | None = None,
    trial_policy: EvalSuiteTrialPolicyV1 | None = None,
) -> EvalRun:
    """Run a typed application-owned workflow as the root eval target."""

    from cayu.evals.execution import WorkflowEvalTarget

    if type(target) is not WorkflowEvalTarget:
        raise TypeError("run_workflow_eval_suite requires an exact WorkflowEvalTarget.")
    target_identity = target.identity()
    app_manifest_fingerprint = target.app.describe().fingerprint
    try:
        execution_profile_fingerprint = await target.app.inspect_run_execution_profile(
            copy_run_request(target.request_base)
        )
    except Exception as exc:
        raise EvalExecutionProfileChangedError(
            f"WorkflowEvalTarget execution profile could not be established ({type(exc).__name__})."
        ) from None
    run, _ = await _run_eval_suite(
        target.app,
        suite,
        retain_trajectory=retain_trajectory,
        retain_final_output=retain_final_output,
        max_concurrency=max_concurrency,
        case_timeout_seconds=case_timeout_seconds,
        trials=trials,
        trial_policy=trial_policy,
        public_output_preview_bytes=None,
        execution_capacity=execution_capacity,
        workflow_target=target,
        workflow_execution_profile_fingerprint=execution_profile_fingerprint,
    )
    try:
        final_execution_profile_fingerprint = await target.app.inspect_run_execution_profile(
            copy_run_request(target.request_base)
        )
    except Exception as exc:
        raise EvalExecutionProfileChangedError(
            f"WorkflowEvalTarget execution profile could not be revalidated ({type(exc).__name__})."
        ) from None
    if (
        target.identity() != target_identity
        or target.app.describe().fingerprint != app_manifest_fingerprint
        or final_execution_profile_fingerprint != execution_profile_fingerprint
    ):
        raise EvalExecutionProfileChangedError(
            "WorkflowEvalTarget identity, application manifest, or execution profile changed "
            "during eval execution."
        )
    return run


async def run_eval_suite(
    app: CayuApp,
    suite: EvalSuite,
    *,
    retain_trajectory: bool = False,
    retain_final_output: bool = True,
    max_concurrency: int = 1,
    case_timeout_seconds: float | None = None,
    trials: int = 1,
    execution_capacity: EvalExecutionCapacity | None = None,
    trial_policy: EvalSuiteTrialPolicyV1 | None = None,
) -> EvalRun:
    """Run every case in the suite and aggregate the results.

    `max_concurrency` runs up to that many concrete case trials at once (default 1 =
    sequential). Results always keep suite and trial order. Note:
    `ScriptedModelProvider` consumes batches by positional request index, so with
    concurrency > 1 interleaved cases may pull each other's batches — keep the
    default for scripted multi-case suites.

    `case_timeout_seconds` bounds each case's run; a case that exceeds it is
    cancelled and recorded as `EvalStatus.ERROR` instead of stalling the suite.

    `trials` runs each case N times and retains every outcome. When supplied,
    `trial_policy` fixes the minimum-pass decision and concurrency ceiling before
    execution; otherwise an all-pass policy records the requested concurrency.

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
        trial_policy=trial_policy,
        public_output_preview_bytes=None,
        execution_capacity=execution_capacity,
    )
    return run


async def _run_eval_suite_with_public_projection(
    app: CayuApp,
    suite: EvalSuite,
    *,
    max_concurrency: int,
    case_timeout_seconds: float | None,
    trials: int,
    trial_policy: EvalSuiteTrialPolicyV1,
    output_preview_bytes: int,
    run_stream: Callable[[RunRequest], AsyncIterator[Event]] | None = None,
    run_id: str | None = None,
    trial_request_transform: TrialRequestTransform | None = None,
    execution_capacity: EvalExecutionCapacity | None = None,
    completed_trials: Mapping[tuple[str, int], tuple[EvalTrialResult, _EvalTrialPublicData]]
    | None = None,
    trial_completed: TrialCompletionCallback | None = None,
    workflow_target: WorkflowEvalTarget | None = None,
    workflow_execution_profile_fingerprint: str | None = None,
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
        trial_policy=trial_policy,
        public_output_preview_bytes=output_preview_bytes,
        run_stream=run_stream,
        run_id=run_id,
        trial_request_transform=trial_request_transform,
        execution_capacity=execution_capacity,
        completed_trials=completed_trials,
        trial_completed=trial_completed,
        workflow_target=workflow_target,
        workflow_execution_profile_fingerprint=workflow_execution_profile_fingerprint,
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
    trial_policy: EvalSuiteTrialPolicyV1 | None,
    public_output_preview_bytes: int | None,
    run_stream: Callable[[RunRequest], AsyncIterator[Event]] | None = None,
    run_id: str | None = None,
    trial_request_transform: TrialRequestTransform | None = None,
    execution_capacity: EvalExecutionCapacity | None = None,
    completed_trials: Mapping[tuple[str, int], tuple[EvalTrialResult, _EvalTrialPublicData]]
    | None = None,
    trial_completed: TrialCompletionCallback | None = None,
    workflow_target: WorkflowEvalTarget | None = None,
    workflow_execution_profile_fingerprint: str | None = None,
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
    if max_concurrency > EVAL_MAX_CONCURRENCY:
        raise ValueError(f"run_eval_suite max_concurrency must be <= {EVAL_MAX_CONCURRENCY}.")
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
    if trial_policy is None:
        validated_trial_policy = EvalSuiteTrialPolicyV1.create(
            trial_count=trials,
            max_concurrency=max_concurrency,
        )
    elif type(trial_policy) is EvalSuiteTrialPolicyV1:
        validated_trial_policy = EvalSuiteTrialPolicyV1.model_validate(
            trial_policy.model_dump(mode="json")
        )
    else:
        raise TypeError("run_eval_suite trial_policy must be an exact policy or None.")
    if validated_trial_policy.trial_count != trials:
        raise ValueError("run_eval_suite trial_policy must match trials.")
    if max_concurrency > validated_trial_policy.max_concurrency:
        raise ValueError("run_eval_suite max_concurrency exceeds its trial policy.")
    _validate_timeout_seconds(case_timeout_seconds, "run_eval_suite case_timeout_seconds")
    run_id = str(uuid4()) if run_id is None else require_durable_clean_nonblank(run_id, "run_id")
    if trial_request_transform is not None and not callable(trial_request_transform):
        raise TypeError("trial_request_transform must be callable or None.")
    if execution_capacity is not None and type(execution_capacity) is not EvalExecutionCapacity:
        raise TypeError("execution_capacity must be an exact EvalExecutionCapacity or None.")
    if completed_trials is None:
        completed_trials = {}
    elif not isinstance(completed_trials, Mapping):
        raise TypeError("completed_trials must be a mapping.")
    if trial_completed is not None and not callable(trial_completed):
        raise TypeError("trial_completed must be callable or None.")
    if (completed_trials or trial_completed is not None) and public_output_preview_bytes is None:
        raise ValueError("Durable trial recovery requires the public trial projection.")
    if workflow_target is not None:
        from cayu.evals.execution import WorkflowEvalTarget

        if type(workflow_target) is not WorkflowEvalTarget:
            raise TypeError("workflow_target must be an exact WorkflowEvalTarget or None.")
        if workflow_target.app is not app:
            raise ValueError("workflow_target app must match the eval runner application.")
        if workflow_target.instance_scope.value == "shared" and max_concurrency > 1:
            raise ValueError(
                "A shared workflow target requires max_concurrency=1; use per_trial "
                "for concurrent trials."
            )
        if run_stream is not None or trial_request_transform is not None:
            raise ValueError("Workflow eval targets own their execution and request identity.")
        if (
            type(workflow_execution_profile_fingerprint) is not str
            or len(workflow_execution_profile_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in workflow_execution_profile_fingerprint
            )
        ):
            raise ValueError(
                "Workflow eval execution requires an exact execution-profile fingerprint."
            )
    elif workflow_execution_profile_fingerprint is not None:
        raise ValueError("workflow_execution_profile_fingerprint requires a workflow eval target.")
    started_at = datetime.now(UTC)
    trial_count = len(suite.cases) * trials
    memory_attribution_bounds = eval_memory_attribution_bounds_for_trial_count(trial_count)
    memory_attribution_source_limit = eval_memory_attribution_source_limit_for_trial_count(
        trial_count
    )
    memory_attribution_max_bytes = eval_memory_attribution_max_bytes_for_trial_count(trial_count)
    memory_attribution_read_lifecycle = _FreshMemoryAttributionReadLifecycle(
        max_operations=max_concurrency
    )
    workflow_instance_tracker = (
        None
        if workflow_target is None
        else _WorkflowInstanceTracker(workflow_target.instance_scope.value)
    )
    async with memory_attribution_read_lifecycle:
        results, public_data_by_case = await _run_suite_cases(
            app,
            suite,
            retain_trajectory=retain_trajectory,
            retain_final_output=retain_final_output,
            max_concurrency=max_concurrency,
            case_timeout_seconds=case_timeout_seconds,
            trials=trials,
            trial_policy=validated_trial_policy,
            public_output_preview_bytes=public_output_preview_bytes,
            memory_attribution_bounds=memory_attribution_bounds,
            memory_attribution_source_limit=memory_attribution_source_limit,
            memory_attribution_max_bytes=memory_attribution_max_bytes,
            memory_attribution_read_lifecycle=memory_attribution_read_lifecycle,
            run_stream=run_stream,
            trial_request_transform=trial_request_transform,
            execution_capacity=execution_capacity,
            completed_trials=completed_trials,
            trial_completed=trial_completed,
            run_id=run_id,
            workflow_target=workflow_target,
            workflow_instance_tracker=workflow_instance_tracker,
            workflow_execution_profile_fingerprint=(workflow_execution_profile_fingerprint),
        )
    observed_started_at = min(result.started_at for result in results)
    observed_completed_at = max(result.completed_at for result in results)
    status = aggregate_eval_status(result.status for result in results)
    score = aggregate_eval_score(result.score for result in results)
    return (
        EvalRun(
            run_id=run_id,
            suite_id=suite.id,
            status=status,
            score=score,
            cases=tuple(results),
            started_at=min(started_at, observed_started_at),
            completed_at=observed_completed_at,
            duration_ms=_duration_ms(min(started_at, observed_started_at), observed_completed_at),
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
    execution_capacity: EvalExecutionCapacity | None = None,
) -> EvalRun | CorpusExecutionResult:
    if type(plan) is not EvalPlan:
        raise TypeError("run_eval_plan requires an EvalPlan.")
    if execution_capacity is not None and type(execution_capacity) is not EvalExecutionCapacity:
        raise TypeError("execution_capacity must be an exact EvalExecutionCapacity or None.")
    portable_target = plan.corpus_target or (
        plan.workflow_target if plan.workflow_target is not None and plan.suite is None else None
    )
    if portable_target is not None:
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
            portable_target,
            corpus,
            suite_id,
            max_concurrency=max_concurrency,
            execution_capacity=execution_capacity,
        )
    if corpus is not None or suite_id is not None:
        raise ValueError("Direct EvalPlan execution does not accept corpus or suite_id.")
    if plan.workflow_target is not None:
        if type(plan.suite) is not EvalSuite:
            raise TypeError("Workflow EvalPlan requires an exact EvalSuite or a corpus.")
        return await run_workflow_eval_suite(
            plan.workflow_target,
            plan.suite,
            retain_trajectory=retain_trajectory,
            max_concurrency=max_concurrency,
            case_timeout_seconds=case_timeout_seconds,
            trials=1 if trials is None else trials,
            execution_capacity=execution_capacity,
        )
    if not isinstance(plan.app, CayuApp) or type(plan.suite) is not EvalSuite:
        raise TypeError("Direct EvalPlan requires a CayuApp and exact EvalSuite.")
    return await run_eval_suite(
        plan.app,
        plan.suite,
        retain_trajectory=retain_trajectory,
        max_concurrency=max_concurrency,
        case_timeout_seconds=case_timeout_seconds,
        trials=1 if trials is None else trials,
        execution_capacity=execution_capacity,
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
    trial_policy: EvalSuiteTrialPolicyV1,
    public_output_preview_bytes: int | None,
    memory_attribution_bounds: MemoryAttributionBounds,
    memory_attribution_source_limit: int,
    memory_attribution_max_bytes: int,
    memory_attribution_read_lifecycle: _FreshMemoryAttributionReadLifecycle,
    run_stream: Callable[[RunRequest], AsyncIterator[Event]] | None,
    trial_request_transform: TrialRequestTransform | None,
    execution_capacity: EvalExecutionCapacity | None,
    completed_trials: Mapping[tuple[str, int], tuple[EvalTrialResult, _EvalTrialPublicData]],
    trial_completed: TrialCompletionCallback | None,
    run_id: str,
    workflow_target: WorkflowEvalTarget | None,
    workflow_instance_tracker: _WorkflowInstanceTracker | None,
    workflow_execution_profile_fingerprint: str | None,
) -> tuple[
    list[EvalCaseResult],
    dict[str, tuple[_EvalTrialPublicData, ...]] | None,
]:
    # Schedule concrete trials, not whole cases, so one repeated case can consume
    # the configured concurrency. Positional slots preserve authored case order
    # and numeric trial order regardless of completion order.
    slots: list[list[tuple[EvalTrialResult, _EvalTrialPublicData | None] | None]] = [
        [None] * trials for _ in suite.cases
    ]
    case_by_id = {case.id: index for index, case in enumerate(suite.cases)}
    for key, execution in completed_trials.items():
        if type(key) is not tuple or len(key) != 2:
            raise ValueError("Completed trial keys must be (case_id, trial_number) pairs.")
        case_id, trial_number = key
        index = case_by_id.get(case_id)
        if index is None or type(trial_number) is not int or not 1 <= trial_number <= trials:
            raise ValueError("Completed trial slot does not belong to the eval suite.")
        result, public_data = execution
        if type(result) is not EvalTrialResult or type(public_data) is not _EvalTrialPublicData:
            raise TypeError("Completed trials require exact result and public-data values.")
        if result.trial_number != trial_number:
            raise ValueError("Completed trial result does not match its slot.")
        slots[index][trial_number - 1] = (
            EvalTrialResult.model_validate(result.model_dump(mode="python", warnings=False)),
            _EvalTrialPublicData.model_validate(public_data.model_dump(mode="python")),
        )
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _run_slot(index: int, case: EvalCase, trial_number: int) -> None:
        async with semaphore:
            capacity_slot = (
                nullcontext() if execution_capacity is None else execution_capacity.slot()
            )
            async with capacity_slot:
                execution = await _run_case_once_with_public_projection(
                    app,
                    case,
                    trial_number=trial_number,
                    suite_id=suite.id,
                    retain_trajectory=retain_trajectory,
                    retain_final_output=retain_final_output,
                    timeout_seconds=case_timeout_seconds,
                    public_output_preview_bytes=public_output_preview_bytes,
                    memory_attribution_bounds=memory_attribution_bounds,
                    memory_attribution_source_limit=memory_attribution_source_limit,
                    memory_attribution_max_bytes=memory_attribution_max_bytes,
                    memory_attribution_read_lifecycle=memory_attribution_read_lifecycle,
                    run_stream=run_stream,
                    trial_request_transform=trial_request_transform,
                    run_id=run_id,
                    workflow_target=workflow_target,
                    workflow_instance_tracker=workflow_instance_tracker,
                    workflow_execution_profile_fingerprint=(workflow_execution_profile_fingerprint),
                )
                result, public_data = execution
                if trial_completed is not None:
                    if public_data is None:
                        raise RuntimeError("Durable trial execution lost its public projection.")
                    await trial_completed(case.id, result, public_data)
                slots[index][trial_number - 1] = execution

    pending_slots = (
        (index, case, trial_number)
        for index, case in enumerate(suite.cases)
        for trial_number in range(1, trials + 1)
        if slots[index][trial_number - 1] is None
    )
    if max_concurrency == 1:
        # Preserve direct cancellation identity and avoid TaskGroup overhead for
        # the common sequential policy while still honoring recovered slots.
        for index, case, trial_number in pending_slots:
            await _run_slot(index, case, trial_number)
    else:
        async with asyncio.TaskGroup() as group:
            for index, case, trial_number in pending_slots:
                group.create_task(_run_slot(index, case, trial_number))

    results: list[EvalCaseResult] = []
    public_data_by_case: dict[str, tuple[_EvalTrialPublicData, ...]] = {}
    for index, case in enumerate(suite.cases):
        executions = [execution for execution in slots[index] if execution is not None]
        if len(executions) != trials:
            raise RuntimeError("Concurrent eval execution lost a trial result.")
        started_at = min(result.started_at for result, _public_data in executions)
        completed_at = max(result.completed_at for result, _public_data in executions)
        results.append(
            _aggregate_trials(
                case,
                [result for result, _ in executions],
                started_at=started_at,
                completed_at=completed_at,
                trial_policy=trial_policy,
            )
        )
        if public_output_preview_bytes is not None:
            retained_public_data: list[_EvalTrialPublicData] = []
            for _, public_data in executions:
                if public_data is None:
                    raise RuntimeError("Corpus trial execution lost its public projection.")
                retained_public_data.append(public_data)
            public_data_by_case[case.id] = tuple(retained_public_data)
    return results, None if public_output_preview_bytes is None else public_data_by_case


async def run_eval_case(
    app: CayuApp,
    case: EvalCase,
    *,
    suite_id: str,
    retain_trajectory: bool = False,
    retain_final_output: bool = True,
    timeout_seconds: float | None = None,
    trials: int = 1,
    trial_policy: EvalSuiteTrialPolicyV1 | None = None,
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
    _validate_trials(trials, "run_eval_case trials")
    if trial_policy is None:
        validated_trial_policy = EvalSuiteTrialPolicyV1.create(trial_count=trials)
    elif type(trial_policy) is EvalSuiteTrialPolicyV1:
        validated_trial_policy = EvalSuiteTrialPolicyV1.model_validate(
            trial_policy.model_dump(mode="json")
        )
    else:
        raise TypeError("run_eval_case trial_policy must be an exact policy or None.")
    if validated_trial_policy.trial_count != trials:
        raise ValueError("run_eval_case trial_policy must match trials.")
    memory_attribution_read_lifecycle = _FreshMemoryAttributionReadLifecycle(max_operations=1)
    async with memory_attribution_read_lifecycle:
        result, _ = await _run_eval_case(
            app,
            case,
            suite_id=suite_id,
            retain_trajectory=retain_trajectory,
            retain_final_output=retain_final_output,
            timeout_seconds=timeout_seconds,
            trials=trials,
            trial_policy=validated_trial_policy,
            public_output_preview_bytes=None,
            memory_attribution_bounds=eval_memory_attribution_bounds_for_trial_count(trials),
            memory_attribution_source_limit=eval_memory_attribution_source_limit_for_trial_count(
                trials
            ),
            memory_attribution_max_bytes=eval_memory_attribution_max_bytes_for_trial_count(trials),
            memory_attribution_read_lifecycle=memory_attribution_read_lifecycle,
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
    trial_policy: EvalSuiteTrialPolicyV1,
    public_output_preview_bytes: int | None,
    memory_attribution_bounds: MemoryAttributionBounds,
    memory_attribution_source_limit: int,
    memory_attribution_max_bytes: int,
    memory_attribution_read_lifecycle: _FreshMemoryAttributionReadLifecycle,
    run_stream: Callable[[RunRequest], AsyncIterator[Event]] | None = None,
    trial_request_transform: TrialRequestTransform | None = None,
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
    if type(trial_policy) is not EvalSuiteTrialPolicyV1:
        raise TypeError("run_eval_case trial_policy must be an exact EvalSuiteTrialPolicyV1.")
    if trial_policy.trial_count != trials:
        raise ValueError("run_eval_case trial_policy must match trials.")
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
            memory_attribution_bounds=memory_attribution_bounds,
            memory_attribution_source_limit=memory_attribution_source_limit,
            memory_attribution_max_bytes=memory_attribution_max_bytes,
            memory_attribution_read_lifecycle=memory_attribution_read_lifecycle,
            run_stream=run_stream,
            trial_request_transform=trial_request_transform,
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
            trial_policy=trial_policy,
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
    memory_attribution_bounds: MemoryAttributionBounds | None = None,
    memory_attribution_source_limit: int | None = None,
    memory_attribution_max_bytes: int | None = None,
) -> EvalTrialResult:
    memory_attribution_read_lifecycle = _FreshMemoryAttributionReadLifecycle(max_operations=1)
    async with memory_attribution_read_lifecycle:
        result, _ = await _run_case_once_with_public_projection(
            app,
            case,
            trial_number=trial_number,
            suite_id=suite_id,
            retain_trajectory=retain_trajectory,
            retain_final_output=retain_final_output,
            timeout_seconds=timeout_seconds,
            public_output_preview_bytes=None,
            memory_attribution_bounds=(
                memory_attribution_bounds or standard_eval_memory_attribution_bounds()
            ),
            memory_attribution_source_limit=(
                EVAL_MEMORY_ATTRIBUTION_MAX_SOURCES
                if memory_attribution_source_limit is None
                else memory_attribution_source_limit
            ),
            memory_attribution_max_bytes=(
                EVAL_MEMORY_ATTRIBUTION_MAX_BYTES
                if memory_attribution_max_bytes is None
                else memory_attribution_max_bytes
            ),
            memory_attribution_read_lifecycle=memory_attribution_read_lifecycle,
        )
    return result


_WORKFLOW_EVAL_EVENT_PAGE_SIZE = 5_000
_WORKFLOW_EVAL_MAX_EVENTS = 10_000


async def _load_workflow_eval_records(
    app: CayuApp,
    *,
    session_id: str,
    workflow_name: str,
) -> tuple[EventRecord, ...]:
    records: list[EventRecord] = []
    after_sequence = 0
    while True:
        page = await app.session_store.query_events(
            EventQuery(
                session_id=session_id,
                workflow_name=workflow_name,
                after_sequence=after_sequence,
                limit=_WORKFLOW_EVAL_EVENT_PAGE_SIZE,
            )
        )
        if not page:
            break
        if any(type(record) is not EventRecord for record in page):
            raise WorkflowEvalFailure(
                WorkflowEvalFailureCode.TARGET_FAILED,
                "Workflow evidence store returned an invalid event record.",
            )
        if any(record.sequence <= after_sequence for record in page):
            raise WorkflowEvalFailure(
                WorkflowEvalFailureCode.COMPLETION_CONFLICT,
                "Workflow evidence contains a non-monotonic event sequence.",
            )
        records.extend(page)
        if len(records) > _WORKFLOW_EVAL_MAX_EVENTS:
            raise WorkflowEvalFailure(
                WorkflowEvalFailureCode.COMPLETION_CONFLICT,
                "Workflow evidence exceeds the bounded event limit.",
            )
        after_sequence = page[-1].sequence
        if len(page) < _WORKFLOW_EVAL_EVENT_PAGE_SIZE:
            break
    if len({record.event.id for record in records}) != len(records):
        raise WorkflowEvalFailure(
            WorkflowEvalFailureCode.COMPLETION_CONFLICT,
            "Workflow evidence contains duplicate event identities.",
        )
    return tuple(records)


def _current_workflow_completion(
    records: tuple[EventRecord, ...],
    *,
    workflow_name: str,
) -> tuple[str, EventRecord]:
    attempts = tuple(
        record
        for record in records
        if record.event.type == WORKFLOW_ATTEMPT_EVENT_TYPE
        and isinstance(record.event.payload.get("attempt_id"), str)
        and record.event.payload.get("attempt_id")
    )
    if not attempts:
        raise WorkflowEvalFailure(
            WorkflowEvalFailureCode.COMPLETION_MISSING,
            "Workflow current-attempt evidence is missing.",
        )
    attempt_id = str(attempts[-1].event.payload["attempt_id"])
    current = tuple(
        record for record in records if record.event.payload.get("attempt_id") == attempt_id
    )
    markers = tuple(
        record for record in current if record.event.type == WORKFLOW_ATTEMPT_EVENT_TYPE
    )
    starts = tuple(record for record in current if record.event.type == EventType.WORKFLOW_STARTED)
    completions = tuple(
        record for record in current if record.event.type == EventType.WORKFLOW_COMPLETED
    )
    if not completions:
        raise WorkflowEvalFailure(
            WorkflowEvalFailureCode.COMPLETION_MISSING,
            "Workflow current-attempt completion evidence is missing.",
        )
    if len(markers) != 1 or len(starts) != 1 or len(completions) != 1:
        raise WorkflowEvalFailure(
            WorkflowEvalFailureCode.COMPLETION_CONFLICT,
            "Workflow current-attempt boundary evidence is conflicting.",
        )
    completion = completions[0]
    if (
        completion.event.workflow_name != workflow_name
        or markers[0].sequence >= starts[0].sequence
        or starts[0].sequence >= completion.sequence
    ):
        raise WorkflowEvalFailure(
            WorkflowEvalFailureCode.COMPLETION_CONFLICT,
            "Workflow current-attempt completion evidence is invalid.",
        )
    if completion.sequence != current[-1].sequence:
        raise WorkflowEvalFailure(
            WorkflowEvalFailureCode.COMPLETION_CONFLICT,
            "Workflow current-attempt evidence continues after completion.",
        )
    return attempt_id, completion


async def _capture_workflow_child_probes(
    app: CayuApp,
    children: tuple[Trajectory, ...],
    requirements: ProbeRequirements,
) -> tuple[Trajectory, ...]:
    captured: list[Trajectory] = []
    for child in children:
        descendants = await _capture_workflow_child_probes(app, child.children, requirements)
        probes = await _capture_probes(app, child.session, requirements)
        captured.append(child.model_copy(update={"children": descendants, "probes": probes}))
    return tuple(captured)


def _workflow_event_count(trajectory: Trajectory) -> int:
    return len(trajectory.events) + sum(
        _workflow_event_count(child) for child in trajectory.children
    )


_WORKFLOW_FAILURE_DIAGNOSTICS = {
    WorkflowEvalFailureCode.TARGET_FAILED: EvalTrialDiagnosticCode.WORKFLOW_TARGET_FAILED,
    WorkflowEvalFailureCode.EXECUTION_FAILED: (EvalTrialDiagnosticCode.WORKFLOW_EXECUTION_FAILED),
    WorkflowEvalFailureCode.COMPLETION_MISSING: (
        EvalTrialDiagnosticCode.WORKFLOW_COMPLETION_MISSING
    ),
    WorkflowEvalFailureCode.COMPLETION_CONFLICT: (
        EvalTrialDiagnosticCode.WORKFLOW_COMPLETION_CONFLICT
    ),
    WorkflowEvalFailureCode.ATTEMPT_SUPERSEDED: (
        EvalTrialDiagnosticCode.WORKFLOW_ATTEMPT_SUPERSEDED
    ),
    WorkflowEvalFailureCode.PROJECTOR_FAILED: EvalTrialDiagnosticCode.WORKFLOW_PROJECTOR_FAILED,
    WorkflowEvalFailureCode.OUTPUT_INVALID: EvalTrialDiagnosticCode.WORKFLOW_OUTPUT_INVALID,
    WorkflowEvalFailureCode.QUIESCENCE_FAILED: (EvalTrialDiagnosticCode.WORKFLOW_QUIESCENCE_FAILED),
}


async def _run_workflow_case_once_with_public_projection(
    target: WorkflowEvalTarget,
    case: EvalCase,
    *,
    trial_number: int,
    suite_id: str,
    run_id: str,
    retain_trajectory: bool,
    retain_final_output: bool,
    timeout_seconds: float | None,
    public_output_preview_bytes: int | None,
    memory_attribution_bounds: MemoryAttributionBounds | None,
    memory_attribution_source_limit: int | None,
    memory_attribution_max_bytes: int | None,
    memory_attribution_read_lifecycle: _FreshMemoryAttributionReadLifecycle,
    workflow_instance_tracker: _WorkflowInstanceTracker,
    workflow_execution_profile_fingerprint: str,
) -> tuple[EvalTrialResult, _EvalTrialPublicData | None]:
    """Execute one workflow trial and bind output to durable current-attempt evidence."""

    from cayu.evals.execution import WorkflowEvalTarget

    if type(target) is not WorkflowEvalTarget:
        raise TypeError("workflow target execution requires an exact WorkflowEvalTarget.")
    started_at = datetime.now(UTC)
    workflow_identity = target.identity()
    app_manifest = target.app.describe()
    root_session_id = workflow_eval_trial_session_id(
        target_revision=workflow_identity.revision,
        run_id=run_id,
        suite_id=suite_id,
        case_id=case.id,
        trial_number=trial_number,
    )
    request = _isolated_trial_request(case.request).model_copy(
        update={"session_id": root_session_id, "causal_budget_id": root_session_id}
    )
    execution: WorkflowEvalExecution | None = None
    runtime_app = target.app
    trajectory: Trajectory | None = None
    final_output = ""
    structured_output: dict[str, Any] | None = None
    assertion_results: list[EvalAssertionResult] = []
    run_error: str | None = None
    diagnostic_code: EvalTrialDiagnosticCode | None = None
    public_output = EvalTrialOutputPreviewV1.unavailable()
    capture_state: _CaptureState | None = None
    publication_attempt_id: str | None = None
    publication_completion_event_id: str | None = None
    publication_record_identity: tuple[tuple[int, str], ...] | None = None
    selected_memory_bounds = MemoryAttributionBounds.model_validate(
        (memory_attribution_bounds or standard_eval_memory_attribution_bounds()).model_dump(
            mode="python"
        )
    )
    selected_memory_source_limit = (
        EVAL_MEMORY_ATTRIBUTION_MAX_SOURCES
        if memory_attribution_source_limit is None
        else memory_attribution_source_limit
    )
    selected_memory_max_bytes = (
        EVAL_MEMORY_ATTRIBUTION_MAX_BYTES
        if memory_attribution_max_bytes is None
        else memory_attribution_max_bytes
    )
    memory_attribution = EvalMemoryAttributionEvidenceV1.unavailable(
        EvalMemoryEvidenceLimitation.MISSING,
        effective_bounds=selected_memory_bounds,
        effective_source_limit=selected_memory_source_limit,
        effective_max_bytes=selected_memory_max_bytes,
    )
    try:
        async with asyncio.timeout(timeout_seconds):
            invocation = WorkflowEvalInvocation(
                run_id=run_id,
                suite_id=suite_id,
                case_id=case.id,
                trial_number=trial_number,
                workflow_run_id=root_session_id,
                idempotency_key=f"cayu-eval:{root_session_id}",
                messages=tuple(request.messages),
                application_context=target.application_context,
            )
            try:
                built = target.workflow_factory(invocation)
                if inspect.isawaitable(built):
                    built = await built
            except Exception as exc:
                raise WorkflowEvalFailure(
                    WorkflowEvalFailureCode.TARGET_FAILED,
                    f"Workflow target factory failed ({type(exc).__name__}).",
                ) from None
            if type(built) is not WorkflowEvalExecution:
                raise WorkflowEvalFailure(
                    WorkflowEvalFailureCode.TARGET_FAILED,
                    "Workflow target factory returned an invalid execution.",
                )
            execution = built
            workflow_instance_tracker.observe(execution)
            runtime_app = execution.app
            if runtime_app.describe().fingerprint != app_manifest.fingerprint:
                raise WorkflowEvalFailure(
                    WorkflowEvalFailureCode.TARGET_FAILED,
                    "Workflow target application manifest does not match its declared identity.",
                )
            try:
                runtime_execution_profile_fingerprint = (
                    await runtime_app.inspect_run_execution_profile(
                        copy_run_request(target.request_base)
                    )
                )
            except Exception as exc:
                raise WorkflowEvalFailure(
                    WorkflowEvalFailureCode.TARGET_FAILED,
                    "Workflow target execution profile could not be established "
                    f"({type(exc).__name__}).",
                ) from None
            if runtime_execution_profile_fingerprint != workflow_execution_profile_fingerprint:
                raise WorkflowEvalFailure(
                    WorkflowEvalFailureCode.TARGET_FAILED,
                    "Workflow target execution profile does not match its declared identity.",
                )
            if execution.workflow.spec != target.workflow_spec:
                raise WorkflowEvalFailure(
                    WorkflowEvalFailureCode.TARGET_FAILED,
                    "Workflow target factory returned a different workflow specification.",
                )
            try:
                async for emitted in execution.workflow.run(root_session_id):
                    if type(emitted) is not Event:
                        raise TypeError("workflow run yielded a non-Event value")
            except Exception as exc:
                if exception_tree_contains(exc, (WorkflowSupersededError,)):
                    raise WorkflowEvalFailure(
                        WorkflowEvalFailureCode.ATTEMPT_SUPERSEDED,
                        "Workflow attempt was superseded during execution.",
                    ) from None
                raise WorkflowEvalFailure(
                    WorkflowEvalFailureCode.EXECUTION_FAILED,
                    f"Workflow execution failed ({type(exc).__name__}).",
                ) from None

            workflow_session = await runtime_app.session_store.load(root_session_id)
            if (
                workflow_session is None
                or workflow_session.status is not SessionStatus.COMPLETED
                or workflow_session.provider_name != WORKFLOW_JOURNAL_PROVIDER
                or workflow_session.agent_name != target.workflow_spec.name
            ):
                raise WorkflowEvalFailure(
                    WorkflowEvalFailureCode.COMPLETION_MISSING,
                    "Workflow journal anchor is missing or invalid.",
                )
            records = await _load_workflow_eval_records(
                runtime_app,
                session_id=root_session_id,
                workflow_name=target.workflow_spec.name,
            )
            attempt_id, completion = _current_workflow_completion(
                records,
                workflow_name=target.workflow_spec.name,
            )
            terminal = WorkflowEvalTerminalEvidence(
                workflow_run_id=root_session_id,
                workflow_name=target.workflow_spec.name,
                attempt_id=attempt_id,
                completion_event=completion.event,
            )
            try:
                projected = target.result_projector(terminal)
                if inspect.isawaitable(projected):
                    projected = await projected
            except Exception as exc:
                raise WorkflowEvalFailure(
                    WorkflowEvalFailureCode.PROJECTOR_FAILED,
                    f"Workflow result projector failed ({type(exc).__name__}).",
                ) from None
            if type(projected) is not WorkflowEvalResult:
                raise WorkflowEvalFailure(
                    WorkflowEvalFailureCode.OUTPUT_INVALID,
                    "Workflow result projector returned an invalid result.",
                )
            try:
                projected = WorkflowEvalResult.model_validate(
                    projected.model_dump(mode="python", round_trip=True, warnings="none")
                )
            except (TypeError, ValueError):
                raise WorkflowEvalFailure(
                    WorkflowEvalFailureCode.OUTPUT_INVALID,
                    "Workflow result projector returned an invalid result.",
                ) from None

            latest_records = await _load_workflow_eval_records(
                runtime_app,
                session_id=root_session_id,
                workflow_name=target.workflow_spec.name,
            )
            latest_attempt_id, latest_completion = _current_workflow_completion(
                latest_records,
                workflow_name=target.workflow_spec.name,
            )
            if latest_attempt_id != attempt_id or latest_completion.event.id != completion.event.id:
                raise WorkflowEvalFailure(
                    WorkflowEvalFailureCode.ATTEMPT_SUPERSEDED,
                    "Workflow attempt changed while its result was being projected.",
                )

            final_output = projected.final_output
            structured_output = projected.structured_output
            output_evidence = WorkflowEvalOutputEvidenceV1(
                target_revision=workflow_identity.revision,
                projector_revision=workflow_identity.result_projector_revision,
                workflow_name=target.workflow_spec.name,
                attempt_id=attempt_id,
                completion_event_id=completion.event.id,
                input_message_count=len(request.messages),
                input_messages_sha256=workflow_eval_input_messages_sha256(tuple(request.messages)),
                final_output_sha256=workflow_eval_output_sha256(final_output),
                structured_output=structured_output,
            )
            capture_state = _CaptureState(bounds=SessionTrajectoryBounds(), strict=False)
            children_incomplete = _IncompleteFlag()
            if not runtime_app.session_store.supports_session_lineage:
                children_incomplete.value = True
            children = await _build_child_trajectories(
                runtime_app,
                root_session_id,
                visited={root_session_id},
                incomplete=children_incomplete,
                parent_terminal_sequence=completion.sequence,
                state=capture_state,
            )
            requirements = _collect_probe_requirements(case.assertions)
            children = await _capture_workflow_child_probes(
                runtime_app,
                children,
                requirements,
            )
            trajectory = _workflow_trajectory_from_session(
                workflow_session,
                workflow_events=tuple(record.event for record in latest_records),
                input_messages=tuple(request.messages),
                output_evidence=output_evidence,
                final_output=final_output,
                children=children,
                children_incomplete=children_incomplete.value,
                metadata=case.metadata,
            )
            if children_incomplete.value:
                raise WorkflowEvalFailure(
                    WorkflowEvalFailureCode.TARGET_FAILED,
                    "Workflow child-session evidence could not be captured completely.",
                )
            memory_alias_key = memory_evidence_key(runtime_app._request_footprint)
            first_memory_projection = await _owned_fresh_memory_attribution_projection(
                runtime_app,
                trajectory,
                bounds=selected_memory_bounds,
                lifecycle=memory_attribution_read_lifecycle,
            )
            if runtime_app.session_store.supports_session_lineage:
                await _owned_fresh_capture_revalidation(
                    runtime_app,
                    capture_state,
                    root_session_id=root_session_id,
                    root_interrupted_observed_events=(),
                    lifecycle=memory_attribution_read_lifecycle,
                )
            revalidated_memory_projection = await _owned_fresh_memory_attribution_projection(
                runtime_app,
                trajectory,
                bounds=selected_memory_bounds,
                lifecycle=memory_attribution_read_lifecycle,
            )
            if _memory_attribution_snapshot(
                first_memory_projection
            ) != _memory_attribution_snapshot(revalidated_memory_projection):
                raise WorkflowEvalFailure(
                    WorkflowEvalFailureCode.COMPLETION_CONFLICT,
                    "Workflow memory evidence changed before evaluation publication.",
                )
            trajectory = revalidated_memory_projection
            memory_attribution = eval_memory_attribution_evidence_from_trajectory(
                trajectory,
                effective_bounds=selected_memory_bounds,
                effective_source_limit=selected_memory_source_limit,
                effective_max_bytes=selected_memory_max_bytes,
                source_alias_key_id=(None if memory_alias_key is None else memory_alias_key.key_id),
                source_alias_key=None if memory_alias_key is None else memory_alias_key.key,
            )
            context = EvalContext(
                trajectory=trajectory,
                suite_id=suite_id,
                case_id=case.id,
                metadata=case.metadata,
                root_evidence_available=True,
            )
            prepared, prepared_error = _prepare_portable_evidence(
                case.assertions,
                context,
                runtime_app=runtime_app,
                memory_attribution_evidence=memory_attribution,
            )
            if public_output_preview_bytes is not None and prepared is not None:
                public_output = EvalTrialOutputPreviewV1.from_retained_evidence(
                    prepared.final_output,
                    prepared.final_output_state,
                    max_preview_bytes=public_output_preview_bytes,
                )
            assertion_results = list(
                await _evaluate_assertions_with_prepared_evidence(
                    case.assertions,
                    context,
                    portable_evidence=prepared,
                    portable_evidence_error=prepared_error,
                )
            )
            assertion_error = _assertion_diagnostic(
                assertion_results,
                EvalOutcome.ERROR,
                "Assertion evaluation failed",
            )
            if assertion_error is not None:
                run_error = assertion_error
                diagnostic_code = EvalTrialDiagnosticCode.ASSERTION_EVALUATION_FAILED
            if runtime_app.session_store.supports_session_lineage:
                await _owned_fresh_capture_revalidation(
                    runtime_app,
                    capture_state,
                    root_session_id=root_session_id,
                    root_interrupted_observed_events=(),
                    lifecycle=memory_attribution_read_lifecycle,
                )
            final_records = await _load_workflow_eval_records(
                runtime_app,
                session_id=root_session_id,
                workflow_name=target.workflow_spec.name,
            )
            final_attempt_id, final_completion = _current_workflow_completion(
                final_records,
                workflow_name=target.workflow_spec.name,
            )
            if final_attempt_id != attempt_id:
                raise WorkflowEvalFailure(
                    WorkflowEvalFailureCode.ATTEMPT_SUPERSEDED,
                    "Workflow attempt changed before evaluation publication.",
                )
            if final_completion.event.id != completion.event.id or tuple(
                (record.sequence, record.event.id) for record in final_records
            ) != tuple((record.sequence, record.event.id) for record in latest_records):
                raise WorkflowEvalFailure(
                    WorkflowEvalFailureCode.COMPLETION_CONFLICT,
                    "Workflow evidence changed before evaluation publication.",
                )
            publication_attempt_id = final_attempt_id
            publication_completion_event_id = final_completion.event.id
            publication_record_identity = tuple(
                (record.sequence, record.event.id) for record in final_records
            )
    except TimeoutError:
        run_error = f"Eval case timed out after {timeout_seconds} seconds."
        diagnostic_code = EvalTrialDiagnosticCode.CASE_TIMEOUT
        public_output = EvalTrialOutputPreviewV1.unavailable()
        trajectory = None
        final_output = ""
        structured_output = None
    except WorkflowEvalFailure as exc:
        run_error = str(exc)
        diagnostic_code = _WORKFLOW_FAILURE_DIAGNOSTICS[exc.code]
        public_output = EvalTrialOutputPreviewV1.unavailable()
        trajectory = None
        final_output = ""
        structured_output = None
    except Exception as exc:
        run_error = f"Workflow eval evidence preparation failed ({type(exc).__name__})."
        diagnostic_code = EvalTrialDiagnosticCode.EVIDENCE_PREPARATION_FAILED
        public_output = EvalTrialOutputPreviewV1.unavailable()
        trajectory = None
        final_output = ""
        structured_output = None
    finally:
        quiescence_succeeded = True
        if execution is not None and execution.close is not None:
            try:
                async with asyncio.timeout(target.close_timeout_seconds):
                    await execution.close()
            except Exception as exc:
                quiescence_succeeded = False
                run_error = f"Workflow target quiescence failed ({type(exc).__name__})."
                diagnostic_code = EvalTrialDiagnosticCode.WORKFLOW_QUIESCENCE_FAILED
                public_output = EvalTrialOutputPreviewV1.unavailable()
                trajectory = None
                final_output = ""
                structured_output = None
        if (
            quiescence_succeeded
            and capture_state is not None
            and publication_attempt_id is not None
            and publication_completion_event_id is not None
            and publication_record_identity is not None
        ):
            try:
                if runtime_app.describe().fingerprint != app_manifest.fingerprint:
                    raise WorkflowEvalFailure(
                        WorkflowEvalFailureCode.TARGET_FAILED,
                        "Workflow target application manifest changed during quiescence.",
                    )
                try:
                    settled_execution_profile_fingerprint = (
                        await runtime_app.inspect_run_execution_profile(
                            copy_run_request(target.request_base)
                        )
                    )
                except Exception as exc:
                    raise WorkflowEvalFailure(
                        WorkflowEvalFailureCode.TARGET_FAILED,
                        "Workflow target execution profile could not be revalidated "
                        f"({type(exc).__name__}).",
                    ) from None
                if settled_execution_profile_fingerprint != workflow_execution_profile_fingerprint:
                    raise WorkflowEvalFailure(
                        WorkflowEvalFailureCode.TARGET_FAILED,
                        "Workflow target execution profile changed during quiescence.",
                    )
                settled_records = await _load_workflow_eval_records(
                    runtime_app,
                    session_id=root_session_id,
                    workflow_name=target.workflow_spec.name,
                )
                settled_attempt_id, settled_completion = _current_workflow_completion(
                    settled_records,
                    workflow_name=target.workflow_spec.name,
                )
                if (
                    settled_attempt_id != publication_attempt_id
                    or settled_completion.event.id != publication_completion_event_id
                    or tuple((record.sequence, record.event.id) for record in settled_records)
                    != publication_record_identity
                ):
                    raise WorkflowEvalFailure(
                        WorkflowEvalFailureCode.COMPLETION_CONFLICT,
                        "Workflow evidence changed during target quiescence.",
                    )
                if runtime_app.session_store.supports_session_lineage:
                    await _owned_fresh_capture_revalidation(
                        runtime_app,
                        capture_state,
                        root_session_id=root_session_id,
                        root_interrupted_observed_events=(),
                        lifecycle=memory_attribution_read_lifecycle,
                    )
            except WorkflowEvalFailure as exc:
                run_error = str(exc)
                diagnostic_code = _WORKFLOW_FAILURE_DIAGNOSTICS[exc.code]
                public_output = EvalTrialOutputPreviewV1.unavailable()
                trajectory = None
                final_output = ""
                structured_output = None
            except SessionTrajectoryError as exc:
                if exc.code is SessionTrajectoryErrorCode.CLOSURE_CHANGED:
                    run_error = "Workflow evidence changed during target quiescence."
                    diagnostic_code = EvalTrialDiagnosticCode.WORKFLOW_COMPLETION_CONFLICT
                else:
                    run_error = (
                        "Workflow evidence could not be revalidated after target quiescence "
                        f"({exc.code.value})."
                    )
                    diagnostic_code = EvalTrialDiagnosticCode.WORKFLOW_QUIESCENCE_FAILED
                public_output = EvalTrialOutputPreviewV1.unavailable()
                trajectory = None
                final_output = ""
                structured_output = None
            except Exception as exc:
                run_error = (
                    "Workflow evidence could not be revalidated after target quiescence "
                    f"({type(exc).__name__})."
                )
                diagnostic_code = EvalTrialDiagnosticCode.WORKFLOW_QUIESCENCE_FAILED
                public_output = EvalTrialOutputPreviewV1.unavailable()
                trajectory = None
                final_output = ""
                structured_output = None

    if run_error is not None:
        assertion_results = list(
            _blocked_assertion_results(
                case.assertions,
                EvalOutcome.ERROR,
                run_error,
                memory_attribution_evidence=memory_attribution,
            )
        )
    completed_at = datetime.now(UTC)
    status = _trial_status(run_error, None, assertion_results)
    if diagnostic_code is None:
        diagnostic_code = (
            EvalTrialDiagnosticCode.PASSED
            if status is EvalStatus.PASSED
            else EvalTrialDiagnosticCode.ASSERTION_FAILED
        )
    public_data = (
        None
        if public_output_preview_bytes is None
        else _EvalTrialPublicData(diagnostic_code=diagnostic_code, output=public_output)
    )
    return (
        EvalTrialResult(
            trial_number=trial_number,
            status=status,
            session_id=root_session_id,
            score=_trial_score(status, assertion_results),
            final_output=final_output if retain_final_output else "",
            structured_output=structured_output if retain_final_output else None,
            assertions=tuple(assertion_results),
            error=run_error,
            evidence_complete=trajectory is not None and not trajectory.children_incomplete,
            events_count=0 if trajectory is None else _workflow_event_count(trajectory),
            usage_summary=(
                None
                if trajectory is None or trajectory.usage_summary is None
                else session_usage_summary_payload(trajectory.usage_summary)
            ),
            memory_attribution=memory_attribution,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=_duration_ms(started_at, completed_at),
            trajectory=trajectory if retain_trajectory else None,
        ),
        public_data,
    )


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
    run_stream: Callable[[RunRequest], AsyncIterator[Event]] | None = None,
    trial_request_transform: TrialRequestTransform | None = None,
    memory_attribution_bounds: MemoryAttributionBounds | None = None,
    memory_attribution_source_limit: int | None = None,
    memory_attribution_max_bytes: int | None = None,
    memory_attribution_read_lifecycle: _FreshMemoryAttributionReadLifecycle,
    run_id: str | None = None,
    workflow_target: WorkflowEvalTarget | None = None,
    workflow_instance_tracker: _WorkflowInstanceTracker | None = None,
    workflow_execution_profile_fingerprint: str | None = None,
) -> tuple[EvalTrialResult, _EvalTrialPublicData | None]:
    if workflow_target is not None:
        if run_id is None:
            raise ValueError("Workflow eval execution requires the enclosing run_id.")
        if workflow_instance_tracker is None:
            raise ValueError("Workflow eval execution requires an instance tracker.")
        if workflow_execution_profile_fingerprint is None:
            raise ValueError("Workflow eval execution requires an execution-profile identity.")
        return await _run_workflow_case_once_with_public_projection(
            workflow_target,
            case,
            trial_number=trial_number,
            suite_id=suite_id,
            run_id=run_id,
            retain_trajectory=retain_trajectory,
            retain_final_output=retain_final_output,
            timeout_seconds=timeout_seconds,
            public_output_preview_bytes=public_output_preview_bytes,
            memory_attribution_bounds=memory_attribution_bounds,
            memory_attribution_source_limit=memory_attribution_source_limit,
            memory_attribution_max_bytes=memory_attribution_max_bytes,
            memory_attribution_read_lifecycle=memory_attribution_read_lifecycle,
            workflow_instance_tracker=workflow_instance_tracker,
            workflow_execution_profile_fingerprint=(workflow_execution_profile_fingerprint),
        )
    started_at = datetime.now(UTC)
    trial_request = _isolated_trial_request(case.request)
    if trial_request_transform is not None:
        transformed = trial_request_transform(
            suite_id,
            case.id,
            trial_number,
            trial_request,
        )
        if type(transformed) is not RunRequest:
            raise TypeError("trial_request_transform must return an exact RunRequest.")
        trial_request = copy_run_request(transformed)
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
    capture_state = _CaptureState(bounds=SessionTrajectoryBounds(), strict=False)
    root_terminal_limits: TerminalSessionEvidenceLimits | None = None
    assertion_results: list[EvalAssertionResult] = []
    public_output = EvalTrialOutputPreviewV1.unavailable()
    diagnostic_code: EvalTrialDiagnosticCode | None = None
    selected_memory_bounds = MemoryAttributionBounds.model_validate(
        (memory_attribution_bounds or standard_eval_memory_attribution_bounds()).model_dump(
            mode="python"
        )
    )
    selected_memory_source_limit = (
        EVAL_MEMORY_ATTRIBUTION_MAX_SOURCES
        if memory_attribution_source_limit is None
        else memory_attribution_source_limit
    )
    selected_memory_max_bytes = (
        EVAL_MEMORY_ATTRIBUTION_MAX_BYTES
        if memory_attribution_max_bytes is None
        else memory_attribution_max_bytes
    )
    memory_alias_key = memory_evidence_key(app._request_footprint)
    memory_attribution = EvalMemoryAttributionEvidenceV1.unavailable(
        EvalMemoryEvidenceLimitation.MISSING,
        effective_bounds=selected_memory_bounds,
        effective_source_limit=selected_memory_source_limit,
        effective_max_bytes=selected_memory_max_bytes,
    )
    deadline: asyncio.Timeout | None = None

    def require_evidence_read_allowed() -> None:
        if deadline is not None and deadline.expired():
            raise _FreshEvalDeadlineExpired

    # asyncio.timeout(None) never expires, so the unbounded default shares the path.
    # Keep the deadline around the full case lifecycle: runtime execution, state/probe
    # capture, child traversal, and assertion evaluation all belong to one trial.
    try:
        async with asyncio.timeout(timeout_seconds) as deadline:
            try:
                stream = app.run(trial_request) if run_stream is None else run_stream(trial_request)
                async for event in stream:
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
                require_evidence_read_allowed()
            except TimeoutError as exc:
                # A provider-originated TimeoutError is an eval run error, not the case
                # deadline. Deadline expiry arrives as cancellation and is handled below.
                run_error = _format_exception(exc)
                diagnostic_code = EvalTrialDiagnosticCode.EXECUTION_FAILED
            except Exception as exc:
                if exception_tree_contains(
                    exc,
                    (EvalExecutionProfileChangedError, ExecutionProfileMismatchError),
                ):
                    raise
                run_error = _format_exception(exc)
                diagnostic_code = EvalTrialDiagnosticCode.EXECUTION_FAILED

            candidate_session_id = observed_session_id or trial_request.session_id
            if candidate_session_id is not None:
                try:
                    root_terminal_limits = capture_state.remaining_terminal_limits(
                        candidate_session_id
                    )
                    require_evidence_read_allowed()
                    terminal_evidence = await _load_terminal_evidence(
                        app,
                        candidate_session_id,
                        limits=root_terminal_limits,
                    )
                    require_evidence_read_allowed()
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
                            if root_terminal_limits is None:
                                raise RuntimeError(
                                    "Exact root evidence limits were lost before interrupted recovery."
                                )
                            terminal_evidence = await _load_fresh_interrupted_evidence(
                                app,
                                candidate_session_id,
                                tuple(emitted_root_events),
                                emitted_events_truncated=emitted_root_events_truncated,
                                limits=root_terminal_limits,
                                before_store_read=require_evidence_read_allowed,
                            )
                            require_evidence_read_allowed()
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
                            external_diagnostic = _external_target_diagnostic(events)
                            failure_reason = _session_failure_reason(events)
                            if external_diagnostic in _EXTERNAL_UNAVAILABLE_DIAGNOSTICS:
                                unavailable_reason = failure_reason
                            else:
                                run_error = failure_reason
                            diagnostic_code = (
                                EvalTrialDiagnosticCode.SESSION_FAILED
                                if external_diagnostic is None
                                else external_diagnostic
                            )
                    final_output = final_output_text(transcript)
                    probe_requirements = _collect_probe_requirements(case.assertions)
                    probes = await _capture_probes(app, session, probe_requirements)
                    require_evidence_read_allowed()
                    children_incomplete = _IncompleteFlag()
                    if terminal_evidence is None:
                        raise RuntimeError("Exact root evidence was lost before child capture.")
                    if root_terminal_limits is None:
                        raise RuntimeError(
                            "Exact root evidence limits were lost before child capture."
                        )
                    if session_id is None:
                        raise RuntimeError(
                            "Exact root session identity was lost before child capture."
                        )
                    capture_state.retain(terminal_evidence, limits=root_terminal_limits)
                    if not app.session_store.supports_session_lineage:
                        children_incomplete.value = True
                    children = await _build_child_trajectories(
                        app,
                        session_id,
                        visited={session_id} if session_id is not None else set(),
                        incomplete=children_incomplete,
                        parent_terminal_sequence=(
                            terminal_evidence.boundary.terminal_event_sequence
                        ),
                        state=capture_state,
                        before_store_read=require_evidence_read_allowed,
                    )
                    require_evidence_read_allowed()
                    trajectory = _trajectory_from_terminal_evidence(
                        terminal_evidence,
                        probes=probes,
                        children=children,
                        children_incomplete=children_incomplete.value,
                        metadata=case.metadata,
                    )
                    try:
                        first_memory_projection = await _owned_fresh_memory_attribution_projection(
                            app,
                            trajectory,
                            bounds=selected_memory_bounds,
                            lifecycle=memory_attribution_read_lifecycle,
                        )
                        require_evidence_read_allowed()
                        if app.session_store.supports_session_lineage:
                            try:
                                await _owned_fresh_capture_revalidation(
                                    app,
                                    capture_state,
                                    root_session_id=session_id,
                                    root_interrupted_observed_events=tuple(emitted_root_events),
                                    lifecycle=memory_attribution_read_lifecycle,
                                )
                            except SessionTrajectoryError as exc:
                                limitation = _fresh_capture_revalidation_limitation(exc)
                                memory_attribution = (
                                    eval_memory_attribution_evidence_from_trajectory(
                                        trajectory,
                                        effective_bounds=selected_memory_bounds,
                                        effective_source_limit=selected_memory_source_limit,
                                        effective_max_bytes=selected_memory_max_bytes,
                                        source_alias_key_id=(
                                            None
                                            if memory_alias_key is None
                                            else memory_alias_key.key_id
                                        ),
                                        source_alias_key=(
                                            None
                                            if memory_alias_key is None
                                            else memory_alias_key.key
                                        ),
                                        unavailable_reason=limitation,
                                    )
                                )
                                raise RuntimeError(
                                    "Fresh eval evidence changed or became unavailable while "
                                    "its durable closure was being revalidated."
                                ) from None
                        require_evidence_read_allowed()
                        revalidated_memory_projection = (
                            await _owned_fresh_memory_attribution_projection(
                                app,
                                trajectory,
                                bounds=selected_memory_bounds,
                                lifecycle=memory_attribution_read_lifecycle,
                            )
                        )
                        require_evidence_read_allowed()
                    except _FreshMemoryAttributionReadFailed:
                        trajectory = _memory_attribution_read_failed_trajectory(trajectory)
                        memory_attribution = eval_memory_attribution_evidence_from_trajectory(
                            trajectory,
                            effective_bounds=selected_memory_bounds,
                            effective_source_limit=selected_memory_source_limit,
                            effective_max_bytes=selected_memory_max_bytes,
                            source_alias_key_id=(
                                None if memory_alias_key is None else memory_alias_key.key_id
                            ),
                            source_alias_key=(
                                None if memory_alias_key is None else memory_alias_key.key
                            ),
                        )
                    else:
                        if _memory_attribution_snapshot(
                            first_memory_projection
                        ) != _memory_attribution_snapshot(revalidated_memory_projection):
                            memory_attribution = eval_memory_attribution_evidence_from_trajectory(
                                trajectory,
                                effective_bounds=selected_memory_bounds,
                                effective_source_limit=selected_memory_source_limit,
                                effective_max_bytes=selected_memory_max_bytes,
                                source_alias_key_id=(
                                    None if memory_alias_key is None else memory_alias_key.key_id
                                ),
                                source_alias_key=(
                                    None if memory_alias_key is None else memory_alias_key.key
                                ),
                                unavailable_reason=EvalMemoryEvidenceLimitation.CLOSURE_CHANGED,
                            )
                            raise RuntimeError(
                                "Memory attribution changed while fresh eval evidence was closing."
                            )
                        trajectory = revalidated_memory_projection
                        memory_attribution = eval_memory_attribution_evidence_from_trajectory(
                            trajectory,
                            effective_bounds=selected_memory_bounds,
                            effective_source_limit=selected_memory_source_limit,
                            effective_max_bytes=selected_memory_max_bytes,
                            source_alias_key_id=(
                                None if memory_alias_key is None else memory_alias_key.key_id
                            ),
                            source_alias_key=(
                                None if memory_alias_key is None else memory_alias_key.key
                            ),
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
                    memory_attribution_evidence=memory_attribution,
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
                    _blocked_assertion_results(
                        case.assertions,
                        EvalOutcome.ERROR,
                        run_error,
                        memory_attribution_evidence=memory_attribution,
                    )
                )
            elif unavailable_reason is not None:
                assertion_results = list(
                    _blocked_assertion_results(
                        case.assertions,
                        EvalOutcome.UNAVAILABLE,
                        unavailable_reason,
                        memory_attribution_evidence=memory_attribution,
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
    except (TimeoutError, _FreshEvalDeadlineExpired) as exc:
        if isinstance(exc, _FreshEvalDeadlineExpired) or (
            deadline is not None and deadline.expired()
        ):
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
        if (
            memory_attribution.completeness.value == "unavailable"
            and not memory_attribution.sources
        ):
            memory_attribution = (
                EvalMemoryAttributionEvidenceV1.unavailable(
                    EvalMemoryEvidenceLimitation.DEADLINE_EXPIRED,
                    effective_bounds=selected_memory_bounds,
                    effective_source_limit=selected_memory_source_limit,
                    effective_max_bytes=selected_memory_max_bytes,
                )
                if trajectory is None
                else eval_memory_attribution_evidence_from_trajectory(
                    trajectory,
                    effective_bounds=selected_memory_bounds,
                    effective_source_limit=selected_memory_source_limit,
                    effective_max_bytes=selected_memory_max_bytes,
                    source_alias_key_id=(
                        None if memory_alias_key is None else memory_alias_key.key_id
                    ),
                    source_alias_key=(None if memory_alias_key is None else memory_alias_key.key),
                    unavailable_reason=EvalMemoryEvidenceLimitation.DEADLINE_EXPIRED,
                )
            )
        # Never expose a partially evaluated assertion prefix after cancellation. Bind
        # blocked portable assertions to the final retained deadline evidence, not the
        # pre-timeout snapshot that may have been replaced above.
        assertion_results = list(
            _blocked_assertion_results(
                case.assertions,
                EvalOutcome.ERROR,
                run_error,
                memory_attribution_evidence=memory_attribution,
            )
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
            memory_attribution=memory_attribution,
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
    limits: TerminalSessionEvidenceLimits,
    before_store_read: Callable[[], None] | None = None,
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
        if before_store_read is not None:
            before_store_read()
        evidence = await app.session_store.load_runner_owned_interrupted_evidence(
            session_id,
            observed_events=emitted_events,
            limits=limits,
        )
        evidence = _validated_terminal_session_evidence(evidence)
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
    memory_attribution_evidence: EvalMemoryAttributionEvidenceV1 | None = None,
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
                memory_attribution_evidence=memory_attribution_evidence,
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
    *,
    memory_attribution_evidence: EvalMemoryAttributionEvidenceV1,
) -> tuple[EvalAssertionResult, ...]:
    if outcome not in (EvalOutcome.ERROR, EvalOutcome.UNAVAILABLE):
        raise ValueError("Blocked assertions require an error or unavailable outcome.")
    results: list[EvalAssertionResult] = []
    from cayu.evals.portable_assertions import _blocked_portable_assertion_result

    for assertion in assertions:
        try:
            result = _blocked_portable_assertion_result(
                assertion,
                outcome,
                message,
                memory_attribution_evidence=memory_attribution_evidence,
            )
            if result is None:
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


def _artifact_matches_probe_requirement(
    artifact: ArtifactMetadata,
    requirement: ArtifactProbeRequirement,
    session: Session,
) -> bool:
    if artifact.scope != requirement.scope:
        return False
    if artifact.scope == ArtifactScope.SESSION and artifact.session_id != session.id:
        return False
    if (
        artifact.scope == ArtifactScope.ENVIRONMENT
        and artifact.environment_name != session.environment_name
    ):
        return False
    if requirement.filename is not None and artifact.filename != requirement.filename:
        return False
    if requirement.content_type is not None and artifact.content_type != requirement.content_type:
        return False
    if requirement.minimum_bytes is not None and artifact.size_bytes < requirement.minimum_bytes:
        return False
    return not (
        requirement.maximum_bytes is not None and artifact.size_bytes > requirement.maximum_bytes
    )


async def _capture_probes(
    app: CayuApp,
    session: Session | None,
    requirements: ProbeRequirements,
) -> TrajectoryProbes:
    # Snapshot exactly what the case's assertions declared, while the environment is still
    # live, so assertions evaluate against the serializable trajectory instead of the app.
    if session is None or not (
        requirements.workspace_paths
        or requirements.workspace_structure_paths
        or requirements.artifact_scopes
        or requirements.artifact_requirements
    ):
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
    workspace_structures: dict[str, WorkspaceStructuralProbe] = {}
    workspace = getattr(env, "workspace", None)
    workspace_paths = requirements.workspace_paths | requirements.workspace_structure_paths
    if workspace_paths and workspace is not None:
        workspace_available = True
        for path in sorted(workspace_paths):
            try:
                # Cap the read so an oversized workspace file can't balloon the trajectory JSON
                # (bytes are base64-encoded there). The full size + a content hash are recorded
                # alongside so a truncated capture is still identifiable and stat-able.
                read_result = await workspace.read_bytes(
                    path,
                    max_bytes=WORKSPACE_PROBE_MAX_BYTES,
                )
                if type(read_result) is not WorkspaceReadResult:
                    raise TypeError("Workspace reads must return WorkspaceReadResult.")
                result = WorkspaceReadResult(
                    content=read_result.content,
                    total_bytes=read_result.total_bytes,
                    truncated=read_result.truncated,
                    offset=read_result.offset,
                    revision=read_result.revision,
                    sha256=read_result.sha256,
                    source_bytes_read=read_result.source_bytes_read,
                    redaction_truncated=read_result.redaction_truncated,
                )
                if (
                    result.offset != 0
                    or len(result.content) > WORKSPACE_PROBE_MAX_BYTES
                    or result.source_bytes_read != len(result.content)
                    or result.redaction_truncated
                ):
                    raise ValueError("Workspace returned data outside the requested read window.")
                content = result.content
                total_bytes = result.total_bytes
                truncated = result.truncated
                captured_sha256 = hashlib.sha256(content).hexdigest()
                if path in requirements.workspace_paths:
                    workspace_files[path] = content
                    workspace_file_stats[path] = WorkspaceFileProbe(
                        total_bytes=total_bytes,
                        truncated=truncated,
                        sha256=captured_sha256,
                    )
                if path in requirements.workspace_structure_paths:
                    if total_bytes > MAX_PORTABLE_JSON_INTEGER:
                        workspace_structures[path] = WorkspaceStructuralProbe(
                            state="unavailable",
                            digest_state="unavailable",
                        )
                    else:
                        workspace_structures[path] = WorkspaceStructuralProbe(
                            state="present",
                            total_bytes=total_bytes,
                            digest_state="limit_exceeded" if truncated else "complete",
                            sha256=None if truncated else captured_sha256,
                        )
            except FileNotFoundError:
                # Confirmed absence is negative evidence. Other failures are retained
                # separately because they did not observe whether the file exists.
                if path in requirements.workspace_paths:
                    workspace_files[path] = None
                if path in requirements.workspace_structure_paths:
                    workspace_structures[path] = WorkspaceStructuralProbe(
                        state="missing",
                        digest_state="unavailable",
                    )
            except Exception:
                if path in requirements.workspace_paths:
                    workspace_unavailable_paths.append(path)
                if path in requirements.workspace_structure_paths:
                    workspace_structures[path] = WorkspaceStructuralProbe(
                        state="unavailable",
                        digest_state="unavailable",
                    )

    artifacts_available = False
    artifacts: list[ArtifactMetadata] = []
    artifact_scopes_captured: list[ArtifactScope] = []
    artifact_scopes_truncated: list[ArtifactScope] = []
    artifact_scopes_unavailable: list[ArtifactScope] = []
    artifact_content_probes: list[ArtifactContentProbe] = []
    artifact_store = getattr(env, "artifact_store", None)
    requested_artifact_scopes = requirements.artifact_scopes | frozenset(
        requirement.scope for requirement in requirements.artifact_requirements
    )
    if requested_artifact_scopes and artifact_store is not None:
        artifacts_available = True
        seen_ids: set[str] = set()
        for scope in sorted(requested_artifact_scopes, key=str):
            try:
                listed_result = await artifact_store.list(
                    scope=scope,
                    session_id=(session.id if scope == ArtifactScope.SESSION else None),
                    environment_name=(
                        session.environment_name if scope == ArtifactScope.ENVIRONMENT else None
                    ),
                    limit=EVAL_ARTIFACT_EVIDENCE_MAX_ARTIFACTS,
                )
                if type(listed_result) is not ArtifactListResult:
                    raise TypeError("Artifact store lists must return ArtifactListResult.")
                listed = ArtifactListResult(
                    artifacts=listed_result.artifacts,
                    total_count=listed_result.total_count,
                    truncated=listed_result.truncated,
                )
                listed_ids = tuple(artifact.id for artifact in listed.artifacts)
                valid_owner_scope = all(
                    artifact.scope == scope
                    and (scope != ArtifactScope.SESSION or artifact.session_id == session.id)
                    and (
                        scope != ArtifactScope.ENVIRONMENT
                        or artifact.environment_name == session.environment_name
                    )
                    for artifact in listed.artifacts
                )
                if (
                    not valid_owner_scope
                    or len(listed.artifacts) > EVAL_ARTIFACT_EVIDENCE_MAX_ARTIFACTS
                    or len(listed_ids) != len(set(listed_ids))
                    or any(artifact_id in seen_ids for artifact_id in listed_ids)
                ):
                    artifact_scopes_unavailable.append(scope)
                    continue
            except Exception:
                artifact_scopes_unavailable.append(scope)
                continue
            if listed.truncated:
                artifact_scopes_truncated.append(scope)
            else:
                artifact_scopes_captured.append(scope)
            for artifact in listed.artifacts:
                matches_requirement = any(
                    _artifact_matches_probe_requirement(artifact, requirement, session)
                    for requirement in requirements.artifact_requirements
                )
                owned_scope_probe = scope in requirements.artifact_scopes and (
                    (scope == ArtifactScope.SESSION and artifact.session_id == session.id)
                    or (
                        scope == ArtifactScope.ENVIRONMENT
                        and artifact.environment_name == session.environment_name
                    )
                )
                if (matches_requirement or owned_scope_probe) and artifact.id not in seen_ids:
                    seen_ids.add(artifact.id)
                    artifacts.append(artifact)

        requested_reads: dict[str, tuple[bool, bool]] = {}
        for artifact in artifacts:
            for requirement in requirements.artifact_requirements:
                if not _artifact_matches_probe_requirement(artifact, requirement, session):
                    continue
                previous = requested_reads.get(artifact.id, (False, False))
                requested_reads[artifact.id] = (
                    previous[0] or requirement.capture_digest,
                    previous[1] or requirement.capture_text,
                )

        artifacts_by_id = {artifact.id: artifact for artifact in artifacts}
        for artifact_id in sorted(requested_reads):
            capture_digest, capture_text = requested_reads[artifact_id]
            if not capture_digest and not capture_text:
                continue
            artifact = artifacts_by_id[artifact_id]
            try:
                read = copy_artifact_read_result(
                    await artifact_store.read_bytes(
                        artifact_id,
                        max_bytes=ARTIFACT_PROBE_MAX_BYTES,
                    ),
                    expected_artifact_id=artifact_id,
                    max_content_bytes=ARTIFACT_PROBE_MAX_BYTES,
                )
                if read.metadata != artifact:
                    raise ValueError(
                        "Artifact store read metadata does not match its scoped listing."
                    )
                if read.source_bytes_read != len(read.content) or read.redaction_truncated:
                    raise ValueError("Artifact store returned an incomplete content projection.")
            except Exception:
                artifact_content_probes.append(
                    ArtifactContentProbe(
                        artifact_id=artifact_id,
                        digest_state="unavailable",
                        text_state="unavailable",
                    )
                )
                continue

            digest_state = "limit_exceeded" if read.truncated else "complete"
            sha256 = None if read.truncated else hashlib.sha256(read.content).hexdigest()
            text_state = "unavailable"
            text = None
            text_supported = _artifact_text_media_type_supported(artifact.content_type)
            if capture_text:
                if not text_supported:
                    text_state = "unsupported"
                elif read.truncated:
                    text_state = "truncated"
                else:
                    try:
                        original = read.content.decode("utf-8")
                    except UnicodeError:
                        text_state = "malformed"
                    else:
                        try:
                            redaction = app.redact_utf8_head(
                                read.content,
                                max_bytes=ARTIFACT_PUBLIC_TEXT_MAX_BYTES,
                                source_complete=True,
                            )
                            if type(redaction) is not tuple or len(redaction) != 2:
                                raise TypeError("Artifact redaction must return a two-item tuple.")
                            redacted, redaction_truncated = redaction
                            if type(redacted) is not str or type(redaction_truncated) is not bool:
                                raise TypeError(
                                    "Artifact redaction must return text and a boolean."
                                )
                            if len(redacted.encode("utf-8")) > ARTIFACT_PUBLIC_TEXT_MAX_BYTES:
                                raise ValueError("Artifact redaction exceeded its byte boundary.")
                        except Exception:
                            text_state = "unavailable"
                        else:
                            if redaction_truncated:
                                text_state = "truncated"
                            elif redacted != original:
                                text_state = "redacted"
                            else:
                                try:
                                    text = require_durable_text(redacted, "artifact text")
                                except ValueError:
                                    text_state = "malformed"
                                    text = None
                                else:
                                    text_state = "available"
            artifact_content_probes.append(
                ArtifactContentProbe(
                    artifact_id=artifact_id,
                    digest_state=digest_state,
                    sha256=sha256,
                    text_state=text_state,
                    text=text,
                )
            )

    return TrajectoryProbes(
        workspace_available=workspace_available,
        workspace_files=workspace_files,
        workspace_file_stats=workspace_file_stats,
        workspace_unavailable_paths=tuple(workspace_unavailable_paths),
        workspace_structures=workspace_structures,
        artifacts_available=artifacts_available,
        artifact_scopes_captured=tuple(artifact_scopes_captured),
        artifact_scopes_truncated=tuple(artifact_scopes_truncated),
        artifact_scopes_unavailable=tuple(artifact_scopes_unavailable),
        artifacts=tuple(artifacts),
        artifact_content_probes=tuple(artifact_content_probes),
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
    trial_policy: EvalSuiteTrialPolicyV1 | None = None,
) -> EvalCaseResult:
    retained = tuple(results)
    validated_policy = (
        EvalSuiteTrialPolicyV1.create(trial_count=len(retained))
        if trial_policy is None
        else trial_policy
    )
    return EvalCaseResult.from_trials(
        case_id=case.id,
        authored_session_id=case.request.session_id,
        trials=retained,
        trial_policy=validated_policy,
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


_EXTERNAL_TARGET_DIAGNOSTIC_BY_PROVIDER_CODE = {
    "external_container_unavailable": EvalTrialDiagnosticCode.EXTERNAL_TARGET_UNAVAILABLE,
    "external_container_cancelled": EvalTrialDiagnosticCode.EXTERNAL_TARGET_CANCELLED,
    "external_container_unknown": EvalTrialDiagnosticCode.EXTERNAL_TARGET_UNKNOWN,
    "external_container_incomplete": EvalTrialDiagnosticCode.EXTERNAL_TARGET_INCOMPLETE,
    "external_container_identity_mismatch": (
        EvalTrialDiagnosticCode.EXTERNAL_TARGET_IDENTITY_MISMATCH
    ),
    "external_container_failed": EvalTrialDiagnosticCode.EXTERNAL_TARGET_FAILED,
    "external_container_oom_killed": EvalTrialDiagnosticCode.EXTERNAL_TARGET_FAILED,
}
_EXTERNAL_UNAVAILABLE_DIAGNOSTICS = frozenset(
    {
        EvalTrialDiagnosticCode.EXTERNAL_TARGET_UNAVAILABLE,
        EvalTrialDiagnosticCode.EXTERNAL_TARGET_CANCELLED,
        EvalTrialDiagnosticCode.EXTERNAL_TARGET_UNKNOWN,
        EvalTrialDiagnosticCode.EXTERNAL_TARGET_INCOMPLETE,
        EvalTrialDiagnosticCode.EXTERNAL_TARGET_IDENTITY_MISMATCH,
    }
)


def _external_target_diagnostic(events: Iterable[Event]) -> EvalTrialDiagnosticCode | None:
    for event in reversed(tuple(events)):
        if event.type != EventType.MODEL_ERROR:
            continue
        code = event.payload.get("provider_error_code")
        if type(code) is str:
            diagnostic = _EXTERNAL_TARGET_DIAGNOSTIC_BY_PROVIDER_CODE.get(code)
            if diagnostic is not None:
                return diagnostic
    return None


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
