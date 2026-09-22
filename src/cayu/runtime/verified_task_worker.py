"""Verifier-aware task orchestration through existing runtime owners."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypeVar, cast

from pydantic import BaseModel, ConfigDict, field_validator

from cayu._exception_groups import exception_cause, iter_exception_tree, set_exception_cause
from cayu._task_wait import (
    CapturedAwaitableOutcome,
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    consume_pending_task_cancellation,
    restore_task_cancellation_requests,
    unexpected_child_cancellation_error,
)
from cayu._validation import require_clean_nonblank, revalidate_model_input
from cayu.deadlines import ExecutionDeadline, ExecutionDeadlineExceeded, effective_deadline
from cayu.runtime._completion_verifier_coordinator import CompletionVerifierOwnedExecution
from cayu.runtime._durable_worker_loop import (
    DurableWorkerStep,
    run_durable_lease_heartbeat,
    run_durable_worker_loop,
    validate_worker_interval,
)
from cayu.runtime._session_request_boundary import prepare_run_request
from cayu.runtime._task_store_operation_boundary import (
    capture_sensitive_result_validation,
    capture_task_store_operation,
    raise_task_store_operation_failure,
    task_store_verified_task_worker_capability_is_complete,
)
from cayu.runtime._verified_task_decision_coordinator import verified_task_operation_id
from cayu.runtime.completion_verifiers import CompletionVerifierExecutionRequest
from cayu.runtime.work_attempt_lifecycle import (
    WorkAttemptLifecycleSettlement,
    WorkAttemptPreEntrySettlementEvidence,
    WorkAttemptPreparationHold,
    WorkAttemptStopReason,
    pre_entry_settlement_authority,
    runtime_stop_reason_for_execution_stop,
    work_attempt_admission_authority_sha256,
)
from cayu.sessions.base import RunRequest, SessionStatus
from cayu.tasks.admission import (
    WORK_ATTEMPT_ADMISSION_LEASE_MAX_SECONDS,
    WorkAttemptAdmission,
    WorkAttemptAdmissionConflict,
    WorkAttemptAdmissionPrepare,
    WorkAttemptAdmissionState,
    WorkAttemptClaimRenewalRequest,
    WorkAttemptExecutionClaimLost,
    WorkAttemptExecutionRequest,
    WorkAttemptProposalRequest,
    WorkAttemptRecoveryRequest,
    WorkAttemptRecoveryRequired,
    WorkAttemptRunRequest,
    _GroupExecutionEntryRefused,
    require_work_attempt_admission_result,
    work_attempt_admission_prepare_sha256,
)
from cayu.tasks.base import (
    Task,
    TaskAggregateFilter,
    TaskClaimLost,
    TaskQuery,
    TaskStatus,
    WorkAttemptLifecycleReceipt,
    WorkAttemptPreparationHoldReceipt,
    copy_task,
    copy_task_query,
)
from cayu.tasks.contracts import (
    CompletionDecision,
    CompletionProposal,
    CompletionProposalCreate,
    CompletionVerificationClaim,
    CompletionVerificationClaimLost,
    WorkAttempt,
    WorkCompletionConflict,
    WorkContract,
    WorkContractRef,
    _GroupVerificationAdmissionRefused,
    completion_proposal_request_sha256,
    copy_completion_decision,
    copy_completion_proposal,
    copy_work_contract,
    require_bounded_work_completion_document,
)
from cayu.tasks.groups import TaskGroupResultResolutionPending

if TYPE_CHECKING:
    from cayu.applications import CayuApp

_T = TypeVar("_T")
_VERIFIER_LEASE_SECONDS = 300
_VERIFIER_TIMEOUT_SECONDS = 30.0


@dataclass(slots=True)
class _RetainedVerifier:
    execution: CompletionVerifierOwnedExecution
    observation: asyncio.Task[CapturedAwaitableOutcome[tuple[BaseException | None]]] | None = None


@dataclass(slots=True)
class _PreparationAdmission:
    request: WorkAttemptExecutionRequest
    contract: WorkContractRef
    session_id: str
    source_request_sha256: str
    execution_owner_id: str
    operation: asyncio.Task[CapturedAwaitableOutcome[WorkAttemptAdmission]]


@dataclass(slots=True)
class _RetainedPreparationSettlement:
    claim: Task
    started: Task | None = None
    settlement: asyncio.Task[CapturedAwaitableOutcome[Task | None]] | None = None
    hold: WorkAttemptPreparationHold | None = None
    admission: _PreparationAdmission | None = None


class _PreparationOwnershipTransferred(WorkAttemptRecoveryRequired):
    """Only store-authenticated admission readback produces this local handoff."""


async def _capture_owned_task_outcome(
    factory: Callable[[], Awaitable[_T]],
) -> CapturedAwaitableOutcome[_T]:
    """Carry owned child cancellation as data to the outward worker boundary.

    Inner cleanup helpers may restore the cancellation they temporarily
    consumed. Letting that pending request override this boxed return would
    discard its cleanup cause when asyncio finishes the Task. Only a directly
    captured current cancellation is consumed here; historical causes/counts
    never change an ordinary outcome's classification.
    """
    outcome = await capture_awaitable_outcome(factory)
    if isinstance(outcome.error, asyncio.CancelledError):
        consume_pending_task_cancellation(outcome.error)
    return outcome


def _merge_worker_failures(
    primary: BaseException | None, secondary: BaseException | None
) -> BaseException | None:
    """Preserve ordered failures while keeping current cancellation outward.

    Causes are inspected only for identity deduplication, never to classify an
    ordinary error as cancellation because of historical causal evidence.
    """
    if primary is None:
        return secondary
    if secondary is None:
        return primary

    def contains(root, target):
        pending = [root]
        seen = set()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            if current is target:
                return True
            pending.extend(item for item in iter_exception_tree(current) if item is not current)
            cause = exception_cause(current)
            if cause is not None:
                pending.append(cause)
        return False

    if contains(primary, secondary):
        return primary
    if contains(secondary, primary):
        return secondary
    if isinstance(primary, asyncio.CancelledError):
        set_exception_cause(primary, _merge_worker_failures(exception_cause(primary), secondary))
        return primary
    if isinstance(secondary, asyncio.CancelledError):
        set_exception_cause(secondary, _merge_worker_failures(primary, exception_cause(secondary)))
        return secondary
    return BaseExceptionGroup("Verified task owned failures", [primary, secondary])


@dataclass(frozen=True, slots=True)
class VerifiedTaskPreparationContext:
    """Detached application data, never a live queue or session lease."""

    task: Task
    contract: WorkContract
    session_id: str


@dataclass(frozen=True, slots=True)
class VerifiedTaskProposalContext:
    """Explicit proposal coordinates after governed execution has quiesced."""

    task: Task
    contract: WorkContract
    attempt: WorkAttempt
    proposal_id: str


class VerifiedTaskHandlerReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    proposal: CompletionProposalCreate

    @field_validator("proposal", mode="before")
    @classmethod
    def copy_proposal(cls, value: object) -> object:
        return revalidate_model_input(value, CompletionProposalCreate)


class VerifiedTaskHandler(ABC):
    """Deterministic, read-only callbacks; retry may call either again.

    Domain effects belong to the governed session. A final model message is
    neither this report nor a verifier decision.
    """

    @abstractmethod
    async def prepare(self, context: VerifiedTaskPreparationContext) -> RunRequest: ...

    @abstractmethod
    async def propose(self, context: VerifiedTaskProposalContext) -> VerifiedTaskHandlerReport: ...


class VerifiedTaskWorkerDraining(RuntimeError):
    """Owned work is still settling; retain the worker and retry aclose()."""


class _LeaseOwner:
    """Local acknowledged lease projection, serialized with durable handoff.

    The store revalidates every mutation. This object is not durable authority
    and cannot authenticate an expired or replaced claim.
    """

    def __init__(self, worker: VerifiedTaskWorker, state: Task | WorkAttemptAdmission) -> None:
        self.worker = worker
        self.state: Task | WorkAttemptAdmission | _RetainedPreparationSettlement = state
        self.lock = asyncio.Lock()

    async def heartbeat(self) -> bool:
        async with self.lock:
            state = self.state
            if isinstance(state, _RetainedPreparationSettlement):
                # Callback work has ended and its exact hold owns the handoff.
                # Renewing now would invalidate the retained lease tuple after
                # a failed or still-pending publication.
                return True
            if type(state) is WorkAttemptAdmission:
                if state.state is WorkAttemptAdmissionState.RELEASED:
                    # A following continuation may acquire the next admission
                    # under this same owner. Only phase completion stops us.
                    return False
                self.state = await self.worker.app.renew_work_attempt_claim(
                    WorkAttemptClaimRenewalRequest(
                        admission_id=state.admission_id,
                        claim_id=state.claim.claim_id,
                        worker_id=state.claim.worker_id,
                        generation=state.claim.generation,
                        lease_seconds=self.worker.lease_seconds,
                    )
                )
                return False
            assert type(state) is Task
            if state.status in {TaskStatus.NEEDS_ATTENTION, TaskStatus.CANCELLED}:
                # Only the validated preparation-hold receipt installs this
                # state; no heartbeat may race that completed handoff.
                return True
            if state.lease_expires_at is None:
                raise TaskClaimLost("Verified task preparation lost its lease.")
            expected_lease = state.lease_expires_at
            updated = await self.worker._operation(
                lambda: self.worker.store.heartbeat(
                    state.id,
                    self.worker.worker_id,
                    lease_expires_at=expected_lease,
                    extend_seconds=self.worker.lease_seconds,
                ),
                name="Verified task preparation heartbeat",
                mutation="heartbeat",
            )
            copied = self.worker._validate(lambda: copy_task(updated), "Verified task heartbeat")
            if (
                copied.id != state.id
                or copied.worker_id != state.worker_id
                or copied.status is not TaskStatus.CLAIMED
                or copied.work_contract != state.work_contract
                or copied.lease_expires_at is None
            ):
                raise TaskClaimLost("Verified task heartbeat returned conflicting authority.")
            self.state = copied
            return False


class VerifiedTaskWorker:
    def __init__(
        self,
        app: CayuApp,
        handler: VerifiedTaskHandler,
        *,
        worker_id: str,
        query: TaskQuery | None = None,
        lease_seconds: int = 300,
        poll_interval_s: float = 1.0,
        max_elapsed_seconds: int = 3600,
        callback_timeout_seconds: float = 30.0,
    ) -> None:
        if app.task_store is None:
            raise ValueError("VerifiedTaskWorker requires a task store.")
        if not isinstance(handler, VerifiedTaskHandler):
            raise TypeError("handler must implement VerifiedTaskHandler.")
        if (
            type(lease_seconds) is not int
            or not 1 <= lease_seconds <= WORK_ATTEMPT_ADMISSION_LEASE_MAX_SECONDS
        ):
            raise ValueError("lease_seconds is outside its supported bounds.")
        if type(max_elapsed_seconds) is not int or not 1 <= max_elapsed_seconds <= 86400:
            raise ValueError("max_elapsed_seconds must be between 1 and 86400.")
        validate_worker_interval(poll_interval_s, "poll_interval_s")
        validate_worker_interval(callback_timeout_seconds, "callback_timeout_seconds")
        if callback_timeout_seconds >= lease_seconds:
            raise ValueError("callback_timeout_seconds must be shorter than lease_seconds.")
        worker_id = require_clean_nonblank(worker_id, "worker_id")
        if app.redact_json(worker_id) != worker_id:
            raise ValueError("worker_id contains a workload secret.")
        selected = copy_task_query(query)
        if selected.has_work_contract is False:
            raise ValueError("VerifiedTaskWorker requires the contracted queue.")
        if (
            selected.q is not None
            or selected.offset
            or selected.status not in {None, TaskStatus.PENDING}
        ):
            raise ValueError("VerifiedTaskWorker accepts queue scope filters, not display filters.")
        self.app = app
        self.store = app.task_store
        self.handler = handler
        self.worker_id = worker_id
        self.query = selected.model_copy(update={"has_work_contract": True})
        self.lease_seconds = lease_seconds
        self.poll_interval_s = poll_interval_s
        self.max_elapsed_seconds = max_elapsed_seconds
        self.callback_timeout_seconds = callback_timeout_seconds
        self._running: asyncio.Task[CapturedAwaitableOutcome[int]] | None = None
        self._verification: _RetainedVerifier | None = None
        self._preparation_settlement: _RetainedPreparationSettlement | None = None
        self._pre_entry_settlement: WorkAttemptLifecycleSettlement | None = None
        self._pre_entry_publication: asyncio.Task | None = None
        self._closed = False
        self._recovery_cursor: str | None = None
        from cayu.tasks._group_maintenance import TaskGroupMaintenance

        self._group_maintenance = TaskGroupMaintenance.for_store(self.store)

    def _validate(self, factory: Callable[[], _T], name: str) -> _T:
        captured = capture_sensitive_result_validation(
            lambda factory=factory: (factory(),),
            operation_name=name,
            redactor=self.app._secret_redactor,
        )
        del factory
        if captured.failure is not None:
            raise_task_store_operation_failure(captured.failure)
        if captured.result is None:
            raise RuntimeError("Verified worker validation returned no result.")
        return captured.result[0]

    async def _operation(
        self,
        factory: Callable[[], Awaitable[_T]],
        *,
        name: str,
        mutation: str | None = None,
    ) -> _T:
        outcome = await capture_task_store_operation(
            lambda factory=factory: self._boxed(factory),
            operation_name=name,
            redactor=self.app._secret_redactor,
            mutation_store=self.store if mutation is not None else None,
            mutation_method_name=mutation,
        )
        del factory
        if outcome.failure is not None:
            raise_task_store_operation_failure(outcome.failure)
        if outcome.result is None:
            raise RuntimeError("Verified worker operation returned no result.")
        return outcome.result[0]

    @staticmethod
    async def _boxed(factory: Callable[[], Awaitable[_T]]) -> tuple[_T]:
        return (await factory(),)

    async def _capture_callback(
        self, factory: Callable[[], Awaitable[_T]], name: str
    ) -> CapturedAwaitableOutcome[_T]:
        # Callback cancellation is owned by _callback, not the storage mutation
        # boundary. Reusing that await boundary would manufacture a second
        # cancellation alongside the one this owner already retains.
        captured = await _capture_owned_task_outcome(factory)
        if captured.error is None or isinstance(captured.error, asyncio.CancelledError):
            return captured

        def detach(error=captured.error):
            raise error

        checked = capture_sensitive_result_validation(
            detach, operation_name=name, redactor=self.app._secret_redactor
        )
        del captured, detach, factory
        return CapturedAwaitableOutcome(error=checked.failure)

    async def _group_cancellation_requested(self, task_id: str) -> bool:
        value = await self._operation(
            lambda: self.store._task_group_cancellation_requested(task_id),
            name="Verified task group cancellation observation",
        )
        if type(value) is not bool:
            raise WorkCompletionConflict(
                "Group cancellation observation must be authoritative boolean evidence."
            )
        return value

    async def _group_owns_callback(self, task: Task) -> bool:
        if not self.store.supports_task_group_quiescence:
            return False
        value = await self._operation(
            lambda: self.store._task_group_retains_execution(task.id),
            name="Verified callback group ownership",
        )
        if type(value) is not bool:
            raise WorkCompletionConflict(
                "Verified callback requires authoritative group ownership."
            )
        return value

    async def _callback(
        self, factory: Callable[[], Awaitable[_T]], name: str, *, retain_until_settled: bool = False
    ) -> _T:
        child = asyncio.create_task(self._capture_callback(factory, name))
        cancellation = None
        expired = False
        try:
            done, _ = await asyncio.wait({child}, timeout=self.callback_timeout_seconds)
            expired = not done
        except asyncio.CancelledError as error:
            cancellation = error
        if (expired or cancellation is not None) and not retain_until_settled:
            child.cancel()
        outcome = await await_shielded_task_outcome(child, cancellation=cancellation)
        restore_task_cancellation_requests(
            outcome.cancellation_requests_consumed,
            cancellation=outcome.cancellation,
        )
        callback_error = outcome.error or (None if outcome.result is None else outcome.result.error)
        if outcome.cancellation is not None:
            if callback_error is not None and not isinstance(
                callback_error, asyncio.CancelledError
            ):
                raise outcome.cancellation from callback_error
            raise outcome.cancellation
        if expired:
            # A late successful callback is discarded, even if it suppressed
            # cancellation. The outer lease heartbeat remains owned until here.
            raise TimeoutError("Verified task callback exceeded its deadline.") from (
                None if isinstance(callback_error, asyncio.CancelledError) else callback_error
            )
        if isinstance(callback_error, asyncio.CancelledError):
            raise unexpected_child_cancellation_error(callback_error, operation=name)
        if callback_error is not None:
            raise callback_error
        if outcome.result is None or outcome.result.result is None:
            raise TypeError("Verified task callback returned no typed result.")
        return outcome.result.result

    async def _settle_cancelled_group_callback(
        self, owner: _LeaseOwner, contract: WorkContract, cancellation: asyncio.CancelledError
    ) -> None:
        """Retain current cancellation while recording a naturally drained callback.

        Only preparation/proposal callers use this after their callback owner
        has settled. Cancellation of model execution is not this evidence.
        """

        async def settle():
            state = owner.state
            assert isinstance(state, (Task, WorkAttemptAdmission))
            if isinstance(state, Task):
                async with owner.lock:
                    # Retain the naturally drained callback's exact proof even
                    # if the election readback or acknowledgement write fails.
                    state = self._retain_drained_preparation(owner)
                    if await self._group_cancellation_requested(state.id):
                        owner.state = state
                        await self._hold_preparation_locked(
                            owner, contract, "work_contract_group_cancelled"
                        )
                    else:
                        await self._consume_preparation_settlement(self.callback_timeout_seconds)
                        owner.state = state
            elif await self._group_cancellation_requested(state.task_id):
                await self._settle_runtime_stop(owner, "work_contract_group_cancelled")

        work = asyncio.create_task(_capture_owned_task_outcome(settle))
        result = await await_shielded_task_outcome(work, cancellation=cancellation)
        cleanup = result.error or (None if result.result is None else result.result.error)
        failure = _merge_worker_failures(cancellation, cleanup)
        failure = _merge_worker_failures(failure, result.cancellation)
        restore_task_cancellation_requests(
            result.cancellation_requests_consumed, cancellation=cancellation
        )
        assert failure is not None
        raise failure from exception_cause(failure)

    def _retain_drained_preparation(self, owner: _LeaseOwner) -> Task:
        """Transfer a completed callback's proof while holding the lease lock."""
        state = owner.state
        assert isinstance(state, Task)
        assert self._preparation_settlement is None
        retained = _RetainedPreparationSettlement(claim=copy_task(state), started=copy_task(state))
        self._preparation_settlement = retained
        owner.state = retained
        return state

    async def _settle_preparation_execution(self, task: Task) -> None:
        authority = copy_task(task)
        await self._operation(
            lambda: self.store._settle_task_group_execution(authority),
            name="Verified preparation execution settlement",
            mutation="_settle_task_group_execution",
        )

    async def _reconcile_preparation_entry(
        self, retained: _RetainedPreparationSettlement
    ) -> Task | None:
        admitted = None
        if retained.admission is not None:
            # Cancellation only stops observation. Absence is not non-admission
            # proof until the one original operation has naturally settled.
            outcome = await asyncio.shield(retained.admission.operation)
            admitted = outcome.result
        if retained.hold is not None or retained.admission is not None:
            admission = await self._operation(
                lambda: self.store.load_latest_work_attempt_admission(retained.claim.id),
                name="Preparation handoff admission reconciliation",
            )
            if admission is not None:
                # A failed admission acknowledgement is not non-dispatch proof.
                # The admission's durable recovery owner must settle that work.
                def require_admission(value=admission):
                    recorded = require_work_attempt_admission_result(
                        value, operation_name="Preparation ownership transfer"
                    )
                    if (
                        recorded.task_id != retained.claim.id
                        or recorded.contract != retained.claim.work_contract
                    ):
                        raise WorkAttemptAdmissionConflict(
                            "Preparation admission changed authority."
                        )
                    if retained.admission is not None:
                        self._require_preparation_admission(recorded, retained.admission)

                try:
                    self._validate(require_admission, "Preparation ownership transfer")
                finally:
                    del admission, require_admission
                raise _PreparationOwnershipTransferred(
                    "Preparation acquired admission ownership; recover the admitted attempt."
                )
            if admitted is not None:
                raise WorkAttemptAdmissionConflict("Acknowledged preparation admission is missing.")
            if retained.hold is not None:
                return await self._publish_preparation_hold(retained.hold)
        if retained.started is not None:
            # Settlement may itself commit before acknowledgement loss. Keep
            # the original execution tuple, not the terminal task's cleared lease.
            await self._settle_preparation_execution(retained.started)
            return
        claim = retained.claim
        value = await self._operation(
            lambda: self.store.load_task(claim.id),
            name="Undispatched preparation readback",
        )

        def require_exact(value=value):
            if type(value) is not Task:
                raise WorkCompletionConflict("Preparation readback lost its exact claim.")
            recorded = copy_task(value)
            # Election or an external cancellation may have changed stop
            # diagnostics, but must not replace the claim or attach execution.
            if (
                claim.started_at is not None
                or recorded.model_copy(
                    update={
                        "started_at": claim.started_at,
                        "updated_at": claim.updated_at,
                        "status_reason": claim.status_reason,
                        "status_payload": claim.status_payload,
                        "error": claim.error,
                    }
                )
                != claim
            ):
                raise WorkCompletionConflict("Preparation readback changed its exact claim.")
            return recorded

        try:
            recorded = self._validate(require_exact, "Undispatched preparation authority")
        finally:
            del value, require_exact
        if recorded.started_at is not None:
            retained.started = recorded
            await self._settle_preparation_execution(recorded)

    async def _consume_preparation_settlement(self, timeout_s: float) -> Task | None:
        retained = self._preparation_settlement
        if retained is None:
            return
        if retained.settlement is None:
            retained.settlement = asyncio.create_task(
                _capture_owned_task_outcome(lambda: self._reconcile_preparation_entry(retained)),
                name="cayu-undispatched-preparation-settlement",
            )
        settlement = retained.settlement
        outcome = await await_shielded_task_outcome(
            settlement, timeout_s=timeout_s, timeout_after_cancellation_s=0
        )
        captured = outcome.result
        failure = outcome.error or (None if captured is None else captured.error)
        if self._preparation_settlement is not retained or retained.settlement is not settlement:
            failure = None
        elif settlement.done():
            if (captured is not None and failure is None) or type(
                failure
            ) is _PreparationOwnershipTransferred:
                self._preparation_settlement = None
            else:
                retained.settlement = None
        if outcome.timed_out:
            failure = _merge_worker_failures(
                failure, VerifiedTaskWorkerDraining("Preparation entry settlement is draining.")
            )
        if outcome.cancellation is not None:
            failure = _merge_worker_failures(failure, outcome.cancellation)
            restore_task_cancellation_requests(
                outcome.cancellation_requests_consumed, cancellation=outcome.cancellation
            )
        if failure is not None:
            raise failure from exception_cause(failure)
        return None if captured is None else captured.result

    def _require_preparation_admission(
        self, value: object, expected: _PreparationAdmission
    ) -> WorkAttemptAdmission:
        admission = require_work_attempt_admission_result(
            value, operation_name="Preparation admission acknowledgement"
        )
        request = expected.request
        assert request.task_id is not None
        # Authenticate the immutable initial intent. A recovery owner may
        # already have replaced the mutable execution claim; that is not a
        # conflicting admission and does not restore preparation ownership.
        prepare = WorkAttemptAdmissionPrepare(
            admission_id=request.admission_id,
            claim_id=request.claim_id,
            attempt_id=request.attempt_id,
            task_id=request.task_id,
            session_id=expected.session_id,
            interaction_id=request.interaction_id,
            worker_id=request.worker_id,
            task_lease_expires_at=request.task_lease_expires_at,
            execution_owner_id=expected.execution_owner_id,
            generation=request.generation,
            lease_seconds=request.lease_seconds,
            kind="initial",
            source_request_sha256=expected.source_request_sha256,
            contract=expected.contract,
            session_invocation=admission.session_invocation,
            source_execution_profile_fingerprint=admission.source_execution_profile_fingerprint,
            run_semantics=admission.run_semantics,
            source_request=admission.source_request,
        )
        if (
            admission.admission_id != request.admission_id
            or admission.task_id != request.task_id
            or admission.session_id != expected.session_id
            or admission.attempt_id != request.attempt_id
            or admission.interaction_id != request.interaction_id
            or admission.kind != "initial"
            or admission.contract != expected.contract
            or admission.source_request_sha256 != expected.source_request_sha256
            or admission.prepare_request_sha256 != work_attempt_admission_prepare_sha256(prepare)
        ):
            raise WorkAttemptAdmissionConflict("Preparation admission changed its exact request.")
        return admission

    async def _admit_prepared(
        self, request: RunRequest, execution: WorkAttemptExecutionRequest
    ) -> WorkAttemptAdmission:
        retained = self._preparation_settlement
        if retained is None:
            return await self.app.admit_work_attempt(request, execution=execution)
        request = self.app._with_application_run_defaults(request)
        assert request.session_id is not None
        assert retained.claim.work_contract is not None
        source = self._validate(
            lambda: self.app._session_engine.work_attempt_source_request_sha256(
                request, kind="initial"
            ),
            "Preparation admission source authority",
        )
        expected = _PreparationAdmission(
            request=execution.model_copy(deep=True),
            contract=retained.claim.work_contract.model_copy(deep=True),
            session_id=request.session_id,
            source_request_sha256=source,
            execution_owner_id=self.app._current_work_attempt_execution_owner_id(),
            operation=asyncio.create_task(
                _capture_owned_task_outcome(
                    lambda: self.app.admit_work_attempt(request, execution=execution)
                ),
                name="cayu-preparation-admission",
            ),
        )
        retained.admission = expected
        outcome = await asyncio.shield(expected.operation)
        if outcome.error is not None:
            if isinstance(outcome.error, asyncio.CancelledError):
                raise unexpected_child_cancellation_error(
                    outcome.error, operation="Preparation admission"
                )
            raise outcome.error from exception_cause(outcome.error)
        admission = self._validate(
            lambda: self._require_preparation_admission(outcome.result, expected),
            "Preparation admission acknowledgement",
        )
        self._preparation_settlement = None
        return admission

    async def _with_heartbeat(self, owner: _LeaseOwner, action: Callable[[], Awaitable[_T]]) -> _T:
        stop = asyncio.Event()

        async def after(update: bool) -> bool | None:
            return True if update else None

        heartbeat = asyncio.create_task(
            _capture_owned_task_outcome(
                lambda: run_durable_lease_heartbeat(
                    owner.heartbeat,
                    lease_seconds=self.lease_seconds,
                    stop=stop,
                    stopped_outcome=True,
                    after_heartbeat=after,
                )
            )
        )
        work = asyncio.create_task(_capture_owned_task_outcome(action))
        failure = None
        result = None
        consumed = 0
        try:
            done, _ = await asyncio.wait({work, heartbeat}, return_when=asyncio.FIRST_COMPLETED)
            if heartbeat in done:
                heartbeat_result = heartbeat.result()
                if heartbeat_result.error is not None:
                    raise heartbeat_result.error
            completed = await asyncio.shield(work)
            if completed.error is not None:
                raise completed.error
            result = completed.result
        except BaseException as error:
            failure = error
            forwarded = work.cancel()
            settled = await await_shielded_task_outcome(
                work,
                cancellation=error if isinstance(error, asyncio.CancelledError) else None,
            )
            consumed += settled.cancellation_requests_consumed
            child_error = settled.error or (
                None if settled.result is None else settled.result.error
            )
            if forwarded and isinstance(child_error, asyncio.CancelledError):
                # This is the cancellation we forwarded, not a second control
                # signal. Its cleanup cause still belongs in the final outcome.
                child_error = exception_cause(child_error)
            failure = _merge_worker_failures(failure, child_error)
            failure = _merge_worker_failures(failure, settled.cancellation)
        finally:
            stop.set()
            settled_heartbeat = await await_shielded_task_outcome(heartbeat)
            consumed += settled_heartbeat.cancellation_requests_consumed
            heartbeat_error = settled_heartbeat.error or (
                None if settled_heartbeat.result is None else settled_heartbeat.result.error
            )
            failure = _merge_worker_failures(failure, heartbeat_error)
            failure = _merge_worker_failures(failure, settled_heartbeat.cancellation)
        restore_task_cancellation_requests(
            consumed, cancellation=failure if isinstance(failure, asyncio.CancelledError) else None
        )
        if failure is not None:
            raise failure from exception_cause(failure)
        return cast("_T", result)

    async def run(self, stop: asyncio.Event | None = None, max_tasks: int | None = None) -> int:
        if self._closed:
            raise RuntimeError("VerifiedTaskWorker is closed.")
        if self._running is not None:
            raise VerifiedTaskWorkerDraining("The prior worker run must settle before another run.")
        if max_tasks is not None and (type(max_tasks) is not int or max_tasks < 0):
            raise ValueError("max_tasks must be a nonnegative integer.")
        if self._pre_entry_settlement is not None:
            await self._consume_pre_entry_settlement(self.callback_timeout_seconds)
            if self._closed or self._running is not None:
                raise VerifiedTaskWorkerDraining("Worker ownership changed during settlement.")
        if self._preparation_settlement is not None:
            await self._consume_preparation_settlement(self.callback_timeout_seconds)
            if self._closed:
                raise RuntimeError("VerifiedTaskWorker is closed.")
            if self._running is not None:
                raise VerifiedTaskWorkerDraining("Another worker run acquired ownership.")
        if self._verification is not None:
            await self._consume_verifier_settlement(self.callback_timeout_seconds)
            if self._closed:
                raise RuntimeError("VerifiedTaskWorker is closed.")
            if self._running is not None:
                raise VerifiedTaskWorkerDraining("Another worker run acquired ownership.")
        if not task_store_verified_task_worker_capability_is_complete(self.store):
            raise NotImplementedError(
                "The task store does not implement the complete verified-worker contract."
            )
        running = asyncio.create_task(
            _capture_owned_task_outcome(
                lambda: run_durable_worker_loop(
                    self._step,
                    poll_interval_s=self.poll_interval_s,
                    stop=stop,
                    max_handled=max_tasks,
                )
            )
        )
        self._running = running
        try:
            completed = await asyncio.shield(running)
            if completed.error is not None:
                raise completed.error
            if completed.result is None:
                raise RuntimeError("Verified worker run returned no outcome.")
            return completed.result
        except asyncio.CancelledError as cancellation:
            running.cancel()
            outcome = await await_shielded_task_outcome(
                running,
                cancellation=cancellation,
                timeout_s=self.callback_timeout_seconds,
            )
            restore_task_cancellation_requests(
                outcome.cancellation_requests_consumed,
                cancellation=outcome.cancellation,
            )
            propagated = outcome.cancellation or cancellation
            child_error = outcome.error or (
                None if outcome.result is None else outcome.result.error
            )
            if isinstance(child_error, asyncio.CancelledError) and child_error is not propagated:
                child_error = exception_cause(child_error)
            propagated = _merge_worker_failures(propagated, child_error)
            assert propagated is not None
            raise propagated from exception_cause(propagated)
        finally:
            if running.done() and self._running is running:
                self._running = None

    async def aclose(self) -> None:
        self._closed = True
        deadline = asyncio.get_running_loop().time() + self.callback_timeout_seconds
        failure = None
        try:
            await self._close_running(self.callback_timeout_seconds)
        except (asyncio.CancelledError, VerifiedTaskWorkerDraining):
            raise
        except BaseException as error:
            failure = error
        try:
            await self._consume_preparation_settlement(
                max(0.0, deadline - asyncio.get_running_loop().time())
            )
        except BaseException as error:
            failure = _merge_worker_failures(failure, error)
        try:
            await self._consume_pre_entry_settlement(
                max(0.0, deadline - asyncio.get_running_loop().time())
            )
        except BaseException as error:
            failure = _merge_worker_failures(failure, error)
        try:
            await self._consume_verifier_settlement(
                max(0.0, deadline - asyncio.get_running_loop().time())
            )
        except BaseException as error:
            failure = _merge_worker_failures(failure, error)
        if failure is not None:
            raise failure from exception_cause(failure)

    async def _close_running(self, timeout_s: float) -> None:
        running = self._running
        if running is None:
            return
        if not running.cancelling():
            running.cancel()
        outcome = await await_shielded_task_outcome(running, timeout_s=timeout_s)
        if not outcome.timed_out and self._running is running:
            self._running = None
        restore_task_cancellation_requests(
            outcome.cancellation_requests_consumed, cancellation=outcome.cancellation
        )
        failure = outcome.error or (None if outcome.result is None else outcome.result.error)
        if isinstance(failure, asyncio.CancelledError):
            # A prior run's cancellation is not a new cancellation of aclose.
            # Deferred cleanup failures remain observable through this handle.
            failure = exception_cause(failure)
        if outcome.cancellation is not None:
            failure = _merge_worker_failures(outcome.cancellation, failure)
            assert failure is not None
            raise failure from exception_cause(failure)
        if outcome.timed_out:
            raise VerifiedTaskWorkerDraining(
                "Owned verified task work is still draining; keep stores open and retry aclose()."
            )
        if failure is not None:
            raise failure

    async def _consume_verifier_settlement(self, timeout_s: float) -> None:
        retained = self._verification
        if retained is None:
            return
        if retained.observation is None:

            async def observe():
                settlement = await retained.execution.settlement()
                return (settlement.failure,)

            retained.observation = asyncio.create_task(
                _capture_owned_task_outcome(observe), name="cayu-worker-verifier-settlement"
            )
        observation = retained.observation
        outcome = await await_shielded_task_outcome(
            observation, timeout_s=timeout_s, timeout_after_cancellation_s=0
        )
        captured = outcome.result
        failure = outcome.error or (None if captured is None else captured.error)
        if isinstance(failure, asyncio.CancelledError):
            failure = unexpected_child_cancellation_error(
                failure, operation="Verified worker verifier settlement observation"
            )
        if self._verification is not retained or retained.observation is not observation:
            # Another waiter already consumed this exact observation. Its
            # cleanup outcome is not a new failure of this caller; the caller's
            # own cancellation below remains independently authoritative.
            failure = None
        elif captured is not None and captured.result is not None:
            failure = _merge_worker_failures(failure, captured.result[0])
            retained.execution.acknowledge_settlement()
            if self._verification is retained:
                self._verification = None
        elif observation.done():
            # Failure of an observation is not proof of underlying settlement.
            # Preserve the exact handle so close can observe it again.
            retained.observation = None
            if failure is None:
                failure = RuntimeError("Verifier observation returned no settlement evidence.")
        if outcome.cancellation is not None:
            failure = _merge_worker_failures(failure, outcome.cancellation)
            restore_task_cancellation_requests(
                outcome.cancellation_requests_consumed, cancellation=outcome.cancellation
            )
        elif outcome.timed_out:
            failure = _merge_worker_failures(
                failure,
                VerifiedTaskWorkerDraining(
                    "Owned completion verification is still draining; keep stores open "
                    "and retry aclose()."
                ),
            )
        if failure is not None:
            raise failure from exception_cause(failure)

    async def __aenter__(self) -> VerifiedTaskWorker:
        if self._closed:
            raise RuntimeError("VerifiedTaskWorker is closed.")
        return self

    async def __aexit__(self, *exc: object) -> None:
        try:
            await self.aclose()
        except BaseException as cleanup:
            primary = exc[1] if len(exc) > 1 and isinstance(exc[1], BaseException) else None
            failure = _merge_worker_failures(primary, cleanup)
            assert failure is not None
            raise failure from exception_cause(failure)

    async def _step(self, _now: float, _handled: int) -> DurableWorkerStep:
        await self._group_maintenance.step(self.store, self.app._secret_redactor, now=_now)
        recovered = await self._discover_unfinished_attempt()
        if recovered is not None:
            return recovered
        await self._operation(
            lambda: self.store.reclaim_expired(
                query=self.query.model_copy(update={"status": TaskStatus.CLAIMED}), max_reclaims=1
            ),
            name="Verified task preparation reclamation",
            mutation="reclaim_expired",
        )
        task = await self._operation(
            lambda: self.store.claim_task(
                self.worker_id, self.query, lease_seconds=self.lease_seconds
            ),
            name="Verified task queue claim",
            mutation="claim_task",
        )
        if task is None:
            return DurableWorkerStep(idle=True)
        task = self._validate(lambda: copy_task(task), "Verified task claim validation")
        if (
            task.work_contract is None
            or task.worker_id != self.worker_id
            or task.status is not TaskStatus.CLAIMED
            or task.lease_expires_at is None
            or task.session_id is not None
        ):
            raise TaskClaimLost("Verified worker queue returned conflicting task authority.")
        contract = await self._load_contract(task.work_contract)
        owner = _LeaseOwner(self, task)
        proposal = await self._with_heartbeat(owner, lambda: self._prepare_and_run(owner, contract))
        if proposal is None or type(proposal) is WorkAttemptLifecycleReceipt:
            return DurableWorkerStep(handled=1, activity=True)
        return await self._finish_proposal(owner, contract, proposal)

    async def _load_contract(self, reference):
        contract = await self._operation(
            lambda: self.store.load_work_contract(reference),
            name="Verified task contract lookup",
        )

        def require_contract(value=contract):
            if type(value) is not WorkContract:
                raise WorkCompletionConflict("Verified task contract is missing.")
            return copy_work_contract(value)

        contract = self._validate(require_contract, "Verified task contract validation")
        if contract.reference() != reference:
            raise WorkCompletionConflict("Verified worker received another work contract.")
        return contract

    async def _discover_unfinished_attempt(self) -> DurableWorkerStep | None:
        # One immutable-ID candidate per step keeps discovery bounded. A live
        # verifier does not prevent this same step from claiming unrelated work.
        scope = TaskAggregateFilter(
            type=self.query.type,
            session_id=self.query.session_id,
            parent_task_id=self.query.parent_task_id,
            assigned_agent_name=self.query.assigned_agent_name,
        )
        rows = await self._operation(
            lambda: self.store.list_unsettled_work_attempt_admissions(
                task_filter=scope, limit=1, after=self._recovery_cursor
            ),
            name="Verified worker recovery discovery",
        )

        def require_candidate():
            if type(rows) is not list or len(rows) > 1:
                raise WorkCompletionConflict("Recovery discovery returned an invalid page.")
            if not rows:
                return None
            admission = require_work_attempt_admission_result(
                rows[0], operation_name="Verified worker recovery discovery"
            )
            if (
                self._recovery_cursor is not None
                and admission.admission_id <= self._recovery_cursor
            ):
                raise WorkCompletionConflict("Recovery discovery did not advance its cursor.")
            return admission

        admission = self._validate(require_candidate, "Verified recovery candidate")
        self._recovery_cursor = None if admission is None else admission.admission_id
        if admission is None:
            return None
        if admission.state in {
            WorkAttemptAdmissionState.PREPARING,
            WorkAttemptAdmissionState.ACTIVE,
            WorkAttemptAdmissionState.RECOVERING,
        }:
            return await self._recover_discovered_execution(admission)
        if admission.state is not WorkAttemptAdmissionState.RELEASED:
            return None
        raw = await self._operation(
            lambda: self.store.load_completion_proposal_for_attempt(admission.attempt_id),
            name="Verified recovery proposal lookup",
        )

        def require_proposal():
            if type(raw) is not CompletionProposal:
                raise WorkCompletionConflict("Released attempt has no published proposal.")
            copied = copy_completion_proposal(raw)
            if (
                copied.attempt_id != admission.attempt_id
                or copied.task_id != admission.task_id
                or copied.contract != admission.contract
            ):
                raise WorkCompletionConflict("Recovery proposal conflicts with its admission.")
            return copied

        proposal = self._validate(require_proposal, "Verified recovery proposal validation")
        contract = await self._load_contract(admission.contract)
        result = await self._finish_proposal(_LeaseOwner(self, admission), contract, proposal)
        return result if result.handled else None

    async def _recover_discovered_execution(
        self, admission: WorkAttemptAdmission
    ) -> DurableWorkerStep | None:
        if admission.claim.lease_expires_at > datetime.now(UTC):
            return None
        if await self._group_cancellation_requested(admission.task_id):
            if (
                admission.state is WorkAttemptAdmissionState.ACTIVE
                and admission.execution_entry is None
                and admission.execution_stop is None
            ):
                if not await self.app._session_engine.settle_work_attempt_admission_handoff(
                    admission
                ):
                    return None
                # Admission positively proves preparation returned. The store's
                # permanent group fence excludes execution entry, including a
                # delayed original worker. Close the admitted session before
                # publishing the receipt that removes it from discovery.
                self._retain_pre_entry_settlement(admission)
                await self._consume_pre_entry_settlement(self.callback_timeout_seconds)
                return DurableWorkerStep(handled=1, activity=True)
            # An ACTIVE admission may already have released its session while
            # its preparation/proposal callback still owns external work.
            # A replacement execution claim is not cleanup acknowledgement.
            # Keep that historical obligation fenced; exact RELEASED proposals
            # are reconciled separately by _finish_proposal. Maintenance turns
            # unresolved owner loss into attention, never fresh execution.
            return None
        release = await self.app._session_engine.load_work_attempt_released_recovery_evidence(
            admission
        )
        if (
            release is None
            and admission.execution_entry is not None
            and not await self.app._session_engine.has_recoverable_work_attempt_model_result(
                admission
            )
            and not await self.app._session_engine.has_recoverable_work_attempt_cleanup(admission)
        ):
            return None
        # Resolve immutable contract data before claiming the replacement lease;
        # a slow lookup must not leave acknowledged ownership without a heartbeat.
        contract = await self._load_contract(admission.contract)
        request = WorkAttemptRecoveryRequest(
            admission_id=admission.admission_id,
            claim_id=verified_task_operation_id(
                "execution-claim", admission.admission_id, str(admission.claim.generation + 1)
            ),
            worker_id=self.worker_id,
            generation=admission.claim.generation + 1,
            lease_seconds=self.lease_seconds,
        )
        try:
            ownership = await self.app._claim_work_attempt_recovery(request)
        except (WorkAttemptExecutionClaimLost, WorkAttemptAdmissionConflict):
            raw = await self._operation(
                lambda: self.store.load_work_attempt_admission(admission.admission_id),
                name="Verified recovery election readback",
            )
            current = self._validate(
                lambda: require_work_attempt_admission_result(
                    raw, operation_name="Verified recovery election readback"
                ),
                "Verified recovery election validation",
            )
            if (
                current.admission_id == admission.admission_id
                and current.prepare_request_sha256 == admission.prepare_request_sha256
                and current.task_id == admission.task_id
                and current.attempt_id == admission.attempt_id
                and current.claim.generation >= request.generation
                and current.claim != admission.claim
                and current.claim.lease_expires_at > datetime.now(UTC)
                and (
                    current.claim.worker_id != self.worker_id
                    or current.claim.execution_owner_id
                    != self.app._current_work_attempt_execution_owner_id()
                )
            ):
                return None
            raise
        owner = _LeaseOwner(self, ownership.admission)
        proposal = await self._with_heartbeat(
            owner, lambda: self._recover_and_process(owner, ownership, contract)
        )
        result = await self._finish_proposal(owner, contract, proposal)
        return result if result.handled else None

    async def _recover_and_process(self, owner, ownership, contract):
        # Recovery may activate local invocation context. Resumed execution
        # must stay in this action Task, not inherit it from the scheduler.
        recovered = await self.app._recover_claimed_work_attempt(ownership)
        async with owner.lock:
            owner.state = recovered
        release = await self.app._session_engine.load_work_attempt_released_recovery_evidence(
            recovered
        )
        if release is not None:
            return await self._propose_after_execution(owner, contract)
        return await self._run_and_propose(owner, contract)

    async def _finish_proposal(self, owner, contract, proposal) -> DurableWorkerStep:
        admission = owner.state
        if type(admission) is not WorkAttemptAdmission:
            raise WorkCompletionConflict("Verified worker lost admitted attempt authority.")
        while True:
            if type(proposal) is WorkAttemptLifecycleReceipt:
                return DurableWorkerStep(handled=1, activity=True)
            if type(admission) is not WorkAttemptAdmission:
                raise WorkCompletionConflict("Verified worker lost its admitted successor.")
            if await self._group_cancellation_requested(admission.task_id):
                # A peer may still own the verifier, including an opaque read
                # that outlived its lease. The group fence prevents a new
                # verifier claim; only an exact committed decision proves the
                # existing verification finished. Merely discovering this
                # admission must neither settle it nor fail the worker loop.
                _, decision = await self._verification_request(proposal.proposal_id, contract)
                if (
                    decision is None
                    and await self._load_verifier_claim(proposal.proposal_id, contract) is not None
                ):
                    return DurableWorkerStep(idle=True)
                if decision is not None:
                    await self.app._reconcile_completion_result_group_settlement(
                        admission.task_id, decision.decision_id
                    )
                try:
                    await self._settle_runtime_stop(
                        owner,
                        "work_contract_group_cancelled",
                        proposal=proposal,
                        decision_id=None if decision is None else decision.decision_id,
                    )
                except TaskGroupResultResolutionPending:
                    return DurableWorkerStep(idle=True)
                return DurableWorkerStep(handled=1, activity=True)
            verification, decision = await self._verification_request(
                proposal.proposal_id, contract
            )
            if admission.run_semantics is None:
                raise WorkCompletionConflict("Verified worker lost its frozen run semantics.")
            if decision is None and admission.run_semantics.deadline.expired:
                try:
                    await self._settle_runtime_stop(
                        owner, "work_contract_elapsed_limit", proposal=proposal
                    )
                except WorkAttemptAdmissionConflict:
                    verification, decision = await self._verification_request(
                        proposal.proposal_id, contract
                    )
                    if decision is None:
                        raise
                else:
                    return DurableWorkerStep(handled=1, activity=True)
            if verification is None:
                return DurableWorkerStep(idle=True)
            try:
                started = await self.app._start_verified_task_decision(
                    admission.admission_id, verification
                )
                self._verification = _RetainedVerifier(started.verification)
                result = await started.result()
            except _GroupVerificationAdmissionRefused:
                if not await self._group_cancellation_requested(admission.task_id):
                    raise
                # Admission runs inside the owned verification operation. Drain
                # that refused operation before re-entering cancellation handling,
                # which independently checks existing verifier/resolver ownership.
                await self._consume_verifier_settlement(self.callback_timeout_seconds)
                continue
            except TaskGroupResultResolutionPending:
                # The election may win after the preliminary cancellation read.
                # A peer's live resolver remains authoritative in either path.
                await self._consume_verifier_settlement(self.callback_timeout_seconds)
                return DurableWorkerStep(idle=True)
            except Exception as failure:
                if isinstance(failure, (CompletionVerificationClaimLost, WorkCompletionConflict)):
                    # A peer may win between discovery and verifier admission.
                    # Only a distinct, positively validated claim can explain
                    # this as contention; our own verifier failures still raise.
                    competing = await self._load_verifier_claim(proposal.proposal_id, contract)
                    if (
                        competing is not None
                        and competing.claim_id != verification.claim_id
                        and verification.claim_id
                        == self._verifier_claim_id(
                            proposal.proposal_id, contract, competing.attempt_number
                        )
                    ):
                        retry, committed = await self._verification_request(
                            proposal.proposal_id, contract
                        )
                        if retry is None or committed is not None:
                            await self._consume_verifier_settlement(self.callback_timeout_seconds)
                            return DurableWorkerStep(idle=True)
                if admission.run_semantics.deadline.expired:
                    try:
                        await self._settle_runtime_stop(
                            owner, "work_contract_elapsed_limit", proposal=proposal
                        )
                    except WorkAttemptAdmissionConflict as conflict:
                        try:
                            _, committed = await self._verification_request(
                                proposal.proposal_id, contract
                            )
                            if committed is None:
                                raise conflict
                        except BaseException as reconciliation:
                            combined = _merge_worker_failures(failure, reconciliation)
                            assert combined is not None
                            raise combined from exception_cause(combined)
                    except BaseException as settlement:
                        combined = _merge_worker_failures(failure, settlement)
                        assert combined is not None
                        raise combined from exception_cause(combined)
                # Expiry is durable state, not a reason to erase a real
                # verifier or publication failure or to claim quiescence.
                raise
            await self._consume_verifier_settlement(self.callback_timeout_seconds)
            if result.settlement is not None:
                return DurableWorkerStep(handled=1, activity=True)
            proposal = await self._with_heartbeat(
                owner,
                lambda owner=owner, predecessor=admission.admission_id, decision_id=result.decision.decision_id: (
                    self._continue_and_run(owner, contract, predecessor, decision_id)
                ),
            )
            admission = owner.state

    async def _continue_and_run(self, owner, contract, predecessor, decision_id):
        # Admission and execution share one asyncio task. Admission-created
        # local fence context must not be stranded in the scheduler's task.
        try:
            async with owner.lock:
                owner.state = await self.app._continue_verified_task(
                    predecessor,
                    decision_id,
                    worker_id=self.worker_id,
                    lease_seconds=self.lease_seconds,
                )
        except Exception as failure:
            admission = owner.state
            if (
                type(admission) is not WorkAttemptAdmission
                or admission.run_semantics is None
                or not admission.run_semantics.deadline.expired
            ):
                raise
            # The store arbitrates against a successor, including admission
            # whose acknowledgement was lost. Never reinterpret that successor
            # as a stop of its predecessor or replace the committed rejection.
            try:
                raw_proposal = await self._operation(
                    lambda: self.store.load_completion_proposal_for_attempt(admission.attempt_id),
                    name="Expired continuation proposal lookup",
                )

                def require_proposal(value=raw_proposal):
                    if type(value) is not CompletionProposal:
                        raise WorkCompletionConflict("Expired continuation has no proposal.")
                    return copy_completion_proposal(value)

                try:
                    proposal = self._validate(require_proposal, "Expired continuation proposal")
                finally:
                    del raw_proposal, require_proposal
                receipt = await self._settle_runtime_stop(
                    owner, "work_contract_elapsed_limit", proposal=proposal, decision_id=decision_id
                )
            except BaseException as settlement:
                combined = _merge_worker_failures(failure, settlement)
                assert combined is not None
                raise combined from exception_cause(combined)
            if isinstance(failure, ExecutionDeadlineExceeded):
                return receipt
            # A coincident deadline must not erase an unrelated admission or
            # acknowledgement failure, even after the durable stop succeeds.
            raise
        return await self._run_and_propose(owner, contract)

    async def _prepare_and_run(self, owner: _LeaseOwner, contract: WorkContract):
        task = owner.state
        assert type(task) is Task
        identity = (task.id, contract.fingerprint)
        session_id = verified_task_operation_id("session", *identity)
        admission_id = verified_task_operation_id("initial-admission", *identity)
        deadline = ExecutionDeadline.after(
            self.max_elapsed_seconds, source="verified-task-worker", scope="task"
        )
        retain_callback = await self._group_owns_callback(task)
        if retain_callback:
            async with owner.lock:
                current = owner.state
                assert type(current) is Task and current.lease_expires_at is not None
                claim_task_id, claim_lease = current.id, current.lease_expires_at
                try:
                    started = await self._operation(
                        lambda: self.store.mark_claimed_task_execution_started(
                            claim_task_id, self.worker_id, claim_lease
                        ),
                        name="Verified preparation execution entry",
                        mutation="mark_claimed_task_execution_started",
                    )

                    def require_started(value=started, expected=current):
                        recorded = copy_task(value)
                        if (
                            recorded.started_at is None
                            or recorded.model_copy(
                                update={
                                    "started_at": expected.started_at,
                                    "updated_at": expected.updated_at,
                                }
                            )
                            != expected
                        ):
                            raise WorkCompletionConflict(
                                "Preparation entry changed its claim authority."
                            )
                        return recorded

                    try:
                        task = owner.state = self._validate(
                            require_started, "Verified preparation execution entry"
                        )
                    finally:
                        del started, require_started
                except BaseException as entry_failure:
                    # The store call has settled, but its acknowledgement may
                    # have been lost. No callback has been dispatched here.
                    retained = _RetainedPreparationSettlement(copy_task(current))
                    self._preparation_settlement = retained
                    owner.state = retained
                    try:
                        await self._consume_preparation_settlement(self.callback_timeout_seconds)
                    except BaseException as settlement_failure:
                        combined = _merge_worker_failures(entry_failure, settlement_failure)
                        assert combined is not None
                        raise combined from exception_cause(combined)
                    raise
        context = VerifiedTaskPreparationContext(
            copy_task(task), copy_work_contract(contract), session_id
        )
        try:
            raw = await self._callback(
                lambda: self.handler.prepare(context),
                "Verified task preparation callback",
                retain_until_settled=retain_callback,
            )

            def prepare(raw=raw):
                if type(raw) is not RunRequest:
                    raise TypeError("Verified task preparation must return RunRequest.")
                if raw.task_id not in {None, task.id} or raw.session_id not in {None, session_id}:
                    raise WorkCompletionConflict(
                        "Handler preparation selected another task or session."
                    )
                if (
                    raw.task_worker_id is not None
                    or raw.task_lease_expires_at is not None
                    or raw.loop_policies
                ):
                    raise WorkCompletionConflict(
                        "Handler preparation cannot choose worker authority or local loop policies."
                    )
                return prepare_run_request(raw, redactor=self.app._secret_redactor)

            try:
                request = self._validate(prepare, "Verified task preparation result")
            finally:
                del raw, prepare
        except asyncio.CancelledError as cancellation:
            if retain_callback:
                await self._settle_cancelled_group_callback(owner, contract, cancellation)
            raise
        except Exception as failure:
            reason = (
                "work_contract_preparation_timed_out"
                if isinstance(failure, TimeoutError)
                else "work_contract_preparation_failed"
            )
            del failure
            try:
                async with owner.lock:
                    if await self._group_cancellation_requested(task.id):
                        reason = "work_contract_group_cancelled"
                    await self._hold_preparation_locked(owner, contract, reason)
            except BaseException as hold_failure:
                if retain_callback and self._preparation_settlement is None:
                    try:
                        async with owner.lock:
                            state = self._retain_drained_preparation(owner)
                            await self._consume_preparation_settlement(
                                self.callback_timeout_seconds
                            )
                            owner.state = state
                    except BaseException as settlement_failure:
                        combined = _merge_worker_failures(hold_failure, settlement_failure)
                        assert combined is not None
                        raise combined from exception_cause(combined)
                raise
            return None
        if retain_callback:
            # Preparation has returned, but even acquiring the lease lock or
            # reading the election can fail before admission starts. Retain its
            # exact execution proof now. Heartbeats may still renew the claim;
            # settlement binds the copied execution marker, not that lease.
            # run/aclose cannot consume this owner until this run has drained.
            assert self._preparation_settlement is None
            self._preparation_settlement = _RetainedPreparationSettlement(
                claim=copy_task(task), started=copy_task(task)
            )
        async with owner.lock:
            current = owner.state
            assert type(current) is Task
            if await self._group_cancellation_requested(task.id):
                await self._hold_preparation_locked(
                    owner, contract, "work_contract_group_cancelled"
                )
                return None
            request = request.model_copy(
                update={
                    "task_id": task.id,
                    "session_id": session_id,
                    "task_worker_id": self.worker_id,
                    "task_lease_expires_at": current.lease_expires_at,
                    "execution_deadline": effective_deadline(deadline, request.execution_deadline),
                }
            )
            execution = WorkAttemptExecutionRequest(
                admission_id=admission_id,
                claim_id=verified_task_operation_id("execution-claim", admission_id, "1"),
                attempt_id=verified_task_operation_id("initial-attempt", *identity),
                interaction_id=verified_task_operation_id("initial-interaction", *identity),
                worker_id=self.worker_id,
                task_id=task.id,
                task_lease_expires_at=current.lease_expires_at,
                generation=1,
                lease_seconds=self.lease_seconds,
            )
            try:
                owner.state = await self._admit_prepared(request, execution)
            except WorkAttemptAdmissionConflict:
                if not retain_callback:
                    raise
                # The callback returned, but admission may have lost a race
                # with election. Retain the exact hold before any readback/write.
                # Its atomic store validation rejects transferred ownership.
                await self._hold_preparation_locked(
                    owner, contract, "work_contract_group_cancelled"
                )
                return None
            except ExecutionDeadlineExceeded:
                await self._hold_preparation_locked(
                    owner, contract, "work_contract_elapsed_limit", request.execution_deadline
                )
                return None
            except BaseException:
                if self._preparation_settlement is not None:
                    owner.state = self._preparation_settlement
                raise
        return await self._run_and_propose(owner, contract)

    async def _hold_preparation_locked(self, owner, contract, reason, deadline=None):
        """Publish through the existing hold owner while the lease lock is held."""
        current = owner.state
        hold = self._preparation_hold_request(current, contract, reason, deadline)
        if reason == "work_contract_group_cancelled" or self._preparation_settlement is not None:
            # Election may be observed before admission, during admission, or
            # while draining cancellation. Every path retains the exact hold
            # before readback/publication, so a transient failure has the same
            # retry owner even when no admission was ever attempted.
            retained = self._preparation_settlement or _RetainedPreparationSettlement(
                claim=copy_task(current)
            )
            retained.hold = hold
            self._preparation_settlement = retained
            owner.state = retained
            held = await self._consume_preparation_settlement(self.callback_timeout_seconds)
            if held is None:
                raise WorkCompletionConflict("Preparation hold returned no settled task.")
            owner.state = held
        else:
            owner.state = await self._publish_preparation_hold(hold)

    def _preparation_hold_request(self, current, contract, reason, deadline=None):
        assert type(current) is Task and current.lease_expires_at is not None
        lease = current.lease_expires_at
        return WorkAttemptPreparationHold(
            hold_id=verified_task_operation_id(
                "preparation-hold", current.id, self.worker_id, lease.isoformat()
            ),
            task_id=current.id,
            contract=contract.reference(),
            worker_id=self.worker_id,
            lease_expires_at=lease,
            reason=reason,
            deadline_expires_at=None if deadline is None else deadline.expires_at,
        )

    async def _publish_preparation_hold(self, hold: WorkAttemptPreparationHold) -> Task:
        receipt = await self._operation(
            lambda: self.store.hold_work_attempt_preparation(hold),
            name="Verified task preparation hold",
            mutation="hold_work_attempt_preparation",
        )

        def require_hold(value=receipt):
            if type(value) is not WorkAttemptPreparationHoldReceipt:
                raise WorkCompletionConflict("Preparation hold returned no receipt.")
            copied = WorkAttemptPreparationHoldReceipt.model_validate(
                value.model_dump(mode="python", warnings=False)
            )
            if copied.request != hold:
                raise WorkCompletionConflict("Preparation hold returned conflicting authority.")
            return copied.task

        try:
            return self._validate(require_hold, "Verified preparation hold receipt")
        finally:
            del receipt, require_hold

    async def _run_and_propose(self, owner: _LeaseOwner, contract: WorkContract):
        admission = owner.state
        assert type(admission) is WorkAttemptAdmission
        try:
            async for _ in self.app._execute_work_attempt(
                WorkAttemptRunRequest(
                    admission_id=admission.admission_id,
                    claim_id=admission.claim.claim_id,
                    worker_id=self.worker_id,
                    generation=admission.claim.generation,
                    lease_seconds=self.lease_seconds,
                )
            ):
                pass
        except Exception as failure:
            reason = (
                "work_contract_elapsed_limit"
                if isinstance(failure, ExecutionDeadlineExceeded)
                else "work_contract_execution_failed"
            )
            return await self._stop_after_failure(owner, reason, failure)
        return await self._propose_after_execution(owner, contract)

    async def _propose_after_execution(self, owner: _LeaseOwner, contract: WorkContract):
        # Execution entry was committed by the engine. Refresh through the
        # existing exact claim owner before requesting release evidence.
        await owner.heartbeat()
        admission = owner.state
        assert type(admission) is WorkAttemptAdmission
        await self.app._session_engine.load_work_attempt_release_evidence(admission)
        if await self._group_cancellation_requested(admission.task_id):
            return await self._settle_runtime_stop(owner, "work_contract_group_cancelled")
        recorded_reason = runtime_stop_reason_for_execution_stop(admission)
        if recorded_reason is not None:
            return await self._settle_runtime_stop(owner, recorded_reason)
        session = await self.app.session_store.load(admission.session_id)
        if session is None:
            raise WorkAttemptRecoveryRequired("Verified attempt has no session to settle.")
        if session.status in {SessionStatus.FAILED, SessionStatus.INTERRUPTED}:
            return await self._settle_runtime_stop(
                owner,
                "work_contract_execution_failed"
                if session.status is SessionStatus.FAILED
                else "work_contract_execution_interrupted",
            )
        if session.status is not SessionStatus.COMPLETED:
            raise WorkAttemptRecoveryRequired("Verified attempt still requires recovery.")
        context = await self._proposal_context(admission, contract)
        retain_callback = await self._group_owns_callback(context.task)
        try:
            proposal = await self._prepare_proposal(
                context, session.execution_deadline, retain_callback=retain_callback
            )
        except asyncio.CancelledError as cancellation:
            if retain_callback:
                await self._settle_cancelled_group_callback(owner, contract, cancellation)
            raise
        except Exception as failure:
            reason = (
                "work_contract_elapsed_limit"
                if isinstance(failure, ExecutionDeadlineExceeded)
                else "work_contract_handler_failed"
            )
            return await self._stop_after_failure(owner, reason, failure)
        if await self._group_cancellation_requested(admission.task_id):
            # The read-only handler has returned, but its proposal is not yet
            # admitted. Do not create verifier work after the group decided.
            return await self._settle_runtime_stop(owner, "work_contract_group_cancelled")
        try:
            async with owner.lock:
                session.execution_deadline.require_admission("completion_proposal")
                await self.app.submit_work_attempt_proposal(
                    WorkAttemptProposalRequest(
                        admission_id=admission.admission_id,
                        claim_id=admission.claim.claim_id,
                        generation=admission.claim.generation,
                        proposal=proposal,
                    )
                )
                updated = await self._operation(
                    lambda: self.store.load_work_attempt_admission(admission.admission_id),
                    name="Verified proposal admission readback",
                )
                if (
                    type(updated) is not WorkAttemptAdmission
                    or updated.state is not WorkAttemptAdmissionState.RELEASED
                ):
                    raise WorkAttemptRecoveryRequired(
                        "Proposal admission release requires reconciliation."
                    )
                owner.state = updated
        except ExecutionDeadlineExceeded as failure:
            return await self._stop_after_failure(owner, "work_contract_elapsed_limit", failure)
        except WorkAttemptAdmissionConflict:
            if await self._group_cancellation_requested(admission.task_id):
                return await self._settle_runtime_stop(owner, "work_contract_group_cancelled")
            raise
        return proposal

    async def _stop_after_failure(self, owner, reason: WorkAttemptStopReason, failure: Exception):
        try:
            if isinstance(failure, _GroupExecutionEntryRefused):
                admission = owner.state
                if type(admission) is not WorkAttemptAdmission:
                    raise WorkAttemptRecoveryRequired("Pre-entry refusal lost its admission owner.")
                # Retain the store-confirmed refusal before any fallible await.
                self._retain_pre_entry_settlement(admission)
            return await self._settle_runtime_stop(owner, reason)
        except asyncio.CancelledError as cancellation:
            combined = _merge_worker_failures(failure, cancellation)
            assert combined is not None
            raise combined from exception_cause(combined)
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                "Verified task failure could not settle", [failure, cleanup]
            ) from None

    async def _settle_runtime_stop(
        self,
        owner: _LeaseOwner,
        reason: WorkAttemptStopReason,
        *,
        proposal: CompletionProposal | CompletionProposalCreate | None = None,
        decision_id: str | None = None,
    ) -> WorkAttemptLifecycleReceipt:
        prior = owner.state
        if self._pre_entry_settlement is not None:
            if type(prior) is not WorkAttemptAdmission:
                raise WorkAttemptRecoveryRequired("Pre-entry settlement lost its admission owner.")
            # Returning from the owned invocation is positive local evidence.
            # Retain it before any lookup, renewal, or lock acquisition can
            # fail. Store readback/settlement still decides whether entry or a
            # replacement owner won; this does not authorize cancellation.
            async with owner.lock:
                receipt = await self._consume_pre_entry_settlement(self.callback_timeout_seconds)
                if receipt is not None:
                    owner.state = prior.model_copy(
                        update={"state": WorkAttemptAdmissionState.RELEASED}
                    )
                    return receipt
        await owner.heartbeat()
        async with owner.lock:
            admission = owner.state
            if type(admission) is not WorkAttemptAdmission:
                raise WorkAttemptRecoveryRequired(
                    "Runtime stop requires exact admission authority."
                )
            reason = runtime_stop_reason_for_execution_stop(admission) or reason
            if proposal is None and await self._group_cancellation_requested(admission.task_id):
                reason = "work_contract_group_cancelled"
            if reason == "work_contract_group_cancelled" and admission.execution_entry is None:
                self._retain_pre_entry_settlement(admission)
                receipt = await self._consume_pre_entry_settlement(self.callback_timeout_seconds)
                assert receipt is not None
                owner.state = admission.model_copy(
                    update={"state": WorkAttemptAdmissionState.RELEASED}
                )
                return receipt
            proposal_digest = (
                None
                if proposal is None
                else self._validate(
                    lambda: (
                        copy_completion_proposal(proposal).request_sha256
                        if type(proposal) is CompletionProposal
                        else completion_proposal_request_sha256(proposal)
                    ),
                    "Expired proposal content identity",
                )
            )
            release = await self.app._session_engine.load_work_attempt_release_evidence(admission)
            request = WorkAttemptLifecycleSettlement(
                settlement_id=verified_task_operation_id("settlement", admission.admission_id),
                task_id=admission.task_id,
                admission_id=admission.admission_id,
                expected_admission_sha256=work_attempt_admission_authority_sha256(admission),
                release_evidence=release,
                kind=(
                    "group_cancellation"
                    if reason == "work_contract_group_cancelled"
                    else "continuation_deadline_stop"
                    if decision_id is not None
                    else "runtime_stop"
                    if proposal is None
                    else "proposal_deadline_stop"
                ),
                decision_id=decision_id,
                application_idempotency_key=(
                    None
                    if decision_id is None or reason == "work_contract_group_cancelled"
                    else verified_task_operation_id(
                        "application", admission.admission_id, decision_id
                    )
                ),
                proposal_id=None if proposal is None else proposal.proposal_id,
                proposal_request_sha256=proposal_digest,
                stop_reason=reason,
            )

            def copy_receipt(value):
                if type(value) is not WorkAttemptLifecycleReceipt:
                    raise WorkCompletionConflict("Runtime stop returned no durable receipt.")
                return WorkAttemptLifecycleReceipt.model_validate(
                    value.model_dump(mode="python", warnings=False)
                )

            def require_receipt(value):
                copied = copy_receipt(value)
                if copied.request != request:
                    raise WorkCompletionConflict(
                        "Runtime stop returned conflicting receipt authority."
                    )
                return copied

            try:
                raw = await self._operation(
                    lambda: self.store.settle_work_attempt_lifecycle(request),
                    name="Verified task runtime stop",
                    mutation="settle_work_attempt_lifecycle",
                )
            except Exception as publication_failure:
                try:
                    raw = await self._operation(
                        lambda: self.store.load_work_attempt_lifecycle_receipt(
                            admission.admission_id
                        ),
                        name="Verified task runtime stop reconciliation",
                    )
                except BaseException as reconciliation_failure:
                    combined = _merge_worker_failures(publication_failure, reconciliation_failure)
                    assert combined is not None
                    raise combined from exception_cause(combined)
                if raw is None:
                    if (
                        isinstance(publication_failure, WorkAttemptAdmissionConflict)
                        and request.kind != "group_cancellation"
                        and proposal is None
                        and await self._group_cancellation_requested(admission.task_id)
                    ):
                        # Only a typed precommit conflict without a receipt may
                        # select the now-authoritative group stop. Ack-loss
                        # replay above keeps its original complete tuple.
                        request = WorkAttemptLifecycleSettlement.model_validate(
                            request.model_copy(
                                update={
                                    "kind": "group_cancellation",
                                    "stop_reason": "work_contract_group_cancelled",
                                }
                            ).model_dump(mode="python", warnings=False)
                        )
                        raw = await self._operation(
                            lambda: self.store.settle_work_attempt_lifecycle(request),
                            name="Verified group stop after decision race",
                            mutation="settle_work_attempt_lifecycle",
                        )
                    else:
                        raise
                if proposal is not None and isinstance(
                    publication_failure, WorkAttemptAdmissionConflict
                ):
                    competing = self._validate(
                        lambda: copy_receipt(raw), "Competing lifecycle receipt"
                    )
                    if (
                        competing.request.kind == "decision_application"
                        and competing.request.admission_id == request.admission_id
                        and competing.request.task_id == request.task_id
                        and competing.request.expected_admission_sha256
                        == request.expected_admission_sha256
                        and competing.request.release_evidence == request.release_evidence
                    ):
                        raise
                try:
                    receipt = self._validate(
                        lambda: require_receipt(raw), "Verified runtime stop receipt"
                    )
                except BaseException as validation_failure:
                    combined = _merge_worker_failures(publication_failure, validation_failure)
                    assert combined is not None
                    raise combined from exception_cause(combined)
            else:
                receipt = self._validate(
                    lambda: require_receipt(raw), "Verified runtime stop receipt"
                )
            # The exact receipt attests this transaction's ACTIVE -> RELEASED
            # transition. This is a local heartbeat projection, not dispatch.
            owner.state = admission.model_copy(update={"state": WorkAttemptAdmissionState.RELEASED})
            return receipt

    def _retain_pre_entry_settlement(self, admission: WorkAttemptAdmission) -> None:
        request = WorkAttemptLifecycleSettlement(
            settlement_id=verified_task_operation_id("settlement", admission.admission_id),
            task_id=admission.task_id,
            admission_id=admission.admission_id,
            expected_admission_sha256=pre_entry_settlement_authority(admission),
            release_evidence=WorkAttemptPreEntrySettlementEvidence(
                session_id=admission.session_id,
                session_instance_id=admission.session_invocation.session_instance_id,
                interaction_id=admission.interaction_id,
                profile_fingerprint=admission.source_execution_profile_fingerprint,
            ),
            kind="group_cancellation",
            stop_reason="work_contract_group_cancelled",
        )
        if self._pre_entry_settlement not in (None, request):
            raise WorkAttemptAdmissionConflict("Another pre-entry settlement is retained.")
        self._pre_entry_settlement = request

    async def _consume_pre_entry_settlement(
        self, timeout_s: float
    ) -> WorkAttemptLifecycleReceipt | None:
        request = self._pre_entry_settlement
        if request is None:
            return None

        async def publish():
            existing = await self._operation(
                lambda: self.store.load_work_attempt_lifecycle_receipt(request.admission_id),
                name="Verified pre-entry receipt lookup",
            )
            if existing is None:
                raw_admission = await self._operation(
                    lambda: self.store.load_work_attempt_admission(request.admission_id),
                    name="Verified pre-entry admission lookup",
                )
                admission = self._validate(
                    lambda: require_work_attempt_admission_result(
                        raw_admission, operation_name="Pre-entry settlement readback"
                    ),
                    "Verified pre-entry admission validation",
                )
                if admission.execution_entry is not None:
                    raise WorkAttemptAdmissionConflict(
                        "Retained pre-entry refusal conflicts with entered execution."
                    )
                if not await self._group_cancellation_requested(request.task_id):
                    raise WorkAttemptAdmissionConflict(
                        "Retained pre-entry refusal has no matching group cancellation."
                    )
                if pre_entry_settlement_authority(admission) != request.expected_admission_sha256:
                    raise WorkAttemptAdmissionConflict("Pre-entry cleanup authority changed.")
                await self.app._session_engine.close_unentered_work_attempt_invocation(admission)
            try:
                raw = (
                    existing
                    if existing is not None
                    else await self._operation(
                        lambda: self.store.settle_work_attempt_lifecycle(request),
                        name="Verified pre-entry group settlement",
                        mutation="settle_work_attempt_lifecycle",
                    )
                )
            except Exception as publication_failure:
                try:
                    raw = await self._operation(
                        lambda: self.store.load_work_attempt_lifecycle_receipt(
                            request.admission_id
                        ),
                        name="Verified pre-entry settlement readback",
                    )
                except BaseException as readback_failure:
                    failure = _merge_worker_failures(publication_failure, readback_failure)
                    assert failure is not None
                    raise failure from exception_cause(failure)
                if raw is None:
                    raise

            def validate():
                if type(raw) is not WorkAttemptLifecycleReceipt:
                    raise WorkCompletionConflict("Pre-entry settlement returned no receipt.")
                receipt = WorkAttemptLifecycleReceipt.model_validate(
                    raw.model_dump(mode="python", warnings=False)
                )
                if receipt.request != request:
                    raise WorkCompletionConflict("Pre-entry settlement receipt conflicts.")
                return receipt

            return self._validate(validate, "Verified pre-entry settlement receipt")

        if self._pre_entry_publication is None:
            self._pre_entry_publication = asyncio.create_task(_capture_owned_task_outcome(publish))
        publication = self._pre_entry_publication
        outcome = await await_shielded_task_outcome(
            publication, timeout_s=timeout_s, timeout_after_cancellation_s=0
        )
        captured = outcome.result
        failure = outcome.error or (None if captured is None else captured.error)
        receipt = None if captured is None else captured.result
        if publication is self._pre_entry_publication and publication.done():
            self._pre_entry_publication = None
            if captured is not None and failure is None:
                self._pre_entry_settlement = None
        if outcome.timed_out:
            failure = _merge_worker_failures(
                failure, VerifiedTaskWorkerDraining("Pre-entry settlement is still draining.")
            )
        failure = _merge_worker_failures(failure, outcome.cancellation)
        restore_task_cancellation_requests(
            outcome.cancellation_requests_consumed, cancellation=outcome.cancellation
        )
        if failure is not None:
            raise failure from exception_cause(failure)
        return receipt

    async def _proposal_context(
        self, admission: WorkAttemptAdmission, contract: WorkContract
    ) -> VerifiedTaskProposalContext:
        task = await self._operation(
            lambda: self.store.load_task(admission.task_id), name="Verified proposal task lookup"
        )

        def require_task(value=task):
            if type(value) is not Task:
                raise WorkCompletionConflict("Verified proposal task is missing.")
            copied = copy_task(value)
            if (
                copied.id != admission.task_id
                or copied.work_contract != admission.contract
                or copied.session_id != admission.session_id
                or copied.session_instance_id != admission.session_invocation.session_instance_id
                or copied.status is not TaskStatus.RUNNING
            ):
                raise WorkCompletionConflict("Verified proposal task conflicts with its admission.")
            return copied

        task = self._validate(require_task, "Verified proposal task validation")
        if admission.attempt is None:
            raise WorkCompletionConflict("Verified proposal has no admitted attempt.")
        proposal_id = verified_task_operation_id("proposal", admission.attempt_id)
        return VerifiedTaskProposalContext(
            copy_task(task),
            copy_work_contract(contract),
            WorkAttempt.model_validate(admission.attempt.model_dump(mode="python", warnings=False)),
            proposal_id,
        )

    async def _prepare_proposal(
        self,
        context: VerifiedTaskProposalContext,
        deadline: ExecutionDeadline,
        *,
        retain_callback: bool,
    ) -> CompletionProposalCreate:
        proposal_id, attempt_id = context.proposal_id, context.attempt.attempt_id
        deadline.require_admission("completion_proposal")
        report = await self._callback(
            lambda: self.handler.propose(context),
            "Verified task proposal callback",
            retain_until_settled=retain_callback,
        )

        def prepare_report(report=report):
            if type(report) is not VerifiedTaskHandlerReport:
                raise TypeError("Verified task proposal requires an explicit typed handler report.")
            copied = VerifiedTaskHandlerReport.model_validate(
                report.model_dump(mode="python", warnings=False)
            )
            if (
                copied.proposal.proposal_id != proposal_id
                or copied.proposal.attempt_id != attempt_id
            ):
                raise WorkCompletionConflict(
                    "Verified task report conflicts with its expected proposal."
                )
            require_bounded_work_completion_document(
                copied.model_dump(mode="json"),
                "verified_task_report",
                max_bytes=65536,
                max_items=4096,
            )
            return copied.proposal

        try:
            proposal = self._validate(prepare_report, "Verified task report validation")
        finally:
            del report, prepare_report
        deadline.require_admission("completion_proposal")
        return proposal

    def _verifier_claim_id(
        self, proposal_id: str, contract: WorkContract, attempt_number: int
    ) -> str:
        return verified_task_operation_id(
            "verifier-claim",
            proposal_id,
            contract.fingerprint,
            str(attempt_number),
            self.worker_id,
            self.app._current_work_attempt_execution_owner_id(),
        )

    async def _load_verifier_claim(
        self, proposal_id: str, contract: WorkContract
    ) -> CompletionVerificationClaim | None:
        raw_claim = await self._operation(
            lambda: self.store.load_completion_verification_claim(proposal_id),
            name="Worker verifier claim lookup",
        )
        try:

            def copy_claim(value=raw_claim):
                if value is None:
                    return None
                if type(value) is not CompletionVerificationClaim:
                    raise WorkCompletionConflict("Worker verifier claim has an invalid type.")
                result = CompletionVerificationClaim.model_validate(
                    value.model_dump(mode="python", warnings=False)
                )
                if result.proposal_id != proposal_id or result.verifier != contract.verifier:
                    raise WorkCompletionConflict(
                        "Worker verifier claim belongs to another proposal."
                    )
                return result

            claim = self._validate(copy_claim, "Worker verifier claim validation")
        finally:
            del raw_claim, copy_claim
        return claim

    async def _verification_request(
        self, proposal_id: str, contract: WorkContract
    ) -> tuple[CompletionVerifierExecutionRequest | None, CompletionDecision | None]:
        claim = await self._load_verifier_claim(proposal_id, contract)

        raw_decision = await self._operation(
            lambda: self.store.load_completion_decision_for_proposal(proposal_id),
            name="Worker verifier decision lookup",
        )
        try:

            def require_decision(value=raw_decision):
                if value is None:
                    return None
                if type(value) is not CompletionDecision:
                    raise WorkCompletionConflict("Worker verifier decision has an invalid type.")
                copied = copy_completion_decision(value)
                if claim is None:
                    raise WorkCompletionConflict("Worker decision has no durable verifier claim.")
                if (
                    copied.proposal_id != proposal_id
                    or copied.claim_id != claim.claim_id
                    or copied.worker_id != claim.worker_id
                    or copied.contract != contract.reference()
                    or copied.verifier != claim.verifier
                    or copied.verifier_profile_fingerprint != claim.verifier_profile_fingerprint
                ):
                    raise WorkCompletionConflict(
                        "Worker verifier decision conflicts with its claim."
                    )
                return copied

            decision = self._validate(require_decision, "Worker verifier decision validation")
        finally:
            del raw_decision, require_decision

        attempt_number = 1
        if claim is not None:
            if decision is not None:
                return CompletionVerifierExecutionRequest(
                    proposal_id=proposal_id,
                    claim_id=claim.claim_id,
                    decision_id=decision.decision_id,
                    worker_id=claim.worker_id,
                    lease_seconds=claim.lease_seconds,
                    # Store-only claims explicitly have no execution timeout.
                    # The coordinator authenticates that None in its committed
                    # decision branch; this bounded placeholder never dispatches.
                    execution_timeout_seconds=(
                        claim.execution_timeout_seconds
                        if claim.execution_timeout_seconds is not None
                        else min(_VERIFIER_TIMEOUT_SECONDS, claim.lease_seconds / 2)
                    ),
                ), decision
            if claim.lease_expires_at > datetime.now(UTC):
                return None, None
            attempt_number = claim.attempt_number + 1
        return CompletionVerifierExecutionRequest(
            proposal_id=proposal_id,
            claim_id=self._verifier_claim_id(proposal_id, contract, attempt_number),
            decision_id=verified_task_operation_id("decision", proposal_id),
            worker_id=self.worker_id,
            lease_seconds=_VERIFIER_LEASE_SECONDS,
            execution_timeout_seconds=_VERIFIER_TIMEOUT_SECONDS,
        ), None
