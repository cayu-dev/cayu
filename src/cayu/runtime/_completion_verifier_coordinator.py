"""Runtime-owned deterministic completion-verifier execution boundary."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TypeVar, cast
from uuid import uuid4

from pydantic import BaseModel

from cayu._exception_groups import exception_group_children
from cayu._task_wait import (
    CapturedAwaitableOutcome,
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    restore_task_cancellation_requests,
)
from cayu._validation import revalidate_model_input
from cayu.runtime._diagnostics import (
    MAX_DIAGNOSTIC_UTF8_BYTES,
    credential_safe_runtime_exception,
    credential_safe_runtime_exception_group,
    exception_diagnostic,
)
from cayu.runtime._task_store_operation_boundary import (
    TaskStoreOperationOutcome,
    capture_sensitive_validation,
    capture_task_store_operation,
)
from cayu.runtime.completion_verifiers import (
    CompletionVerifierExecutionError,
    CompletionVerifierExecutionRequest,
    CompletionVerifierRequest,
    CompletionVerifierUnavailable,
    DeterministicCompletionVerifier,
    copy_completion_verifier_execution_request,
)
from cayu.runtime.tasks import TaskClaimLost, TaskStore
from cayu.runtime.work_contracts import (
    CompletionDecision,
    CompletionDecisionCreate,
    CompletionProposal,
    CompletionProposalCreate,
    CompletionVerificationClaim,
    CompletionVerificationClaimLost,
    CompletionVerificationClaimRequest,
    CompletionVerifierDecision,
    CompletionVerifierKind,
    CompletionVerifierRef,
    TaskCompletionDecisionRequired,
    WorkAttempt,
    WorkAttemptCreate,
    WorkCompletionConflict,
    WorkContract,
    WorkContractConflict,
    completion_decision_request_sha256,
    completion_gap_fingerprint,
    completion_proposal_request_sha256,
    completion_verification_claim_request_sha256,
    copy_completion_decision,
    copy_completion_proposal,
    copy_completion_verification_claim,
    copy_completion_verifier_decision,
    copy_work_attempt,
    copy_work_contract,
    validate_completion_decision_contract,
    work_attempt_request_sha256,
)
from cayu.runtime.workspace_observation_recovery import (
    retain_workspace_observation_pending_cancellation_requests,
    workspace_observation_pending_cancellation_requests,
)
from cayu.vaults import SecretRedactor

_ModelT = TypeVar("_ModelT", bound=BaseModel)
_ExecutionKey = str
_MAX_ACTIVE_COMPLETION_VERIFIERS = 64
_BASE_EXCEPTION_ARGS_DESCRIPTOR = BaseException.__dict__["args"]


class _ClaimHeartbeatCancellationMarker:
    """Per-heartbeat provenance for cancellation forwarded to its adapter."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<completion-verifier-heartbeat-cancellation>"

    __str__ = __repr__


class _ClaimHeartbeatShutdownMarker:
    """Per-heartbeat provenance for runtime-owned heartbeat shutdown."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<completion-verifier-heartbeat-shutdown>"

    __str__ = __repr__


@dataclass(slots=True)
class _SingleFlightLock:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


@dataclass(frozen=True, slots=True)
class _VerifierAuthority:
    proposal: CompletionProposal = field(repr=False)
    attempt: WorkAttempt = field(repr=False)
    contract: WorkContract = field(repr=False)
    adapter_request: CompletionVerifierRequest = field(repr=False)


@dataclass(frozen=True, slots=True)
class _TaskStoreOwner:
    store: TaskStore = field(repr=False)


@dataclass(slots=True)
class _ClaimHeartbeat:
    stop: asyncio.Event
    task: asyncio.Task[None]
    cancellation_marker: _ClaimHeartbeatCancellationMarker
    shutdown_marker: _ClaimHeartbeatShutdownMarker
    retained_for_drain: bool = False
    observed_failure_id: int | None = None
    shutdown_requested: bool = False


@dataclass(frozen=True, slots=True)
class _ClaimHeartbeatSettlement:
    failure: BaseException | None = None
    caller_cancellation: asyncio.CancelledError | None = None
    cancellation_requests_consumed: int = 0


@dataclass(slots=True)
class _DrainingAdapter:
    task: asyncio.Task[CapturedAwaitableOutcome[CompletionVerifierDecision]]
    heartbeat: _ClaimHeartbeat
    settlement_task: asyncio.Task[_ClaimHeartbeatSettlement] | None = None


def _verifier_key(reference: CompletionVerifierRef) -> tuple[str, str, str, str]:
    return (
        reference.verifier_id,
        reference.version,
        reference.kind.value,
        reference.configuration_fingerprint,
    )


def _contains_workload_secret(
    values: tuple[str, ...],
    *,
    redactor: SecretRedactor,
) -> bool:
    return any(redactor.redact_text(value) != value for value in values)


def _copy_exact_model(
    value: object,
    model_type: type[_ModelT],
    *,
    operation_name: str,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[_ModelT]:
    if type(value) is not model_type:
        return TaskStoreOperationOutcome()
    copier: Callable[[object], object]
    if model_type is WorkContract:
        copier = cast("Callable[[object], object]", copy_work_contract)
    elif model_type is WorkAttempt:
        copier = cast("Callable[[object], object]", copy_work_attempt)
    elif model_type is CompletionProposal:
        copier = cast("Callable[[object], object]", copy_completion_proposal)
    elif model_type is CompletionVerificationClaim:
        copier = cast("Callable[[object], object]", copy_completion_verification_claim)
    elif model_type is CompletionDecision:
        copier = cast("Callable[[object], object]", copy_completion_decision)
    else:

        def copier(candidate: object) -> object:
            return revalidate_model_input(candidate, model_type)

    return capture_sensitive_validation(
        lambda: cast("_ModelT", copier(value)),
        operation_name=operation_name,
        redactor=redactor,
    )


def _validate_decision_contract(
    contract: WorkContract,
    request: CompletionDecisionCreate,
) -> bool:
    validate_completion_decision_contract(contract, request)
    return True


def _detached_verifier_failure(
    error: BaseException,
    *,
    redactor: SecretRedactor,
) -> BaseException:
    """Detach an adapter failure from its traceback, context, and argument graph."""

    if isinstance(error, BaseExceptionGroup):
        return credential_safe_runtime_exception_group(
            error,
            group_message="Completion verifier reported multiple failures.",
            leaf_mapper=lambda leaf: _detached_verifier_failure(leaf, redactor=redactor),
            invalid_leaf_factory=lambda: CompletionVerifierExecutionError(
                "Completion verifier reported invalid failure evidence."
            ),
            truncated_leaf_factory=lambda: CompletionVerifierExecutionError(
                "Additional completion verifier failures were omitted."
            ),
            fallback_leaf_mapper=lambda leaf: _generic_verifier_failure(
                leaf,
                redactor=redactor,
            ),
            redactor=redactor,
        )
    diagnostic = exception_diagnostic(
        error,
        empty_message="completion verifier failed",
        nonportable_message="Completion verifier failed with a non-portable diagnostic.",
        redactor=redactor,
    )
    message = diagnostic.message
    if isinstance(error, GeneratorExit):
        return credential_safe_runtime_exception(
            GeneratorExit,
            message,
            redactor=redactor,
            fallback_message="Completion verifier process-control diagnostic was redacted.",
        )
    if isinstance(error, KeyboardInterrupt):
        return credential_safe_runtime_exception(
            KeyboardInterrupt,
            message,
            redactor=redactor,
            fallback_message="Completion verifier process-control diagnostic was redacted.",
        )
    if isinstance(error, SystemExit):
        return credential_safe_runtime_exception(
            SystemExit,
            message,
            redactor=redactor,
            fallback_message="Completion verifier process-control diagnostic was redacted.",
        )
    if isinstance(error, asyncio.CancelledError):
        return credential_safe_runtime_exception(
            CompletionVerifierExecutionError,
            "Completion verifier was cancelled without caller cancellation.",
            redactor=redactor,
            fallback_message="Completion verifier cancellation diagnostic was redacted.",
        )
    return credential_safe_runtime_exception(
        CompletionVerifierExecutionError,
        redactor.redact_text_bounded(
            f"{diagnostic.error_type}: {message}",
            max_bytes=MAX_DIAGNOSTIC_UTF8_BYTES,
        ),
        redactor=redactor,
        fallback_message="Completion verifier failure diagnostic was redacted.",
    )


def _generic_verifier_failure(
    error: BaseException,
    *,
    redactor: SecretRedactor,
) -> BaseException:
    """Retain runtime failure classification without retaining diagnostic text."""

    if isinstance(error, GeneratorExit):
        safe_type: type[BaseException] = GeneratorExit
    elif isinstance(error, KeyboardInterrupt):
        safe_type = KeyboardInterrupt
    elif isinstance(error, SystemExit):
        safe_type = SystemExit
    elif isinstance(error, asyncio.CancelledError):
        safe_type = asyncio.CancelledError
    elif type(error) in {
        CompletionVerifierExecutionError,
        CompletionVerifierUnavailable,
        TaskClaimLost,
        CompletionVerificationClaimLost,
        TaskCompletionDecisionRequired,
        WorkCompletionConflict,
        WorkContractConflict,
        TimeoutError,
        ConnectionError,
        ValueError,
        TypeError,
        NotImplementedError,
        RuntimeError,
    }:
        safe_type = type(error)
    elif isinstance(error, Exception):
        safe_type = CompletionVerifierExecutionError
    else:
        safe_type = BaseException
    return credential_safe_runtime_exception(
        safe_type,
        "Completion verifier failure diagnostic was redacted.",
        redactor=redactor,
        fallback_message="Completion verifier failure diagnostic was withheld.",
    )


def _is_claim_heartbeat_cancellation(
    error: BaseException | None,
    marker: _ClaimHeartbeatCancellationMarker,
) -> bool:
    """Recognize only the cancellation minted for one exact heartbeat failure."""

    return isinstance(error, asyncio.CancelledError) and _is_exception_marker(error, marker)


def _is_exception_marker(error: BaseException, marker: object) -> bool:
    """Read one base-owned exception argument without extension dispatch."""

    if not isinstance(error, BaseException):
        return False
    try:
        args = _BASE_EXCEPTION_ARGS_DESCRIPTOR.__get__(error, BaseException)
    except BaseException:
        return False
    return type(args) is tuple and len(args) == 1 and args[0] is marker


def _exception_tree_has_cancellation(error: BaseException) -> bool:
    pending = [error]
    observed: set[int] = set()
    while pending:
        candidate = pending.pop()
        if id(candidate) in observed:
            continue
        observed.add(id(candidate))
        if isinstance(candidate, asyncio.CancelledError):
            return True
        if isinstance(candidate, BaseExceptionGroup):
            children = exception_group_children(candidate)
            if children is not None:
                pending.extend(children)
    return False


def _adapter_failure_after_heartbeat(
    error: BaseException | None,
    *,
    heartbeat: _ClaimHeartbeat,
    heartbeat_failure: BaseException | None,
) -> BaseException | None:
    """Discard only the control signal caused by the preserved heartbeat failure."""

    if heartbeat_failure is not None and error is not None:
        return _prune_exception_graph(
            error,
            should_prune=lambda leaf: _is_claim_heartbeat_cancellation(
                leaf,
                heartbeat.cancellation_marker,
            ),
        )
    return error


def _prune_exception_graph(
    error: BaseException,
    *,
    should_prune: Callable[[BaseException], bool],
) -> BaseException | None:
    """Remove authenticated leaves without recursive or extension-owned traversal."""

    next_token = 1
    pending: list[tuple[int, BaseException, bool]] = [(0, error, False)]
    child_tokens_by_group: dict[int, tuple[int, ...]] = {}
    rebuilt_by_token: dict[int, BaseException | None] = {}
    observed: set[int] = set()
    while pending:
        token, candidate, expanded = pending.pop()
        if expanded:
            children = [
                rebuilt_by_token.pop(child_token, None)
                for child_token in child_tokens_by_group.pop(token, ())
            ]
            retained = [child for child in children if child is not None]
            rebuilt_by_token[token] = (
                BaseExceptionGroup(
                    "Completion verifier reported multiple failures.",
                    retained,
                )
                if retained
                else None
            )
            continue

        candidate_id = id(candidate)
        if candidate_id in observed:
            rebuilt_by_token[token] = None
            continue
        observed.add(candidate_id)
        if not isinstance(candidate, BaseExceptionGroup):
            try:
                prune = should_prune(candidate)
            except BaseException:
                prune = False
            rebuilt_by_token[token] = None if prune else candidate
            continue
        children = exception_group_children(candidate)
        if children is None:
            rebuilt_by_token[token] = candidate
            continue
        child_tokens = tuple(range(next_token, next_token + len(children)))
        next_token += len(children)
        child_tokens_by_group[token] = child_tokens
        pending.append((token, candidate, True))
        pending.extend(
            (child_token, child, False)
            for child_token, child in reversed(tuple(zip(child_tokens, children, strict=True)))
        )
    return rebuilt_by_token.get(0)


def _ordered_execution_failures(
    *,
    adapter_failures: tuple[BaseException | None, ...] = (),
    boundary_failures: tuple[BaseException | None, ...] = (),
    redactor: SecretRedactor,
) -> tuple[BaseException, ...]:
    """Detach adapter failures and preserve every distinct boundary failure once."""

    ordered: list[BaseException] = []
    observed: set[int] = set()
    for failure in adapter_failures:
        if failure is None or id(failure) in observed:
            continue
        observed.add(id(failure))
        ordered.append(_detached_verifier_failure(failure, redactor=redactor))
    for failure in boundary_failures:
        if failure is None or id(failure) in observed:
            continue
        observed.add(id(failure))
        ordered.append(failure)
    return tuple(ordered)


def _safe_caller_cancellation(
    cancellation: asyncio.CancelledError,
    *,
    redactor: SecretRedactor,
) -> asyncio.CancelledError:
    diagnostic = exception_diagnostic(
        cancellation,
        empty_message="completion verifier execution cancelled",
        nonportable_message="Completion verifier execution cancellation had a non-portable diagnostic.",
        redactor=redactor,
    )
    return credential_safe_runtime_exception(
        asyncio.CancelledError,
        diagnostic.message,
        redactor=redactor,
        fallback_message="Completion verifier cancellation diagnostic was redacted.",
    )


def _ordered_failure_group(
    message: str,
    *failures: BaseException,
    redactor: SecretRedactor,
) -> BaseExceptionGroup:
    """Preserve distinct failures in operation order and cancellation authority."""

    ordered: list[BaseException] = []
    for failure in failures:
        if all(failure is not existing for existing in ordered):
            ordered.append(failure)
    group = (
        ExceptionGroup(message, cast("list[Exception]", ordered))
        if all(isinstance(failure, Exception) for failure in ordered)
        else BaseExceptionGroup(message, ordered)
    )
    pending_cancellation_requests = max(
        (workspace_observation_pending_cancellation_requests(failure) for failure in ordered),
        default=0,
    )
    if pending_cancellation_requests:
        retain_workspace_observation_pending_cancellation_requests(
            group,
            pending_cancellation_requests,
        )
    group = credential_safe_runtime_exception_group(
        group,
        group_message=message,
        leaf_mapper=lambda leaf: leaf,
        invalid_leaf_factory=lambda: CompletionVerifierExecutionError(
            "Concurrent failure group was invalid."
        ),
        truncated_leaf_factory=lambda: CompletionVerifierExecutionError(
            "Additional concurrent failures were omitted."
        ),
        fallback_leaf_mapper=lambda leaf: _generic_verifier_failure(
            leaf,
            redactor=redactor,
        ),
        redactor=redactor,
    )
    if pending_cancellation_requests:
        retain_workspace_observation_pending_cancellation_requests(
            group,
            pending_cancellation_requests,
        )
    return group


class CompletionVerifierCoordinator:
    """Resolve application adapters and bind their output to durable authority."""

    def __init__(
        self,
        *,
        task_store: TaskStore | None,
        secret_redactor: SecretRedactor,
    ) -> None:
        if task_store is not None and not isinstance(task_store, TaskStore):
            raise TypeError("Completion verifier coordinator requires a TaskStore.")
        if not isinstance(secret_redactor, SecretRedactor):
            raise TypeError("Completion verifier coordinator requires a SecretRedactor.")
        self._task_store = task_store
        self._secret_redactor = secret_redactor
        self._execution_owner_process_id = os.getpid()
        self._execution_owner_id = f"cver_{uuid4().hex}"
        self._verifiers: dict[tuple[str, str, str, str], DeterministicCompletionVerifier] = {}
        self._locks: dict[_ExecutionKey, _SingleFlightLock] = {}
        self._adapter_tasks: set[
            asyncio.Task[CapturedAwaitableOutcome[CompletionVerifierDecision]]
        ] = set()
        self._adapter_capacity_reservations: set[object] = set()
        self._draining_adapter_tasks: dict[_ExecutionKey, _DrainingAdapter] = {}
        self._claim_heartbeat_tasks: set[asyncio.Task[None]] = set()

    def _ensure_process_local_generation(self) -> None:
        """Refresh quiescent runtime ownership after a process fork."""

        process_id = os.getpid()
        if process_id == self._execution_owner_process_id:
            return
        if (
            self._locks
            or self._adapter_tasks
            or self._adapter_capacity_reservations
            or self._draining_adapter_tasks
            or self._claim_heartbeat_tasks
        ):
            raise CompletionVerifierExecutionError(
                "Completion verifier coordinator inherited active execution state "
                "across a process boundary; rebuild the application in this worker."
            ) from None
        self._execution_owner_process_id = process_id
        self._execution_owner_id = f"cver_{uuid4().hex}"

    def register(
        self,
        reference: CompletionVerifierRef,
        verifier: DeterministicCompletionVerifier,
    ) -> CompletionVerifierRef:
        self._ensure_process_local_generation()
        if type(reference) is not CompletionVerifierRef:
            del reference, verifier
            raise TypeError("Completion verifier registration requires a CompletionVerifierRef.")
        if not isinstance(verifier, DeterministicCompletionVerifier):
            del reference, verifier
            raise TypeError(
                "Completion verifier registration requires a DeterministicCompletionVerifier."
            )
        reference_value = reference
        del reference
        validation = capture_sensitive_validation(
            lambda value=reference_value: cast(
                "CompletionVerifierRef",
                revalidate_model_input(value, CompletionVerifierRef),
            ),
            operation_name="Completion verifier reference validation",
            redactor=self._secret_redactor,
        )
        del reference_value
        if validation.failure is not None:
            failure = validation.failure
            del validation, verifier
            raise failure from None
        copied = validation.result
        del validation
        if copied is None:
            del verifier
            raise ValueError("Completion verifier reference is invalid.") from None
        if copied.kind is not CompletionVerifierKind.DETERMINISTIC:
            del copied, verifier
            raise ValueError("Only deterministic completion verifiers can be registered.")
        if _contains_workload_secret(
            (
                copied.verifier_id,
                copied.version,
                copied.configuration_fingerprint,
            ),
            redactor=self._secret_redactor,
        ):
            del copied, verifier
            raise ValueError(
                "Completion verifier identity contains a workload secret and cannot be "
                "registered as durable authority."
            ) from None
        key = _verifier_key(copied)
        if key in self._verifiers:
            del copied, key, verifier
            raise credential_safe_runtime_exception(
                ValueError,
                "Completion verifier identity is already registered.",
                redactor=self._secret_redactor,
                fallback_message="Completion verifier registration conflict.",
            ) from None
        self._verifiers[key] = verifier
        del verifier
        return copied

    async def verify(
        self,
        request: CompletionVerifierExecutionRequest,
    ) -> CompletionDecision:
        self._ensure_process_local_generation()
        request_value = request
        del request
        validation = capture_sensitive_validation(
            lambda value=request_value: copy_completion_verifier_execution_request(value),
            operation_name="Completion verifier execution request validation",
            redactor=self._secret_redactor,
        )
        del request_value
        if validation.failure is not None:
            raise validation.failure from None
        copied = validation.result
        del validation
        if copied is None:
            raise ValueError("Completion verifier execution request is invalid.") from None
        if _contains_workload_secret(
            (
                copied.proposal_id,
                copied.claim_id,
                copied.decision_id,
                copied.worker_id,
            ),
            redactor=self._secret_redactor,
        ):
            del copied
            raise ValueError(
                "Completion-verifier execution identity contains a workload secret."
            ) from None
        key = copied.proposal_id
        entry = self._locks.get(key)
        if entry is None:
            entry = _SingleFlightLock()
            self._locks[key] = entry
        entry.users += 1
        try:
            safe_cancellation: asyncio.CancelledError | None = None
            try:
                await entry.lock.acquire()
            except asyncio.CancelledError as cancellation:
                safe_cancellation = _safe_caller_cancellation(
                    cancellation,
                    redactor=self._secret_redactor,
                )
                del cancellation, copied
            if safe_cancellation is not None:
                raise safe_cancellation from None
            try:
                return await self._verify_locked(copied)
            finally:
                entry.lock.release()
        finally:
            entry.users -= 1
            if entry.users == 0 and self._locks.get(key) is entry:
                self._locks.pop(key, None)

    async def _verify_locked(
        self,
        request: CompletionVerifierExecutionRequest,
    ) -> CompletionDecision:
        store_owner = _TaskStoreOwner(self._require_store())
        authority = await self._load_verifier_authority(store_owner, request)

        existing = await self._load_converged_decision(
            store_owner,
            decision_id=request.decision_id,
            proposal_id=request.proposal_id,
        )
        if existing is not None:
            claim: CompletionVerificationClaim | None = None
            existing_validation = None
            try:
                claim = await self._load_required_claim(
                    store_owner,
                    request.proposal_id,
                )
                self._require_exact_claim(
                    claim,
                    request,
                    authority.contract.verifier,
                    execution_owner_id=claim.execution_owner_id,
                    accept_legacy_timeout=True,
                )
                existing_validation = capture_sensitive_validation(
                    lambda: self._require_existing_decision(
                        existing,
                        request=request,
                        proposal=authority.proposal,
                        attempt=authority.attempt,
                        contract=authority.contract,
                        claim=cast("CompletionVerificationClaim", claim),
                    ),
                    operation_name="Existing completion decision validation",
                    redactor=self._secret_redactor,
                )
                if existing_validation.failure is not None:
                    raise existing_validation.failure from None
                if existing_validation.result is not True:
                    raise WorkCompletionConflict(
                        "Durable completion decision has conflicting authority."
                    ) from None
                return existing
            except BaseException as failure:
                del existing
                del claim, existing_validation, authority, store_owner
                raise failure from None

        if authority.contract.verifier.kind is not CompletionVerifierKind.DETERMINISTIC:
            raise credential_safe_runtime_exception(
                CompletionVerifierUnavailable,
                "Provider-backed completion verifiers are not supported by this runtime slice.",
                redactor=self._secret_redactor,
                fallback_message="The required completion verifier is unavailable.",
            ) from None
        verifier_key = _verifier_key(authority.contract.verifier)
        if verifier_key not in self._verifiers:
            raise credential_safe_runtime_exception(
                CompletionVerifierUnavailable,
                "The exact deterministic completion verifier required by the work contract "
                "is not registered.",
                redactor=self._secret_redactor,
                fallback_message="The required completion verifier is unavailable.",
            ) from None

        operation_key = self._execution_key(request)
        draining = self._draining_adapter_tasks.get(operation_key)
        if draining is not None and not draining.task.done():
            raise CompletionVerifierExecutionError(
                "The prior exact completion-verifier execution is still draining."
            ) from None
        if draining is not None:
            settlement_task = self._ensure_draining_adapter_settlement(
                operation_key,
                draining,
            )
            if not settlement_task.done():
                raise CompletionVerifierExecutionError(
                    "The prior exact completion-verifier execution is still draining."
                ) from None
            if not self._finalize_draining_adapter_settlement(
                operation_key,
                draining,
                settlement_task,
            ):
                raise CompletionVerifierExecutionError(
                    "The prior exact completion-verifier execution is still draining."
                ) from None

        capacity_reservation = self._reserve_adapter_capacity()
        verifier = self._verifiers[verifier_key]
        heartbeat: _ClaimHeartbeat | None = None
        try:
            claim_request = CompletionVerificationClaimRequest(
                claim_id=request.claim_id,
                proposal_id=request.proposal_id,
                worker_id=request.worker_id,
                execution_owner_id=self._execution_owner_id,
                verifier=authority.contract.verifier,
                lease_seconds=request.lease_seconds,
                execution_timeout_seconds=request.execution_timeout_seconds,
            )
            claim_outcome = await capture_task_store_operation(
                lambda: store_owner.store.claim_completion_verification(claim_request),
                operation_name="Completion verification claim",
                redactor=self._secret_redactor,
                mutation_store=store_owner.store,
                mutation_method_name="claim_completion_verification",
            )
            if claim_outcome.failure is not None:
                raise claim_outcome.failure from None
            claim_validation = _copy_exact_model(
                claim_outcome.result,
                CompletionVerificationClaim,
                operation_name="Completion verification claim result validation",
                redactor=self._secret_redactor,
            )
            del claim_outcome
            if claim_validation.failure is not None:
                raise claim_validation.failure from None
            claim = claim_validation.result
            del claim_validation
            if claim is None:
                raise WorkCompletionConflict(
                    "Task store returned an invalid completion verification claim."
                ) from None
            self._require_exact_claim(
                claim,
                request,
                authority.contract.verifier,
                execution_owner_id=self._execution_owner_id,
            )
            claim = await self._renew_claim(
                store_owner,
                claim_request,
                request,
                prior_claim=claim,
            )
            heartbeat = self._start_claim_heartbeat(
                store_owner,
                claim_request,
                request,
                claim,
            )

            outcome = await self._invoke_adapter(
                verifier,
                authority.adapter_request,
                operation_key=operation_key,
                timeout_seconds=request.execution_timeout_seconds,
                capacity_reservation=capacity_reservation,
                heartbeat=heartbeat,
            )
        except BaseException as failure:
            if heartbeat is not None and not heartbeat.retained_for_drain:
                settlement = await self._settle_claim_heartbeat(heartbeat)
                propagated = self._failure_after_heartbeat_settlement(
                    failure,
                    settlement,
                    group_message=(
                        "Completion verifier execution and heartbeat settlement failed."
                    ),
                )
                del settlement
            else:
                propagated = failure
            del verifier, authority, store_owner
            if propagated is None:  # pragma: no cover - primary failure is authoritative
                raise AssertionError("Completion verifier failure was lost.") from None
            raise propagated from None
        finally:
            self._release_adapter_capacity_reservation(capacity_reservation)
        try:
            decision = await self._publish_adapter_outcome(
                store_owner=store_owner,
                request=request,
                proposal=authority.proposal,
                attempt=authority.attempt,
                contract=authority.contract,
                outcome=outcome,
            )
        except BaseException as failure:
            if heartbeat is not None:
                settlement = await self._settle_claim_heartbeat(heartbeat)
                propagated = self._failure_after_heartbeat_settlement(
                    failure,
                    settlement,
                    group_message=(
                        "Completion decision publication and heartbeat settlement failed."
                    ),
                )
                del settlement
            else:
                propagated = failure
            del outcome, verifier, claim, claim_request, authority, store_owner
            if propagated is None:  # pragma: no cover - primary failure is authoritative
                raise AssertionError("Completion decision publication failure was lost.") from None
            raise propagated from None
        if heartbeat is not None:
            settlement = await self._settle_claim_heartbeat(heartbeat)
            propagated = self._failure_after_heartbeat_settlement(
                None,
                settlement,
                group_message="Completion verifier heartbeat settlement failed.",
            )
            del settlement
            if propagated is not None:
                del decision, outcome, verifier, claim, claim_request, authority, store_owner
                raise propagated from None
        return decision

    async def _load_verifier_authority(
        self,
        store_owner: _TaskStoreOwner,
        request: CompletionVerifierExecutionRequest,
    ) -> _VerifierAuthority:
        proposal: CompletionProposal | None = None
        attempt: WorkAttempt | None = None
        contract: WorkContract | None = None
        context_validation = None
        adapter_request: CompletionVerifierRequest | None = None
        try:
            proposal = await self._load_required(
                lambda owner=store_owner, proposal_id=request.proposal_id: (
                    owner.store.load_completion_proposal(proposal_id)
                ),
                CompletionProposal,
                identity=request.proposal_id,
                identity_field="proposal_id",
                operation_name="Completion proposal lookup",
            )
            attempt = await self._load_required(
                lambda owner=store_owner, attempt_id=proposal.attempt_id: (
                    owner.store.load_work_attempt(attempt_id)
                ),
                WorkAttempt,
                identity=proposal.attempt_id,
                identity_field="attempt_id",
                operation_name="Work attempt lookup",
            )
            contract = await self._load_required(
                lambda owner=store_owner, reference=proposal.contract: (
                    owner.store.load_work_contract(reference)
                ),
                WorkContract,
                identity=proposal.contract.contract_id,
                identity_field="contract_id",
                operation_name="Work contract lookup",
            )
            if _contains_workload_secret(
                (
                    contract.contract_id,
                    contract.verifier.verifier_id,
                    contract.verifier.version,
                    contract.verifier.configuration_fingerprint,
                    *(criterion.criterion_id for criterion in contract.criteria),
                    *(constraint.constraint_id for constraint in contract.constraints),
                    *(requirement.requirement_id for requirement in contract.evidence_requirements),
                ),
                redactor=self._secret_redactor,
            ):
                raise ValueError(
                    "Durable completion-verifier context contains a workload secret in public "
                    "identity."
                ) from None
            context_validation = capture_sensitive_validation(
                lambda contract_value=contract, attempt_value=attempt, proposal_value=proposal: (
                    CompletionVerifierRequest(
                        contract=contract_value,
                        attempt=attempt_value,
                        proposal=proposal_value,
                    )
                ),
                operation_name="Completion verifier context validation",
                redactor=self._secret_redactor,
            )
            if context_validation.failure is not None:
                raise context_validation.failure from None
            adapter_request = context_validation.result
            context_validation = None
            if adapter_request is None:
                raise WorkCompletionConflict(
                    "Durable verifier context has conflicting authority."
                ) from None
            self._require_context_integrity(
                proposal=proposal,
                attempt=attempt,
            )
            return _VerifierAuthority(
                proposal=proposal,
                attempt=attempt,
                contract=contract,
                adapter_request=adapter_request,
            )
        except BaseException as failure:
            del proposal, attempt, contract, context_validation, adapter_request, store_owner
            raise failure from None

    async def _publish_adapter_outcome(
        self,
        *,
        store_owner: _TaskStoreOwner,
        request: CompletionVerifierExecutionRequest,
        proposal: CompletionProposal,
        attempt: WorkAttempt,
        contract: WorkContract,
        outcome: CompletionVerifierDecision,
    ) -> CompletionDecision:
        decision_request = CompletionDecisionCreate(
            decision_id=request.decision_id,
            proposal_id=request.proposal_id,
            claim_id=request.claim_id,
            worker_id=request.worker_id,
            verifier=contract.verifier,
            verdict=outcome.verdict,
            criterion_outcomes=outcome.criterion_outcomes,
            constraint_outcomes=outcome.constraint_outcomes,
            gaps=outcome.gaps,
            evidence_references=outcome.evidence_references,
        )
        del outcome
        record_outcome = None
        reconciled: CompletionDecision | None = None
        decision_validation = None
        decision: CompletionDecision | None = None
        indexed_decision: CompletionDecision | None = None
        contract_validation = None
        try:
            contract_validation = capture_sensitive_validation(
                lambda: _validate_decision_contract(contract, decision_request),
                operation_name="Completion verifier contract validation",
                redactor=self._secret_redactor,
            )
            if contract_validation.failure is not None:
                raise contract_validation.failure from None
            if contract_validation.result is not True:
                raise WorkCompletionConflict(
                    "Completion verifier decision conflicts with the frozen contract."
                ) from None
            contract_validation = None
            record_outcome = await capture_task_store_operation(
                lambda owner=store_owner, value=decision_request: (
                    owner.store.record_completion_decision(value)
                ),
                operation_name="Completion decision publication",
                redactor=self._secret_redactor,
                mutation_store=store_owner.store,
                mutation_method_name="record_completion_decision",
            )
            if record_outcome.failure is not None:
                failure = record_outcome.failure
                record_outcome = None
                # Read-only acknowledgement-loss reconciliation is valid only
                # for ordinary failures.  Cancellation and process-control
                # signals remain authoritative even if the mutation committed.
                if not isinstance(failure, Exception):
                    raise failure from None
                try:
                    reconciled = await self._load_converged_decision(
                        store_owner,
                        decision_id=request.decision_id,
                        proposal_id=request.proposal_id,
                    )
                    reconciliation_matches = (
                        reconciled is not None
                        and self._decision_matches_request(
                            reconciled,
                            decision_request,
                            proposal=proposal,
                            attempt=attempt,
                            contract=contract,
                        )
                    )
                except BaseException as reconciliation_failure:
                    combined = _ordered_failure_group(
                        "Completion decision publication and reconciliation failed.",
                        failure,
                        reconciliation_failure,
                        redactor=self._secret_redactor,
                    )
                    del failure, reconciliation_failure
                    raise combined from None
                if reconciliation_matches:
                    return reconciled
                raise failure from None
            decision_validation = _copy_exact_model(
                record_outcome.result,
                CompletionDecision,
                operation_name="Completion decision result validation",
                redactor=self._secret_redactor,
            )
            record_outcome = None
            if decision_validation.failure is not None:
                raise decision_validation.failure from None
            decision = decision_validation.result
            decision_validation = None
            if decision is None or not self._decision_matches_request(
                decision,
                decision_request,
                proposal=proposal,
                attempt=attempt,
                contract=contract,
            ):
                raise WorkCompletionConflict(
                    "Task store returned a completion decision other than the exact publication."
                ) from None
            indexed_decision = await self._load_converged_decision(
                store_owner,
                decision_id=request.decision_id,
                proposal_id=request.proposal_id,
            )
            if indexed_decision != decision:
                raise WorkCompletionConflict(
                    "Completion decision publication did not converge in both durable indexes."
                ) from None
            return decision
        except BaseException as failure:
            del decision_request, contract, attempt, proposal, store_owner
            del record_outcome, reconciled, decision_validation, decision, indexed_decision
            del contract_validation
            raise failure from None

    def _require_store(self) -> TaskStore:
        store = self._task_store
        if store is None:
            raise RuntimeError("task_store is required to execute completion verifiers.") from None
        if not store.supports_verified_work_contracts:
            raise NotImplementedError(
                f"{type(store).__name__} does not support verified work contracts."
            ) from None
        return store

    async def _load_required(
        self,
        operation: Callable[[], Awaitable[object]],
        model_type: type[_ModelT],
        *,
        identity: str,
        identity_field: str,
        operation_name: str,
    ) -> _ModelT:
        outcome = await capture_task_store_operation(
            operation,
            operation_name=operation_name,
            redactor=self._secret_redactor,
        )
        if outcome.failure is not None:
            raise outcome.failure from None
        if outcome.result is None:
            raise KeyError(f"{operation_name.removesuffix(' lookup')} not found.") from None
        validation = _copy_exact_model(
            outcome.result,
            model_type,
            operation_name=f"{operation_name} result validation",
            redactor=self._secret_redactor,
        )
        del outcome
        if validation.failure is not None:
            raise validation.failure from None
        value = validation.result
        del validation
        if value is None or getattr(value, identity_field, None) != identity:
            del value, operation
            raise WorkCompletionConflict(
                f"Task store returned {operation_name.lower()} authority for another identity."
            ) from None
        return value

    async def _load_required_claim(
        self,
        store_owner: _TaskStoreOwner,
        proposal_id: str,
    ) -> CompletionVerificationClaim:
        return await self._load_required(
            lambda: store_owner.store.load_completion_verification_claim(proposal_id),
            CompletionVerificationClaim,
            identity=proposal_id,
            identity_field="proposal_id",
            operation_name="Completion verification claim lookup",
        )

    async def _load_optional_decision(
        self,
        store_owner: _TaskStoreOwner,
        decision_id: str,
    ) -> CompletionDecision | None:
        outcome = await capture_task_store_operation(
            lambda: store_owner.store.load_completion_decision(decision_id),
            operation_name="Completion decision lookup",
            redactor=self._secret_redactor,
        )
        if outcome.failure is not None:
            raise outcome.failure from None
        if outcome.result is None:
            return None
        validation = _copy_exact_model(
            outcome.result,
            CompletionDecision,
            operation_name="Completion decision lookup result validation",
            redactor=self._secret_redactor,
        )
        del outcome
        if validation.failure is not None:
            raise validation.failure from None
        decision = validation.result
        del validation
        if decision is None or decision.decision_id != decision_id:
            del decision, store_owner
            raise WorkCompletionConflict(
                "Task store returned completion decision authority for another identity."
            ) from None
        return decision

    async def _load_optional_decision_for_proposal(
        self,
        store_owner: _TaskStoreOwner,
        proposal_id: str,
    ) -> CompletionDecision | None:
        outcome = await capture_task_store_operation(
            lambda: store_owner.store.load_completion_decision_for_proposal(proposal_id),
            operation_name="Completion proposal decision lookup",
            redactor=self._secret_redactor,
        )
        if outcome.failure is not None:
            raise outcome.failure from None
        if outcome.result is None:
            return None
        validation = _copy_exact_model(
            outcome.result,
            CompletionDecision,
            operation_name="Completion proposal decision lookup result validation",
            redactor=self._secret_redactor,
        )
        del outcome
        if validation.failure is not None:
            raise validation.failure from None
        decision = validation.result
        del validation
        if decision is None or decision.proposal_id != proposal_id:
            del decision, store_owner
            raise WorkCompletionConflict(
                "Task store returned completion decision authority for another proposal."
            ) from None
        return decision

    async def _load_converged_decision(
        self,
        store_owner: _TaskStoreOwner,
        *,
        decision_id: str,
        proposal_id: str,
    ) -> CompletionDecision | None:
        by_id = await self._load_optional_decision(store_owner, decision_id)
        by_proposal = await self._load_optional_decision_for_proposal(store_owner, proposal_id)
        if by_id is None and by_proposal is not None:
            if by_proposal.decision_id != decision_id:
                del by_id, by_proposal, store_owner
                raise WorkCompletionConflict(
                    "Completion proposal has conflicting authority from a different durable "
                    "decision."
                ) from None
            # Both indexes are insert-only and must publish atomically, but
            # separate reads can straddle a concurrent exact publication.
            # Confirm the first miss once before classifying durable corruption.
            by_id = await self._load_optional_decision(store_owner, decision_id)
            if by_id is not None:
                if by_id != by_proposal:
                    del by_id, by_proposal, store_owner
                    raise WorkCompletionConflict(
                        "Task store returned non-convergent completion decision indexes with "
                        "conflicting authority."
                    ) from None
                return by_id
            del by_id, by_proposal, store_owner
            raise WorkCompletionConflict(
                "Task store omitted the completion decision from its identity index."
            ) from None
        if by_id is not None and by_proposal is None:
            by_proposal = await self._load_optional_decision_for_proposal(
                store_owner,
                proposal_id,
            )
            if by_proposal is not None:
                if by_id != by_proposal:
                    del by_id, by_proposal, store_owner
                    raise WorkCompletionConflict(
                        "Task store returned non-convergent completion decision indexes with "
                        "conflicting authority."
                    ) from None
                return by_id
            del by_id, by_proposal, store_owner
            raise WorkCompletionConflict(
                "Task store omitted the completion decision from its proposal index."
            ) from None
        if by_id is not None and by_id != by_proposal:
            del by_id, by_proposal, store_owner
            raise WorkCompletionConflict(
                "Task store returned non-convergent completion decision indexes with "
                "conflicting authority."
            ) from None
        return by_id

    def _require_exact_claim(
        self,
        claim: CompletionVerificationClaim,
        request: CompletionVerifierExecutionRequest,
        verifier: CompletionVerifierRef,
        *,
        execution_owner_id: str | None,
        accept_legacy_timeout: bool = False,
    ) -> None:
        if execution_owner_id is not None and _contains_workload_secret(
            (execution_owner_id,),
            redactor=self._secret_redactor,
        ):
            del claim, request, verifier, execution_owner_id
            raise WorkCompletionConflict(
                "Completion verification claim contains unsafe execution authority."
            ) from None
        expected_timeout = (
            None
            if accept_legacy_timeout and claim.execution_timeout_seconds is None
            else request.execution_timeout_seconds
        )
        expected = CompletionVerificationClaimRequest(
            claim_id=request.claim_id,
            proposal_id=request.proposal_id,
            worker_id=request.worker_id,
            execution_owner_id=execution_owner_id,
            verifier=verifier,
            lease_seconds=request.lease_seconds,
            execution_timeout_seconds=expected_timeout,
        )
        if (
            claim.claim_id != request.claim_id
            or claim.proposal_id != request.proposal_id
            or claim.worker_id != request.worker_id
            or claim.execution_owner_id != execution_owner_id
            or claim.execution_timeout_seconds != expected_timeout
            or claim.verifier != verifier
            or claim.request_sha256 != completion_verification_claim_request_sha256(expected)
        ):
            del claim, request, verifier, expected, expected_timeout
            raise WorkCompletionConflict(
                "Completion verification claim conflicts with the exact execution request."
            ) from None

    def _require_existing_decision(
        self,
        decision: CompletionDecision,
        *,
        request: CompletionVerifierExecutionRequest,
        proposal: CompletionProposal,
        attempt: WorkAttempt,
        contract: WorkContract,
        claim: CompletionVerificationClaim,
    ) -> bool:
        if (
            decision.decision_id != request.decision_id
            or decision.proposal_id != proposal.proposal_id
            or decision.claim_id != claim.claim_id
            or decision.worker_id != request.worker_id
            or decision.verifier != contract.verifier
            or decision.task_id != proposal.task_id
            or decision.attempt_id != attempt.attempt_id
            or decision.contract != contract.reference()
        ):
            raise WorkCompletionConflict(
                "Durable completion decision conflicts with the exact execution request."
            ) from None
        durable_request = CompletionDecisionCreate(
            decision_id=decision.decision_id,
            proposal_id=decision.proposal_id,
            claim_id=decision.claim_id,
            worker_id=decision.worker_id,
            verifier=decision.verifier,
            decision_version=decision.decision_version,
            verdict=decision.verdict,
            criterion_outcomes=decision.criterion_outcomes,
            constraint_outcomes=decision.constraint_outcomes,
            gaps=decision.gaps,
            evidence_references=decision.evidence_references,
        )
        if decision.request_sha256 != completion_decision_request_sha256(
            durable_request
        ) or decision.gap_fingerprint != completion_gap_fingerprint(durable_request):
            raise WorkCompletionConflict(
                "Durable completion decision has conflicting integrity evidence."
            ) from None
        validate_completion_decision_contract(contract, durable_request)
        return True

    def _decision_matches_request(
        self,
        decision: CompletionDecision,
        request: CompletionDecisionCreate,
        *,
        proposal: CompletionProposal,
        attempt: WorkAttempt,
        contract: WorkContract,
    ) -> bool:
        returned_request = CompletionDecisionCreate(
            decision_id=decision.decision_id,
            proposal_id=decision.proposal_id,
            claim_id=decision.claim_id,
            worker_id=decision.worker_id,
            verifier=decision.verifier,
            decision_version=decision.decision_version,
            verdict=decision.verdict,
            criterion_outcomes=decision.criterion_outcomes,
            constraint_outcomes=decision.constraint_outcomes,
            gaps=decision.gaps,
            evidence_references=decision.evidence_references,
        )
        validate_completion_decision_contract(contract, returned_request)
        return (
            returned_request == request
            and decision.request_sha256 == completion_decision_request_sha256(returned_request)
            and decision.gap_fingerprint == completion_gap_fingerprint(returned_request)
            and decision.proposal_id == proposal.proposal_id
            and decision.task_id == proposal.task_id
            and decision.attempt_id == attempt.attempt_id
            and decision.contract == contract.reference()
        )

    @staticmethod
    def _require_context_integrity(
        *,
        proposal: CompletionProposal,
        attempt: WorkAttempt,
    ) -> None:
        attempt_request = WorkAttemptCreate(
            attempt_id=attempt.attempt_id,
            task_id=attempt.task_id,
            session_id=attempt.session_id,
            contract=attempt.contract,
            execution_profile_fingerprint=attempt.execution_profile_fingerprint,
            worker_id=attempt.worker_id,
        )
        proposal_request = CompletionProposalCreate(
            proposal_id=proposal.proposal_id,
            attempt_id=proposal.attempt_id,
            result=proposal.result,
            evidence_references=proposal.evidence_references,
        )
        if attempt.request_sha256 != work_attempt_request_sha256(attempt_request):
            del attempt_request, proposal_request, proposal, attempt
            raise WorkCompletionConflict(
                "Durable work attempt has conflicting request integrity evidence."
            ) from None
        if proposal.request_sha256 != completion_proposal_request_sha256(proposal_request):
            del attempt_request, proposal_request, proposal, attempt
            raise WorkCompletionConflict(
                "Durable completion proposal has conflicting request integrity evidence."
            ) from None

    async def _renew_claim(
        self,
        store_owner: _TaskStoreOwner,
        claim_request: CompletionVerificationClaimRequest,
        execution_request: CompletionVerifierExecutionRequest,
        *,
        prior_claim: CompletionVerificationClaim,
    ) -> CompletionVerificationClaim:
        outcome = await capture_task_store_operation(
            lambda: store_owner.store.renew_completion_verification_claim(claim_request),
            operation_name="Completion verification claim renewal",
            redactor=self._secret_redactor,
            mutation_store=store_owner.store,
            mutation_method_name="renew_completion_verification_claim",
        )
        if outcome.failure is not None:
            raise outcome.failure from None
        validation = _copy_exact_model(
            outcome.result,
            CompletionVerificationClaim,
            operation_name="Completion verification claim renewal result validation",
            redactor=self._secret_redactor,
        )
        del outcome
        if validation.failure is not None:
            raise validation.failure from None
        renewed = validation.result
        del validation
        if renewed is None:
            raise WorkCompletionConflict(
                "Task store returned an invalid renewed completion verification claim."
            ) from None
        self._require_exact_claim(
            renewed,
            execution_request,
            claim_request.verifier,
            execution_owner_id=claim_request.execution_owner_id,
        )
        if (
            renewed.attempt_number != prior_claim.attempt_number
            or renewed.claimed_at != prior_claim.claimed_at
            or renewed.lease_expires_at < prior_claim.lease_expires_at
        ):
            del renewed, prior_claim, store_owner, claim_request, execution_request
            raise WorkCompletionConflict(
                "Completion verification claim renewal did not compare-and-extend "
                "the existing authority."
            ) from None
        return renewed

    def _start_claim_heartbeat(
        self,
        store_owner: _TaskStoreOwner,
        claim_request: CompletionVerificationClaimRequest,
        execution_request: CompletionVerifierExecutionRequest,
        claim: CompletionVerificationClaim,
    ) -> _ClaimHeartbeat:
        stop = asyncio.Event()
        task = asyncio.create_task(
            self._heartbeat_claim(
                store_owner,
                claim_request,
                execution_request,
                claim,
                stop,
            ),
            name="cayu-completion-verifier-claim-heartbeat",
        )
        control = _ClaimHeartbeat(
            stop=stop,
            task=task,
            cancellation_marker=_ClaimHeartbeatCancellationMarker(),
            shutdown_marker=_ClaimHeartbeatShutdownMarker(),
        )
        self._claim_heartbeat_tasks.add(task)

        def settled(completed: asyncio.Task[None]) -> None:
            self._claim_heartbeat_tasks.discard(completed)
            with suppress(BaseException):
                completed.result()

        task.add_done_callback(settled)
        return control

    async def _heartbeat_claim(
        self,
        store_owner: _TaskStoreOwner,
        claim_request: CompletionVerificationClaimRequest,
        execution_request: CompletionVerifierExecutionRequest,
        claim: CompletionVerificationClaim,
        stop: asyncio.Event,
    ) -> None:
        interval = min(max(claim_request.lease_seconds / 4.0, 0.05), 30.0)
        current = claim
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
                return
            except TimeoutError:
                pass
            current = await self._renew_claim(
                store_owner,
                claim_request,
                execution_request,
                prior_claim=current,
            )

    @staticmethod
    def _request_claim_heartbeat_stop(heartbeat: _ClaimHeartbeat) -> bool:
        """Request owned shutdown and retain the task until it settles."""

        heartbeat.stop.set()
        if heartbeat.task.done():
            return False
        heartbeat.shutdown_requested = True
        return heartbeat.task.cancel(heartbeat.shutdown_marker)

    async def _settle_claim_heartbeat(
        self,
        heartbeat: _ClaimHeartbeat,
    ) -> _ClaimHeartbeatSettlement:
        cancel_requested = self._request_claim_heartbeat_stop(heartbeat)
        owned_shutdown_requested = heartbeat.shutdown_requested
        shielded = await await_shielded_task_outcome(
            heartbeat.task,
            timeout_s=None,
            timeout_after_cancellation_s=None,
        )
        failure = shielded.error
        if failure is not None and id(failure) == heartbeat.observed_failure_id:
            failure = None
        elif failure is not None and (cancel_requested or owned_shutdown_requested):
            removed_shutdown_cancellation = False

            def is_owned_shutdown(leaf: BaseException) -> bool:
                nonlocal removed_shutdown_cancellation
                if not isinstance(leaf, asyncio.CancelledError):
                    return False
                if _is_exception_marker(leaf, heartbeat.shutdown_marker):
                    removed_shutdown_cancellation = True
                    return True
                # The task-store boundary detaches cancellation arguments.  A
                # successful Task.cancel() against this private task therefore
                # authenticates the one resulting cancellation leaf.
                if not removed_shutdown_cancellation:
                    removed_shutdown_cancellation = True
                    return True
                return False

            failure = _prune_exception_graph(failure, should_prune=is_owned_shutdown)
        caller_cancellation = (
            None
            if shielded.cancellation is None
            else _safe_caller_cancellation(
                shielded.cancellation,
                redactor=self._secret_redactor,
            )
        )
        return _ClaimHeartbeatSettlement(
            failure=failure,
            caller_cancellation=caller_cancellation,
            cancellation_requests_consumed=shielded.cancellation_requests_consumed,
        )

    def _failure_after_heartbeat_settlement(
        self,
        primary: BaseException | None,
        settlement: _ClaimHeartbeatSettlement,
        *,
        group_message: str,
    ) -> BaseException | None:
        failures: list[BaseException] = []
        if primary is not None:
            failures.append(primary)
        if settlement.failure is not None and all(
            settlement.failure is not failure for failure in failures
        ):
            failures.append(settlement.failure)
        caller_cancellation = settlement.caller_cancellation
        if caller_cancellation is not None and (
            primary is None or not _exception_tree_has_cancellation(primary)
        ):
            failures.append(caller_cancellation)

        if not failures:
            propagated: BaseException | None = None
        elif len(failures) == 1:
            propagated = failures[0]
        else:
            propagated = _ordered_failure_group(
                group_message,
                *failures,
                redactor=self._secret_redactor,
            )
        if settlement.cancellation_requests_consumed:
            if propagated is not None:
                retain_workspace_observation_pending_cancellation_requests(
                    propagated,
                    max(
                        settlement.cancellation_requests_consumed,
                        workspace_observation_pending_cancellation_requests(propagated),
                    ),
                )
            restore_task_cancellation_requests(
                settlement.cancellation_requests_consumed,
                cancellation=caller_cancellation,
            )
        return propagated

    @staticmethod
    def _claim_heartbeat_failure(heartbeat: _ClaimHeartbeat) -> BaseException | None:
        if not heartbeat.task.done() or heartbeat.task.cancelled():
            return None
        try:
            heartbeat.task.result()
        except BaseException as failure:
            heartbeat.observed_failure_id = id(failure)
            return failure
        return None

    def _cancel_adapter_for_failed_heartbeat(
        self,
        heartbeat_task: asyncio.Task[None],
        adapter_task: asyncio.Task[CapturedAwaitableOutcome[CompletionVerifierDecision]],
        cancellation_marker: _ClaimHeartbeatCancellationMarker,
    ) -> None:
        if heartbeat_task.cancelled():
            return
        try:
            heartbeat_task.result()
        except BaseException:
            if not adapter_task.done():
                adapter_task.cancel(cancellation_marker)

    async def _invoke_adapter(
        self,
        verifier: DeterministicCompletionVerifier,
        request: CompletionVerifierRequest,
        *,
        operation_key: _ExecutionKey,
        timeout_seconds: float,
        capacity_reservation: object,
        heartbeat: _ClaimHeartbeat,
    ) -> CompletionVerifierDecision:
        try:
            task = asyncio.create_task(
                capture_awaitable_outcome(
                    lambda adapter=verifier, value=request: adapter.verify(value)
                ),
                name="cayu-completion-verifier",
            )
        except BaseException:
            self._release_adapter_capacity_reservation(capacity_reservation)
            del verifier, request
            raise
        del verifier, request
        self._release_adapter_capacity_reservation(capacity_reservation)
        self._adapter_tasks.add(task)
        task.add_done_callback(
            lambda completed: self._release_adapter_task(
                operation_key,
                completed,
            )
        )
        heartbeat.task.add_done_callback(
            lambda completed, adapter_task=task, marker=heartbeat.cancellation_marker: (
                self._cancel_adapter_for_failed_heartbeat(
                    completed,
                    adapter_task,
                    marker,
                )
            )
        )
        shielded = await await_shielded_task_outcome(
            task,
            timeout_s=timeout_seconds,
            timeout_after_cancellation_s=0,
        )
        if shielded.cancellation is not None:
            cancellation = shielded.cancellation
            cancellation_requests_consumed = shielded.cancellation_requests_consumed
            captured_during_cancellation = shielded.result
            heartbeat_failure = self._claim_heartbeat_failure(heartbeat)
            execution_failures = _ordered_execution_failures(
                adapter_failures=(
                    _adapter_failure_after_heartbeat(
                        shielded.error,
                        heartbeat=heartbeat,
                        heartbeat_failure=heartbeat_failure,
                    ),
                    _adapter_failure_after_heartbeat(
                        (
                            None
                            if captured_during_cancellation is None
                            else captured_during_cancellation.error
                        ),
                        heartbeat=heartbeat,
                        heartbeat_failure=heartbeat_failure,
                    ),
                ),
                boundary_failures=(heartbeat_failure,),
                redactor=self._secret_redactor,
            )
            self._cancel_and_retain_adapter_task(operation_key, task, heartbeat)
            safe = _safe_caller_cancellation(
                cancellation,
                redactor=self._secret_redactor,
            )
            if not execution_failures:
                propagated: BaseException = safe
            else:
                propagated = _ordered_failure_group(
                    "Completion verifier failed concurrently with caller cancellation.",
                    *execution_failures,
                    safe,
                    redactor=self._secret_redactor,
                )
                retain_workspace_observation_pending_cancellation_requests(
                    propagated,
                    max(cancellation_requests_consumed, 1),
                )
            del cancellation, captured_during_cancellation, heartbeat_failure
            del execution_failures
            del shielded, task
            restore_task_cancellation_requests(
                cancellation_requests_consumed,
                cancellation=safe,
            )
            raise propagated from None
        if shielded.timed_out:
            heartbeat_failure = self._claim_heartbeat_failure(heartbeat)
            self._cancel_and_retain_adapter_task(operation_key, task, heartbeat)
            timeout_failure = CompletionVerifierExecutionError(
                "Completion verifier exceeded its bounded execution timeout."
            )
            execution_failures = _ordered_execution_failures(
                boundary_failures=(timeout_failure, heartbeat_failure),
                redactor=self._secret_redactor,
            )
            del heartbeat_failure, timeout_failure
            del shielded, task
            if len(execution_failures) == 1:
                raise execution_failures[0] from None
            raise _ordered_failure_group(
                "Completion verifier timed out while claim renewal failed.",
                *execution_failures,
                redactor=self._secret_redactor,
            ) from None
        task_error = shielded.error
        captured = shielded.result
        self._release_adapter_task(operation_key, task)
        del shielded, task
        heartbeat_failure = self._claim_heartbeat_failure(heartbeat)
        captured_error = None if captured is None else captured.error
        captured_result = None if captured is None else captured.result
        captured_was_missing = captured is None
        execution_failures = _ordered_execution_failures(
            adapter_failures=(
                _adapter_failure_after_heartbeat(
                    task_error,
                    heartbeat=heartbeat,
                    heartbeat_failure=heartbeat_failure,
                ),
                _adapter_failure_after_heartbeat(
                    captured_error,
                    heartbeat=heartbeat,
                    heartbeat_failure=heartbeat_failure,
                ),
            ),
            boundary_failures=(heartbeat_failure,),
            redactor=self._secret_redactor,
        )
        del task_error, captured_error, captured, heartbeat_failure
        if execution_failures:
            del captured_result, captured_was_missing
            if len(execution_failures) == 1:
                raise execution_failures[0] from None
            raise _ordered_failure_group(
                "Completion verifier execution and claim renewal failed.",
                *execution_failures,
                redactor=self._secret_redactor,
            ) from None
        del execution_failures
        if captured_was_missing:
            del captured_result
            raise CompletionVerifierExecutionError(
                "Completion verifier returned no captured outcome."
            ) from None
        del captured_was_missing
        validation = capture_sensitive_validation(
            lambda value=captured_result: copy_completion_verifier_decision(
                cast("CompletionVerifierDecision", value)
            ),
            operation_name="Completion verifier decision validation",
            redactor=self._secret_redactor,
        )
        del captured_result
        if validation.failure is not None:
            raise validation.failure from None
        decision = validation.result
        del validation
        if decision is None:
            raise CompletionVerifierExecutionError(
                "Completion verifier returned an invalid decision."
            ) from None
        return decision

    def _reserve_adapter_capacity(self) -> object:
        if (
            len(self._adapter_tasks) + len(self._adapter_capacity_reservations)
            >= _MAX_ACTIVE_COMPLETION_VERIFIERS
        ):
            raise CompletionVerifierExecutionError(
                "Completion verifier execution capacity is exhausted."
            ) from None
        reservation = object()
        self._adapter_capacity_reservations.add(reservation)
        return reservation

    def _release_adapter_capacity_reservation(self, reservation: object) -> None:
        self._adapter_capacity_reservations.discard(reservation)

    def _cancel_and_retain_adapter_task(
        self,
        operation_key: _ExecutionKey,
        task: asyncio.Task[CapturedAwaitableOutcome[CompletionVerifierDecision]],
        heartbeat: _ClaimHeartbeat,
    ) -> None:
        heartbeat.retained_for_drain = True
        draining = _DrainingAdapter(
            task=task,
            heartbeat=heartbeat,
        )
        self._draining_adapter_tasks[operation_key] = draining
        if task.done():
            self._ensure_draining_adapter_settlement(operation_key, draining)
            return
        task.cancel("Completion verifier execution no longer owns publication authority.")

    def _ensure_draining_adapter_settlement(
        self,
        operation_key: _ExecutionKey,
        draining: _DrainingAdapter,
    ) -> asyncio.Task[_ClaimHeartbeatSettlement]:
        settlement_task = draining.settlement_task
        if settlement_task is not None:
            return settlement_task
        settlement_task = asyncio.create_task(
            self._settle_claim_heartbeat(draining.heartbeat),
            name="cayu-completion-verifier-drain-settlement",
        )
        draining.settlement_task = settlement_task
        settlement_task.add_done_callback(
            lambda completed, expected=draining: self._finalize_draining_adapter_settlement(
                operation_key,
                expected,
                completed,
            )
        )
        return settlement_task

    def _finalize_draining_adapter_settlement(
        self,
        operation_key: _ExecutionKey,
        draining: _DrainingAdapter,
        completed: asyncio.Task[_ClaimHeartbeatSettlement],
    ) -> bool:
        try:
            completed.result()
        except BaseException:
            return False
        current = self._draining_adapter_tasks.get(operation_key)
        if current is draining and current.settlement_task is completed:
            self._draining_adapter_tasks.pop(operation_key, None)
        return True

    def _release_adapter_task(
        self,
        operation_key: _ExecutionKey,
        completed: asyncio.Task[CapturedAwaitableOutcome[CompletionVerifierDecision]],
    ) -> None:
        self._adapter_tasks.discard(completed)
        draining = self._draining_adapter_tasks.get(operation_key)
        if draining is not None and draining.task is completed:
            self._ensure_draining_adapter_settlement(operation_key, draining)
        with suppress(BaseException):
            completed.result()

    @staticmethod
    def _execution_key(request: CompletionVerifierExecutionRequest) -> _ExecutionKey:
        return request.proposal_id


__all__ = ["CompletionVerifierCoordinator"]
