"""Cancellation-aware, secret-safe boundary for new verified-work store calls."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import FunctionType, MethodType
from typing import Generic, NoReturn, TypeVar

from cayu._exception_groups import exception_group_children, rebuild_exception_group
from cayu._task_wait import CapturedAwaitableOutcome, capture_awaitable_outcome
from cayu.runtime._diagnostics import (
    MAX_DIAGNOSTIC_EXCEPTION_GROUP_NODES,
    MAX_DIAGNOSTIC_UTF8_BYTES,
    credential_safe_runtime_exception,
    credential_safe_runtime_exception_group,
    exception_diagnostic,
    runtime_owned_exception_renderings_are_credential_safe,
)
from cayu.runtime.tasks import TaskClaimLost, TaskStore
from cayu.runtime.work_attempt_admission import (
    WorkAttemptAdmissionConflict,
    WorkAttemptExecutionClaimLost,
    WorkAttemptRecoveryRequired,
)
from cayu.runtime.work_contracts import (
    CompletionVerificationClaimLost,
    TaskCompletionDecisionRequired,
    WorkCompletionConflict,
    WorkContractConflict,
)
from cayu.runtime.workspace_observation_recovery import (
    retain_workspace_observation_pending_cancellation_requests,
)
from cayu.vaults import SecretRedactor

_ResultT = TypeVar("_ResultT")
_BASE_EXCEPTION_ARGS_DESCRIPTOR = BaseException.__dict__["args"]
_BASE_EXCEPTION_CAUSE_DESCRIPTOR = BaseException.__dict__["__cause__"]
_WORK_ATTEMPT_ADMISSION_MUTATIONS = (
    "prepare_work_attempt_admission",
    "activate_work_attempt_admission",
    "renew_work_attempt_execution_claim",
    "claim_work_attempt_recovery",
    "activate_work_attempt_recovery",
    "submit_admitted_completion_proposal",
)
_WORK_ATTEMPT_ADMISSION_READS = (
    "load_work_attempt_admission",
    "load_work_attempt_execution_claim",
)
_INTERRUPTED_TASK_HANDOFF_MUTATIONS = (
    "release_interrupted_task_worker",
    "recover_interrupted_task_worker",
)
_INTERRUPTED_TASK_HANDOFF_READS = (
    "load_interrupted_task_handoff_receipt",
    "list_expired_interrupted_task_handoff_candidates",
)
_TASK_CANCELLATION_RECONCILIATION_METHODS = (
    "mark_claimed_task_execution_started",
    "request_claimed_task_cancellation",
    "reconcile_task_cancellation",
    "terminalize_task",
    "load_task_terminalization_receipt",
)


class _TaskStoreOperationCancellationMarker:
    """Per-operation provenance for cancellation forwarded to the owned child."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<task-store-operation-cancellation>"

    __str__ = __repr__


@dataclass(frozen=True, slots=True)
class TaskStoreOperationOutcome(Generic[_ResultT]):
    """One detached store result or failure, returned as data to its payload owner."""

    result: _ResultT | None = None
    failure: BaseException | None = None


def task_store_mutation_is_cancellation_quiescent(
    task_store: TaskStore,
    method_name: str,
) -> bool:
    """Return positive structural proof for one exact mutation implementation."""

    if not method_name:
        return False
    task_store_type = type(task_store)
    mro = type.__getattribute__(task_store_type, "__mro__")

    # Dynamic lookup and instance shadowing can replace the implementation after
    # its declaring class has supplied proof.  Neither shape is stable positive
    # evidence for the callable that the operation boundary will actually invoke.
    getattribute_owner = next(
        (
            candidate
            for candidate in mro
            if "__getattribute__" in type.__getattribute__(candidate, "__dict__")
        ),
        None,
    )
    if getattribute_owner is None:
        return False
    getattribute_implementation = type.__getattribute__(
        getattribute_owner,
        "__dict__",
    )["__getattribute__"]
    if getattribute_implementation is not object.__getattribute__:
        return False
    try:
        instance_state = object.__getattribute__(task_store, "__dict__")
    except AttributeError:
        instance_state = None
    if isinstance(instance_state, dict) and method_name in instance_state:
        return False

    # An inherited public method can still dispatch through helpers, wrappers,
    # or readiness hooks replaced by the concrete subclass.  Inherited proof is
    # therefore not positive evidence for that concrete implementation graph:
    # every subclass must opt in explicitly after reviewing its dependencies.
    concrete_declarations = type.__getattribute__(task_store_type, "__dict__")
    if concrete_declarations.get("verified_work_mutations_are_cancellation_quiescent") is not True:
        return False
    implementation_owner = next(
        (
            candidate
            for candidate in mro
            if method_name in type.__getattribute__(candidate, "__dict__")
        ),
        None,
    )
    if implementation_owner is None:
        return False
    owner_declarations = type.__getattribute__(implementation_owner, "__dict__")
    implementation = owner_declarations[method_name]
    if (
        owner_declarations.get("verified_work_mutations_are_cancellation_quiescent") is not True
        or type(implementation) is not FunctionType
    ):
        return False
    resolved = object.__getattribute__(task_store, method_name)
    return (
        type(resolved) is MethodType
        and resolved.__self__ is task_store
        and resolved.__func__ is implementation
    )


def _task_store_method_has_stable_concrete_implementation(
    task_store: TaskStore,
    method_name: str,
) -> bool:
    """Prove that one method is not the base placeholder or dynamic dispatch."""

    task_store_type = type(task_store)
    mro = type.__getattribute__(task_store_type, "__mro__")
    getattribute_owner = next(
        (
            candidate
            for candidate in mro
            if "__getattribute__" in type.__getattribute__(candidate, "__dict__")
        ),
        None,
    )
    if getattribute_owner is None:
        return False
    getattribute_implementation = type.__getattribute__(
        getattribute_owner,
        "__dict__",
    )["__getattribute__"]
    if getattribute_implementation is not object.__getattribute__:
        return False
    try:
        instance_state = object.__getattribute__(task_store, "__dict__")
    except AttributeError:
        instance_state = None
    if isinstance(instance_state, dict) and method_name in instance_state:
        return False
    implementation_owner = next(
        (
            candidate
            for candidate in mro
            if method_name in type.__getattribute__(candidate, "__dict__")
        ),
        None,
    )
    if implementation_owner is None:
        return False
    implementation = type.__getattribute__(implementation_owner, "__dict__")[method_name]
    base_implementation = type.__getattribute__(TaskStore, "__dict__").get(method_name)
    if type(implementation) is not FunctionType or implementation is base_implementation:
        return False
    resolved = object.__getattribute__(task_store, method_name)
    return (
        type(resolved) is MethodType
        and resolved.__self__ is task_store
        and resolved.__func__ is implementation
    )


def task_store_work_attempt_admission_capability_is_complete(
    task_store: TaskStore,
) -> bool:
    """Return positive proof for the complete work-attempt extension family."""

    try:
        supported = object.__getattribute__(task_store, "supports_work_attempt_admission")
    except BaseException:
        return False
    if supported is not True:
        return False
    if not all(
        _task_store_method_has_stable_concrete_implementation(task_store, method_name)
        for method_name in (*_WORK_ATTEMPT_ADMISSION_MUTATIONS, *_WORK_ATTEMPT_ADMISSION_READS)
    ):
        return False
    return all(
        task_store_mutation_is_cancellation_quiescent(task_store, method_name)
        for method_name in _WORK_ATTEMPT_ADMISSION_MUTATIONS
    )


def task_store_interrupted_handoff_capability_is_complete(
    task_store: TaskStore,
) -> bool:
    """Return positive proof for the exact interrupted-task handoff family."""

    try:
        declarations = type.__getattribute__(type(task_store), "__dict__")
    except BaseException:
        return False
    if declarations.get("supports_interrupted_task_handoffs") is not True:
        return False
    if not all(
        _task_store_method_has_stable_concrete_implementation(task_store, method_name)
        for method_name in (
            *_INTERRUPTED_TASK_HANDOFF_MUTATIONS,
            *_INTERRUPTED_TASK_HANDOFF_READS,
        )
    ):
        return False
    return all(
        task_store_mutation_is_cancellation_quiescent(task_store, method_name)
        for method_name in _INTERRUPTED_TASK_HANDOFF_MUTATIONS
    )


def task_store_exact_interrupted_handoff_capability_is_complete(
    task_store: TaskStore,
) -> bool:
    """Return positive proof for exact task-selected handoff and continuation."""

    try:
        declarations = type.__getattribute__(type(task_store), "__dict__")
    except BaseException:
        return False
    return bool(
        declarations.get("supports_exact_interrupted_task_handoffs") is True
        and task_store_interrupted_handoff_capability_is_complete(task_store)
        and _task_store_method_has_stable_concrete_implementation(
            task_store,
            "load_expired_interrupted_task_handoff_candidate",
        )
        and _task_store_method_has_stable_concrete_implementation(
            task_store,
            "claim_interrupted_task_continuation",
        )
        and task_store_mutation_is_cancellation_quiescent(
            task_store,
            "claim_interrupted_task_continuation",
        )
    )


def task_store_cancellation_reconciliation_capability_is_complete(
    task_store: TaskStore,
) -> bool:
    """Return positive proof for owner-lost ordinary-task settlement."""

    try:
        cancellation_supported = object.__getattribute__(
            task_store,
            "supports_task_cancellation_reconciliation",
        )
        terminalization_supported = object.__getattribute__(
            task_store,
            "supports_idempotent_terminalization",
        )
    except BaseException:
        return False
    if cancellation_supported is not True or terminalization_supported is not True:
        return False
    return all(
        _task_store_method_has_stable_concrete_implementation(task_store, method_name)
        for method_name in _TASK_CANCELLATION_RECONCILIATION_METHODS
    )


def _task_store_cancellation(
    cancellation: asyncio.CancelledError,
    *,
    operation_name: str,
    redactor: SecretRedactor,
) -> asyncio.CancelledError:
    diagnostic = exception_diagnostic(
        cancellation,
        empty_message=f"{operation_name.lower()} cancelled",
        nonportable_message=f"{operation_name} cancellation had a non-portable diagnostic.",
        redactor=redactor,
    )
    return credential_safe_runtime_exception(
        asyncio.CancelledError,
        diagnostic.message,
        redactor=redactor,
        fallback_message=f"{operation_name} cancellation diagnostic was redacted.",
    )


def _task_store_forwarded_cancellation(
    caller_cancellation: asyncio.CancelledError,
    forwarded_cancellation: asyncio.CancelledError,
    *,
    operation_name: str,
    redactor: SecretRedactor,
) -> asyncio.CancelledError:
    """Detach one authenticated cancellation and its settlement evidence."""

    safe_cancellation = _task_store_cancellation(
        caller_cancellation,
        operation_name=operation_name,
        redactor=redactor,
    )
    try:
        settlement_evidence = _BASE_EXCEPTION_CAUSE_DESCRIPTOR.__get__(
            forwarded_cancellation,
            BaseException,
        )
    except BaseException:
        settlement_evidence = RuntimeError(
            f"{operation_name} cancellation settlement evidence was inaccessible."
        )
    if settlement_evidence is None:
        return safe_cancellation
    if isinstance(settlement_evidence, BaseExceptionGroup):
        safe_evidence: BaseException = _detached_task_store_group(
            settlement_evidence,
            operation_name=operation_name,
            redactor=redactor,
        )
    else:
        safe_evidence = _detached_task_store_failure(
            settlement_evidence,
            operation_name=operation_name,
            redactor=redactor,
        )
    safe_cancellation.__cause__ = safe_evidence
    if runtime_owned_exception_renderings_are_credential_safe(
        safe_cancellation,
        redactor=redactor,
    ):
        return safe_cancellation

    # The evidence and cancellation can each be safe while Python's fixed
    # direct-cause separator composes their renderings into a registered
    # secret. Preserve the evidence classification with a generic diagnostic,
    # then validate the complete chain again before it crosses the boundary.
    generic_evidence = _generic_task_store_settlement_evidence(
        safe_evidence,
        operation_name=operation_name,
        redactor=redactor,
    )
    safe_cancellation.__cause__ = generic_evidence
    if runtime_owned_exception_renderings_are_credential_safe(
        safe_cancellation,
        redactor=redactor,
    ):
        return safe_cancellation

    fallback_evidence = credential_safe_runtime_exception(
        RuntimeError,
        f"{operation_name} cancellation settlement evidence was redacted.",
        redactor=redactor,
        fallback_message="Task-store cancellation settlement evidence was withheld.",
    )
    safe_cancellation.__cause__ = fallback_evidence
    if runtime_owned_exception_renderings_are_credential_safe(
        safe_cancellation,
        redactor=redactor,
    ):
        return safe_cancellation

    # A registry can deliberately contain every fixed cause-chain rendering.
    # In that degenerate case, keep caller cancellation authoritative and omit
    # the optional diagnostic evidence rather than publish a registered secret.
    safe_cancellation.__cause__ = None
    return safe_cancellation


def _detached_task_store_failure(
    error: BaseException,
    *,
    operation_name: str,
    redactor: SecretRedactor,
) -> BaseException:
    """Copy one extension failure without its traceback, context, or payload locals."""

    diagnostic = exception_diagnostic(
        error,
        empty_message=f"{operation_name.lower()} failed",
        nonportable_message=f"{operation_name} failed with a non-portable diagnostic.",
        redactor=redactor,
    )
    message = diagnostic.message
    safe_type: type[BaseException]
    if isinstance(error, GeneratorExit):
        safe_type = GeneratorExit
    elif isinstance(error, KeyboardInterrupt):
        safe_type = KeyboardInterrupt
    elif isinstance(error, SystemExit):
        safe_type = SystemExit
    elif type(error) in {
        TimeoutError,
        ConnectionError,
        TaskClaimLost,
        WorkAttemptAdmissionConflict,
        WorkAttemptExecutionClaimLost,
        WorkAttemptRecoveryRequired,
        CompletionVerificationClaimLost,
        TaskCompletionDecisionRequired,
        WorkContractConflict,
        WorkCompletionConflict,
        ValueError,
        TypeError,
        NotImplementedError,
        RuntimeError,
    }:
        safe_type = type(error)
    elif isinstance(error, Exception):
        safe_type = RuntimeError
        message = redactor.redact_text_bounded(
            f"{diagnostic.error_type}: {message}",
            max_bytes=MAX_DIAGNOSTIC_UTF8_BYTES,
        )
    else:
        safe_type = BaseException
        message = redactor.redact_text_bounded(
            f"{diagnostic.error_type}: {message}",
            max_bytes=MAX_DIAGNOSTIC_UTF8_BYTES,
        )
    return credential_safe_runtime_exception(
        safe_type,
        message,
        redactor=redactor,
        fallback_message=f"{operation_name} failure diagnostic was redacted.",
    )


def _detached_task_store_group(
    error: BaseExceptionGroup,
    *,
    operation_name: str,
    redactor: SecretRedactor,
    caller_cancellation: asyncio.CancelledError | None = None,
    child_cancellation_marker: _TaskStoreOperationCancellationMarker | None = None,
) -> BaseExceptionGroup:
    """Detach a group and represent current caller cancellation exactly once."""

    cancellation_claimed = False

    def map_leaf(leaf: BaseException) -> BaseException:
        nonlocal cancellation_claimed
        if isinstance(leaf, asyncio.CancelledError):
            if (
                caller_cancellation is not None
                and not cancellation_claimed
                and _is_forwarded_task_store_cancellation(
                    leaf,
                    child_cancellation_marker,
                )
            ):
                cancellation_claimed = True
                return _task_store_cancellation(
                    caller_cancellation,
                    operation_name=operation_name,
                    redactor=redactor,
                )
            return RuntimeError(
                f"{operation_name} was cancelled without "
                + (
                    "caller cancellation."
                    if caller_cancellation is None
                    else "distinct caller cancellation."
                )
            )
        return _detached_task_store_failure(
            leaf,
            operation_name=operation_name,
            redactor=redactor,
        )

    rebuilt = rebuild_exception_group(
        error,
        group_message=f"{operation_name} reported multiple failures.",
        leaf_mapper=map_leaf,
        invalid_leaf_factory=lambda: RuntimeError(
            f"{operation_name} reported an invalid failure group."
        ),
        max_nodes=MAX_DIAGNOSTIC_EXCEPTION_GROUP_NODES,
        truncated_leaf_factory=lambda: RuntimeError(
            f"{operation_name} omitted additional failure evidence."
        ),
    )
    if caller_cancellation is not None and not cancellation_claimed:
        rebuilt = BaseExceptionGroup(
            f"{operation_name} failed after caller cancellation.",
            [
                rebuilt,
                _task_store_cancellation(
                    caller_cancellation,
                    operation_name=operation_name,
                    redactor=redactor,
                ),
            ],
        )
    deduplicated = _deduplicate_detached_task_store_group(
        rebuilt,
        operation_name=operation_name,
    )
    return _credential_safe_task_store_group(
        deduplicated,
        operation_name=operation_name,
        group_message=f"{operation_name} reported multiple failures.",
        redactor=redactor,
    )


def _credential_safe_task_store_group(
    error: BaseExceptionGroup,
    *,
    operation_name: str,
    group_message: str,
    redactor: SecretRedactor,
) -> BaseExceptionGroup:
    """Apply the final diagnostic boundary after runtime group composition."""

    return credential_safe_runtime_exception_group(
        error,
        group_message=group_message,
        leaf_mapper=lambda leaf: leaf,
        invalid_leaf_factory=lambda: RuntimeError(
            f"{operation_name} reported an invalid failure group."
        ),
        truncated_leaf_factory=lambda: RuntimeError(
            f"{operation_name} omitted additional failure evidence."
        ),
        fallback_leaf_mapper=lambda leaf: _generic_task_store_failure(
            leaf,
            operation_name=operation_name,
            redactor=redactor,
        ),
        redactor=redactor,
    )


def _generic_task_store_failure(
    error: BaseException,
    *,
    operation_name: str,
    redactor: SecretRedactor,
) -> BaseException:
    """Preserve a detached store failure's public classification without text."""

    if isinstance(error, GeneratorExit):
        safe_type: type[BaseException] = GeneratorExit
    elif isinstance(error, KeyboardInterrupt):
        safe_type = KeyboardInterrupt
    elif isinstance(error, SystemExit):
        safe_type = SystemExit
    elif isinstance(error, asyncio.CancelledError):
        safe_type = asyncio.CancelledError
    elif type(error) in {
        TimeoutError,
        ConnectionError,
        TaskClaimLost,
        WorkAttemptAdmissionConflict,
        WorkAttemptExecutionClaimLost,
        WorkAttemptRecoveryRequired,
        CompletionVerificationClaimLost,
        TaskCompletionDecisionRequired,
        WorkContractConflict,
        WorkCompletionConflict,
        ValueError,
        TypeError,
        NotImplementedError,
        RuntimeError,
    }:
        safe_type = type(error)
    elif isinstance(error, Exception):
        safe_type = RuntimeError
    else:
        safe_type = BaseException
    return credential_safe_runtime_exception(
        safe_type,
        f"{operation_name} failure diagnostic was redacted.",
        redactor=redactor,
        fallback_message="Task-store failure diagnostic was withheld.",
    )


def _generic_task_store_settlement_evidence(
    error: BaseException,
    *,
    operation_name: str,
    redactor: SecretRedactor,
) -> BaseException:
    """Discard evidence text while retaining its bounded failure structure."""

    if not isinstance(error, BaseExceptionGroup):
        return _generic_task_store_failure(
            error,
            operation_name=operation_name,
            redactor=redactor,
        )
    return credential_safe_runtime_exception_group(
        error,
        group_message=f"{operation_name} cancellation settlement evidence was redacted.",
        leaf_mapper=lambda leaf: _generic_task_store_failure(
            leaf,
            operation_name=operation_name,
            redactor=redactor,
        ),
        invalid_leaf_factory=lambda: RuntimeError(
            f"{operation_name} reported invalid cancellation settlement evidence."
        ),
        truncated_leaf_factory=lambda: RuntimeError(
            f"{operation_name} omitted additional cancellation settlement evidence."
        ),
        fallback_leaf_mapper=lambda leaf: _generic_task_store_failure(
            leaf,
            operation_name=operation_name,
            redactor=redactor,
        ),
        redactor=redactor,
    )


def _deduplicate_detached_task_store_group(
    error: BaseExceptionGroup,
    *,
    operation_name: str,
) -> BaseExceptionGroup:
    """Retain each detached failure identity once across a shared group graph."""

    next_token = 1
    pending: list[tuple[int, BaseException, bool]] = [(0, error, False)]
    child_tokens_by_group: dict[int, tuple[int, ...]] = {}
    rebuilt_by_token: dict[int, BaseException] = {}
    observed_failure_ids: set[int] = set()

    def repeated_failure() -> RuntimeError:
        return RuntimeError(f"{operation_name} reported repeated failure evidence.")

    while pending:
        token, candidate, expanded = pending.pop()
        if expanded:
            child_tokens = child_tokens_by_group.pop(token, ())
            children: list[BaseException] = []
            for child_token in child_tokens:
                child = rebuilt_by_token.pop(child_token, None)
                children.append(child if child is not None else repeated_failure())
            rebuilt_by_token[token] = BaseExceptionGroup(
                f"{operation_name} reported multiple failures.",
                children or [repeated_failure()],
            )
            continue

        candidate_id = id(candidate)
        if candidate_id in observed_failure_ids:
            rebuilt_by_token[token] = repeated_failure()
            continue
        observed_failure_ids.add(candidate_id)
        if not isinstance(candidate, BaseExceptionGroup):
            rebuilt_by_token[token] = candidate
            continue

        children = exception_group_children(candidate)
        if children is None:
            rebuilt_by_token[token] = RuntimeError(
                f"{operation_name} reported an invalid failure group."
            )
            continue
        child_tokens = tuple(range(next_token, next_token + len(children)))
        next_token += len(children)
        child_tokens_by_group[token] = child_tokens
        pending.append((token, candidate, True))
        pending.extend(
            (child_token, child, False)
            for child_token, child in reversed(tuple(zip(child_tokens, children, strict=True)))
        )

    rebuilt = rebuilt_by_token.get(0)
    if isinstance(rebuilt, BaseExceptionGroup):
        return rebuilt
    return BaseExceptionGroup(
        f"{operation_name} reported multiple failures.",
        [
            rebuilt
            if isinstance(rebuilt, BaseException)
            else RuntimeError(f"{operation_name} reported an invalid failure group.")
        ],
    )


def _is_forwarded_task_store_cancellation(
    cancellation: asyncio.CancelledError,
    marker: _TaskStoreOperationCancellationMarker | None,
) -> bool:
    """Authenticate the exact cancellation this boundary sent to its child."""

    if marker is None:
        return False
    try:
        args = _BASE_EXCEPTION_ARGS_DESCRIPTOR.__get__(cancellation, BaseException)
    except BaseException:
        return False
    return type(args) is tuple and len(args) == 1 and args[0] is marker


def capture_sensitive_validation(
    operation: Callable[[], _ResultT],
    *,
    operation_name: str,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[_ResultT]:
    """Run synchronous validation without exporting hostile fatal diagnostics.

    Ordinary validation exceptions remain represented by an empty outcome so
    the public API can publish its stable invalid-input error. Fatal signals and
    groups are detached because callers and extension-owned return models can be
    mutated after construction to raise them from copying or serialization.
    """

    try:
        return TaskStoreOperationOutcome(result=operation())
    except BaseExceptionGroup as error:
        failure = _detached_task_store_group(
            error,
            operation_name=operation_name,
            redactor=redactor,
        )
    except asyncio.CancelledError:
        failure = RuntimeError(f"{operation_name} was cancelled without caller cancellation.")
    except Exception:
        return TaskStoreOperationOutcome()
    except BaseException as error:
        failure = _detached_task_store_failure(
            error,
            operation_name=operation_name,
            redactor=redactor,
        )
    return TaskStoreOperationOutcome(failure=failure)


def capture_sensitive_result_validation(
    operation: Callable[[], _ResultT],
    *,
    operation_name: str,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[_ResultT]:
    """Detach every failure raised while validating an extension result.

    Unlike caller-input validation, a successful extension call can return a
    well-formed but conflicting value. Preserve that stable classification and
    message after redaction, while discarding the extension-owned model and the
    validation traceback that retained it.
    """

    try:
        return TaskStoreOperationOutcome(result=operation())
    except BaseExceptionGroup as error:
        failure = _detached_task_store_group(
            error,
            operation_name=operation_name,
            redactor=redactor,
        )
    except asyncio.CancelledError:
        failure = RuntimeError(f"{operation_name} was cancelled without caller cancellation.")
    except BaseException as error:
        failure = _detached_task_store_failure(
            error,
            operation_name=operation_name,
            redactor=redactor,
        )
    return TaskStoreOperationOutcome(failure=failure)


def _task_store_mutation_quiescence_failure(
    mutation_store: TaskStore | None,
    mutation_method_name: str | None,
) -> BaseException | None:
    if mutation_store is None:
        return None
    if mutation_method_name is None:
        return RuntimeError("Verified-work mutation settlement requires an exact method identity.")
    if not task_store_mutation_is_cancellation_quiescent(
        mutation_store,
        mutation_method_name,
    ):
        return NotImplementedError(
            "The task-store mutation implementation must explicitly guarantee that "
            "verified-work mutations are cancellation-quiescent."
        )
    return None


def raise_task_store_operation_failure(failure: BaseException) -> NoReturn:
    """Raise one detached failure without erasing safe explicit evidence."""

    try:
        cause = _BASE_EXCEPTION_CAUSE_DESCRIPTOR.__get__(failure, BaseException)
    except BaseException:
        cause = None
    if cause is None:
        raise failure from None
    raise failure from cause


async def _capture_validated_task_store_operation(
    operation: Callable[[], Awaitable[_ResultT]],
    *,
    mutation_store: TaskStore | None,
    mutation_method_name: str | None,
) -> CapturedAwaitableOutcome[_ResultT]:
    """Revalidate mutation identity in its child immediately before invocation."""

    failure = _task_store_mutation_quiescence_failure(
        mutation_store,
        mutation_method_name,
    )
    if failure is not None:
        return CapturedAwaitableOutcome(error=failure)
    return await capture_awaitable_outcome(operation)


async def capture_task_store_operation(
    operation: Callable[[], Awaitable[_ResultT]],
    *,
    operation_name: str,
    redactor: SecretRedactor,
    mutation_store: TaskStore | None = None,
    mutation_method_name: str | None = None,
) -> TaskStoreOperationOutcome[_ResultT]:
    """Capture a store call while authenticating caller cancellation provenance."""

    mutation_quiescence_failure = _task_store_mutation_quiescence_failure(
        mutation_store,
        mutation_method_name,
    )
    if mutation_quiescence_failure is not None:
        return TaskStoreOperationOutcome(failure=mutation_quiescence_failure)

    current_task = asyncio.current_task()
    observed_cancellation_requests = 0 if current_task is None else current_task.cancelling()
    caller_cancellation: asyncio.CancelledError | None = None
    caller_cancellation_count = 0

    # Authenticate an already-pending request in the caller before the
    # extension is dispatched. Historical requests that were handled by an
    # earlier boundary do not raise at this checkpoint and are not reclaimed.
    try:
        await asyncio.sleep(0)
    except asyncio.CancelledError as error:
        current_cancellation_requests = 0 if current_task is None else current_task.cancelling()
        caller_cancellation = error
        caller_cancellation_count = max(
            current_cancellation_requests - observed_cancellation_requests,
            1,
        )
        observed_cancellation_requests = current_cancellation_requests
        del operation
        safe_cancellation = _task_store_cancellation(
            caller_cancellation,
            operation_name=operation_name,
            redactor=redactor,
        )
        retain_workspace_observation_pending_cancellation_requests(
            safe_cancellation,
            caller_cancellation_count,
        )
        return TaskStoreOperationOutcome(failure=safe_cancellation)

    # The extension must not share the caller task: otherwise it can call
    # current_task().cancel()/uncancel() and forge or erase the signal used as
    # caller authority. A fixed child-only request still lets a declared
    # quiescent mutation settle before this boundary returns.
    operation_task = asyncio.create_task(
        _capture_validated_task_store_operation(
            operation,
            mutation_store=mutation_store,
            mutation_method_name=mutation_method_name,
        ),
        name="cayu-task-store-operation",
    )
    child_cancellation_marker = _TaskStoreOperationCancellationMarker()
    del operation
    while not operation_task.done():
        try:
            await asyncio.shield(operation_task)
        except asyncio.CancelledError as error:
            current_cancellation_requests = 0 if current_task is None else current_task.cancelling()
            if current_cancellation_requests > observed_cancellation_requests:
                caller_cancellation_count += (
                    current_cancellation_requests - observed_cancellation_requests
                )
                observed_cancellation_requests = current_cancellation_requests
                if caller_cancellation is None:
                    caller_cancellation = error
                    operation_task.cancel(child_cancellation_marker)
                continue
            if operation_task.done():
                break
            raise

    # Completion and cancellation can become ready in the same loop turn. The
    # caller gets one final real delivery point before a successful result can
    # escape the ownership boundary.
    try:
        await asyncio.sleep(0)
    except asyncio.CancelledError as error:
        current_cancellation_requests = 0 if current_task is None else current_task.cancelling()
        if current_cancellation_requests <= observed_cancellation_requests:
            raise
        caller_cancellation_count += current_cancellation_requests - observed_cancellation_requests
        observed_cancellation_requests = current_cancellation_requests
        if caller_cancellation is None:
            caller_cancellation = error

    try:
        captured = operation_task.result()
    except BaseException as error:
        captured_result: _ResultT | None = None
        captured_error: BaseException | None = error
    else:
        captured_result = captured.result
        captured_error = captured.error
        del captured
    del operation_task
    caller_cancelled = caller_cancellation is not None

    if captured_error is None:
        result = captured_result
        if not caller_cancelled:
            return TaskStoreOperationOutcome(result=result)
        del result
        if caller_cancellation is None:  # pragma: no cover - construction invariant
            raise AssertionError("Caller cancellation was not retained.")
        safe_cancellation = _task_store_cancellation(
            caller_cancellation,
            operation_name=operation_name,
            redactor=redactor,
        )
        retain_workspace_observation_pending_cancellation_requests(
            safe_cancellation,
            caller_cancellation_count,
        )
        return TaskStoreOperationOutcome(failure=safe_cancellation)

    error = captured_error
    if isinstance(error, asyncio.CancelledError):
        if caller_cancelled and _is_forwarded_task_store_cancellation(
            error,
            child_cancellation_marker,
        ):
            if caller_cancellation is None:  # pragma: no cover - construction invariant
                raise AssertionError("Caller cancellation was not retained.")
            failure: BaseException = _task_store_forwarded_cancellation(
                caller_cancellation,
                error,
                operation_name=operation_name,
                redactor=redactor,
            )
        else:
            child_cancellation_failure = RuntimeError(
                f"{operation_name} was cancelled without "
                + (
                    "caller cancellation."
                    if caller_cancellation is None
                    else "distinct caller cancellation."
                )
            )
            if caller_cancelled:
                if caller_cancellation is None:  # pragma: no cover - construction invariant
                    raise AssertionError("Caller cancellation was not retained.")
                group_message = f"{operation_name} failed after caller cancellation."
                failure = _credential_safe_task_store_group(
                    BaseExceptionGroup(
                        group_message,
                        [
                            child_cancellation_failure,
                            _task_store_cancellation(
                                caller_cancellation,
                                operation_name=operation_name,
                                redactor=redactor,
                            ),
                        ],
                    ),
                    operation_name=operation_name,
                    group_message=group_message,
                    redactor=redactor,
                )
            else:
                failure = child_cancellation_failure
    elif isinstance(error, BaseExceptionGroup):
        failure = _detached_task_store_group(
            error,
            operation_name=operation_name,
            redactor=redactor,
            caller_cancellation=caller_cancellation,
            child_cancellation_marker=(child_cancellation_marker if caller_cancelled else None),
        )
    else:
        safe_error = _detached_task_store_failure(
            error,
            operation_name=operation_name,
            redactor=redactor,
        )
        if caller_cancelled:
            if caller_cancellation is None:  # pragma: no cover - construction invariant
                raise AssertionError("Caller cancellation was not retained.")
            safe_cancellation = _task_store_cancellation(
                caller_cancellation,
                operation_name=operation_name,
                redactor=redactor,
            )
            group_message = f"{operation_name} failed after caller cancellation."
            failure = _credential_safe_task_store_group(
                BaseExceptionGroup(
                    group_message,
                    [safe_error, safe_cancellation],
                ),
                operation_name=operation_name,
                group_message=group_message,
                redactor=redactor,
            )
        else:
            failure = safe_error
    del error
    if caller_cancelled:
        retain_workspace_observation_pending_cancellation_requests(
            failure,
            caller_cancellation_count,
        )
    return TaskStoreOperationOutcome(failure=failure)


__all__ = [
    "TaskStoreOperationOutcome",
    "capture_sensitive_validation",
    "capture_task_store_operation",
    "raise_task_store_operation_failure",
    "task_store_cancellation_reconciliation_capability_is_complete",
    "task_store_mutation_is_cancellation_quiescent",
    "task_store_work_attempt_admission_capability_is_complete",
]
