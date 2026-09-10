"""Cancellation ownership for work-attempt session-store mutations."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast

from cayu._task_wait import (
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    restore_task_cancellation_requests,
)
from cayu._validation import copy_json_value
from cayu.core.messages import Message, detach_message
from cayu.runtime._checkpoint_redaction import durable_value_contains_secret
from cayu.runtime._diagnostics import (
    credential_safe_runtime_exception,
    credential_safe_runtime_exception_group,
    exception_diagnostic,
    runtime_owned_exception_renderings_are_credential_safe,
)
from cayu.runtime._task_store_operation_boundary import (
    TaskStoreOperationOutcome,
    capture_sensitive_result_validation,
    capture_task_store_operation,
    raise_task_store_operation_failure,
)
from cayu.runtime.invocation import SessionInvocationBinding
from cayu.runtime.sessions import (
    DeferredInteractionInput,
    EventRecord,
    Session,
    SessionStore,
    TranscriptSnapshot,
    _initial_transcript_pending_interaction_id,
    copy_session,
)
from cayu.runtime.work_attempt_admission import (
    WorkAttemptAdmission,
    WorkAttemptAdmissionState,
    WorkAttemptExecutionStopReason,
    WorkAttemptExecutionStopRequest,
    require_work_attempt_admission_result,
    require_work_attempt_execution_stop_result,
)
from cayu.vaults import SecretRedactor

_ResultT = TypeVar("_ResultT")

if TYPE_CHECKING:
    from cayu.runtime.tasks import TaskStore


async def record_work_attempt_execution_stop(
    store: TaskStore,
    *,
    admission: WorkAttemptAdmission,
    reason: WorkAttemptExecutionStopReason,
    redactor: SecretRedactor,
) -> None:
    """Preserve the first authoritative stop before terminal cleanup effects."""
    if admission.execution_stop is not None:
        return
    entry = admission.execution_entry
    if entry is None:
        raise RuntimeError("Governed execution stop has no execution entry.")
    request = WorkAttemptExecutionStopRequest(
        admission_id=admission.admission_id,
        prepare_request_sha256=admission.prepare_request_sha256,
        claim_id=admission.claim.claim_id,
        worker_id=admission.claim.worker_id,
        execution_owner_id=admission.claim.execution_owner_id,
        generation=admission.claim.generation,
        execution_entry=entry,
        reason=reason,
    )
    outcome = await capture_task_store_operation(
        lambda: store.record_work_attempt_execution_stop(request),
        operation_name="Governed execution stop",
        redactor=redactor,
        mutation_store=store,
        mutation_method_name="record_work_attempt_execution_stop",
    )
    if outcome.failure is not None:
        raise_task_store_operation_failure(outcome.failure)
    validation = capture_sensitive_result_validation(
        lambda: require_work_attempt_execution_stop_result(outcome.result, admission, request),
        operation_name="Governed execution stop validation",
        redactor=redactor,
    )
    if validation.failure is not None:
        raise_task_store_operation_failure(validation.failure)


async def reconcile_work_attempt_initial_publication(
    store: SessionStore,
    *,
    admission: WorkAttemptAdmission,
    expected_source: list[Message],
    expected_transcript: list[Message],
    redactor: SecretRedactor,
) -> bool:
    """Recognize only the exact committed initial projection after a lost reply.

    This is readback inside the existing run owner, not dispatch or release
    authority. A remaining deferred record must still use the atomic publisher.
    """
    deferred = await read_work_attempt_session_store(
        lambda: store.load_deferred_interaction_input(admission.session_id),
        operation_name="Initial publication deferred-input readback",
        redactor=redactor,
    )
    if deferred is not None:
        return False
    session = await read_work_attempt_session_store(
        lambda: store.load(admission.session_id),
        operation_name="Initial publication session readback",
        redactor=redactor,
    )
    snapshot = await read_work_attempt_session_store(
        lambda: store.load_transcript_snapshot(admission.session_id),
        operation_name="Initial publication transcript readback",
        redactor=redactor,
    )
    checkpoint = await read_work_attempt_session_store(
        lambda: store.load_checkpoint(admission.session_id),
        operation_name="Initial publication checkpoint readback",
        redactor=redactor,
    )

    def validate() -> bool:
        if type(session) is not Session or type(snapshot) is not TranscriptSnapshot:
            raise TypeError("Initial publication readback returned invalid records.")
        current = copy_session(session)
        transcript = TranscriptSnapshot.model_validate(
            snapshot.model_dump(mode="python", warnings=False)
        )
        prefix_count = len(expected_transcript) - len(expected_source)
        entry = admission.execution_entry
        if (
            admission.kind != "initial"
            or entry is None
            or current.run_epoch != entry.request.run_epoch
            or SessionInvocationBinding(
                id=current.id,
                session_instance_id=current.instance_id,
                invocation=current.invocation,
            )
            != admission.session_invocation
            or _initial_transcript_pending_interaction_id(checkpoint) is not None
            or prefix_count < 0
            or expected_transcript[prefix_count:] != expected_source
            or transcript.cursor != len(expected_transcript)
            or len(transcript.records) != len(expected_transcript)
            or any(
                record.index != index
                or record.message != expected_transcript[index]
                or record.interaction_id
                != (None if index < prefix_count else admission.interaction_id)
                for index, record in enumerate(transcript.records)
            )
        ):
            raise RuntimeError("Initial publication readback conflicts with admitted input.")
        return True

    outcome = capture_sensitive_result_validation(
        validate,
        operation_name="Initial publication exact readback validation",
        redactor=redactor,
    )
    if outcome.failure is not None:
        raise_task_store_operation_failure(outcome.failure)
    return outcome.result is True


@dataclass(frozen=True, slots=True)
class WorkAttemptExecutionInput:
    """Detached transcript projection; materialization stays with the run owner."""

    messages: tuple[Message, ...]
    messages_to_append: tuple[Message, ...]
    deferred: bool


def capture_work_attempt_execution_input_result(
    transcript: object,
    deferred: object,
    *,
    admission: WorkAttemptAdmission,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[WorkAttemptExecutionInput]:
    def prepare_input() -> WorkAttemptExecutionInput:
        admitted = require_work_attempt_admission_result(
            admission, operation_name="Execution-input admission"
        )
        initial = admitted.kind == "initial"
        if type(transcript) is not list:
            raise TypeError("Work-attempt transcript must be a list.")
        detached: list[Message] = []
        for message in transcript:
            if type(message) is not Message:
                raise TypeError("Work-attempt transcript contains an invalid message.")
            detached.append(detach_message(message))
        messages = tuple(detached)
        if deferred is None:
            if initial and not (
                admitted.state is WorkAttemptAdmissionState.ACTIVE
                and admitted.claim.generation > 1
                and admitted.recovery_evidence_sha256 is not None
            ):
                raise RuntimeError("Initial work-attempt execution lacks deferred input.")
            return WorkAttemptExecutionInput(messages, (), False)
        if type(deferred) is not DeferredInteractionInput:
            raise TypeError("Work-attempt deferred input has an invalid type.")
        copied = DeferredInteractionInput.model_validate(
            deferred.model_dump(mode="python", warnings=False)
        )
        if copied.interaction_id != admitted.interaction_id:
            raise RuntimeError("Work-attempt deferred input belongs to another interaction.")
        source = tuple(detach_message(message) for message in copied.source_messages)
        if initial:
            if messages or copied.initial_transcript_messages is None:
                raise RuntimeError("Initial work-attempt transcript conflicts with deferred input.")
            messages = tuple(
                detach_message(message) for message in copied.initial_transcript_messages
            )
        else:
            if copied.initial_transcript_messages is not None:
                raise RuntimeError("Continuation input unexpectedly replaces the transcript.")
            messages = (*messages, *source)
        return WorkAttemptExecutionInput(messages, source, True)

    return capture_sensitive_result_validation(
        prepare_input,
        operation_name="Work-attempt execution-input validation",
        redactor=redactor,
    )


def capture_work_attempt_session_result(
    value: object,
    *,
    operation_name: str,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[Session]:
    """Detach and revalidate one successful session-store result."""

    def copy_result() -> Session:
        copied = copy_session(cast("Session", value))
        payload = copied.model_dump(mode="json", warnings=False)
        if redactor.redact_json(payload) != payload:
            raise RuntimeError(f"{operation_name} returned secret-bearing session authority.")
        return copied

    return capture_sensitive_result_validation(
        copy_result,
        operation_name=f"{operation_name} result validation",
        redactor=redactor,
    )


def capture_work_attempt_checkpoint_result(
    value: object,
    *,
    operation_name: str,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[dict[str, Any] | None]:
    """Detach and revalidate one checkpoint used as work-attempt authority."""

    def copy_checkpoint() -> dict[str, Any] | None:
        if value is None:
            return None
        copied = copy_json_value(value, "checkpoint")
        if type(copied) is not dict:
            raise TypeError("Work-attempt checkpoint must be a JSON object or None.")
        if durable_value_contains_secret(copied, redactor=redactor):
            raise RuntimeError(f"{operation_name} returned a secret-bearing checkpoint.")
        return copied

    return capture_sensitive_result_validation(
        copy_checkpoint,
        operation_name=f"{operation_name} result validation",
        redactor=redactor,
    )


def capture_work_attempt_deferred_input_result(
    value: object,
    *,
    operation_name: str,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[tuple[str, bool] | None]:
    """Detach deferred input and retain its identity plus exact-prefix evidence."""

    def copy_deferred_input() -> tuple[str, bool] | None:
        if value is None:
            return None
        if type(value) is not DeferredInteractionInput:
            raise TypeError("Work-attempt deferred input has an invalid type.")
        copied = DeferredInteractionInput.model_validate(
            value.model_dump(mode="python", warnings=False)
        )
        return copied.interaction_id, copied.initial_transcript_messages is not None

    return capture_sensitive_result_validation(
        copy_deferred_input,
        operation_name=f"{operation_name} result validation",
        redactor=redactor,
    )


def capture_work_attempt_event_records_result(
    value: object,
    *,
    operation_name: str,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[tuple[EventRecord, ...]]:
    """Detach and revalidate event records used for admission handoff."""

    def copy_records() -> tuple[EventRecord, ...]:
        if type(value) is not list:
            raise TypeError("Work-attempt event lookup must return a list.")
        copied: list[EventRecord] = []
        for record in value:
            if type(record) is not EventRecord:
                raise TypeError("Work-attempt event lookup returned an invalid record.")
            copied.append(
                EventRecord.model_validate(record.model_dump(mode="python", warnings=False))
            )
        payload = [record.model_dump(mode="json", warnings=False) for record in copied]
        if redactor.redact_json(payload) != payload:
            raise RuntimeError(f"{operation_name} returned secret-bearing event authority.")
        return tuple(copied)

    return capture_sensitive_result_validation(
        copy_records,
        operation_name=f"{operation_name} result validation",
        redactor=redactor,
    )


def _session_mutation_failure_type(
    error: BaseException,
    *,
    preserved_failure_types: tuple[type[BaseException], ...],
) -> type[BaseException]:
    if isinstance(error, GeneratorExit):
        return GeneratorExit
    if isinstance(error, KeyboardInterrupt):
        return KeyboardInterrupt
    if isinstance(error, SystemExit):
        return SystemExit
    if isinstance(error, asyncio.CancelledError):
        return RuntimeError
    if type(error) in preserved_failure_types:
        return type(error)
    if type(error) in {
        TimeoutError,
        ConnectionError,
        KeyError,
        ValueError,
        TypeError,
        NotImplementedError,
        RuntimeError,
    }:
        return type(error)
    return RuntimeError if isinstance(error, Exception) else BaseException


def _generic_session_mutation_failure(
    error: BaseException,
    *,
    operation_name: str,
    preserved_failure_types: tuple[type[BaseException], ...],
    redactor: SecretRedactor,
) -> BaseException:
    """Preserve bounded failure structure after aggregate rendering redaction."""

    return credential_safe_runtime_exception(
        _session_mutation_failure_type(
            error,
            preserved_failure_types=preserved_failure_types,
        ),
        f"{operation_name} failure diagnostic was redacted.",
        redactor=redactor,
        fallback_message="Session mutation failure diagnostic was withheld.",
    )


def _detached_session_mutation_failure(
    error: BaseException,
    *,
    operation_name: str,
    preserved_failure_types: tuple[type[BaseException], ...],
    redactor: SecretRedactor,
) -> BaseException:
    """Return bounded evidence without retaining extension traceback locals."""

    if isinstance(error, BaseExceptionGroup):
        return credential_safe_runtime_exception_group(
            error,
            group_message=f"{operation_name} reported multiple failures.",
            leaf_mapper=lambda leaf: _detached_session_mutation_failure(
                leaf,
                operation_name=operation_name,
                preserved_failure_types=preserved_failure_types,
                redactor=redactor,
            ),
            invalid_leaf_factory=lambda: RuntimeError(
                f"{operation_name} reported invalid failure evidence."
            ),
            truncated_leaf_factory=lambda: RuntimeError(
                f"{operation_name} omitted additional failure evidence."
            ),
            fallback_leaf_mapper=lambda leaf: _generic_session_mutation_failure(
                leaf,
                operation_name=operation_name,
                preserved_failure_types=preserved_failure_types,
                redactor=redactor,
            ),
            redactor=redactor,
        )
    if isinstance(error, asyncio.CancelledError):
        return credential_safe_runtime_exception(
            RuntimeError,
            f"{operation_name} was cancelled without caller cancellation.",
            redactor=redactor,
            fallback_message="Session mutation cancellation was reclassified.",
        )
    diagnostic = exception_diagnostic(
        error,
        empty_message=f"{operation_name} failed",
        nonportable_message=f"{operation_name} failed with a non-portable diagnostic.",
        redactor=redactor,
    )
    message = f"{diagnostic.error_type}: {diagnostic.message}"
    return credential_safe_runtime_exception(
        _session_mutation_failure_type(
            error,
            preserved_failure_types=preserved_failure_types,
        ),
        message,
        redactor=redactor,
        fallback_message=f"{operation_name} failure diagnostic was redacted.",
    )


def _session_mutation_cancellation(
    cancellation: asyncio.CancelledError,
    *,
    operation_name: str,
    preserved_failure_types: tuple[type[BaseException], ...],
    redactor: SecretRedactor,
    settlement_error: BaseException | None = None,
) -> asyncio.CancelledError:
    """Detach caller cancellation and any completed mutation failure evidence."""

    diagnostic = exception_diagnostic(
        cancellation,
        empty_message=f"{operation_name.lower()} cancelled",
        nonportable_message=f"{operation_name} cancellation had a non-portable diagnostic.",
        redactor=redactor,
    )
    safe_cancellation = credential_safe_runtime_exception(
        asyncio.CancelledError,
        diagnostic.message,
        redactor=redactor,
        fallback_message=f"{operation_name} cancellation diagnostic was redacted.",
    )
    if settlement_error is None:
        return safe_cancellation
    safe_cancellation.__cause__ = _detached_session_mutation_failure(
        settlement_error,
        operation_name=operation_name,
        preserved_failure_types=preserved_failure_types,
        redactor=redactor,
    )
    if runtime_owned_exception_renderings_are_credential_safe(
        safe_cancellation,
        redactor=redactor,
    ):
        return safe_cancellation
    safe_cancellation.__cause__ = credential_safe_runtime_exception(
        RuntimeError,
        f"{operation_name} settlement failure diagnostic was redacted.",
        redactor=redactor,
        fallback_message="Session mutation settlement evidence was withheld.",
    )
    if runtime_owned_exception_renderings_are_credential_safe(
        safe_cancellation,
        redactor=redactor,
    ):
        return safe_cancellation
    safe_cancellation.__cause__ = None
    return safe_cancellation


async def read_work_attempt_session_store(
    operation: Callable[[], Awaitable[_ResultT]],
    *,
    operation_name: str,
    redactor: SecretRedactor,
) -> _ResultT:
    """Own and detach one extension-provided session-store authority read."""

    outcome = await capture_task_store_operation(
        operation,
        operation_name=operation_name,
        redactor=redactor,
    )
    if outcome.failure is not None:
        raise_task_store_operation_failure(outcome.failure)
    return cast("_ResultT", outcome.result)


async def settle_work_attempt_owned_operation(
    operation: Callable[[], Awaitable[_ResultT]],
    *,
    operation_name: str,
    preserved_failure_types: tuple[type[BaseException], ...] = (),
    redactor: SecretRedactor,
) -> _ResultT:
    """Keep one dispatched work-attempt operation owned until its outcome is definite.

    Session-store extensions may delegate to threads, subprocesses, remote SDKs,
    or other cancellation-opaque work.  Once dispatched, caller cancellation is
    therefore retained while the exact child settles; returning earlier would
    make the corresponding task-store generation replaceable while the old
    session mutation could still commit.
    """

    # Honor cancellation that was already pending before this boundary without
    # starting a mutation that the caller never dispatched. Detach that signal
    # here so neither its arguments nor the undispatched operation remain
    # reachable through the public traceback.
    try:
        await asyncio.sleep(0)
    except asyncio.CancelledError as cancellation:
        safe_cancellation = _session_mutation_cancellation(
            cancellation,
            operation_name=operation_name,
            preserved_failure_types=preserved_failure_types,
            redactor=redactor,
        )
        del cancellation, operation
        raise safe_cancellation from None
    owner = asyncio.create_task(
        capture_awaitable_outcome(operation),
        name=f"cayu-{operation_name}",
    )
    del operation
    outcome = await await_shielded_task_outcome(owner)
    del owner
    if outcome.error is not None:
        safe_error = _detached_session_mutation_failure(
            outcome.error,
            operation_name=operation_name,
            preserved_failure_types=preserved_failure_types,
            redactor=redactor,
        )
        del outcome
        raise safe_error from None
    captured = outcome.result
    if captured is None:  # pragma: no cover - captured child invariant
        raise AssertionError("Work-attempt session mutation returned no captured outcome.")
    if outcome.cancellation is not None:
        safe_cancellation = _session_mutation_cancellation(
            outcome.cancellation,
            operation_name=operation_name,
            preserved_failure_types=preserved_failure_types,
            redactor=redactor,
            settlement_error=captured.error,
        )
        restore_task_cancellation_requests(
            outcome.cancellation_requests_consumed,
            cancellation=safe_cancellation,
        )
        del captured, outcome
        if safe_cancellation.__cause__ is not None:
            raise safe_cancellation from safe_cancellation.__cause__
        raise safe_cancellation from None
    if captured.error is not None:
        safe_error = _detached_session_mutation_failure(
            captured.error,
            operation_name=operation_name,
            preserved_failure_types=preserved_failure_types,
            redactor=redactor,
        )
        del captured, outcome
        raise safe_error from None
    if captured.result is None:
        raise RuntimeError("Work-attempt owned operation returned no durable result.")
    return captured.result


async def settle_work_attempt_session_mutation(
    operation: Callable[[], Awaitable[_ResultT]],
    *,
    operation_name: str,
    preserved_failure_types: tuple[type[BaseException], ...] = (),
    redactor: SecretRedactor,
) -> _ResultT:
    """Own one dispatched session mutation through its definite settlement."""

    return await settle_work_attempt_owned_operation(
        operation,
        operation_name=operation_name,
        preserved_failure_types=preserved_failure_types,
        redactor=redactor,
    )


__all__ = [
    "capture_work_attempt_checkpoint_result",
    "capture_work_attempt_deferred_input_result",
    "capture_work_attempt_event_records_result",
    "capture_work_attempt_session_result",
    "read_work_attempt_session_store",
    "settle_work_attempt_owned_operation",
    "settle_work_attempt_session_mutation",
]
