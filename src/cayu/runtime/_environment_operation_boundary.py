"""Secret-safe boundary for extension-owned environment operations."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, cast

from cayu._exception_groups import exception_tree_contains, rebuild_exception_group
from cayu.runtime._binding_cleanup import (
    BindingCleanupStatus,
    BindingFinalizeFailure,
    attach_binding_cleanup_status,
    attach_binding_finalize_safe_payload,
    binding_cleanup_status,
    binding_finalize_safe_payload,
    binding_finalize_status,
    record_binding_finalize_failures,
)
from cayu.runtime._diagnostics import exception_diagnostic
from cayu.vaults import SecretRedactor

_BASE_EXCEPTION_ARGS_DESCRIPTOR = BaseException.__dict__["args"]
_BASE_EXCEPTION_CAUSE_DESCRIPTOR = BaseException.__dict__["__cause__"]
_BASE_EXCEPTION_CONTEXT_DESCRIPTOR = BaseException.__dict__["__context__"]
_BASE_EXCEPTION_DICT_DESCRIPTOR = BaseException.__dict__["__dict__"]
_BASE_EXCEPTION_TRACEBACK_DESCRIPTOR = BaseException.__dict__["__traceback__"]


async def _completed_environment_operation() -> None:
    """Replace extension-owned callables retained by propagated tracebacks."""


completed_environment_operation = _completed_environment_operation


def _binding_status_field(status: object, name: str, default: object) -> object:
    try:
        return object.__getattribute__(status, name)
    except BaseException:
        return default


def _environment_failure_redactor(
    error: BaseException,
    *,
    redactor: SecretRedactor,
) -> SecretRedactor:
    """Include binding-owned credentials before any propagated failure is copied."""

    status = binding_finalize_status(error)
    if status is None:
        return redactor
    supplemental = _binding_status_field(status, "supplemental_redactor", None)
    if isinstance(supplemental, SecretRedactor):
        return redactor.merged_with(supplemental)
    return redactor


def _detached_binding_cleanup_status(
    status: BindingCleanupStatus,
    *,
    redactor: SecretRedactor,
) -> BindingCleanupStatus:
    """Copy an authenticated cleanup handoff without its original failures."""

    initial_error = _binding_status_field(
        status,
        "initial_error",
        RuntimeError("Binding cleanup failure metadata was invalid."),
    )
    if not isinstance(initial_error, BaseException):
        initial_error = RuntimeError("Binding cleanup failure metadata was invalid.")
    retry_value = _binding_status_field(status, "retry", _completed_environment_operation)
    if not callable(retry_value):
        retry_value = _completed_environment_operation
    retry = cast("Callable[[], Awaitable[None]]", retry_value)
    retry_error = _binding_status_field(status, "retry_error", None)
    if retry_error is not None and not isinstance(retry_error, BaseException):
        retry_error = RuntimeError("Binding cleanup retry metadata was invalid.")
    return BindingCleanupStatus(
        initial_error=_detached_environment_failure(
            initial_error,
            redactor=redactor,
        ),
        retry=retry,
        retry_attempted=_binding_status_field(status, "retry_attempted", False) is True,
        retry_error=(
            None
            if retry_error is None
            else _detached_environment_failure(
                retry_error,
                redactor=redactor,
            )
        ),
    )


def _attach_detached_binding_statuses(
    source: BaseException,
    target: BaseException,
    *,
    redactor: SecretRedactor,
) -> None:
    """Move lifecycle ownership to a safe exception without raw failure objects."""

    if (cleanup_status := binding_cleanup_status(source)) is not None:
        attach_binding_cleanup_status(
            target,
            _detached_binding_cleanup_status(
                cleanup_status,
                redactor=redactor,
            ),
        )
    if (finalize_status := binding_finalize_status(source)) is not None:
        supplemental_redactor = finalize_status.supplemental_redactor
        status_redactor = (
            redactor
            if supplemental_redactor is None
            else redactor.merged_with(supplemental_redactor)
        )
        record_binding_finalize_failures(
            target,
            tuple(
                BindingFinalizeFailure(
                    phase=failure.phase,
                    error=(
                        target
                        if failure.error is source
                        else _detached_environment_failure(
                            failure.error,
                            redactor=status_redactor,
                        )
                    ),
                )
                for failure in finalize_status.failures
            ),
        )
    if (safe_payload := binding_finalize_safe_payload(source)) is not None:
        attach_binding_finalize_safe_payload(target, safe_payload)


def _detached_environment_cancellation(
    cancellation: asyncio.CancelledError,
    *,
    redactor: SecretRedactor,
) -> asyncio.CancelledError:
    """Return a secret-safe cancellation without extension-owned attachments."""

    redactor = _environment_failure_redactor(cancellation, redactor=redactor)
    safe_args: tuple[Any, ...] = ("Environment operation cancelled",)
    try:
        raw_args = _BASE_EXCEPTION_ARGS_DESCRIPTOR.__get__(
            cancellation,
            BaseException,
        )
        if type(raw_args) is tuple:
            safe_args = tuple(
                redactor.redact_text(value)
                if type(value) is str
                else "Environment operation cancelled"
                for value in raw_args
            )
        safe_cancellation = asyncio.CancelledError(*safe_args)
        _attach_detached_binding_statuses(
            cancellation,
            safe_cancellation,
            redactor=redactor,
        )
        _BASE_EXCEPTION_ARGS_DESCRIPTOR.__set__(cancellation, ())
        _BASE_EXCEPTION_TRACEBACK_DESCRIPTOR.__set__(cancellation, None)
        _BASE_EXCEPTION_CAUSE_DESCRIPTOR.__set__(cancellation, None)
        _BASE_EXCEPTION_CONTEXT_DESCRIPTOR.__set__(cancellation, None)
        namespace = _BASE_EXCEPTION_DICT_DESCRIPTOR.__get__(
            cancellation,
            BaseException,
        )
        if type(namespace) is dict:
            dict.clear(namespace)
        if type(cancellation) is asyncio.CancelledError:
            _BASE_EXCEPTION_ARGS_DESCRIPTOR.__set__(cancellation, safe_args)
            _attach_detached_binding_statuses(
                safe_cancellation,
                cancellation,
                redactor=redactor,
            )
            return cancellation
    except BaseException:
        safe_args = ("Environment operation cancelled",)
        safe_cancellation = asyncio.CancelledError(*safe_args)
    return safe_cancellation


def _detach_exact_environment_failure_in_place(
    error: BaseException,
    *,
    message: str,
    redactor: SecretRedactor,
) -> BaseException | None:
    """Scrub a mutable built-in failure while preserving its phase identity."""

    if type(error) not in {
        Exception,
        RuntimeError,
        TypeError,
        ValueError,
    }:
        return None
    status_carrier = RuntimeError("detached environment failure status")
    try:
        _attach_detached_binding_statuses(
            error,
            status_carrier,
            redactor=redactor,
        )
        # ``message`` has already been redacted and byte-bounded by
        # ``exception_diagnostic``. Reprocessing the original arguments here
        # can expand a short repeated secret far beyond the diagnostic limit.
        _BASE_EXCEPTION_ARGS_DESCRIPTOR.__set__(error, (message,))
        _BASE_EXCEPTION_TRACEBACK_DESCRIPTOR.__set__(error, None)
        _BASE_EXCEPTION_CAUSE_DESCRIPTOR.__set__(error, None)
        _BASE_EXCEPTION_CONTEXT_DESCRIPTOR.__set__(error, None)
        namespace = _BASE_EXCEPTION_DICT_DESCRIPTOR.__get__(error, BaseException)
        if type(namespace) is dict:
            dict.clear(namespace)
        _attach_detached_binding_statuses(
            status_carrier,
            error,
            redactor=redactor,
        )
    except BaseException:
        return None
    return error


def _environment_operation_group(
    error: BaseExceptionGroup,
    *,
    operation: str,
    redactor: SecretRedactor,
    caller_cancelled: bool,
) -> BaseExceptionGroup:
    """Detach every extension-owned leaf while preserving ordered failure shape."""

    redactor = _environment_failure_redactor(error, redactor=redactor)
    group_message = f"{operation} reported multiple failures."
    try:
        raw_args = _BASE_EXCEPTION_ARGS_DESCRIPTOR.__get__(error, BaseException)
        if type(raw_args) is tuple and raw_args and type(raw_args[0]) is str:
            group_message = redactor.redact_text(raw_args[0])
    except BaseException:
        pass
    cancellation_claimed = False

    def map_leaf(leaf: BaseException) -> BaseException:
        nonlocal cancellation_claimed
        if isinstance(leaf, asyncio.CancelledError):
            cancellation_claimed = True
        return _detached_environment_failure(
            leaf,
            redactor=redactor,
        )

    rebuilt = rebuild_exception_group(
        error,
        group_message=group_message,
        leaf_mapper=map_leaf,
        invalid_leaf_factory=lambda: RuntimeError(
            f"{operation} reported an invalid failure group."
        ),
    )
    if caller_cancelled and not cancellation_claimed:
        rebuilt = BaseExceptionGroup(
            f"{operation} failed after caller cancellation.",
            [rebuilt, asyncio.CancelledError("Environment operation cancelled")],
        )
    _attach_detached_binding_statuses(
        error,
        rebuilt,
        redactor=redactor,
    )
    return rebuilt


def _detached_environment_failure(
    error: BaseException,
    *,
    redactor: SecretRedactor,
) -> BaseException:
    """Return a fresh, bounded failure without extension traceback state."""

    redactor = _environment_failure_redactor(error, redactor=redactor)
    if isinstance(error, asyncio.CancelledError):
        return _detached_environment_cancellation(
            error,
            redactor=redactor,
        )
    if not redactor.has_values:
        return error
    diagnostic = exception_diagnostic(
        error,
        empty_message="environment operation failed",
        nonportable_message=("Environment operation failed with a non-portable diagnostic."),
        redactor=redactor,
    )
    message = diagnostic.message
    if (
        detached := _detach_exact_environment_failure_in_place(
            error,
            message=message,
            redactor=redactor,
        )
    ) is not None:
        return detached
    if isinstance(error, GeneratorExit):
        return GeneratorExit(message)
    if isinstance(error, KeyboardInterrupt):
        return KeyboardInterrupt(message)
    if isinstance(error, SystemExit):
        return SystemExit(message)
    if type(error) is TimeoutError:
        detached_timeout = TimeoutError(message)
        _attach_detached_binding_statuses(
            error,
            detached_timeout,
            redactor=redactor,
        )
        return detached_timeout
    if type(error) is ValueError:
        return ValueError(message)
    if type(error) is TypeError:
        return TypeError(message)
    if type(error) is RuntimeError:
        return RuntimeError(message)
    if isinstance(error, Exception):
        return RuntimeError(f"{diagnostic.error_type}: {message}")
    return BaseException(f"{diagnostic.error_type}: {message}")


async def await_environment_operation(
    operation: Callable[[], Awaitable[Any]],
    *,
    operation_name: str,
    redactor: SecretRedactor,
) -> Any:
    """Await an extension while distinguishing caller and child cancellation."""

    current_task = asyncio.current_task()
    cancellation_baseline = 0 if current_task is None else current_task.cancelling()
    cancellation_pending_at_entry = bool(
        current_task is not None and getattr(current_task, "_must_cancel", False)
    )
    result: Any = None
    sanitized_failure: BaseException | None = None
    try:
        result = await operation()
    except asyncio.CancelledError as cancellation:
        caller_cancelled = (
            current_task is not None and current_task.cancelling() > cancellation_baseline
        ) or cancellation_pending_at_entry
        safe_cancellation = _detached_environment_cancellation(
            cancellation,
            redactor=redactor,
        )
        del cancellation
        if caller_cancelled:
            sanitized_failure = safe_cancellation
        else:
            child_cancellation_failure = RuntimeError(
                f"{operation_name} was cancelled without caller cancellation."
            )
            _attach_detached_binding_statuses(
                safe_cancellation,
                child_cancellation_failure,
                redactor=redactor,
            )
            sanitized_failure = child_cancellation_failure
    except BaseExceptionGroup as error:
        caller_cancelled = (
            current_task is not None and current_task.cancelling() > cancellation_baseline
        ) or cancellation_pending_at_entry
        if not caller_cancelled and not exception_tree_contains(
            error,
            asyncio.CancelledError,
        ):
            raise
        safe_group = _environment_operation_group(
            error,
            operation=operation_name,
            redactor=redactor,
            caller_cancelled=caller_cancelled,
        )
        del error
        sanitized_failure = safe_group
    except BaseException as error:
        caller_cancelled = (
            current_task is not None and current_task.cancelling() > cancellation_baseline
        ) or cancellation_pending_at_entry
        if not caller_cancelled:
            raise
        cancel_message = (
            None if current_task is None else getattr(current_task, "_cancel_message", None)
        )
        safe_cancellation = _detached_environment_cancellation(
            asyncio.CancelledError(cancel_message),
            redactor=redactor,
        )
        safe_error = _detached_environment_failure(
            error,
            redactor=redactor,
        )
        source_error = error
        del error
        sanitized_failure = BaseExceptionGroup(
            f"{operation_name} failed after caller cancellation.",
            [safe_error, safe_cancellation],
        )
        _attach_detached_binding_statuses(
            source_error,
            sanitized_failure,
            redactor=redactor,
        )
        source_error = None
    operation = _completed_environment_operation
    if sanitized_failure is not None:
        raise sanitized_failure from None
    if current_task is not None and current_task.cancelling() > cancellation_baseline:
        safe_cancellation = _detached_environment_cancellation(
            asyncio.CancelledError(getattr(current_task, "_cancel_message", None)),
            redactor=redactor,
        )
        result = None
        raise safe_cancellation from None
    return result


__all__ = ["await_environment_operation", "completed_environment_operation"]
