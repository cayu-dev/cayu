"""Internal handoff for structured binding lifecycle failures.

Bindings own their rollback, while the session engine and environment lifecycle
own last-resort cleanup and durable failure events. This small status object carries the
structured failure across that boundary without replacing original exception
objects. Finalization metadata uses the same controlled handoff for durable
phase diagnostics.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from cayu._exception_groups import (
    exception_cause,
    exception_context,
    exception_group_children,
    iter_exception_tree,
)
from cayu._exception_state import exception_state, set_exception_state
from cayu._validation import require_durable_text
from cayu.runtime._diagnostics import (
    _attach_runtime_exception_payload,
    _copy_runtime_exception_payload,
    _runtime_exception_payload,
    exception_diagnostic,
)
from cayu.vaults import SecretRedactor

_STATUS_ATTRIBUTE = "_cayu_binding_cleanup_status"
_FINALIZE_STATUS_ATTRIBUTE = "_cayu_binding_finalize_status"
_FINALIZE_SAFE_PAYLOAD_ATTRIBUTE = "_cayu_binding_finalize_safe_payload"
_BINDING_STATUS_TOKEN = object()
_MISSING_STATUS_FIELD = object()
BINDING_FINALIZE_ERROR_TEXT_MAX_BYTES = 512
_BINDING_FINALIZE_ERROR_TYPE_MAX_BYTES = 128
_TRUNCATED_SUFFIX = "... [truncated]"
_BINDING_FINALIZE_EMPTY_MESSAGE = "binding finalization failed"
_BINDING_FINALIZE_NONPORTABLE_MESSAGE = (
    "Binding finalization failed with a non-portable diagnostic."
)
_BINDING_FINALIZE_REDACTION_FAILURE_MESSAGE = (
    "Binding finalization failed and its diagnostic could not be safely redacted."
)

BindingFinalizePhase = Literal[
    "cancellation",
    "workspace_finalize",
    "managed_resource_cleanup",
]


@dataclass
class BindingCleanupStatus:
    """Mutable cleanup status attached to the original binding exception."""

    initial_error: BaseException
    retry: Callable[[], Awaitable[None]]
    retry_attempted: bool = False
    retry_error: BaseException | None = None

    @property
    def incomplete(self) -> bool:
        return not self.retry_attempted or self.retry_error is not None


@dataclass(frozen=True)
class BindingFinalizeFailure:
    """One controlled finalization phase failure retained for diagnostics."""

    phase: BindingFinalizePhase
    error: BaseException


@dataclass(frozen=True)
class BindingFinalizeStatus:
    """Ordered finalization failures attached to the propagated exception."""

    failures: tuple[BindingFinalizeFailure, ...]
    supplemental_redactor: SecretRedactor | None = None


@dataclass(frozen=True, slots=True)
class _BindingStatusHandoff:
    """Authenticated runtime-owned attachment for in-process lifecycle state."""

    attribute_name: str
    status: object
    token: object


def attach_binding_finalize_safe_payload(
    error: BaseException,
    payload: dict[str, Any],
) -> None:
    """Attach validated finalization evidence without trusting exception accessors."""

    _attach_runtime_exception_payload(
        error,
        attribute_name=_FINALIZE_SAFE_PAYLOAD_ATTRIBUTE,
        payload=payload,
    )


def binding_finalize_safe_payload(error: BaseException) -> dict[str, Any] | None:
    """Return only evidence attached through the runtime-owned handoff."""

    return _runtime_exception_payload(
        error,
        attribute_name=_FINALIZE_SAFE_PAYLOAD_ATTRIBUTE,
    )


def copy_binding_finalize_safe_payload(
    source: BaseException,
    target: BaseException,
) -> None:
    """Propagate an authenticated immutable handoff to an aggregate error."""

    _copy_runtime_exception_payload(
        source,
        target,
        attribute_name=_FINALIZE_SAFE_PAYLOAD_ATTRIBUTE,
    )


def record_binding_cleanup_failure(
    bind_error: BaseException,
    cleanup_error: BaseException,
    *,
    retry: Callable[[], Awaitable[None]],
) -> BindingCleanupStatus:
    """Attach cleanup state while preserving the binding exception identity."""

    if not isinstance(bind_error, BaseException):
        raise TypeError("bind_error must be an exception.")
    if not isinstance(cleanup_error, BaseException):
        raise TypeError("cleanup_error must be an exception.")
    if not callable(retry):
        raise TypeError("retry must be callable.")
    status = BindingCleanupStatus(initial_error=cleanup_error, retry=retry)
    attach_binding_cleanup_status(bind_error, status)
    return status


def attach_binding_cleanup_status(
    error: BaseException,
    status: BindingCleanupStatus,
) -> None:
    """Attach existing cleanup state to an aggregate that preserves its source."""

    if type(status) is not BindingCleanupStatus:
        raise TypeError("status must be a BindingCleanupStatus.")
    _attach_binding_status(
        error,
        attribute_name=_STATUS_ATTRIBUTE,
        status=status,
    )


def binding_cleanup_status(error: BaseException) -> BindingCleanupStatus | None:
    """Return runtime-owned cleanup state attached by a binding, if present."""

    status = _binding_status(
        error,
        attribute_name=_STATUS_ATTRIBUTE,
        expected_type=BindingCleanupStatus,
    )
    return status if type(status) is BindingCleanupStatus else None


def binding_cleanup_payload(
    error: BaseException,
    *,
    redactor: SecretRedactor,
) -> dict[str, Any] | None:
    """Serialize cleanup state without exposing exception objects."""

    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    status = binding_cleanup_status(error)
    if status is None:
        return None
    initial_error = _binding_status_field(status, "initial_error", _MISSING_STATUS_FIELD)
    if not isinstance(initial_error, BaseException):
        initial_error = RuntimeError("Binding cleanup failure metadata was invalid.")
    retry_attempted = (
        _binding_status_field(status, "retry_attempted", _MISSING_STATUS_FIELD) is True
    )
    retry_error = _binding_status_field(status, "retry_error", _MISSING_STATUS_FIELD)
    if retry_error is _MISSING_STATUS_FIELD:
        retry_error = (
            RuntimeError("Binding cleanup retry metadata was invalid.") if retry_attempted else None
        )
    if retry_error is not None and not isinstance(retry_error, BaseException):
        retry_error = RuntimeError("Binding cleanup retry metadata was invalid.")
    payload: dict[str, Any] = {
        "incomplete": not retry_attempted or retry_error is not None,
        **exception_diagnostic(
            initial_error,
            empty_message="binding cleanup failed",
            nonportable_message=("Binding cleanup failed with a non-portable diagnostic."),
            redactor=redactor,
        ).payload_fields(
            message_key="initial_error",
            error_type_key="initial_error_type",
            detail_prefix="initial_",
        ),
        "retry_attempted": retry_attempted,
    }
    if retry_attempted:
        payload["retry_completed"] = retry_error is None
    if retry_error is not None:
        payload.update(
            exception_diagnostic(
                retry_error,
                empty_message="binding cleanup retry failed",
                nonportable_message=(
                    "Binding cleanup retry failed with a non-portable diagnostic."
                ),
                redactor=redactor,
            ).payload_fields(
                message_key="retry_error",
                error_type_key="retry_error_type",
                detail_prefix="retry_",
            )
        )
    return payload


def record_binding_finalize_failures(
    error: BaseException,
    failures: tuple[BindingFinalizeFailure, ...],
    *,
    supplemental_redactor: SecretRedactor | None = None,
) -> BindingFinalizeStatus:
    """Attach trusted phase ordering without replacing any child exception."""

    if type(failures) is not tuple or not failures:
        raise ValueError("Binding finalization failures cannot be empty.")
    for failure in failures:
        if type(failure) is not BindingFinalizeFailure:
            raise TypeError("failures must contain BindingFinalizeFailure values.")
        if failure.phase not in {
            "cancellation",
            "workspace_finalize",
            "managed_resource_cleanup",
        }:
            raise ValueError("Binding finalization failure phase is invalid.")
        if not isinstance(failure.error, BaseException):
            raise TypeError("Binding finalization failure errors must be exceptions.")
    if supplemental_redactor is not None and not isinstance(supplemental_redactor, SecretRedactor):
        raise TypeError("supplemental_redactor must be a SecretRedactor or None.")
    status = BindingFinalizeStatus(
        failures=failures,
        supplemental_redactor=supplemental_redactor,
    )
    _attach_binding_status(
        error,
        attribute_name=_FINALIZE_STATUS_ATTRIBUTE,
        status=status,
    )
    return status


def binding_finalize_status(error: BaseException) -> BindingFinalizeStatus | None:
    """Return trusted phase ordering attached by a binding, if present."""

    status = _binding_status(
        error,
        attribute_name=_FINALIZE_STATUS_ATTRIBUTE,
        expected_type=BindingFinalizeStatus,
    )
    return status if type(status) is BindingFinalizeStatus else None


def binding_finalize_cancellation(error: BaseException) -> asyncio.CancelledError | None:
    """Find cancellation without invoking extension-defined accessors."""

    pending: list[BaseException] = [error]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop()
        if id(candidate) in visited:
            continue
        visited.add(id(candidate))
        if isinstance(candidate, asyncio.CancelledError):
            return candidate
        context = exception_context(candidate)
        if context is not None:
            pending.append(context)
        cause = exception_cause(candidate)
        if cause is not None:
            pending.append(cause)
        if isinstance(candidate, BaseExceptionGroup):
            children = exception_group_children(candidate)
            if children is not None:
                pending.extend(reversed(children))
    return None


def binding_finalize_explicit_cancellation(
    error: BaseException,
) -> asyncio.CancelledError | None:
    """Find cancellation explicitly propagated by an error or exception group."""

    for candidate in iter_exception_tree(error):
        if isinstance(candidate, asyncio.CancelledError):
            return candidate
    return None


def is_containable_cleanup_error(error: BaseException) -> bool:
    """Return whether best-effort cleanup may contain every propagated leaf."""

    pending = [error]
    visited: set[int] = set()
    while pending:
        candidate = pending.pop()
        if id(candidate) in visited:
            continue
        visited.add(id(candidate))
        if isinstance(candidate, BaseExceptionGroup):
            children = exception_group_children(candidate)
            if children is None:
                return False
            pending.extend(children)
        elif not isinstance(candidate, Exception | asyncio.CancelledError):
            return False
    return True


def binding_finalize_fatal_signal(error: BaseException) -> BaseException | None:
    """Retain fatal interpreter/control-flow signals carried by a group."""

    fatal_types = (GeneratorExit, KeyboardInterrupt, SystemExit)
    if isinstance(error, fatal_types):
        return error
    if isinstance(error, BaseExceptionGroup):
        fatal_signals = [
            candidate
            for candidate in iter_exception_tree(error)
            if isinstance(candidate, fatal_types)
        ]
        if fatal_signals:
            return BaseExceptionGroup(
                "Binding finalization carried fatal control-flow signals.",
                fatal_signals,
            )
    return None


def append_binding_finalize_cancellation(
    error: BaseException,
    cancellation: asyncio.CancelledError,
) -> BaseException:
    """Retain one cancellation observed after binding phase failures."""

    status = binding_finalize_status(error)
    failures = (
        status.failures
        if status is not None
        else (BindingFinalizeFailure(phase="workspace_finalize", error=error),)
    )
    if binding_finalize_explicit_cancellation(error) is not None or any(
        failure.phase == "cancellation" or isinstance(failure.error, asyncio.CancelledError)
        for failure in failures
    ):
        return error
    failures = (
        *failures,
        BindingFinalizeFailure(phase="cancellation", error=cancellation),
    )
    aggregate = BaseExceptionGroup(
        "Binding finalization reported failures before cancellation.",
        [failure.error for failure in failures],
    )
    copy_binding_finalize_safe_payload(error, aggregate)
    record_binding_finalize_failures(
        aggregate,
        failures,
        supplemental_redactor=(status.supplemental_redactor if status is not None else None),
    )
    return aggregate


def binding_finalize_error_details(
    error: BaseException,
    *,
    redactor: SecretRedactor,
) -> dict[str, str]:
    """Return bounded, redacted fields safe for durable diagnostics."""

    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    status = binding_finalize_status(error)
    redactors = _diagnostic_redactors(redactor, status=status)
    return _binding_finalize_error_details(error, redactors=redactors)


def binding_finalize_failure_payload(
    error: BaseException,
    *,
    redactor: SecretRedactor,
) -> list[dict[str, str]] | None:
    """Serialize controlled finalization metadata without exception attributes."""

    status = binding_finalize_status(error)
    if status is None:
        return None
    redactors = _diagnostic_redactors(redactor, status=status)
    return [
        {
            "phase": failure.phase,
            **_binding_finalize_error_details(failure.error, redactors=redactors),
        }
        for failure in status.failures
    ]


def _binding_finalize_error_details(
    error: BaseException,
    *,
    redactors: tuple[SecretRedactor, ...],
) -> dict[str, str]:
    combined_redactor = redactors[0]
    for redactor in redactors[1:]:
        combined_redactor = combined_redactor.merged_with(redactor)
    diagnostic = exception_diagnostic(
        error,
        empty_message=_BINDING_FINALIZE_EMPTY_MESSAGE,
        nonportable_message=_BINDING_FINALIZE_NONPORTABLE_MESSAGE,
        preserve_empty_message=True,
        redactor=combined_redactor,
    )
    return {
        "error": _bounded_redacted_text(
            diagnostic.message,
            redactors=redactors,
            max_bytes=BINDING_FINALIZE_ERROR_TEXT_MAX_BYTES,
            fallback=_BINDING_FINALIZE_REDACTION_FAILURE_MESSAGE,
        ),
        "error_type": _bounded_redacted_text(
            diagnostic.error_type,
            redactors=redactors,
            max_bytes=_BINDING_FINALIZE_ERROR_TYPE_MAX_BYTES,
            fallback="Exception",
        ),
    }


def _diagnostic_redactors(
    redactor: SecretRedactor,
    *,
    status: BindingFinalizeStatus | None,
) -> tuple[SecretRedactor, ...]:
    supplemental_redactor = (
        None if status is None else _binding_status_field(status, "supplemental_redactor", None)
    )
    if not isinstance(supplemental_redactor, SecretRedactor):
        return (redactor,)
    # Exact binding-owned credentials must be removed before an application
    # secret that happens to be a substring can make the exact value unmatchable.
    return (supplemental_redactor, redactor)


def _bounded_redacted_text(
    value: str,
    *,
    redactors: tuple[SecretRedactor, ...],
    max_bytes: int,
    fallback: str,
) -> str:
    """Redact first, then retain valid UTF-8 within one durable byte bound."""

    try:
        redacted = value
        for redactor in redactors:
            redacted = SecretRedactor.redact_text(redactor, redacted)
        require_durable_text(redacted, "binding finalization diagnostic")
    except BaseException:
        redacted = fallback
    encoded = redacted.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return encoded.decode("utf-8")
    suffix = _TRUNCATED_SUFFIX.encode("utf-8")
    prefix = encoded[: max_bytes - len(suffix)].decode("utf-8", errors="ignore")
    return prefix + _TRUNCATED_SUFFIX


def _attach_binding_status(
    error: BaseException,
    *,
    attribute_name: str,
    status: object,
) -> None:
    if not isinstance(error, BaseException):
        raise TypeError("error must be an exception.")
    handoff = _BindingStatusHandoff(
        attribute_name=attribute_name,
        status=status,
        token=_BINDING_STATUS_TOKEN,
    )
    if not set_exception_state(error, attribute_name, handoff):
        raise RuntimeError("Could not attach binding lifecycle status.")


def _binding_status(
    error: BaseException,
    *,
    attribute_name: str,
    expected_type: type[object],
) -> object | None:
    handoff = exception_state(error, attribute_name)
    if (
        type(handoff) is not _BindingStatusHandoff
        or handoff.token is not _BINDING_STATUS_TOKEN
        or handoff.attribute_name != attribute_name
        or type(handoff.status) is not expected_type
    ):
        return None
    return handoff.status


def _binding_status_field(status: object, name: str, default: object) -> object:
    try:
        return object.__getattribute__(status, name)
    except BaseException:
        return default
