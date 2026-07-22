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

from cayu.vaults import SecretRedactor

_STATUS_ATTRIBUTE = "_cayu_binding_cleanup_status"
_FINALIZE_STATUS_ATTRIBUTE = "_cayu_binding_finalize_status"
BINDING_FINALIZE_SAFE_PAYLOAD_ATTRIBUTE = "_cayu_binding_finalize_safe_payload"
BINDING_FINALIZE_ERROR_TEXT_MAX_BYTES = 512
_BINDING_FINALIZE_ERROR_TYPE_MAX_BYTES = 128
_TRUNCATED_SUFFIX = "... [truncated]"

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


def record_binding_cleanup_failure(
    bind_error: BaseException,
    cleanup_error: BaseException,
    *,
    retry: Callable[[], Awaitable[None]],
) -> BindingCleanupStatus:
    """Attach cleanup state while preserving the binding exception identity."""

    status = BindingCleanupStatus(initial_error=cleanup_error, retry=retry)
    attach_binding_cleanup_status(bind_error, status)
    return status


def attach_binding_cleanup_status(
    error: BaseException,
    status: BindingCleanupStatus,
) -> None:
    """Attach existing cleanup state to an aggregate that preserves its source."""

    setattr(error, _STATUS_ATTRIBUTE, status)


def binding_cleanup_status(error: BaseException) -> BindingCleanupStatus | None:
    """Return runtime-owned cleanup state attached by a binding, if present."""

    status = getattr(error, _STATUS_ATTRIBUTE, None)
    return status if isinstance(status, BindingCleanupStatus) else None


def binding_cleanup_payload(error: BaseException) -> dict[str, Any] | None:
    """Serialize cleanup state without exposing exception objects."""

    status = binding_cleanup_status(error)
    if status is None:
        return None
    payload: dict[str, Any] = {
        "incomplete": status.incomplete,
        "initial_error": str(status.initial_error),
        "initial_error_type": type(status.initial_error).__name__,
        "retry_attempted": status.retry_attempted,
    }
    if status.retry_attempted:
        payload["retry_completed"] = status.retry_error is None
    if status.retry_error is not None:
        payload["retry_error"] = str(status.retry_error)
        payload["retry_error_type"] = type(status.retry_error).__name__
    return payload


def record_binding_finalize_failures(
    error: BaseException,
    failures: tuple[BindingFinalizeFailure, ...],
    *,
    supplemental_redactor: SecretRedactor | None = None,
) -> BindingFinalizeStatus:
    """Attach trusted phase ordering without replacing any child exception."""

    if not failures:
        raise ValueError("Binding finalization failures cannot be empty.")
    if supplemental_redactor is not None and not isinstance(supplemental_redactor, SecretRedactor):
        raise TypeError("supplemental_redactor must be a SecretRedactor or None.")
    status = BindingFinalizeStatus(
        failures=failures,
        supplemental_redactor=supplemental_redactor,
    )
    setattr(error, _FINALIZE_STATUS_ATTRIBUTE, status)
    return status


def binding_finalize_status(error: BaseException) -> BindingFinalizeStatus | None:
    """Return trusted phase ordering attached by a binding, if present."""

    status = getattr(error, _FINALIZE_STATUS_ATTRIBUTE, None)
    return status if isinstance(status, BindingFinalizeStatus) else None


def binding_finalize_cancellation(error: BaseException) -> asyncio.CancelledError | None:
    """Find the first cancellation in an exception group or causal chain."""

    visited: set[int] = set()

    def find(candidate: BaseException | None) -> asyncio.CancelledError | None:
        if candidate is None or id(candidate) in visited:
            return None
        visited.add(id(candidate))
        if isinstance(candidate, asyncio.CancelledError):
            return candidate
        if isinstance(candidate, BaseExceptionGroup):
            for child in candidate.exceptions:
                cancellation = find(child)
                if cancellation is not None:
                    return cancellation
        cancellation = find(candidate.__cause__)
        if cancellation is not None:
            return cancellation
        return find(candidate.__context__)

    return find(error)


def binding_finalize_explicit_cancellation(
    error: BaseException,
) -> asyncio.CancelledError | None:
    """Find cancellation explicitly propagated by an error or exception group."""

    if isinstance(error, asyncio.CancelledError):
        return error
    if isinstance(error, BaseExceptionGroup):
        for child in error.exceptions:
            cancellation = binding_finalize_explicit_cancellation(child)
            if cancellation is not None:
                return cancellation
    return None


def is_containable_cleanup_error(error: BaseException) -> bool:
    """Return whether best-effort cleanup may contain every propagated leaf."""

    if isinstance(error, Exception | asyncio.CancelledError):
        return True
    if isinstance(error, BaseExceptionGroup):
        return all(is_containable_cleanup_error(child) for child in error.exceptions)
    return False


def binding_finalize_fatal_signal(error: BaseException) -> BaseException | None:
    """Retain fatal interpreter/control-flow signals carried by a group."""

    fatal_types = (GeneratorExit, KeyboardInterrupt, SystemExit)
    if isinstance(error, fatal_types):
        return error
    if isinstance(error, BaseExceptionGroup):
        return error.subgroup(lambda candidate: isinstance(candidate, fatal_types))
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
    safe_payload = getattr(error, BINDING_FINALIZE_SAFE_PAYLOAD_ATTRIBUTE, None)
    if isinstance(safe_payload, dict):
        setattr(aggregate, BINDING_FINALIZE_SAFE_PAYLOAD_ATTRIBUTE, safe_payload)
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
    return {
        "error": _bounded_redacted_text(
            str(error),
            redactors=redactors,
            max_bytes=BINDING_FINALIZE_ERROR_TEXT_MAX_BYTES,
        ),
        "error_type": _bounded_redacted_text(
            type(error).__name__,
            redactors=redactors,
            max_bytes=_BINDING_FINALIZE_ERROR_TYPE_MAX_BYTES,
        ),
    }


def _diagnostic_redactors(
    redactor: SecretRedactor,
    *,
    status: BindingFinalizeStatus | None,
) -> tuple[SecretRedactor, ...]:
    if status is None or status.supplemental_redactor is None:
        return (redactor,)
    # Exact binding-owned credentials must be removed before an application
    # secret that happens to be a substring can make the exact value unmatchable.
    return (status.supplemental_redactor, redactor)


def _bounded_redacted_text(
    value: str,
    *,
    redactors: tuple[SecretRedactor, ...],
    max_bytes: int,
) -> str:
    """Redact first, then retain valid UTF-8 within one durable byte bound."""

    redacted = value
    for redactor in redactors:
        redacted = redactor.redact_text(redacted)
    encoded = redacted.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return encoded.decode("utf-8")
    suffix = _TRUNCATED_SUFFIX.encode("utf-8")
    prefix = encoded[: max_bytes - len(suffix)].decode("utf-8", errors="ignore")
    return prefix + _TRUNCATED_SUFFIX
