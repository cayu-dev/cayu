"""Internal handoff for binding rollback failures.

Bindings own their rollback, while ``CayuApp`` owns the last-resort runner
cleanup and the durable failure events.  This small status object carries the
structured failure across that boundary without replacing the original bind
exception.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

_STATUS_ATTRIBUTE = "_cayu_binding_cleanup_status"


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


def record_binding_cleanup_failure(
    bind_error: BaseException,
    cleanup_error: BaseException,
    *,
    retry: Callable[[], Awaitable[None]],
) -> BindingCleanupStatus:
    """Attach cleanup state while preserving the binding exception identity."""

    status = BindingCleanupStatus(initial_error=cleanup_error, retry=retry)
    setattr(bind_error, _STATUS_ATTRIBUTE, status)
    return status


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
