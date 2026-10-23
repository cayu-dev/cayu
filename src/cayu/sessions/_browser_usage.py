"""Exact private ownership of an incarnation-bound native browser-use receipt."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

BROWSER_USAGE_KEY = "__cayu_browser_usage_v1__"
_OWNER: ContextVar[tuple[str, str] | None] = ContextVar("browser_usage_publication", default=None)


def browser_usage_receipt(session_id: str, instance_id: str) -> dict[str, str]:
    return {"session_id": session_id, "session_instance_id": instance_id}


@contextmanager
def browser_usage_publication_scope(session_id: str, instance_id: str) -> Iterator[None]:
    """Only the exact built-in tool publication owner enters this scope."""
    token = _OWNER.set((session_id, instance_id))
    try:
        yield
    finally:
        _OWNER.reset(token)


def require_browser_usage_owner(key: str, record: object = None) -> None:
    if key != BROWSER_USAGE_KEY:
        return
    owner = _OWNER.get()
    if owner is None or (record is not None and record != browser_usage_receipt(*owner)):
        raise ValueError("Browser usage requires its exact native publication owner.")
