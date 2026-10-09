"""Access rule for runtime-owned tool-call replay records."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

STORAGE_KEY = "cayu:tool-call-replay:v1:"
_owned: ContextVar[bool] = ContextVar("tool_call_replay_owned", default=False)


@contextmanager
def owner_scope() -> Iterator[None]:
    token = _owned.set(True)
    try:
        yield
    finally:
        _owned.reset(token)


def require_operation_key_access(key: str, *, read: bool) -> None:
    del read
    if key.startswith(STORAGE_KEY) and not _owned.get():
        raise ValueError("Tool-call replay records are runtime-owned.")
