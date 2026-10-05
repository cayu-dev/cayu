"""Receiving-owner scope for a durable managed-task hint handoff."""

from contextlib import contextmanager
from contextvars import ContextVar

from cayu.sessions.external_waits import external_wait_digest

_TIMER: ContextVar[str | None] = ContextVar("external_wait_timer", default=None)


@contextmanager
def timer_scope(command):
    if command.kind not in {"prepare_timer", "publish_timer"} or command.timer is None:
        raise PermissionError("External timer requires its exact scheduling operation.")
    token = _TIMER.set(external_wait_digest(command))
    try:
        yield
    finally:
        _TIMER.reset(token)


def require_timer_scope(command):
    if _TIMER.get() != external_wait_digest(command):
        raise PermissionError("External timer requires its registered receiving owner.")
