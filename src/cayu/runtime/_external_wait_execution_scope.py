"""Private runtime preparation scope; SessionStore mints the retained identity."""

from contextlib import contextmanager
from contextvars import ContextVar

from cayu.sessions._external_wait_transition import ExternalWaitMutation
from cayu.sessions.external_waits import external_wait_digest

_EXECUTION: ContextVar[str | None] = ContextVar("external_wait_execution", default=None)


@contextmanager
def execution_preparation_scope(command: ExternalWaitMutation):
    if (
        command.kind not in {"prepare_execution", "exclude_execution"}
        or command.execution_intent is None
    ):
        raise PermissionError("External execution preparation requires an exact runtime intent.")
    token = _EXECUTION.set(external_wait_digest(command))
    try:
        yield
    finally:
        _EXECUTION.reset(token)


def require_execution_preparation(command: ExternalWaitMutation) -> None:
    if _EXECUTION.get() != external_wait_digest(command):
        raise PermissionError("External execution preparation requires its runtime owner.")
