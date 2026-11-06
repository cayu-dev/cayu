"""Positive evidence that an invocation create or admission did not commit.

A create or admission command that fails replays its exact receipt before it
raises. The receipt is written atomically with the commit, so a replay that
completes without one proves the command did not commit. Only that proof
marks the failure; a failure of the replay itself proves nothing.
"""

from __future__ import annotations

from cayu._exception_state import exception_state, set_exception_state

_NOT_COMMITTED_STATE = "_cayu_invocation_command_not_committed"
_NOT_COMMITTED = object()


def mark_invocation_command_not_committed(error: BaseException) -> None:
    set_exception_state(error, _NOT_COMMITTED_STATE, _NOT_COMMITTED)


def invocation_command_proven_not_committed(error: BaseException) -> bool:
    return exception_state(error, _NOT_COMMITTED_STATE) is _NOT_COMMITTED
