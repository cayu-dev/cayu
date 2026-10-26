"""Keep test runners away from Cayu's real interpreter-exit hooks (never runtime API)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from cayu.runners import _exit_cleanup


@contextmanager
def isolated_interpreter_exit_cleanup() -> Iterator[None]:
    """Restore the exit-cleanup registries to what they held on entry.

    Tests use fake runners and container IDs; at interpreter exit the hooks
    would otherwise run real `docker rm -f` commands against them.
    """

    unsettled = dict(_exit_cleanup._unsettled)
    holders = dict(_exit_cleanup._holders)
    try:
        yield
    finally:
        _exit_cleanup._unsettled.clear()
        _exit_cleanup._unsettled.update(unsettled)
        _exit_cleanup._holders.clear()
        _exit_cleanup._holders.update(holders)
