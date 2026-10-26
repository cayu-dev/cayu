"""Best-effort sandbox cleanup at normal interpreter exit.

Nothing here replaces ``await app.aclose()``. It only covers sandboxes whose
release Cayu had already decided on, or that nothing could ever reconnect to,
when a process ends without closing its app:

- A runner whose close or kill started but never settled (its close deadline
  passed, or the event loop stopped while the removal was still running) is
  finished at exit through ``Runner._close_all_at_interpreter_exit``. The
  runtime asked for that removal after any workspace copy-back, so finishing
  it is always safe.
- An application whose session store does not outlive the process registers
  its own hook (``EnvironmentLifecycle.declare_process_local_session_store``):
  allocations recorded only in that store can never be reconnected, so they
  are released too.

Exit hooks only run on a normal interpreter exit (including an unhandled
exception or ``sys.exit``). A process killed by a signal runs none of them. A
durable store recovers its allocations through ``recover_incomplete_sessions``;
for a process-local store, ``reap_orphaned_docker_coding_containers`` removes
Docker coding containers once their owner process is gone.
"""

from __future__ import annotations

import atexit
import logging
import os
import weakref
from collections.abc import Callable, Iterable
from contextlib import suppress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cayu.runners.base import Runner

logger = logging.getLogger(__name__)

_owner_pid = os.getpid()
# Runners whose terminal lifecycle started but has not settled, with its action.
# Held strongly: a runner its owner dropped still owes this removal.
_unsettled: dict[int, tuple[Runner, str]] = {}
# Process-local allocation holders: each returns the runners it still holds.
_holders: dict[int, Callable[[], Iterable[Runner]]] = {}
# Runners a forked child inherited from its parent; held so ids stay unique.
_inherited: dict[int, Runner] = {}


def track_unsettled_terminal_lifecycle(runner: Runner, action: str) -> None:
    """Remember that ``runner`` started terminal ``action`` and has not settled it."""

    _unsettled[id(runner)] = (runner, action)


def forget_unsettled_terminal_lifecycle(runner: Runner) -> None:
    """Drop ``runner`` once its terminal lifecycle settled successfully."""

    # The entry holds the runner strongly, so its id cannot have been reused.
    _unsettled.pop(id(runner), None)


def release_held_runners_at_exit(owner: object, held: Callable[[], Iterable[Runner]]) -> None:
    """Apply each runner's close action at exit for allocations nothing can reconnect to.

    ``held`` must not reference ``owner``: it outlives it. When ``owner`` is
    collected while ``held`` returns nothing, the registration is dropped;
    otherwise it stays until exit, because those runners still owe a release.
    """

    key = id(held)
    _holders[key] = held

    def owner_collected() -> None:
        if _holders.get(key) is held and not any(True for _ in held()):
            del _holders[key]

    weakref.finalize(owner, owner_collected).atexit = False


def close_runners_at_exit(entries: Iterable[tuple[Runner, str | None]]) -> int:
    """Run each adapter's synchronous exit close; return how many it released.

    Entries are grouped by runner type so an adapter can batch them (Docker
    runs one ``docker rm -f`` per daemon). Never raises: this runs during
    interpreter shutdown.
    """

    by_type: dict[type[Runner], list[tuple[Runner, str | None]]] = {}
    for runner, action in entries:
        by_type.setdefault(type(runner), []).append((runner, action))
    released = 0
    for runner_type, group in by_type.items():
        try:
            released += int(runner_type._close_all_at_interpreter_exit(group))
        except Exception:
            logger.debug("Closing runners at interpreter exit failed.", exc_info=True)
    return released


def owned_by_this_process(owner_pid: int) -> bool:
    """Whether an exit hook registered in ``owner_pid`` may act in this process.

    A forked child that exits normally must never release its parent's sandboxes.
    """

    return owner_pid == os.getpid()


def _finish_unsettled_at_exit() -> None:
    if not owned_by_this_process(_owner_pid):
        return
    entries = [(runner, action) for runner, action in _unsettled.values() if not runner.is_closed]
    _unsettled.clear()
    finished = close_runners_at_exit(entries)
    if finished:
        with suppress(Exception):
            logger.warning(
                "Finished %d sandbox removal(s) at interpreter exit that did not settle "
                "before the event loop stopped; close the app with `await app.aclose()` "
                "to settle cleanup inside the loop.",
                finished,
            )


def _release_held_at_exit() -> None:
    if not owned_by_this_process(_owner_pid):
        return
    holders = list(_holders.values())
    _holders.clear()
    runners: dict[int, Runner] = {}
    for held in holders:
        with suppress(Exception):
            runners.update(
                (id(runner), runner)
                for runner in held()
                # A forked child keeps its own allocations, never its parent's.
                if _inherited.get(id(runner)) is not runner
            )
    released = close_runners_at_exit((runner, None) for runner in runners.values())
    if released:
        with suppress(Exception):
            logger.warning(
                "Released %d sandbox(es) at interpreter exit that an unclosed app with a "
                "process-local session store still held; close the app with "
                "`await app.aclose()` to release them inside the loop.",
                released,
            )


def _forget_in_child() -> None:
    """In a forked child, give up the parent's sandboxes but keep registrations.

    An inherited application keeps its holder, so allocations the child makes
    are still released at the child's exit; the runners it held at the fork
    belong to the parent and are excluded.
    """

    global _owner_pid
    _owner_pid = os.getpid()
    _unsettled.clear()
    for held in list(_holders.values()):
        with suppress(Exception):
            _inherited.update((id(runner), runner) for runner in held())


# atexit runs hooks last-registered first: release held allocations, then
# finish removals that were already under way.
atexit.register(_finish_unsettled_at_exit)
atexit.register(_release_held_at_exit)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_in_child)


__all__ = [
    "close_runners_at_exit",
    "forget_unsettled_terminal_lifecycle",
    "owned_by_this_process",
    "release_held_runners_at_exit",
    "track_unsettled_terminal_lifecycle",
]
