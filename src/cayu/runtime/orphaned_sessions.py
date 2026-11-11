"""Reconcile, on a schedule, the sessions a lost worker left ``running``.

:func:`run_orphaned_session_reconciliation` runs
``CayuApp.reconcile_orphaned_sessions`` at once and then on an interval. ``cayu
serve`` (``create_server``) and ``cayu worker`` run it; a host that runs Cayu
another way runs it beside its own work. Nothing here starts on import.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from math import isfinite
from typing import TYPE_CHECKING

from cayu.runtime.application_lifecycle import ApplicationAdmissionsSealed
from cayu.sessions.recovery import IncompleteSessionRecoveryAction

if TYPE_CHECKING:
    from cayu.applications import CayuApp

logger = logging.getLogger(__name__)

DEFAULT_ORPHANED_SESSION_RECONCILE_INTERVAL_SECONDS = 600.0
DEFAULT_ORPHANED_SESSION_INACTIVE_AFTER_SECONDS = 3600


async def run_orphaned_session_reconciliation(
    app: CayuApp,
    stop: asyncio.Event,
    *,
    interval_seconds: float = DEFAULT_ORPHANED_SESSION_RECONCILE_INTERVAL_SECONDS,
    inactive_for_seconds: int = DEFAULT_ORPHANED_SESSION_INACTIVE_AFTER_SECONDS,
) -> None:
    """Reconcile orphaned sessions now, then every ``interval_seconds``, until ``stop`` is set.

    Each pass is ``app.reconcile_orphaned_sessions(inactive_for_seconds=...)``. The
    first runs at once, so a process that starts after a deploy, or wakes from sleep,
    ends what a lost worker left ``running`` without waiting an interval first. A
    pass that is running when ``stop`` is set finishes; cancel the call to stop it
    sooner. A failed pass is logged and retried at the next interval, and the loop
    returns once the application stops admitting work. Run it beside a worker's
    entrypoint, as ``cayu worker`` does.
    """

    if not isinstance(stop, asyncio.Event):
        raise TypeError("stop must be an asyncio.Event.")
    if isinstance(interval_seconds, bool) or not isinstance(interval_seconds, int | float):
        raise TypeError("interval_seconds must be a number.")
    if not isfinite(interval_seconds) or interval_seconds <= 0:
        raise ValueError("interval_seconds must be finite and positive.")
    if type(inactive_for_seconds) is not int:
        raise TypeError("inactive_for_seconds must be an integer.")
    if inactive_for_seconds < 0:
        raise ValueError("inactive_for_seconds must not be negative.")
    while not stop.is_set():
        try:
            results = await app.reconcile_orphaned_sessions(
                inactive_for_seconds=inactive_for_seconds
            )
        except ApplicationAdmissionsSealed:
            return
        except Exception:
            logger.exception("Orphaned-session reconciliation failed; it retries next interval.")
        else:
            if results:
                logger.info(
                    "Reconciled %d orphaned running session(s); %d could not be reconciled.",
                    sum(IncompleteSessionRecoveryAction.FAILED not in r.actions for r in results),
                    sum(IncompleteSessionRecoveryAction.FAILED in r.actions for r in results),
                )
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
