"""Re-read after a durable save that ended cancelled.

A save that commits in its own child task can be cancelled after its write
committed, alone or with cleanup failures grouped beside the cancellation. Its
caller would then treat a committed write as never made. Every committing child
task runs its save through :func:`save_reconciling_cancellation`, which
reconciles that outcome the way the save already reconciles an ordinary
failure: it re-reads durable state, or exactly replays the save, before the
caller decides.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from cayu._exception_groups import exception_tree_leaves

_ResultT = TypeVar("_ResultT")


def save_ended_cancelled(error: BaseException) -> bool:
    """Whether a save ended by cancellation, alone or grouped with ordinary failures.

    Its write may have committed. A save that ended with only ordinary failures
    reconciles itself; process control and ``GeneratorExit`` stop reconciliation.
    """

    if isinstance(error, asyncio.CancelledError):
        return True
    if not isinstance(error, BaseExceptionGroup):
        return False
    leaves = exception_tree_leaves(error)
    return any(isinstance(leaf, asyncio.CancelledError) for leaf in leaves) and all(
        isinstance(leaf, Exception | asyncio.CancelledError) for leaf in leaves
    )


async def save_reconciling_cancellation(
    save: Callable[[], Awaitable[_ResultT]],
    *,
    reconcile: Callable[[], Awaitable[_ResultT | None]],
) -> _ResultT:
    """Run a committing save as the body of its own child task.

    When the save ends cancelled, ``reconcile`` re-reads durable state (or
    exactly replays the save) and its committed result is returned, so the caller
    sees the write it made. When ``reconcile`` returns ``None`` (nothing
    committed), fails or is itself cancelled, the original failure propagates.
    Only a dedicated child task may run this: it consumes that child's own
    cancellation, while its caller keeps the caller's.
    """

    try:
        return await save()
    except BaseException as failure:
        if not save_ended_cancelled(failure):
            raise
        committed: _ResultT | None = None
        try:
            committed = await reconcile()
        except BaseException as reconcile_failure:
            # A failed or cancelled re-read decides nothing: the save's own
            # failure, with this one noted, stays the outcome. Process control
            # still propagates.
            if not isinstance(reconcile_failure, Exception) and not save_ended_cancelled(
                reconcile_failure
            ):
                raise
            failure.add_note(
                f"Reconciling the cancelled save also failed: {type(reconcile_failure).__name__}."
            )
        if committed is None:
            raise
        return committed
