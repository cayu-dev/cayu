"""Runtime-owned admission lifetime, separate from retained tool provenance."""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from cayu.tools.base import ToolContext


@dataclass
class _Lifetime:
    context: ToolContext
    active: bool = True
    # Event-loop time at which the run's time limit stops this call, if it does.
    run_time_stops_at: float | None = None


_CURRENT: ContextVar[_Lifetime | None] = ContextVar("tool_invocation_lifetime", default=None)


@contextmanager
def tool_invocation_lifetime(
    context: ToolContext,
    *,
    run_time_stops_at: float | None = None,
) -> Iterator[None]:
    lifetime = _Lifetime(context, run_time_stops_at=run_time_stops_at)
    token = _CURRENT.set(lifetime)
    try:
        yield
    finally:
        # Child tasks inherit this object, not an independent grant. Closing the
        # invocation also closes new admission through their copied contexts.
        lifetime.active = False
        _CURRENT.reset(token)


def has_live_tool_invocation(context: ToolContext) -> bool:
    lifetime = _CURRENT.get()
    return lifetime is not None and lifetime.active and lifetime.context is context


def run_time_left_for_call(context: ToolContext) -> float | None:
    """Seconds before the run's time limit stops this live call, or ``None``.

    ``None`` when the call runs outside its invocation or the run's time
    limit doesn't bound it (no limit, or a shorter tool timeout does).
    """

    lifetime = _CURRENT.get()
    if lifetime is None or not lifetime.active or lifetime.context is not context:
        return None
    if lifetime.run_time_stops_at is None:
        return None
    return lifetime.run_time_stops_at - asyncio.get_running_loop().time()
