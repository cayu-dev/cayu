"""Runtime-owned admission lifetime, separate from retained tool provenance."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from cayu.tools.base import ToolContext


@dataclass
class _Lifetime:
    context: ToolContext
    active: bool = True


_CURRENT: ContextVar[_Lifetime | None] = ContextVar("tool_invocation_lifetime", default=None)


@contextmanager
def tool_invocation_lifetime(context: ToolContext) -> Iterator[None]:
    lifetime = _Lifetime(context)
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
