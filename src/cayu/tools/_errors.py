"""Shared error contract for framework-native tools.

Built-in tools validate model-supplied arguments eagerly and raise
``ValueError`` (or pydantic ``ValidationError``) on bad input. Without a
shared boundary those exceptions escape into the generic framework
exception path, which loses the structured ``{"error": "invalid_arguments"}``
contract the knowledge tools already expose. The decorator below converts
argument-validation failures explicitly marked inside a tool's ``run`` into
structured ``is_error`` tool results. Operational failures raised after that
phase retain their original exception type and runtime meaning.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any, TypeVar

from pydantic import ValidationError

from cayu.core.tools import ToolContext, ToolResult

_ToolT = TypeVar("_ToolT")

ToolRunMethod = Callable[[_ToolT, ToolContext, dict[str, Any]], Awaitable[ToolResult]]


class _InvalidToolArguments(Exception):
    def __init__(self, error: ValueError) -> None:
        super().__init__(str(error))
        self.error = error


def invalid_tool_arguments_result(exc: Exception) -> ToolResult:
    """Build the shared structured result for model-supplied bad arguments."""
    return ToolResult(
        content=str(exc),
        structured={"error": "invalid_arguments"},
        is_error=True,
    )


@contextmanager
def tool_argument_validation() -> Iterator[None]:
    """Mark model-controlled argument validation without catching later failures."""

    try:
        yield
    except (ValidationError, ValueError) as exc:
        raise _InvalidToolArguments(exc) from exc


def structured_invalid_arguments(run: ToolRunMethod[_ToolT]) -> ToolRunMethod[_ToolT]:
    """Convert explicitly marked argument failures into structured results.

    Apply to a tool's ``run`` method and wrap only its model-input parsing in
    :func:`tool_argument_validation`. Unmarked ``ValueError``, ``TypeError``,
    and other exceptions propagate as operational or host failures.
    """

    @wraps(run)
    async def wrapper(self: _ToolT, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        try:
            return await run(self, ctx, args)
        except _InvalidToolArguments as exc:
            return invalid_tool_arguments_result(exc.error)

    return wrapper
