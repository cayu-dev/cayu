"""Shared error contract for framework-native tools.

Built-in tools validate model-supplied arguments eagerly and raise
``ValueError`` (or pydantic ``ValidationError``) on bad input. Without a
shared boundary those exceptions escape into the generic framework
exception path, which loses the structured ``{"error": "invalid_arguments"}``
contract the knowledge tools already expose. The decorator below converts
argument-validation failures raised by a tool's ``run`` into structured
``is_error`` tool results so every built-in tool reports bad arguments the
same way.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any, TypeVar

from pydantic import ValidationError

from cayu.core.tools import ToolContext, ToolResult

_ToolT = TypeVar("_ToolT")

ToolRunMethod = Callable[[_ToolT, ToolContext, dict[str, Any]], Awaitable[ToolResult]]


def invalid_tool_arguments_result(exc: Exception) -> ToolResult:
    """Build the shared structured result for model-supplied bad arguments."""
    return ToolResult(
        content=str(exc),
        structured={"error": "invalid_arguments"},
        is_error=True,
    )


def structured_invalid_arguments(run: ToolRunMethod[_ToolT]) -> ToolRunMethod[_ToolT]:
    """Convert argument ``ValueError``/``ValidationError`` into structured results.

    Apply to a tool's ``run`` method. Only ``ValueError`` (and pydantic
    ``ValidationError``) are converted -- they signal model-supplied bad
    arguments in built-in tools. ``TypeError`` and other exceptions still
    propagate because they indicate host misconfiguration, not model error.
    """

    @wraps(run)
    async def wrapper(self: _ToolT, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        try:
            return await run(self, ctx, args)
        except (ValidationError, ValueError) as exc:
            return invalid_tool_arguments_result(exc)

    return wrapper
