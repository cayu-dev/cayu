"""Shared error contract for native Python tools.

Built-in tools validate model-supplied arguments eagerly and raise
``ValueError`` (or pydantic ``ValidationError``) on bad input. Without a
shared boundary those exceptions escape into the generic runtime
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
        structured={
            "error": "invalid_arguments",
            **(exc.details if isinstance(exc, ToolArgumentShapeError) else {}),
        },
        is_error=True,
    )


# Only source-owned schema names may be reflected; syntactically valid keys can
# still be credentials or other sensitive caller content.
_SAFE_FIELD_NAMES = frozenset(
    {
        "operations",
        "type",
        "path",
        "from_path",
        "to_path",
        "expected_revision",
        "content",
        "edits",
        "old_text",
        "new_text",
        "expected_replacements",
        "pattern",
        "limit",
        "offset",
        "max_result_bytes",
    }
)


class ToolArgumentShapeError(ValueError):
    """Bounded, content-free diagnostics for an object schema mismatch."""

    def __init__(
        self,
        args: object,
        *,
        allowed: frozenset[str],
        required: frozenset[str] = frozenset(),
        hint: str = "",
    ) -> None:
        self.details: dict[str, Any] = {}
        if not isinstance(args, dict):
            super().__init__("Tool arguments must be an object.")
            return
        missing = sorted(required - args.keys())
        safe_names = allowed | required | _SAFE_FIELD_NAMES
        unknown = sorted(name for name in safe_names if name in args and name not in allowed)
        omitted = sum(1 for name in args if name not in allowed and name not in safe_names)
        self.details = {"missing_fields": missing, "unknown_fields": unknown}
        if omitted:
            self.details["unknown_fields_omitted"] = omitted
        notes = []
        if missing:
            notes.append("Missing fields: " + ", ".join(missing) + ".")
        if unknown or omitted:
            notes.append(
                "Tool arguments contain unknown fields"
                + (": " + ", ".join(unknown) if unknown else "")
                + "."
            )
        if omitted:
            notes.append("Additional unrecognized field names were withheld.")
        if hint:
            notes.append(hint)
        super().__init__(" ".join(notes))


def reject_unknown_tool_arguments(
    args: object,
    *,
    allowed: frozenset[str],
    required: frozenset[str] = frozenset(),
    hint: str = "",
) -> None:
    """Reject invalid object shapes with bounded, trusted field names only."""
    if not isinstance(args, dict) or args.keys() - allowed or required - args.keys():
        raise ToolArgumentShapeError(args, allowed=allowed, required=required, hint=hint)


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
