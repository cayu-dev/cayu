"""Name the query, path and header parameters a control-plane request got wrong.

Bounded control-plane routes answer a validation failure without reflecting the rejected
input. Parameter names and their rules come from the route's own declaration, so naming
them reflects nothing the caller sent, and tells a caller of `GET /api/knowledge?limit=500`
which parameter to fix instead of only "Invalid control-plane request." Body failures stay
unnamed: a body's field locations can carry the caller's own keys.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi.exceptions import RequestValidationError

_PARAMETER_LOCATIONS = {"query": "query parameter", "path": "path parameter", "header": "header"}
_MAX_NAMED_PARAMETERS = 5
# Pydantic error types that carry a bound in their context: (context key, rule).
_BOUNDS = {
    "less_than_equal": ("le", "must be at most {}"),
    "greater_than_equal": ("ge", "must be at least {}"),
    "less_than": ("lt", "must be less than {}"),
    "greater_than": ("gt", "must be greater than {}"),
    "string_too_long": ("max_length", "must be at most {} characters"),
    "string_too_short": ("min_length", "must be at least {} characters"),
    "too_long": ("max_length", "takes at most {} values"),
    "too_short": ("min_length", "takes at least {} values"),
}


def invalid_parameters_detail(error: RequestValidationError) -> str | None:
    """``"query parameter limit must be at most 100"`` and so on, or ``None``.

    ``None`` unless every failure is a declared query, path or header parameter.
    """

    named: list[str] = []
    for item in error.errors():
        location = item.get("loc")
        if (
            not isinstance(location, tuple | list)
            or len(location) < 2
            or location[0] not in _PARAMETER_LOCATIONS
            or not isinstance(location[1], str)
        ):
            return None
        description = f"{_PARAMETER_LOCATIONS[location[0]]} {location[1]} {_rule(item)}"
        if description not in named:
            named.append(description)
    if not named:
        return None
    if len(named) > _MAX_NAMED_PARAMETERS:
        named = [*named[:_MAX_NAMED_PARAMETERS], f"and {len(named) - _MAX_NAMED_PARAMETERS} more"]
    return "; ".join(named)


def _rule(item: Mapping[str, Any]) -> str:
    kind = item.get("type")
    context = item.get("ctx")
    context = context if isinstance(context, Mapping) else {}
    if kind == "missing":
        return "is required"
    if kind in _BOUNDS:
        key, text = _BOUNDS[kind]
        value = context.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return text.format(value)
        return "is out of range"
    if kind in {"int_parsing", "int_type", "int_from_float"}:
        return "must be an integer"
    if kind in {"float_parsing", "float_type", "decimal_parsing"}:
        return "must be a number"
    if kind in {"bool_parsing", "bool_type"}:
        return "must be true or false"
    if kind in {"enum", "literal_error"} and isinstance(context.get("expected"), str):
        return f"must be {context['expected']}"
    return "is invalid"
