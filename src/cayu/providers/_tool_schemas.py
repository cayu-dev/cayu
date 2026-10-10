"""Provider-facing serialization of tool input schemas."""

from __future__ import annotations

from typing import Any

_NUMBER_BOUND_KEYWORDS = frozenset(
    {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}
)
# Keywords whose value is one subschema, a list of subschemas, or a map of
# name to subschema. Every other keyword holds literal data (`default`,
# `const`, `enum`, `examples`, ...) and is copied unchanged.
_SUBSCHEMA_KEYWORDS = frozenset(
    {
        "additionalItems",
        "additionalProperties",
        "contains",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
_SUBSCHEMA_LIST_KEYWORDS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_SUBSCHEMA_MAP_KEYWORDS = frozenset(
    {"$defs", "definitions", "dependentSchemas", "patternProperties", "properties"}
)
# Integers in this range convert to a float without changing their value.
_MAX_EXACT_FLOAT_INTEGER = 2**53


def float_number_bounds(schema: Any) -> Any:
    """Return a copy of ``schema`` whose ``number`` bounds are written as floats.

    Durable copies store an integral float such as ``100.0`` as ``100``. OpenAI
    renders an integer bound on a ``number`` field as ``100`` on some upstream
    replicas and ``100.0`` on others, so identical requests alternate between
    two token counts and the prompt cache never reaches past the tools. Float
    bounds render the same way every time. ``integer`` fields, integers a float
    can't represent exactly, and literal values are copied unchanged.
    """

    if type(schema) is not dict:
        return schema
    copied: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _SUBSCHEMA_KEYWORDS:
            copied[key] = (
                [float_number_bounds(item) for item in value]
                if type(value) is list
                else float_number_bounds(value)
            )
        elif key in _SUBSCHEMA_LIST_KEYWORDS and type(value) is list:
            copied[key] = [float_number_bounds(item) for item in value]
        elif key in _SUBSCHEMA_MAP_KEYWORDS and type(value) is dict:
            copied[key] = {name: float_number_bounds(item) for name, item in value.items()}
        else:
            copied[key] = value
    if _is_number_schema(copied):
        for keyword in _NUMBER_BOUND_KEYWORDS & copied.keys():
            value = copied[keyword]
            if type(value) is int and abs(value) <= _MAX_EXACT_FLOAT_INTEGER:
                copied[keyword] = float(value)
    return copied


def _is_number_schema(schema: dict[str, Any]) -> bool:
    declared = schema.get("type")
    if declared == "number":
        return True
    return type(declared) is list and "number" in declared and "integer" not in declared


__all__ = ["float_number_bounds"]
