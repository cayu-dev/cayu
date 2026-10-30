"""JSON-pointer selection and comparison for structured-output field assertions."""

from __future__ import annotations

import re
from decimal import Decimal
from enum import StrEnum
from operator import ge, gt, le, lt
from typing import Any, Literal, TypeAlias, cast

from cayu._validation import (
    MAX_DURABLE_JSON_NESTING,
    copy_bounded_durable_json_value,
    unescape_json_pointer_segment,
)
from cayu.evals._bounded_pattern import compile_bounded_pattern
from cayu.evals.json_subset import EVAL_TOOL_JSON_MAX_DEPTH, _json_number
from cayu.vaults import REDACTED_SECRET

StructuredOutputFieldOperator: TypeAlias = Literal[
    "equals",
    "in",
    "exists",
    "matches",
    "gt",
    "gte",
    "lt",
    "lte",
]

EVAL_STRUCTURED_OUTPUT_POINTER_MAX_CHARS = 1_024
EVAL_STRUCTURED_OUTPUT_FIELD_VALUE_MAX_BYTES = 4_096
EVAL_STRUCTURED_OUTPUT_FIELD_VALUE_MAX_NODES = 256
# Published once per trial; a larger observed value is compared but withheld.
EVAL_STRUCTURED_OUTPUT_FIELD_OBSERVED_MAX_BYTES = 1_024
# Bounds one pattern search to at most this many characters times the program size.
EVAL_STRUCTURED_OUTPUT_FIELD_MATCH_MAX_CHARS = 4_096
# Matches the projected workflow structured-output limit.
EVAL_STRUCTURED_OUTPUT_EVIDENCE_MAX_BYTES = 256 << 10

_NUMERIC_COMPARISONS = {"gt": gt, "gte": ge, "lt": lt, "lte": le}
_INVALID_POINTER_ESCAPE = re.compile(r"~(?![01])")
_ARRAY_INDEX = re.compile(r"0|[1-9][0-9]*", re.ASCII)


class StructuredOutputFieldOutcome(StrEnum):
    """Exact result of comparing one resolved structured-output field value."""

    MATCHED = "matched"
    MISMATCHED = "mismatched"
    REDACTED = "redacted"
    LIMIT_EXCEEDED = "limit_exceeded"


def json_pointer_tokens(pointer: str, field_name: str) -> tuple[str, ...]:
    """Validate one RFC 6901 pointer and return its decoded reference tokens."""

    if pointer == "":
        return ()
    if not pointer.startswith("/"):
        raise ValueError(f"`{field_name}` must be empty or start with '/'.")
    if _INVALID_POINTER_ESCAPE.search(pointer) is not None:
        raise ValueError(f"`{field_name}` may escape only '~' as '~0' and '/' as '~1'.")
    return tuple(unescape_json_pointer_segment(token) for token in pointer[1:].split("/"))


def copy_structured_output_field_value(value: object, field_name: str) -> Any:
    """Copy one bounded expected value of a structured-output field comparison."""

    return copy_bounded_durable_json_value(
        value,
        field_name,
        max_bytes=EVAL_STRUCTURED_OUTPUT_FIELD_VALUE_MAX_BYTES,
        max_nodes=EVAL_STRUCTURED_OUTPUT_FIELD_VALUE_MAX_NODES,
        max_nesting=EVAL_TOOL_JSON_MAX_DEPTH,
    )


def validate_structured_output_field_comparison(
    operator: StructuredOutputFieldOperator,
    value: object,
) -> None:
    """Require the expected-value shape that ``operator`` compares."""

    if operator == "exists":
        if value is not None:
            raise ValueError("An `exists` comparison takes no value.")
    elif operator == "in":
        if type(value) is not list or not value:
            raise ValueError("An `in` comparison requires a non-empty array of values.")
    elif operator == "matches":
        if type(value) is not str:
            raise ValueError("A `matches` comparison requires a pattern string.")
        compile_bounded_pattern(value)
    elif operator in _NUMERIC_COMPARISONS and not _json_number(value):
        raise ValueError(f"A `{operator}` comparison requires a JSON number.")


def copy_structured_output_evidence(value: object, field_name: str) -> Any:
    """Copy one structured output within the bounded evidence limit."""

    return copy_bounded_durable_json_value(
        value,
        field_name,
        max_bytes=EVAL_STRUCTURED_OUTPUT_EVIDENCE_MAX_BYTES,
        max_nodes=EVAL_STRUCTURED_OUTPUT_EVIDENCE_MAX_BYTES,
        max_nesting=MAX_DURABLE_JSON_NESTING,
    )


def copy_observed_field_value(value: object, field_name: str) -> Any:
    """Copy one observed field value within its per-trial publication limit."""

    return copy_bounded_durable_json_value(
        value,
        field_name,
        max_bytes=EVAL_STRUCTURED_OUTPUT_FIELD_OBSERVED_MAX_BYTES,
        max_nodes=EVAL_STRUCTURED_OUTPUT_FIELD_OBSERVED_MAX_BYTES,
        max_nesting=EVAL_TOOL_JSON_MAX_DEPTH,
    )


def _redacted(value: str) -> bool:
    # Redaction rewrites secret substrings, so any marker makes the original unknown.
    return REDACTED_SECRET in value


def resolve_json_pointer(
    document: object,
    tokens: tuple[str, ...],
) -> tuple[Literal["resolved", "unresolved", "redacted"], object]:
    """Select one value; redaction preserves JSON types, so only keys can hide a path."""

    current = document
    for token in tokens:
        if type(current) is dict:
            selected = cast("dict[str, object]", current)
            if token in selected:
                if _redacted(token):
                    return "redacted", None
                current = selected[token]
                continue
            if any(_redacted(key) for key in selected):
                return "redacted", None
            return "unresolved", None
        if type(current) is list and _ARRAY_INDEX.fullmatch(token) and int(token) < len(current):
            current = current[int(token)]
            continue
        return "unresolved", None
    return "resolved", current


def _compare_exact(expected: object, observed: object) -> StructuredOutputFieldOutcome:
    """Compare JSON exactly, with integer and float values compared as decimals."""

    pending: list[tuple[object, object]] = [(expected, observed)]
    redacted = False
    while pending:
        expected_value, observed_value = pending.pop()
        if type(observed_value) is str and _redacted(observed_value):
            if type(expected_value) is not str:
                return StructuredOutputFieldOutcome.MISMATCHED
            redacted = True
            continue
        if type(expected_value) is dict:
            if type(observed_value) is not dict:
                return StructuredOutputFieldOutcome.MISMATCHED
            expected_object = cast("dict[str, object]", expected_value)
            observed_object = cast("dict[str, object]", observed_value)
            if any(_redacted(key) for key in observed_object):
                redacted = True
                continue
            if expected_object.keys() != observed_object.keys():
                return StructuredOutputFieldOutcome.MISMATCHED
            pending.extend((child, observed_object[key]) for key, child in expected_object.items())
            continue
        if type(expected_value) is list:
            if type(observed_value) is not list or len(expected_value) != len(observed_value):
                return StructuredOutputFieldOutcome.MISMATCHED
            pending.extend(zip(expected_value, observed_value, strict=True))
            continue
        if _json_number(expected_value) and _json_number(observed_value):
            if Decimal(str(expected_value)) != Decimal(str(observed_value)):
                return StructuredOutputFieldOutcome.MISMATCHED
            continue
        if type(expected_value) is not type(observed_value) or expected_value != observed_value:
            return StructuredOutputFieldOutcome.MISMATCHED
    return (
        StructuredOutputFieldOutcome.REDACTED if redacted else StructuredOutputFieldOutcome.MATCHED
    )


def compare_structured_output_field(
    operator: StructuredOutputFieldOperator,
    expected: Any,
    observed: object,
) -> StructuredOutputFieldOutcome:
    """Compare one resolved value; a type the operator cannot compare is a mismatch."""

    if operator == "exists":
        return StructuredOutputFieldOutcome.MATCHED
    if operator == "equals":
        return _compare_exact(expected, observed)
    if operator == "in":
        outcomes = {_compare_exact(choice, observed) for choice in expected}
        if StructuredOutputFieldOutcome.MATCHED in outcomes:
            return StructuredOutputFieldOutcome.MATCHED
        if StructuredOutputFieldOutcome.REDACTED in outcomes:
            return StructuredOutputFieldOutcome.REDACTED
        return StructuredOutputFieldOutcome.MISMATCHED
    if operator == "matches":
        if type(observed) is not str:
            return StructuredOutputFieldOutcome.MISMATCHED
        if _redacted(observed):
            return StructuredOutputFieldOutcome.REDACTED
        if len(observed) > EVAL_STRUCTURED_OUTPUT_FIELD_MATCH_MAX_CHARS:
            return StructuredOutputFieldOutcome.LIMIT_EXCEEDED
        matched = compile_bounded_pattern(expected).search(observed)
        return (
            StructuredOutputFieldOutcome.MATCHED
            if matched
            else StructuredOutputFieldOutcome.MISMATCHED
        )
    if not _json_number(observed):
        return StructuredOutputFieldOutcome.MISMATCHED
    passed = _NUMERIC_COMPARISONS[operator](Decimal(str(observed)), Decimal(str(expected)))
    return (
        StructuredOutputFieldOutcome.MATCHED if passed else StructuredOutputFieldOutcome.MISMATCHED
    )
