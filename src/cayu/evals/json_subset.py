from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Any, cast

from cayu._validation import (
    DurableValueError,
    copy_bounded_durable_json_value,
    json_utf8_size_within_limit,
)
from cayu.vaults import REDACTED_SECRET

EVAL_TOOL_JSON_MAX_BYTES = 4 * 1024
EVAL_TOOL_JSON_MAX_DEPTH = 12
EVAL_TOOL_JSON_MAX_NODES = 128


class JsonSubsetOutcome(StrEnum):
    """Exact outcome of comparing one bounded expected JSON subset."""

    MATCHED = "matched"
    MISMATCHED = "mismatched"
    REDACTED = "redacted"


def copy_eval_tool_json_object(value: object, field_name: str) -> dict[str, Any]:
    """Copy one portable bounded JSON object used by a tool assertion."""

    copied = copy_bounded_durable_json_value(
        value,
        field_name,
        max_bytes=EVAL_TOOL_JSON_MAX_BYTES,
        max_nodes=EVAL_TOOL_JSON_MAX_NODES,
        max_nesting=EVAL_TOOL_JSON_MAX_DEPTH,
    )
    if type(copied) is not dict:
        raise ValueError(f"`{field_name}` must be a JSON object.")
    if not json_utf8_size_within_limit(
        copied,
        EVAL_TOOL_JSON_MAX_BYTES,
        ensure_ascii=False,
    ):
        raise DurableValueError("json_value_too_large", field_name)
    return copied


def compare_json_subset(expected: object, actual: object) -> JsonSubsetOutcome:
    """Compare JSON with recursive object-subset and exact-array semantics.

    Object keys in ``expected`` may be a subset of ``actual``. Arrays retain
    positional meaning and therefore require the same length. JSON scalars are
    compared by kind and value, except that integer and finite-float values use
    their exact decimal JSON value. A redaction marker only makes a comparison
    unavailable when it occurs on a path selected by the expected subset.
    """

    pending: list[tuple[object, object]] = [(expected, actual)]
    selected_path_was_redacted = False
    while pending:
        expected_value, actual_value = pending.pop()
        if actual_value == REDACTED_SECRET:
            selected_path_was_redacted = True
            continue
        if type(expected_value) is dict:
            if type(actual_value) is not dict:
                return JsonSubsetOutcome.MISMATCHED
            expected_object = cast("dict[str, object]", expected_value)
            actual_object = cast("dict[str, object]", actual_value)
            for key, child in expected_object.items():
                if key not in actual_object:
                    return JsonSubsetOutcome.MISMATCHED
                pending.append((child, actual_object[key]))
            continue
        if type(expected_value) is list:
            if type(actual_value) is not list or len(expected_value) != len(actual_value):
                return JsonSubsetOutcome.MISMATCHED
            pending.extend(zip(expected_value, actual_value, strict=True))
            continue
        if _json_number(expected_value) and _json_number(actual_value):
            if Decimal(str(expected_value)) != Decimal(str(actual_value)):
                return JsonSubsetOutcome.MISMATCHED
            continue
        if type(expected_value) is not type(actual_value) or expected_value != actual_value:
            return JsonSubsetOutcome.MISMATCHED
    return JsonSubsetOutcome.REDACTED if selected_path_was_redacted else JsonSubsetOutcome.MATCHED


def equal_json_values(left: object, right: object) -> bool:
    """Compare two retained JSON values exactly without redaction semantics."""

    pending: list[tuple[object, object]] = [(left, right)]
    while pending:
        left_value, right_value = pending.pop()
        if type(left_value) is dict:
            if type(right_value) is not dict or left_value.keys() != right_value.keys():
                return False
            left_object = cast("dict[str, object]", left_value)
            right_object = cast("dict[str, object]", right_value)
            pending.extend((child, right_object[key]) for key, child in left_object.items())
            continue
        if type(left_value) is list:
            if type(right_value) is not list or len(left_value) != len(right_value):
                return False
            pending.extend(zip(left_value, right_value, strict=True))
            continue
        if _json_number(left_value) and _json_number(right_value):
            if Decimal(str(left_value)) != Decimal(str(right_value)):
                return False
            continue
        if type(left_value) is not type(right_value) or left_value != right_value:
            return False
    return True


def _json_number(value: object) -> bool:
    return type(value) in {int, float}
