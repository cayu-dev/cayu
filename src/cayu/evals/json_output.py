"""Strict JSON parsing and value comparison for deterministic output assertions."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any


def parse_json_output(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError("JSON output contains a nonfinite number.")

    def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("JSON output contains duplicate object keys.")
            result[key] = value
        return result

    try:
        return json.loads(
            text,
            parse_int=Decimal,
            parse_float=Decimal,
            parse_constant=reject_constant,
            object_pairs_hook=object_from_pairs,
        )
    except (ValueError, ArithmeticError, RecursionError) as error:
        raise ValueError("Output must be one valid, unambiguous JSON value.") from error


def json_outputs_equal(expected: str, actual: str) -> bool:
    pending = [(parse_json_output(expected), parse_json_output(actual))]
    while pending:
        left, right = pending.pop()
        if type(left) is not type(right):
            return False
        if isinstance(left, dict):
            if left.keys() != right.keys():
                return False
            pending.extend((value, right[key]) for key, value in left.items())
        elif isinstance(left, list):
            if len(left) != len(right):
                return False
            pending.extend(zip(left, right, strict=True))
        elif left != right:
            return False
    return True
