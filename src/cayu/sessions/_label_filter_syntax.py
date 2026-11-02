"""Text syntax for exact label filters and label selectors.

HTTP query parameters and CLI options share these parsers, so a filter written
as text means the same thing on every surface. Errors are ``ValueError``s; the
caller decides how to report them.
"""

from __future__ import annotations

from collections.abc import Iterable

from cayu._validation import copy_label_map
from cayu.sessions.queries import LabelSelectorOperator, LabelSelectorRequirement


def parse_label_filters(values: Iterable[str] | None) -> dict[str, str]:
    """Parse repeated ``key=value`` exact label filters."""

    if values is None:
        return {}
    labels: dict[str, str] = {}
    for raw in values:
        if type(raw) is not str or "=" not in raw:
            raise ValueError("Session label filters must use `key=value`.")
        key, value = raw.split("=", 1)
        parsed_key, parsed_value = next(iter(copy_label_map({key: value}, "label").items()))
        if parsed_key in labels:
            raise ValueError(f"Duplicate session label filter: {parsed_key}")
        labels[parsed_key] = parsed_value
    return labels


def parse_label_selectors(values: Iterable[str] | None) -> tuple[LabelSelectorRequirement, ...]:
    """Parse repeated selector strings; each may hold comma-separated requirements."""

    if values is None:
        return ()
    selectors: list[LabelSelectorRequirement] = []
    for raw in values:
        if type(raw) is not str:
            raise ValueError("Label selector must be a string.")
        for expression in _split_label_selector(raw):
            selectors.append(_parse_label_selector_expression(expression))
    return tuple(selectors)


def _split_label_selector(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Invalid label selector.")
        elif char == "," and depth == 0:
            part = value[start:index].strip()
            if not part:
                raise ValueError("Invalid label selector.")
            parts.append(part)
            start = index + 1
    if depth != 0:
        raise ValueError("Invalid label selector.")
    part = value[start:].strip()
    if not part:
        raise ValueError("Invalid label selector.")
    parts.append(part)
    return parts


def _parse_label_selector_expression(expression: str) -> LabelSelectorRequirement:
    if expression.startswith("!"):
        return LabelSelectorRequirement(
            key=expression[1:].strip(), operator=LabelSelectorOperator.NOT_EXISTS
        )
    for separator, operator in (
        (" notin ", LabelSelectorOperator.NOT_IN),
        (" in ", LabelSelectorOperator.IN),
    ):
        if separator in expression:
            key, raw_values = expression.split(separator, 1)
            return LabelSelectorRequirement(
                key=key.strip(),
                operator=operator,
                values=_parse_label_selector_values(raw_values),
            )
    for separator, operator in (
        ("!=", LabelSelectorOperator.NOT_IN),
        ("==", LabelSelectorOperator.IN),
        ("=", LabelSelectorOperator.IN),
    ):
        if separator in expression:
            key, value = expression.split(separator, 1)
            return LabelSelectorRequirement(
                key=key.strip(), operator=operator, values=(value.strip(),)
            )
    return LabelSelectorRequirement(key=expression.strip(), operator=LabelSelectorOperator.EXISTS)


def _parse_label_selector_values(raw_values: str) -> tuple[str, ...]:
    raw_values = raw_values.strip()
    if not raw_values.startswith("(") or not raw_values.endswith(")"):
        raise ValueError("Label selector values must use `(a,b)`.")
    values = tuple(value.strip() for value in raw_values[1:-1].split(","))
    if any(not value for value in values):
        raise ValueError("Label selector values cannot be blank.")
    return values
