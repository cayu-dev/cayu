"""Filters on a task's top-level input and metadata values.

`GET /api/tasks` (and `client.js`'s `watchTasks`) take them as repeated `input=key=value`
and `metadata=key=value`; `cayu task list` takes them as `--input`, `--metadata` and
`--run` and matches a local store's tasks with the same rule.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

# At most this many `input` and `metadata` filters on one task list read.
VALUE_FILTERS_MAX = 8


@dataclass(frozen=True, slots=True)
class TaskValueFilters:
    """Repeated `key=value` filters on a task's top-level input and metadata values."""

    input: tuple[tuple[str, str], ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    @classmethod
    def parse(cls, input: Iterable[str] | None, metadata: Iterable[str] | None) -> TaskValueFilters:
        """Filters from `key=value` texts; a malformed one, or too many, raise ``ValueError``."""

        return cls(_parse_pairs(input, "input"), _parse_pairs(metadata, "metadata"))

    @classmethod
    def of(
        cls, input: Mapping[str, str] | None = None, metadata: Mapping[str, str] | None = None
    ) -> TaskValueFilters:
        """Filters from key-value mappings, checked as ``parse`` checks them."""

        return cls.parse(_texts(input, "input"), _texts(metadata, "metadata"))

    def __bool__(self) -> bool:
        return bool(self.input or self.metadata)

    def binding(self) -> dict[str, Any]:
        return {"input": self.input, "metadata": self.metadata}

    def matches(self, task: Any, redact: Callable[[Any, str], Any]) -> bool:
        """Whether each named value matches it as the task detail shows it.

        The whole input or metadata is redacted first, as the detail does, since
        redaction can rename keys too.
        """
        for source, pairs, field_name in (
            (task.input, self.input, "task.input"),
            (task.metadata, self.metadata, "task.metadata"),
        ):
            if not pairs:
                continue
            projected = redact(source, field_name)
            if not isinstance(projected, dict):
                return False
            for key, expected in pairs:
                if key not in projected or not value_matches(projected[key], expected):
                    return False
        return True


def _texts(values: Mapping[str, str] | None, name: str) -> list[str]:
    texts: list[str] = []
    for key, value in (values or {}).items():
        # A filter travels as key=value and is split at the first "=", so such a key
        # would be read as a shorter one and select other tasks.
        if "=" in key:
            raise ValueError(f'A task {name} filter key must not contain "=": {key}')
        texts.append(f"{key}={value}")
    return texts


def _parse_pairs(values: Iterable[str] | None, name: str) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for raw in values or ():
        key, separator, value = raw.partition("=")
        if not separator or not key or key != key.strip():
            raise ValueError(f"Each {name} filter is key=value, such as {name}=key=value.")
        pairs.append((key, value))
    if len(pairs) > VALUE_FILTERS_MAX:
        raise ValueError(f"At most {VALUE_FILTERS_MAX} {name} filters are allowed.")
    return tuple(pairs)


def value_matches(value: Any, expected: str) -> bool:
    """Strings compare as text, booleans as true/false, numbers by value (`2` is `2.0`)."""
    if type(value) is str:
        return value == expected
    if type(value) is bool:
        return expected == ("true" if value else "false")
    if type(value) in (int, float):
        try:
            if type(value) is int and expected.lstrip("-").isdigit():
                return value == int(expected)
            return value == float(expected)
        except (ValueError, OverflowError):
            return False
    return False
