"""SAP OData V2: filters built from typed values, and answers made plain for the model."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from datetime import date, datetime, timedelta, timezone
from typing import Any

#: The most values one `or` chain compares, so a filter stays well within URL limits.
CHUNK = 20
_FIELD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DATE = re.compile(r"/Date\((-?\d{1,15})(?:([+-])(\d{4}))?\)/")
_EPOCH = datetime(1970, 1, 1)
#: The keys SAP adds to an entity that are about the protocol, not the data.
_PROTOCOL = frozenset({"__metadata", "__deferred"})


def literal(text: str) -> str:
    """An OData string literal: quoted, with its quotes doubled."""

    return "'" + text.replace("'", "''") + "'"


def any_of(field: str, values: Iterable[str]) -> str:
    if not _FIELD.fullmatch(field):
        raise ValueError(f"{field!r} isn't an OData field name.")
    return "(" + " or ".join(f"{field} eq {literal(value)}" for value in values) + ")"


def on_or_after(field: str, day: date) -> str:
    return f"{field} ge datetime'{day.isoformat()}T00:00:00'"


def before(field: str, day: date) -> str:
    return f"{field} lt datetime'{day.isoformat()}T00:00:00'"


def chunked(values: Sequence[str]) -> list[Sequence[str]]:
    return [values[start : start + CHUNK] for start in range(0, len(values), CHUNK)]


def plain(value: Any, incomplete: set[str] | None = None, name: str = "") -> Any:
    """SAP's value as the model reads it: protocol keys dropped, an expanded list's
    ``{"results": [...]}`` unwrapped, and ``/Date(...)/`` as an ISO date. Decimals stay the
    strings SAP sends, so no digit is lost. An expanded list SAP paged (its ``__next``) is
    only its first part: its property's name is added to ``incomplete``."""

    if isinstance(value, dict):
        if set(value) <= {"results", "__count", "__next"} and isinstance(
            value.get("results"), list
        ):
            if value.get("__next") and incomplete is not None:
                incomplete.add(name)
            return [plain(item, incomplete, name) for item in value["results"]]
        return {
            key: plain(item, incomplete, key)
            for key, item in value.items()
            if key not in _PROTOCOL and not _deferred(item)
        }
    if isinstance(value, list):
        return [plain(item, incomplete, name) for item in value]
    if isinstance(value, str):
        return iso(value)
    return value


def _deferred(value: Any) -> bool:
    """A navigation property that wasn't expanded: only a link, no data."""

    return isinstance(value, dict) and set(value) == {"__deferred"}


def iso(text: str) -> str:
    """``/Date(ms)/`` as ``YYYY-MM-DD`` (or with its time, when it has one) and
    ``/Date(ms±mmmm)/``, an offset in minutes, as the instant in that offset; any other text
    unchanged."""

    match = _DATE.fullmatch(text)
    if match is None:
        return text
    milliseconds, sign, offset = match.groups()
    try:
        instant = _EPOCH + timedelta(milliseconds=int(milliseconds))
        if sign is None:
            if instant.time() == datetime.min.time():
                return instant.date().isoformat()
            return _timestamp(instant)
        shift = timedelta(minutes=int(offset)) * (-1 if sign == "-" else 1)
        return _timestamp((instant + shift).replace(tzinfo=timezone(shift)))
    except (OverflowError, ValueError):
        return text


def _timestamp(moment: datetime) -> str:
    return moment.isoformat(timespec="milliseconds" if moment.microsecond else "seconds")
