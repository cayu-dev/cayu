from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta


def normalize_utc_datetime(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
    return value.astimezone(UTC)


def utc_clock(clock: Callable[[], datetime] | None) -> Callable[[], datetime]:
    if clock is None:
        return lambda: datetime.now(UTC)
    if not callable(clock):
        raise TypeError("clock must be callable.")

    def checked_clock() -> datetime:
        return normalize_utc_datetime(clock(), "clock()")

    return checked_clock


def utc_duration_cutoff(now: datetime, duration_seconds: int) -> datetime | None:
    """Return ``now - duration``, or ``None`` when no timestamp can match."""

    normalized_now = normalize_utc_datetime(now, "now")
    if type(duration_seconds) is not int or duration_seconds < 0:
        raise ValueError("duration_seconds must be a non-negative integer.")
    try:
        return normalized_now - timedelta(seconds=duration_seconds)
    except OverflowError:
        # ``datetime.min`` is itself a valid store timestamp.  Returning it as
        # a saturated cutoff would therefore expire a record stamped at that
        # exact instant even though the requested duration has not elapsed.
        return None
