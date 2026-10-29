"""Bounded calendar windows shared by event usage and session-start projections."""

from __future__ import annotations

from bisect import bisect_right
from datetime import UTC, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_TIME_BUCKETS = 512
BucketUnit = Literal["hour", "day", "week"]


class TimeSeriesSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    unit: BucketUnit
    timezone: str = Field(default="UTC", min_length=1, max_length=128)

    @field_validator("timezone")
    @classmethod
    def valid_zone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ValueError, ZoneInfoNotFoundError):
            raise ValueError("timezone must name an available IANA time zone.") from None
        return value


class TimeBucket(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start_at: datetime
    end_at: datetime

    @field_validator("start_at", "end_at")
    @classmethod
    def aware_instant(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Bucket boundaries must be timezone-aware.")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def positive_window(self) -> TimeBucket:
        if self.start_at >= self.end_at:
            raise ValueError("Buckets must have a positive duration.")
        return self


def time_buckets(start: datetime, end: datetime, spec: TimeSeriesSpec) -> tuple[TimeBucket, ...]:
    """Clip aligned local calendar buckets to an aware, half-open input window.

    Hours advance on the UTC timeline, retaining both hours of a fall-back fold.
    Days and seven-day weeks start at local midnight, anchored to the first day.
    A nonexistent midnight advances to the first corresponding real instant;
    a repeated midnight uses its first occurrence. Skipped days have no bucket.
    """
    window = TimeBucket(start_at=start, end_at=end)
    if window.end_at - window.start_at > timedelta(days=366):
        raise ValueError("Series windows cannot exceed 366 days.")
    spec = TimeSeriesSpec.model_validate(spec.model_dump())
    try:
        local = window.start_at.astimezone(ZoneInfo(spec.timezone))
        anchor = local.replace(minute=0, second=0, microsecond=0)
        if spec.unit != "hour":
            anchor = anchor.replace(hour=0, fold=0)
        anchor_utc = anchor.astimezone(UTC)
    except OverflowError:
        raise ValueError("Window cannot be represented in the requested time zone.") from None
    at = window.start_at
    buckets: list[TimeBucket] = []
    step = 1
    while at < window.end_at:
        try:
            boundary = (
                anchor_utc + timedelta(hours=step)
                if spec.unit == "hour"
                else (anchor + timedelta(days=step * (7 if spec.unit == "week" else 1))).astimezone(
                    UTC
                )
            )
        except OverflowError:
            boundary = window.end_at
        step += 1
        if boundary <= at:
            continue
        if len(buckets) == MAX_TIME_BUCKETS:
            raise ValueError(f"Series cannot contain more than {MAX_TIME_BUCKETS} buckets.")
        end_at = min(boundary, window.end_at)
        buckets.append(TimeBucket(start_at=at, end_at=end_at))
        at = end_at
    return tuple(buckets)


def bucket_index(starts: tuple[datetime, ...], instant: datetime) -> int:
    """Find the bucket for an instant already known to be inside the full window."""
    return bisect_right(starts, instant) - 1
