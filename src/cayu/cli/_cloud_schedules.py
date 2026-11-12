"""The rules Cayu Cloud applies to a `[[schedules]]` entry, checked before anything is uploaded.

Cloud runs an Agent's schedules with EventBridge Scheduler on ECS and as CronJobs on Kubernetes,
and admits only expressions that keep their exact meaning on both. These are Cloud's own rules
(`cayu_cloud.schedule_expressions`); keep them identical, so `cayu cloud init --check` and the
deploy answer what Cloud would:

- `rate(<n> minute|minutes|hour|hours|day|days)` where the interval divides the clock: a minute
  rate that divides an hour, an hour rate that divides a day, or `rate(1 day)`;
- `cron(<minutes> <hours> <day-of-month> <month> <day-of-week> <year>)` with exactly one of the
  day fields `?`, any year (`*` or `?`), and no `L`, `W`, `#` or stepped day of the week.

A schedule's `timezone` is an IANA zone name (`America/Los_Angeles`) its `cron()` expression is
read in, so it fires at the same local time all year; without one it is UTC. A `rate()`
expression counts an interval, not a time of day, so it takes no zone. Nor may a zoned schedule run
in an hour a daylight-saving change skips or repeats in its zone (1 AM to 2 AM in
America/Los_Angeles): EventBridge runs a repeated time once where a CronJob runs it twice.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from functools import cache
from zoneinfo import ZoneInfo, available_timezones

from cayu.cli._cloud_api import CloudApiError

#: The zone a schedule without `timezone` runs in, and the names Cloud treats as that zone.
DEFAULT_SCHEDULE_TIMEZONE = "UTC"
UTC_NAMES = frozenset({"UTC", "Etc/UTC"})

_RATE = re.compile(r"rate\((\d+) (minute|minutes|hour|hours|day|days)\)")
_CRON = re.compile(r"cron\((.*)\)")
_UNSUPPORTED_CRON = re.compile(r"[LW#]", re.IGNORECASE)
# Month and day names, which may contain L or W without being those tokens.
_NAMES = re.compile(
    r"JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC|SUN|MON|TUE|WED|THU|FRI|SAT", re.IGNORECASE
)
_DOW_NUMBER = re.compile(r"\d+")
_FORMS = (
    "use rate(<n> minutes|hours|days) or "
    "cron(<minutes> <hours> <day-of-month> <month> <day-of-week> <year>)"
)
# The years whose daylight-saving changes a zoned schedule is checked against, as Cloud does.
_CHANGE_YEARS = range(2020, 2046)
# An IANA name's shape, for a machine whose zone database can't be listed.
_ZONE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*(/[A-Za-z0-9][A-Za-z0-9_+-]*){0,2}")
_ZONE_NAME_MAX = 64


def _invalid(message: str) -> CloudApiError:
    return CloudApiError("manifest_invalid", message)


def check_schedule_expression(expression: str) -> None:
    """Refuse ``expression`` unless Cayu Cloud runs it on ECS and Kubernetes alike."""

    value = expression.strip()
    rate = _RATE.fullmatch(value)
    if rate is not None:
        _check_rate(int(rate.group(1)), rate.group(2), expression)
        return
    cron = _CRON.fullmatch(value)
    if cron is None:
        raise _invalid(f"Schedule {expression!r} can't run on Cayu Cloud: {_FORMS}.")
    fields = cron.group(1).split()
    _check_cron(fields, expression)
    if (fields[2] == "?") == (fields[4] == "?"):
        raise _invalid(
            f"Schedule {expression!r} must set exactly one of the day-of-month and day-of-week "
            "fields to ?, as EventBridge requires: for example cron(0 3 * * ? *) or "
            "cron(0 3 ? * MON-FRI *)."
        )


def _check_rate(count: int, unit: str, expression: str) -> None:
    singular = unit.rstrip("s")
    if count < 1 or (count == 1) != (unit == singular):
        raise _invalid(f"Schedule {expression!r} is not a valid rate expression.")
    if (
        (singular == "minute" and 60 % count == 0 and count < 60)
        or (singular == "hour" and 24 % count == 0 and count < 24)
        or (singular == "day" and count == 1)
    ):
        return
    raise _invalid(
        f"Schedule {expression!r} can't keep its exact interval on Kubernetes, which runs "
        "schedules on the clock: use a minute rate that divides an hour, an hour rate that "
        "divides a day, rate(1 day), or a cron() expression."
    )


def _check_cron(fields: list[str], expression: str) -> None:
    if len(fields) != 6:
        raise _invalid(f"Schedule {expression!r} must have six cron fields.")
    year, day_of_week = fields[5], fields[4]
    if year not in {"*", "?"}:
        raise _invalid(f"Schedule {expression!r} limits the year, which Kubernetes can't run.")
    if any(_UNSUPPORTED_CRON.search(_NAMES.sub("", field)) for field in fields):
        raise _invalid(
            f"Schedule {expression!r} uses L, W or #, which Kubernetes schedules don't support."
        )
    if "/" in day_of_week:
        raise _invalid(f"Schedule {expression!r} steps the day of the week; list the days instead.")
    if day_of_week != "?":
        for match in _DOW_NUMBER.finditer(day_of_week):
            if not 1 <= int(match.group(0)) <= 7:
                raise _invalid(f"Schedule {expression!r} has a day of the week outside 1-7.")


@cache
def _known_zones() -> frozenset[str]:
    # `localtime` links to the host's own zone and `Factory` is tzdata's placeholder: neither
    # names a place. Cloud refuses both too.
    return frozenset(available_timezones()) - {"localtime", "Factory"}


@cache
def _changed_hours(timezone: str) -> frozenset[int]:
    """The local hours a daylight-saving (or other offset) change skips or repeats in
    ``timezone``, over `_CHANGE_YEARS`."""

    zone = ZoneInfo(timezone)
    hours: set[int] = set()
    day = date(_CHANGE_YEARS.start, 1, 1)
    end = date(_CHANGE_YEARS.stop, 1, 1)
    while day < end:
        following = day + timedelta(days=1)
        if (
            datetime(day.year, day.month, day.day, tzinfo=zone).utcoffset()
            != datetime(following.year, following.month, following.day, tzinfo=zone).utcoffset()
        ):
            for changed in (day, following):
                for hour in range(24):
                    for minute in (0, 15, 30, 45):
                        local = datetime(
                            changed.year, changed.month, changed.day, hour, minute, tzinfo=zone
                        )
                        repeated = local.utcoffset() != local.replace(fold=1).utcoffset()
                        skipped = local.astimezone(UTC).astimezone(zone).replace(
                            tzinfo=None
                        ) != local.replace(tzinfo=None)
                        if repeated or skipped:
                            hours.add(hour)
        day = following
    return frozenset(hours)


def _cron_hours(field: str) -> frozenset[int] | None:
    """The hours a cron() hours field names, or None for one this doesn't read."""

    hours: set[int] = set()
    for part in field.split(","):
        base, _, step = part.partition("/")
        if base in {"*", "?"}:
            first, last = 0, 23
        elif re.fullmatch(r"\d{1,2}", base):
            first = last = int(base)
            if step:
                last = 23
        elif match := re.fullmatch(r"(\d{1,2})-(\d{1,2})", base):
            first, last = int(match.group(1)), int(match.group(2))
        else:
            return None
        # A step of 0 (or one that isn't a number) names no hours this can read.
        if step and (not step.isdigit() or int(step) == 0):
            return None
        hours.update(range(first, last + 1, int(step) if step else 1))
    return frozenset(hours)


def check_schedule_timezone(timezone: object, *, expression: str, name: str) -> None:
    """Refuse a `timezone` that isn't an IANA zone name, or one on a `rate()` schedule."""

    if type(timezone) is not str or not timezone:
        raise _invalid(
            f'Schedule "{name}" timezone must be an IANA time zone name, such as '
            "America/Los_Angeles."
        )
    known = _known_zones()
    # Without a zone database here (Windows without tzdata), only the name's shape is checked;
    # Cloud still refuses a name it doesn't know.
    unknown = (
        timezone not in known
        if known
        else len(timezone) > _ZONE_NAME_MAX or _ZONE_NAME.fullmatch(timezone) is None
    )
    if unknown:
        raise _invalid(
            f'Schedule "{name}" timezone {timezone!r} is not an IANA time zone name, such as '
            "America/Los_Angeles."
        )
    if timezone not in UTC_NAMES and _CRON.fullmatch(expression.strip()) is None:
        raise _invalid(
            f'Schedule "{name}" sets timezone {timezone!r} on {expression!r}: a rate() '
            "expression counts an interval, not a time of day, so it takes no time zone. "
            "Use a cron() expression, or remove timezone."
        )
    if timezone in UTC_NAMES or not known:
        return
    fields = cron.group(1).split() if (cron := _CRON.fullmatch(expression.strip())) else []
    hours = _cron_hours(fields[1]) if len(fields) == 6 else None
    changed = _changed_hours(timezone)
    clash = changed if hours is None else changed & hours
    if clash:
        shown = ", ".join(f"{hour}:00-{hour}:59" for hour in sorted(clash))
        raise _invalid(
            f'Schedule "{name}" ({expression!r}) can run at {shown} in {timezone}, which a '
            "daylight-saving change skips or repeats there. EventBridge and Kubernetes run "
            "those times differently (a repeated time runs once on one and twice on the "
            "other), so choose hours outside them, or write the schedule in UTC."
        )
