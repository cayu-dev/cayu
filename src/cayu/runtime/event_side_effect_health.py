"""Read-only operational projections for durable event fan-out.

No event payload or execution authority is part of this contract.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt

PERSISTED_EVENT_SIDE_EFFECT_MAX_ATTEMPTS = 3
Status = Literal["pending", "leased", "failed", "delivered", "dead_lettered"]


class PersistedEventSideEffectHealth(BaseModel):
    model_config = ConfigDict(extra="forbid")
    observed_at: datetime
    pending: int = 0
    leased_live: int = 0
    leased_expired: int = 0
    failed_retryable: int = 0
    failed_deferred: int = 0
    dead_lettered: int = 0
    delivered: int = 0
    claimable_total: int = 0
    outstanding_total: int = 0
    repeatedly_failing: int = 0
    final_attempt_boundary: int = 0
    max_outstanding_attempts: int = 0
    max_automatic_attempts: int = PERSISTED_EVENT_SIDE_EFFECT_MAX_ATTEMPTS
    oldest_claimable_at: datetime | None = None
    oldest_claimable_age_seconds: float | None = None
    oldest_pending_at: datetime | None = None
    oldest_pending_age_seconds: float | None = None
    oldest_failed_at: datetime | None = None
    oldest_failed_age_seconds: float | None = None
    oldest_dead_letter_at: datetime | None = None
    oldest_dead_letter_age_seconds: float | None = None
    earliest_live_lease_expires_at: datetime | None = None


class PersistedEventSideEffectQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    statuses: set[Status] | None = None
    claimable_only: bool = False
    outstanding_only: bool = True
    limit: StrictInt = Field(default=100, ge=1, le=1000)
    cursor: str | None = Field(default=None, max_length=16384)


class PersistedEventSideEffectInspection(BaseModel):
    session_id: str
    event_id: str
    event_sequence: int
    status: Status
    claimable: bool
    attempts: int
    lease_expires_at: datetime | None
    next_attempt_at: datetime | None
    updated_at: datetime
    last_error: str | None


class PersistedEventSideEffectPage(BaseModel):
    observed_at: datetime
    deliveries: list[PersistedEventSideEffectInspection]
    next_cursor: str | None = None


def safe_error(value: object) -> str | None:
    # Exception text is arbitrary application data. Without the application's
    # sink-boundary redactor it cannot safely be distinguished from a secret.
    return (
        None
        if value is None
        else "Side effect failed; inspect access-controlled application diagnostics."
    )


def cursor_key(query: PersistedEventSideEffectQuery) -> tuple[str, str] | None:
    if query.cursor is None:
        return None
    try:
        data = json.loads(base64.b64decode(query.cursor, altchars=b"-_", validate=True))
        if (
            not isinstance(data, list)
            or len(data) != 4
            or data[0] != 1
            or data[1] != filter_key(query)
            or not all(type(x) is str for x in data[2:])
        ):
            raise ValueError
        return data[2], data[3]
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ValueError("Invalid event side-effect cursor or mismatched filters.") from exc


def filter_key(query: PersistedEventSideEffectQuery) -> list[Any]:
    return [
        None if query.statuses is None else sorted(query.statuses),
        query.claimable_only,
        query.outstanding_only,
    ]


def claimable(row: Any, now: datetime) -> bool:
    return (
        row.status == "pending"
        or (row.status == "failed" and (row.next_attempt_at is None or row.next_attempt_at <= now))
        or (
            row.status == "leased"
            and row.lease_expires_at is not None
            and row.lease_expires_at <= now
        )
    )


def page(
    rows: list[Any], query: PersistedEventSideEffectQuery, now: datetime
) -> PersistedEventSideEffectPage:
    next_cursor = None
    if len(rows) > query.limit:
        last = rows[query.limit - 1]
        next_cursor = base64.urlsafe_b64encode(
            json.dumps(
                [1, filter_key(query), last.session_id, last.event_id],
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode()
        ).decode()
    return PersistedEventSideEffectPage(
        observed_at=now,
        next_cursor=next_cursor,
        deliveries=[
            PersistedEventSideEffectInspection(
                **row.model_dump(
                    include={
                        "session_id",
                        "event_id",
                        "event_sequence",
                        "status",
                        "attempts",
                        "lease_expires_at",
                        "next_attempt_at",
                        "updated_at",
                    }
                ),
                claimable=claimable(row, now),
                last_error=safe_error(row.last_error),
            )
            for row in rows[: query.limit]
        ],
    )


def finish_health(values: dict[str, Any], now: datetime) -> PersistedEventSideEffectHealth:
    for key, value in list(values.items()):
        if value is None and not key.endswith("_at"):
            values[key] = 0
    for name in ("claimable", "pending", "failed", "dead_letter"):
        key = f"oldest_{name}_at"
        value = values.get(key)
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        if value is not None:
            value = value.astimezone(UTC)
            values[key] = value
        values[f"oldest_{name}_age_seconds"] = (
            None if value is None else max(0.0, (now - value).total_seconds())
        )
    return PersistedEventSideEffectHealth(observed_at=now, **values)


# SQL and in-memory implementations share the same public classification.
# A one-row CTE binds one instant for the whole statement.
CLAIMABLE_SQL = "(status = 'pending' OR (status = 'failed' AND (next_attempt_at IS NULL OR next_attempt_at <= observed_at)) OR (status = 'leased' AND lease_expires_at <= observed_at))"
CONDITIONS = {
    "pending": "status = 'pending'",
    "leased_live": "status = 'leased' AND lease_expires_at > observed_at",
    "leased_expired": "status = 'leased' AND lease_expires_at <= observed_at",
    "failed_retryable": "status = 'failed'",
    "failed_deferred": "status = 'failed' AND next_attempt_at > observed_at",
    "dead_lettered": "status = 'dead_lettered'",
    "delivered": "status = 'delivered'",
    "claimable_total": CLAIMABLE_SQL,
    "outstanding_total": "status <> 'delivered'",
    "repeatedly_failing": "status = 'failed' AND attempts > 1",
    "final_attempt_boundary": (
        f"(status IN ('pending', 'failed') OR (status = 'leased' AND lease_expires_at <= observed_at)) AND attempts >= {PERSISTED_EVENT_SIDE_EFFECT_MAX_ATTEMPTS - 1}"
        f" OR (status = 'leased' AND lease_expires_at > observed_at AND attempts >= {PERSISTED_EVENT_SIDE_EFFECT_MAX_ATTEMPTS})"
    ),
}
AGGREGATES = {
    name: f"SUM(CASE WHEN {condition} THEN 1 ELSE 0 END)" for name, condition in CONDITIONS.items()
}
AGGREGATES.update(
    {
        "max_outstanding_attempts": "MAX(CASE WHEN status <> 'delivered' THEN attempts ELSE 0 END)",
        "oldest_claimable_at": f"MIN(CASE WHEN {CLAIMABLE_SQL} THEN CASE WHEN status = 'leased' THEN lease_expires_at WHEN status = 'failed' AND next_attempt_at > updated_at THEN next_attempt_at ELSE updated_at END END)",
        "oldest_pending_at": "MIN(CASE WHEN status = 'pending' THEN updated_at END)",
        "oldest_failed_at": "MIN(CASE WHEN status = 'failed' THEN updated_at END)",
        "oldest_dead_letter_at": "MIN(CASE WHEN status = 'dead_lettered' THEN updated_at END)",
        "earliest_live_lease_expires_at": "MIN(CASE WHEN status = 'leased' AND lease_expires_at > observed_at THEN lease_expires_at END)",
    }
)


def health_sql(placeholder: str) -> str:
    return (
        f"WITH observation AS (SELECT {placeholder} AS observed_at) SELECT "
        + ", ".join(f"{expression} AS {name}" for name, expression in AGGREGATES.items())
        + " FROM cayu_persisted_event_side_effects CROSS JOIN observation"
    )


def page_sql(
    query: PersistedEventSideEffectQuery, now: Any, placeholder: str
) -> tuple[str, list[Any]]:
    key = cursor_key(query)
    # Python and SQLite use code-point/binary order. PostgreSQL must not inherit
    # a deployment locale for the same opaque keyset cursor contract.
    ordered_keys = (
        'session_id COLLATE "C", event_id COLLATE "C"'
        if placeholder == "%s"
        else "session_id, event_id"
    )
    params: list[Any] = [now]
    clauses = []
    if query.outstanding_only:
        clauses.append("status <> 'delivered'")
    if query.claimable_only:
        clauses.append(CLAIMABLE_SQL)
    if query.statuses is not None:
        clauses.append(
            "status IN (" + ", ".join(placeholder for _ in query.statuses) + ")"
            if query.statuses
            else "1 = 0"
        )
        params.extend(sorted(query.statuses))
    if key is not None:
        clauses.append(f"({ordered_keys}) > ({placeholder}, {placeholder})")
        params.extend(key)
    params.append(query.limit + 1)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    return (
        f"WITH observation AS (SELECT {placeholder} AS observed_at) SELECT "
        "session_id, event_id, event_sequence, status, attempts, claim_id, "
        "lease_expires_at, next_attempt_at, last_error, updated_at "
        "FROM cayu_persisted_event_side_effects CROSS JOIN observation"
        + where
        + f" ORDER BY {ordered_keys} LIMIT {placeholder}",
        params,
    )
