"""Read bounded cost observations within the native usage-rollup snapshot."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from cayu.budgets.reported import (
    MAX_REPORTED_COST_RECORDS,
    ReportedCostPage,
    reported_cost_observation,
)


def reported_cost_statement(where: str, dialect: Literal["sqlite", "postgres"]) -> str:
    paths = {
        "request_id": ("id",),
        "provider_name": ("provider_name",),
        "model": ("model",),
        "cost": ("usage", "cost"),
        "currency": ("usage", "cost_currency"),
        "status": ("usage", "cost_status"),
    }
    fields = []
    for name, path in paths.items():
        if dialect == "sqlite":
            expr = f"json_extract(event.payload_json, '$.{'.'.join(path)}')"
            typed = f"json_type(event.payload_json, '$.{'.'.join(path)}') = 'text'"
        else:
            expr = f"event.payload #>> '{{{','.join(path)}}}'"
            typed = f"jsonb_typeof(event.payload #> '{{{','.join(path)}}}') = 'string'"
        invalid = f"WHEN {expr} IS NOT NULL THEN ''" if name == "cost" else ""
        fields.append(
            f"CASE WHEN {typed} AND length({expr}) <= 256 THEN {expr} {invalid} END AS {name}"
        )
    marker = "?" if dialect == "sqlite" else "%s"
    present = (
        "json_type(event.payload_json, '$.usage.cost_status') IS NOT NULL"
        if dialect == "sqlite"
        else "event.payload #> '{usage,cost_status}' IS NOT NULL"
    )
    return f"""
        WITH matched AS (SELECT id FROM cayu_sessions {where})
        SELECT event.session_id, event.event_id, event.timestamp, {", ".join(fields)}
        FROM cayu_events AS event JOIN matched ON matched.id = event.session_id
        WHERE event.event_type = 'model.completed' AND {present}
          AND event.timestamp >= {marker} AND event.timestamp < {marker}
        ORDER BY event.timestamp DESC, event.session_id DESC, event.event_id DESC
        LIMIT {MAX_REPORTED_COST_RECORDS + 1}
    """


def reported_cost_page(rows) -> ReportedCostPage:
    return ReportedCostPage(
        records=tuple(
            reported_cost_observation(
                session_id=row[0],
                event_id=row[1],
                timestamp=(datetime.fromisoformat(row[2]) if isinstance(row[2], str) else row[2]),
                values=dict(
                    zip(
                        ("request_id", "provider_name", "model", "cost", "currency", "status"),
                        row[3:],
                        strict=True,
                    )
                ),
            )
            for row in rows[:MAX_REPORTED_COST_RECORDS]
        ),
        truncated=len(rows) > MAX_REPORTED_COST_RECORDS,
    )
