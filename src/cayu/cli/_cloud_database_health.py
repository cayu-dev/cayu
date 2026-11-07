"""`cayu cloud service database --application NAME`: where an Agent's database time goes.

The command reads Cloud's Agent database health (`GET /v1/applications/{id}/service/database`)
with the `cayu cloud login` or a selected context's API key; anyone who may see the Agent may
read it. It tells apart what an Agent can't see from its own session events: the round trip to
the instance itself, PgBouncer's query times and its
waits for a server connection, the Agent role's connections, and the shared instance's CPU and
CPU credits.

- Round trips: measured by Cloud as it answers, from its API in the Agent's network straight to
  the instance (connecting, then ten ``SELECT 1``).
- Connections: the Agent role's connections to its database now, by state.
- Pooler: what PgBouncer measured for the Agent's database over the window (`--window`, 15
  minutes by default, at most 6 hours): average query, transaction and wait times, the longest
  wait, and the most waiting clients and server connections in any minute.
- Instance: the shared instance's CPU, CPU credit balance and connections over the window.

A part Cloud can't read is shown as unavailable with its reason. Nothing is changed, and the
answer names no host or credential: the command keeps only the fields it knows.

The command prints a table by default; `--json` prints the JSON envelope the other `cayu cloud`
commands print, with `schema_version` in its result.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any, cast

from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_releases import _when, add_output_argument, render_failure
from cayu.cli._cloud_settings import _cell, _table

SCHEMA_VERSION = 1
DEFAULT_WINDOW_MINUTES = 15
MAX_WINDOW_MINUTES = 360
_INSTANCE_CLASS = re.compile(r"db\.[a-z0-9]+\.[a-z0-9]+")
_REASON = re.compile(r"[a-z_]{1,64}")
_SPAN = re.compile(r"(\d+)([mh])")
_ROUND_TRIPS = ("connect_ms", "samples", "median_ms", "min_ms", "max_ms")
# The paths Cloud measures a round trip on.
_PATHS = frozenset({"direct"})
_CONNECTIONS = ("total", "active", "idle", "idle_in_transaction", "other", "waiting_on_locks")
_POOLER = (
    "samples",
    "tasks",
    "queries",
    "queries_per_second",
    "avg_query_ms",
    "avg_transaction_ms",
    "avg_wait_ms",
    "max_wait_ms",
    "max_client_waiting",
    "max_server_connections",
)
_PARTS = ("round_trips", "connections", "pooler", "instance")
_REASONS = {
    "database_unreachable": "Cloud couldn't reach the database",
    "no_pooler": "PgBouncer doesn't front this install's Agents",
    "no_samples": "no PgBouncer lines for this Agent in the window (no traffic, or an older pooler)",
    "no_instance_metrics": "this install reports no instance metrics",
    "metrics_not_permitted": "Cloud may not read these metrics yet",
    "metrics_unreadable": "Cloud couldn't read these metrics",
}


def window_minutes(value: str) -> int:
    """`--window`: a span of whole minutes or hours (`15m`, `2h`), from 1 minute to 6 hours."""

    span = _SPAN.fullmatch(value)
    if span is None:
        raise argparse.ArgumentTypeError(f"{value!r} is not a span like 15m or 2h")
    minutes = int(span[1]) * (60 if span[2] == "h" else 1)
    if not 1 <= minutes <= MAX_WINDOW_MINUTES:
        raise argparse.ArgumentTypeError("must be from 1m to 6h")
    return minutes


def add_service_database_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--window",
        type=window_minutes,
        default=DEFAULT_WINDOW_MINUTES,
        metavar="SPAN",
        help="How far back the pooler and instance figures reach: 15m, 1h, up to 6h (15m).",
    )
    add_output_argument(parser, render_database_health, failure=render_failure)


def service_database(
    arguments: argparse.Namespace, *, client: CloudApiClient, application_id: str
) -> dict[str, Any]:
    try:
        answer = client.request(
            "GET",
            f"/v1/applications/{application_id}/service/database",
            query={"window_minutes": str(arguments.window)},
        )
    except CloudApiError as exc:
        raise _health_error(exc) from None
    return {"operation": "service.database", "result": health_summary(answer, application_id)}


def health_summary(answer: Mapping[str, Any], application_id: str) -> dict[str, Any]:
    """The fields the command knows, each checked; anything else Cloud sends is left out."""

    window = _mapping(answer.get("window"))
    instance = _mapping(answer.get("instance"))
    unavailable = _mapping(answer.get("unavailable"))
    return {
        "schema_version": SCHEMA_VERSION,
        "application_id": application_id,
        "observed_at": _time(answer.get("observed_at")),
        "window": {"start": _time(window.get("start")), "end": _time(window.get("end"))},
        "round_trips": _round_trips(answer.get("round_trips")),
        "connections": _fields(answer.get("connections"), _CONNECTIONS),
        "pooler": _fields(answer.get("pooler"), _POOLER),
        "instance": None
        if not instance
        else {
            "shared": instance.get("shared") is True,
            "instance_class": _instance_class(instance.get("instance_class")),
            "cpu_percent": _fields(instance.get("cpu_percent"), ("average", "max")),
            "cpu_credit_balance": _fields(instance.get("cpu_credit_balance"), ("latest", "min")),
            "connections": _fields(instance.get("connections"), ("max",)),
        },
        "unavailable": {
            part: reason
            for part in _PARTS
            if isinstance(reason := unavailable.get(part), str) and _REASON.fullmatch(reason)
        },
    }


def render_database_health(result: Mapping[str, Any]) -> str:
    shown = result["result"]
    window = shown["window"]
    unavailable = shown["unavailable"]
    lines = [
        f"Agent database: {_cell(shown['application_id'])}",
        f"  Observed: {_when(shown['observed_at'])}",
        f"  Window: {_when(window['start'])} to {_when(window['end'])}",
        "",
    ]
    trips = shown["round_trips"]
    if trips is None:
        lines.append(_missing("Round trips", unavailable.get("round_trips")))
    else:
        lines += [
            f"Round trips from Cloud to the instance ({_cell(trips['path'])}):",
            f"  Connect: {_ms(trips['connect_ms'])}",
            f"  SELECT 1 x {_cell(trips['samples'])}: median {_ms(trips['median_ms'])}, "
            f"min {_ms(trips['min_ms'])}, max {_ms(trips['max_ms'])}",
        ]
    connections = shown["connections"]
    if connections is None:
        lines.append(_missing("Connections", unavailable.get("connections")))
    else:
        lines += [
            "",
            "Connections of the Agent's role now:",
            *(
                "  " + line
                for line in _table(
                    ("TOTAL", "ACTIVE", "IDLE", "IDLE IN TX", "OTHER", "WAITING ON LOCKS"),
                    [tuple(_cell(connections[name]) for name in _CONNECTIONS)],
                )
            ),
        ]
    pooler = shown["pooler"]
    lines.append("")
    if pooler is None:
        lines.append(_missing("PgBouncer", unavailable.get("pooler")))
    else:
        lines += [
            f"PgBouncer for this Agent ({_cell(pooler['samples'])} samples from "
            f"{_cell(pooler['tasks'])} tasks):",
            f"  Queries: {_cell(pooler['queries'])} ({_cell(pooler['queries_per_second'])}/s), "
            f"average {_ms(pooler['avg_query_ms'])}",
            f"  Transactions: average {_ms(pooler['avg_transaction_ms'])}",
            f"  Wait for a server connection: average {_ms(pooler['avg_wait_ms'])}, "
            f"longest {_ms(pooler['max_wait_ms'])}",
            f"  Most in a minute: {_cell(pooler['max_client_waiting'])} clients waiting, "
            f"{_cell(pooler['max_server_connections'])} server connections",
        ]
    instance = shown["instance"]
    lines.append("")
    if instance is None:
        lines.append(_missing("Instance", unavailable.get("instance")))
    else:
        cpu = instance["cpu_percent"]
        credits = instance["cpu_credit_balance"]
        connections_max = instance["connections"]
        name = _cell(instance["instance_class"]) if instance["instance_class"] else "instance"
        lines += [
            f"Shared {name} (every Agent of the install):",
            "  CPU: "
            + (
                "-"
                if cpu is None
                else f"average {_cell(cpu['average'])}%, max {_cell(cpu['max'])}%"
            ),
            "  CPU credits: "
            + (
                "- (not a burstable class)"
                if credits is None
                else f"now {_cell(credits['latest'])}, lowest {_cell(credits['min'])}"
            ),
            "  Connections: "
            + ("-" if connections_max is None else f"at most {_cell(connections_max['max'])}"),
        ]
    return "\n".join(lines)


def _missing(part: str, reason: object) -> str:
    text = _REASONS.get(reason, _cell(reason)) if isinstance(reason, str) else "not reported"
    return f"{part}: unavailable ({text})."


def _ms(value: object) -> str:
    return "-" if value is None else f"{_cell(value)} ms"


def _health_error(error: CloudApiError) -> CloudApiError:
    """Say what a refused read means; Cloud's own code and message win where it sends one."""

    if error.code == "agent_database_not_found":
        message = "The Agent has no database yet; deploy it first."
    elif error.code == "agent_database_health_unavailable":
        message = "This Cayu Cloud gives its Agents no database, so there is nothing to show."
    elif error.status_code == 404:
        message = (
            "Cayu Cloud has no such Agent, or this Cayu Cloud doesn't report an Agent's "
            "database health yet; it needs an update."
        )
    elif error.status_code == 403:
        message = (
            "This login can't read the Agent: it needs to be one of the Agent's people, a "
            "developer or an admin, or an API key with the read scope."
        )
    else:
        return error
    return CloudApiError(
        error.category,
        message,
        status_code=error.status_code,
        detail=error.detail,
        code=error.code,
        request=error.request,
    )


def _fields(value: object, names: tuple[str, ...]) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    fields = _mapping(value)
    return {name: _scalar(fields.get(name)) for name in names}


def _round_trips(value: object) -> dict[str, Any] | None:
    fields = _fields(value, _ROUND_TRIPS)
    if fields is None:
        return None
    path = _mapping(value).get("path")
    return {"path": path if isinstance(path, str) and path in _PATHS else None, **fields}


def _scalar(value: object) -> int | float | None:
    """A metric: a number, never text, so nothing else Cloud sends can reach the output."""

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def _time(value: object) -> str | None:
    if not isinstance(value, str) or len(value) > 40:
        return None
    try:
        return datetime.fromisoformat(value).isoformat()
    except ValueError:
        return None


def _instance_class(value: object) -> str | None:
    return value if isinstance(value, str) and _INSTANCE_CLASS.fullmatch(value) else None


def _mapping(value: object) -> Mapping[str, Any]:
    return cast("Mapping[str, Any]", value) if isinstance(value, Mapping) else {}
