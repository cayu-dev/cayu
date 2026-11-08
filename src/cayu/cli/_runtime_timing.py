"""`cayu session runtime-timing`: the runtime timing records a Cayu process holds for a session.

The runtime times each tool round and the gap before each model step, with the database
time inside every phase, and keeps the most recent records in the memory of the process
that ran them. This command reads them from a running Cayu server (``--server-url``) or a
Cayu Cloud Agent (``--application``) through ``GET /api/sessions/{id}/runtime-timing``
(`cayu.server._runtime_timing_routes`). A session store holds none, so ``--sqlite`` and
``--postgres`` are refused with where to look instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from urllib.parse import quote

from cayu.cli._output import add_output_options, output_destination, print_table, terminal_safe
from cayu.cli._read_targets import ReadTarget, add_read_target_options, run_read_command

ERROR_CODE = "RUNTIME_TIMING_READ_FAILED"
MAX_RECORDS = 1024

_NOT_IN_A_STORE = (
    "Runtime timing records are kept only in the memory of the process that ran the "
    "session, so a session store holds none. Read a running Cayu server with --server-url "
    "URL or a Cayu Cloud Agent with --application NAME; in your own process, call "
    "app.inspect_recent_tool_round_timing() and "
    "app.inspect_recent_model_step_preparation_timing(), or log them with "
    "LoggingEventSink(log_runtime_timing=True)."
)


def add_runtime_timing_parser(commands: Any) -> None:
    """Register ``cayu session runtime-timing`` on the ``cayu session`` subcommands."""

    parser = commands.add_parser(
        "runtime-timing",
        help="Show the runtime timing records a Cayu process holds for a session.",
        description=(
            "Show the recent runtime timing records of one session: each tool round and "
            "each model step's preparation, split into phases, with the database "
            "transactions, lock and pool waits, SQL and commit time inside them. The "
            "records live only in the memory of the process that ran the work, in a "
            "bounded buffer shared by its sessions, so this reads a running Cayu server "
            "(--server-url) or Cayu Cloud Agent (--application) and shows what the "
            "answering process holds. A session run by another process, such as a task "
            "worker, has none there. `cayu session show` reports a different breakdown, "
            "computed from the session's stored events."
        ),
    )
    parser.add_argument("session_id")
    add_read_target_options(parser)
    parser.add_argument(
        "--limit",
        type=_record_limit,
        default=50,
        help=f"How many of the newest records to show (1-{MAX_RECORDS}, default 50).",
    )
    add_output_options(parser, formats=("json", "table"))


def run_runtime_timing(args: argparse.Namespace) -> int:
    from cayu.cli.session import CLI_SCHEMA_VERSION

    if args.server_url is None and args.application is None:
        try:
            with output_destination(args.output):
                _print_error(args.output_format, _NOT_IN_A_STORE, schema_version=CLI_SCHEMA_VERSION)
        except OSError as exc:
            print(f"error: could not write output: {exc}", file=sys.stderr)
        return 1
    return run_read_command(
        args, _runtime_timing, error_code=ERROR_CODE, schema_version=CLI_SCHEMA_VERSION
    )


async def _runtime_timing(args: argparse.Namespace, target: ReadTarget) -> int:
    from cayu.cli._cloud_agent_api import CayuApiNotFound
    from cayu.cli.session import CLI_SCHEMA_VERSION
    from cayu.sessions._call_inspection import redact_inspection_value

    api, source = _open_api(target)
    try:
        body = api.get(
            f"/sessions/{quote(args.session_id, safe='')}/runtime-timing",
            query={"limit": str(args.limit)},
            resource="session runtime timing",
        )
    except CayuApiNotFound:
        raise ValueError(f"Session not found: {args.session_id}") from None
    finally:
        api.close()
    if not isinstance(body, dict) or not isinstance(body.get("records"), list):
        raise ValueError("The server's runtime timing answer is not a Cayu one.")
    payload = redact_inspection_value(
        {
            "schema_version": CLI_SCHEMA_VERSION,
            "source": source,
            **body,
            "phase_totals": phase_totals(body["records"]),
        }
    )
    if args.output_format == "json":
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        _print_runtime_timing(payload)
    return 0


def phase_totals(records: list[Any]) -> list[dict[str, Any]]:
    """Each phase summed over the records' own phases (a round's phases include its calls'),
    in the order the phases first appear: how many records had it, its time, and the
    database work inside that time."""

    totals: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        for phase in record.get("phases") or ():
            if not isinstance(phase, dict) or not isinstance(phase.get("name"), str):
                continue
            entry = totals.setdefault(
                phase["name"],
                {
                    "phase": phase["name"],
                    "records": 0,
                    "duration_seconds": 0.0,
                    "store_transaction_count": 0,
                    "store_lock_wait_seconds": 0.0,
                    "store_execution_seconds": 0.0,
                    "store_commit_seconds": 0.0,
                },
            )
            entry["records"] += 1
            for key in (
                "duration_seconds",
                "store_transaction_count",
                "store_lock_wait_seconds",
                "store_execution_seconds",
                "store_commit_seconds",
            ):
                value = phase.get(key)
                if isinstance(value, int | float) and not isinstance(value, bool):
                    entry[key] += value
    return list(totals.values())


def _open_api(target: ReadTarget) -> tuple[Any, dict[str, str]]:
    from cayu.cli._cloud_agent_api import CayuApi, open_cloud_agent_api

    if target.server_url is not None:
        from cayu.cli.diagnostics import read_authorization

        api = CayuApi(
            target.server_url,
            authorization=read_authorization(target.authorization_env),
            timeout_seconds=target.timeout_seconds,
        )
        return api, {"kind": "server", "url": target.server_url}
    if target.application is None:
        raise ValueError(_NOT_IN_A_STORE)
    api = open_cloud_agent_api(
        target.application,
        context=target.context,
        timeout_seconds=target.timeout_seconds,
        wake_timeout_seconds=target.wake_timeout_seconds,
        offers_wake_option=True,
    )
    return api, {"kind": "cayu_cloud", "application": target.application}


def _print_runtime_timing(payload: dict[str, Any]) -> None:
    process = payload.get("process") or {}
    execution = payload.get("session_execution") or {}
    recorder = payload.get("recorder") or {}
    records = payload.get("records") or []
    print(
        terminal_safe(
            f"Runtime timing of session {payload.get('session_id')}, as process "
            f"{process.get('owner_id')} holds it in memory."
        )
    )
    owner = execution.get("owner_id")
    if execution.get("in_this_process") is not True and owner is not None:
        print(
            terminal_safe(
                f"The session is {execution.get('state')} in another process ({owner}); "
                "records of the work that process ran are not here."
            )
        )
    if recorder.get("enabled") is False:
        print("This process doesn't record runtime timing (RuntimeTimingConfig(enabled=False)).")
    print(
        f"The process keeps the newest {recorder.get('recent_capacity')} records of all its "
        f"sessions and holds {recorder.get('recent_records')}; "
        f"{recorder.get('dropped_records')} were dropped before delivery to sinks and "
        f"{recorder.get('failed_deliveries')} deliveries failed."
    )
    if not records:
        print("\nNo records for this session in this process.")
        return
    print()
    print_table(
        ("started_at", "kind", "time", "calls", "store", "slowest phases"),
        [_record_row(record) for record in records if isinstance(record, dict)],
    )
    totals = payload.get("phase_totals") or []
    if totals:
        print("\nPhases over these records")
        print_table(
            ("phase", "records", "time", "transactions", "lock_wait", "sql", "commit"),
            [
                {
                    "phase": entry["phase"],
                    "records": entry["records"],
                    "time": _seconds(entry["duration_seconds"]),
                    "transactions": entry["store_transaction_count"],
                    "lock_wait": _seconds(entry["store_lock_wait_seconds"]),
                    "sql": _seconds(entry["store_execution_seconds"]),
                    "commit": _seconds(entry["store_commit_seconds"]),
                }
                for entry in totals
            ],
        )
        print(
            "lock_wait is local lock and connection-pool admission plus database lock "
            "acquisition, which sql also counts; the columns overlap and aren't a sum."
        )


def _record_row(record: dict[str, Any]) -> dict[str, Any]:
    phases = [phase for phase in record.get("phases") or () if isinstance(phase, dict)]
    slowest = sorted(phases, key=lambda phase: -_number(phase.get("duration_seconds")))[:3]
    transactions = sum(int(_number(phase.get("store_transaction_count"))) for phase in phases)
    # Shown apart, not added up: a database lock's acquisition counts in both the
    # lock wait and the SQL time.
    sql, commit, wait = (
        sum(_number(phase.get(key)) for phase in phases)
        for key in ("store_execution_seconds", "store_commit_seconds", "store_lock_wait_seconds")
    )
    calls = record.get("calls")
    kind = "tool round" if record.get("kind") == "tool_round" else "model step preparation"
    return {
        "started_at": record.get("started_at"),
        "kind": kind + (" (incomplete)" if record.get("incomplete") is True else ""),
        "time": _seconds(_number(record.get("duration_seconds"))),
        "calls": len(calls) if isinstance(calls, list) else None,
        "store": (
            f"{transactions} tx: sql {_seconds(sql)}, commit {_seconds(commit)}, "
            f"lock/pool wait {_seconds(wait)}"
        )
        if transactions
        else None,
        "slowest phases": ", ".join(
            f"{phase.get('name')} {_seconds(_number(phase.get('duration_seconds')))}"
            for phase in slowest
            if _number(phase.get("duration_seconds")) > 0
        )
        or None,
    }


def _seconds(value: float) -> str:
    if value < 1:
        return f"{value * 1000:.1f}ms"
    if value < 60:
        return f"{value:.2f}s"
    minutes, seconds = divmod(int(value), 60)
    return f"{minutes}m{seconds:02d}s"


def _number(value: object) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def _record_limit(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"limit must be between 1 and {MAX_RECORDS}.") from None
    if not 1 <= parsed <= MAX_RECORDS:
        raise argparse.ArgumentTypeError(f"limit must be between 1 and {MAX_RECORDS}.")
    return parsed


def _print_error(output_format: str, message: str, *, schema_version: str) -> None:
    if output_format == "json":
        print(
            json.dumps(
                {
                    "schema_version": schema_version,
                    "error": {"code": ERROR_CODE, "message": message},
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    else:
        print(f"error: {message}", file=sys.stderr)
