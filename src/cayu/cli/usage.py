"""Event-window usage and session-start counts from a running local or Cloud agent."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cayu.cli._output import add_output_options, print_table, terminal_safe
from cayu.cli._read_targets import ReadTarget, _positive_seconds, run_read_command


def add_usage_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "usage",
        help="Inspect event-window usage and session starts.",
        description="Read the same usage and session-start data as the Developer dashboard. Choose rollup for activity or counts for starts.",
    )
    commands = parser.add_subparsers(dest="usage_command", required=True)
    for name, description in (
        (
            "rollup",
            "Sum activity in an event-time window, including older sessions' recent work. Add --bucket hour, day or week for a counter series; use counts to inspect session starts.",
        ),
        (
            "counts",
            "Count session starts and their current failures, excluding eval trials and forks by default. Add --bucket for a start histogram; use rollup for event-window activity.",
        ),
    ):
        command = commands.add_parser(
            name, help=description.split(".", 1)[0] + ".", description=description
        )
        target = command.add_mutually_exclusive_group(required=True)
        target.add_argument("--server-url", help="Running local or deployed Cayu server mount URL.")
        target.add_argument("--application", help="Cayu Cloud Agent slug or name.")
        command.set_defaults(sqlite=None, postgres=None)
        command.add_argument(
            "--authorization-env",
            default="CAYU_API_AUTHORIZATION",
            help="Environment variable containing the server Authorization header.",
        )
        command.add_argument(
            "--context", type=Path, help="Private Cloud context file for --application."
        )
        command.add_argument("--timeout-seconds", type=_positive_seconds, default=30.0)
        command.add_argument(
            "--since", type=_instant, required=True, help="Inclusive start, ISO8601 with timezone."
        )
        command.add_argument(
            "--until", type=_instant, required=True, help="Exclusive end, ISO8601 with timezone."
        )
        command.add_argument(
            "--bucket", choices=("hour", "day", "week"), help="Optional bounded counter series."
        )
        command.add_argument(
            "--timezone", default="UTC", help="IANA zone for calendar buckets (default UTC)."
        )
        for option in ("agent", "environment", "provider", "model"):
            command.add_argument(f"--{option}", help=f"Exact current session {option} filter.")
        if name == "rollup":
            command.add_argument(
                "--price-book",
                type=Path,
                help="Optional price-book JSON for the existing headline cost. Series contain counters only.",
            )
            command.add_argument(
                "--top-sessions",
                type=_top_limit,
                help="Include up to 100 sessions, using the current token order.",
            )
        else:
            command.add_argument("--include-eval", action="store_true")
            command.add_argument("--include-forks", action="store_true")
        add_output_options(command, formats=("json", "table"), default="table")


def _instant(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError
        return result.astimezone(UTC)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an ISO8601 timestamp with timezone") from None


def _top_limit(value: str) -> int:
    try:
        number = int(value)
        if not 1 <= number <= 100:
            raise ValueError
        return number
    except ValueError:
        raise argparse.ArgumentTypeError("must be between 1 and 100") from None


def run_usage(args: argparse.Namespace) -> int:
    # Check before the shared output context opens/truncates its destination.
    if args.output is not None:
        output = Path(args.output)
        for source in (args.context, getattr(args, "price_book", None)):
            if source is not None and (
                output.resolve() == source.resolve()
                or (output.exists() and source.exists() and output.samefile(source))
            ):
                print("error: output must not refer to an input file.", file=sys.stderr)
                return 1
    return run_read_command(
        args, _run_usage, error_code="USAGE_INSPECTION_FAILED", schema_version="1"
    )


async def _run_usage(args: argparse.Namespace, target: ReadTarget) -> int:
    from cayu._time_buckets import TimeSeriesSpec, time_buckets
    from cayu.cli._cloud_agent_api import CayuApi, open_cloud_agent_api
    from cayu.cli.diagnostics import read_authorization

    if args.since >= args.until:
        raise ValueError("--since must be before --until.")
    series = (
        None if args.bucket is None else TimeSeriesSpec(unit=args.bucket, timezone=args.timezone)
    )
    if series is not None:
        time_buckets(args.since, args.until, series)
    if target.server_url is not None:
        api = CayuApi(
            target.server_url,
            authorization=read_authorization(target.authorization_env),
            timeout_seconds=target.timeout_seconds,
        )
    else:
        assert target.application is not None
        api = open_cloud_agent_api(
            target.application, context=target.context, timeout_seconds=target.timeout_seconds
        )
    filters = {
        key: value
        for key, value in (
            ("agent_name", args.agent),
            ("environment_name", args.environment),
            ("provider_name", args.provider),
            ("model", args.model),
        )
        if value is not None
    }
    try:
        surface = "usage_series" if args.usage_command == "rollup" else "session_overview_series"
        if series is not None or (args.usage_command == "counts" and filters):
            capability = api.get("/contract", resource="server capabilities")
            for key in ("capabilities", "surfaces", surface, "read", "enabled"):
                capability = capability.get(key) if isinstance(capability, dict) else None
            if capability is not True:
                raise ValueError(
                    f"The server does not advertise {surface} support for the requested options."
                )
        if args.usage_command == "rollup":
            body: dict[str, Any] = {
                "start_at": args.since.isoformat(),
                "end_at": args.until.isoformat(),
                "session_filter": filters,
            }
            if series is not None:
                body["series"] = series.model_dump()
            if args.top_sessions is not None:
                body["session_group_limit"] = args.top_sessions
            if args.price_book is not None:
                with args.price_book.open("rb") as stream:
                    raw = stream.read(2 * 1024 * 1024 + 1)
                if len(raw) > 2 * 1024 * 1024:
                    raise ValueError("Price book exceeds the 2 MiB request limit.")
                body["pricing"] = json.loads(raw)
            result = api.post("/usage/rollup", body, resource="usage rollup")
        else:
            query = {
                "since": args.since.isoformat(),
                "until": args.until.isoformat(),
                **filters,
                "include_eval": str(args.include_eval).lower(),
                "include_forks": str(args.include_forks).lower(),
            }
            if series is not None:
                query.update(bucket_unit=series.unit, timezone=series.timezone)
            result = api.get(
                "/sessions/overview/counts", query=query, resource="session-start counts"
            )
        if not isinstance(result, dict):
            raise ValueError("The server returned an invalid usage response.")
        if series is not None and not isinstance(result.get("series"), dict):
            raise ValueError("The server did not return the requested usage series.")
        if args.output_format == "json":
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        else:
            _render_table(result, starts=args.usage_command == "counts")
        return 0
    finally:
        api.close()


def _render_table(result: dict[str, Any], *, starts: bool) -> None:
    if starts:
        _print_values(
            ("Metric", "Count"),
            [
                ["Sessions started", result["total_count"]],
                ["Currently failed", result["failed"]],
                ["Running now", result["running_now"]],
            ],
        )
    else:
        totals = result["totals"]
        _print_values(
            ("Metric", "Count"),
            [
                ["Sessions with activity", totals["session_count"]],
                ["Model steps", totals["model_steps"]],
                ["Tool calls", totals["tool_calls"]],
                ["Tokens", totals["usage"]["total_tokens"]],
            ],
        )
        cost = result.get("cost")
        if cost is not None:
            amounts = ", ".join(
                f"{row['total_cost']} {row['currency']}" for row in cost["currencies"]
            )
            print(
                terminal_safe(
                    f"Estimated cost (priced work): {amounts or 'unavailable'}; unpriced steps: {cost['unpriced_model_steps']}; unevaluated steps: {cost['unevaluated_model_steps']}"
                )
            )
    series = result.get("series")
    if series is not None:
        print(
            "Session-start buckets"
            if starts
            else "Event-activity buckets (sessions may occur in several buckets)"
        )
        headers = (
            ("Start (UTC)", "End (UTC)", "Starts", "Failed")
            if starts
            else ("Start (UTC)", "End (UTC)", "Active sessions", "Tokens")
        )
        rows = [
            [
                b["start_at"],
                b["end_at"],
                b["total_count"] if starts else b["totals"]["session_count"],
                b["failed"] if starts else b["totals"]["usage"]["total_tokens"],
            ]
            for b in series["buckets"]
        ]
        _print_values(headers, rows)


def _print_values(headers: tuple[str, ...], rows: list[list[object]]) -> None:
    print_table(headers, [dict(zip(headers, row, strict=True)) for row in rows])
