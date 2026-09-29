"""Read live request-cost diagnostics from a running Cayu server."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import sys
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from cayu.cli._output import add_output_options, output_destination
from cayu.runtime.request_costs import (
    DEFAULT_REQUEST_COST_WINDOW_SECONDS,
    MAX_REQUEST_COST_WINDOW_SECONDS,
    RequestCostSummary,
)

CLI_SCHEMA_VERSION = "1"
DEFAULT_SERVER_URL = "http://127.0.0.1:8000"
DEFAULT_AUTHORIZATION_ENV = "CAYU_API_AUTHORIZATION"
EXIT_BUDGET_EXCEEDED = 1
EXIT_UNAVAILABLE = 2
_REQUEST_TIMEOUT_SECONDS = 10.0
_MAX_RESPONSE_BYTES = 1024 * 1024
_DURATION_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)([smh]?)$")
_DURATION_UNITS = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0}


class DiagnosticsError(RuntimeError):
    """The request summary could not be read."""


def add_diagnostics_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "diagnostics",
        help="Read live diagnostics from a running Cayu server.",
        description="Read live diagnostics from a running Cayu server.",
    )
    commands = parser.add_subparsers(dest="diagnostics_command", required=True)
    requests_parser = commands.add_parser(
        "requests",
        help="Summarize recent request rate, latency, and CPU by route.",
        description=(
            "Summarize the server's in-memory request-timing buffer: requests per "
            "minute, wall time, and CPU seconds per minute by route template, the "
            "top routes by CPU, and the estimated share of --vcpu CPUs. With "
            "--budget-idle-cpu the command exits 1 when that share exceeds the "
            "budget; it exits 2 when the summary cannot be read. Authorization is "
            f"read from {DEFAULT_AUTHORIZATION_ENV} by default and is never printed."
        ),
    )
    requests_parser.add_argument(
        "--server-url",
        default=DEFAULT_SERVER_URL,
        help=(
            "Cayu server root URL, including any mount prefix but excluding /api "
            f"(default: {DEFAULT_SERVER_URL}; use http://127.0.0.1:8000/cayu for "
            "an app that calls mount_cayu at /cayu)."
        ),
    )
    requests_parser.add_argument(
        "--since",
        type=duration_seconds,
        default=float(DEFAULT_REQUEST_COST_WINDOW_SECONDS),
        metavar="DURATION",
        help="Window to summarize, such as 90s, 5m, or 1h (default: 5m).",
    )
    requests_parser.add_argument(
        "--vcpu",
        type=_positive_float,
        default=1.0,
        help="CPUs available to the web process, used for the share (default: 1).",
    )
    requests_parser.add_argument(
        "--budget-idle-cpu",
        type=_fraction,
        metavar="FRACTION",
        help=("Exit 1 when request CPU exceeds this fraction of --vcpu, for example 0.01 for 1%%."),
    )
    requests_parser.add_argument(
        "--top",
        type=_positive_int,
        default=10,
        help="Number of routes to show in table output (default: 10).",
    )
    requests_parser.add_argument(
        "--authorization-env",
        default=DEFAULT_AUTHORIZATION_ENV,
        metavar="NAME",
        help="Environment variable containing the complete Authorization header value.",
    )
    add_output_options(requests_parser, formats=("json", "table"), default="table")


def run_diagnostics(args: argparse.Namespace) -> int:
    try:
        with output_destination(args.output):
            return _run_requests(args)
    except OSError as exc:
        print(f"error: could not write output: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE


def _run_requests(args: argparse.Namespace) -> int:
    try:
        endpoint = request_cost_endpoint(args.server_url)
        authorization = read_authorization(args.authorization_env)
        summary = fetch_request_cost_summary(
            endpoint,
            authorization=authorization,
            since_seconds=args.since,
            vcpu=args.vcpu,
        )
    except (DiagnosticsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    if not summary.enabled:
        print(
            "error: request timing is off on this server. It is on by default under "
            "`cayu serve` and for local development; otherwise pass "
            "request_timing=True to mount_cayu() or a RequestTimingConfig to ServerConfig.",
            file=sys.stderr,
        )
        return EXIT_UNAVAILABLE
    budget = args.budget_idle_cpu
    exceeded = budget is not None and summary.vcpu_share > budget
    if args.output_format == "json":
        print(
            json.dumps(
                {
                    "schema_version": CLI_SCHEMA_VERSION,
                    "summary": summary.model_dump(mode="json"),
                    "budget": None
                    if budget is None
                    else {
                        "idle_cpu": budget,
                        "vcpu": summary.vcpu,
                        "vcpu_share": summary.vcpu_share,
                        "exceeded": exceeded,
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(render_request_cost_table(summary, budget=budget, top=args.top), end="")
    return EXIT_BUDGET_EXCEEDED if exceeded else 0


def duration_seconds(value: str) -> float:
    match = _DURATION_PATTERN.fullmatch(value.strip())
    if match is None:
        raise argparse.ArgumentTypeError("duration must look like 90s, 5m, or 1h.")
    seconds = float(match.group(1)) * _DURATION_UNITS[match.group(2)]
    if not 0 < seconds <= MAX_REQUEST_COST_WINDOW_SECONDS:
        raise argparse.ArgumentTypeError("duration must be greater than 0 and at most 24h.")
    return seconds


def request_cost_endpoint(server_url: str) -> str:
    parsed = urlsplit(server_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("A canonical HTTP(S) Cayu server URL is required.")
    if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
        raise ValueError("Cayu server URL must use HTTPS outside loopback.")
    path = parsed.path.rstrip("/") + "/api/diagnostics/requests"
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def read_authorization(env_name: str) -> str | None:
    authorization = os.environ.get(env_name)
    if authorization is not None and (
        not authorization.strip()
        or authorization != authorization.strip()
        or "\r" in authorization
        or "\n" in authorization
    ):
        raise ValueError(f"{env_name} contains an invalid Authorization value.")
    return authorization


def fetch_request_cost_summary(
    endpoint: str,
    *,
    authorization: str | None,
    since_seconds: float,
    vcpu: float,
    timeout_seconds: float = _REQUEST_TIMEOUT_SECONDS,
) -> RequestCostSummary:
    import httpx

    headers = {"Accept": "application/json"}
    if authorization is not None:
        headers["Authorization"] = authorization
    try:
        with (
            httpx.Client(follow_redirects=False, timeout=timeout_seconds) as client,
            client.stream(
                "GET",
                endpoint,
                params={"since_seconds": since_seconds, "vcpu": vcpu},
                headers=headers,
            ) as response,
        ):
            if response.status_code in {401, 403}:
                raise DiagnosticsError(
                    f"Cayu server returned HTTP {response.status_code}; set the operator "
                    "Authorization header in the --authorization-env variable."
                )
            if response.status_code == 404:
                raise DiagnosticsError(
                    "Cayu server returned HTTP 404; check --server-url (include the "
                    "mount prefix, exclude /api) and that the server runs this Cayu version."
                )
            if response.status_code != 200:
                raise DiagnosticsError(f"Cayu server returned HTTP {response.status_code}.")
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > _MAX_RESPONSE_BYTES:
                    raise DiagnosticsError("Cayu server returned an oversized summary.")
    except httpx.RequestError:
        raise DiagnosticsError(f"Cayu server is unavailable at {endpoint}.") from None
    try:
        return RequestCostSummary.model_validate_json(bytes(body))
    except ValueError:
        raise DiagnosticsError("Cayu server returned an invalid request summary.") from None


def render_request_cost_table(
    summary: RequestCostSummary,
    *,
    budget: float | None,
    top: int,
) -> str:
    lines = [
        (
            f"Requests in the last {_duration(summary.requested_window_seconds)} "
            f"(covered {_duration(summary.window_seconds)}"
            f"{', older records evicted' if summary.truncated else ''}): "
            f"{summary.requests} requests, {summary.requests_per_minute:.1f}/min"
        ),
        (
            f"Request CPU: {summary.cpu_seconds:.3f} s, "
            f"{summary.cpu_seconds_per_minute:.3f} s/min, "
            f"{summary.vcpu_share:.2%} of {summary.vcpu:g} vCPU"
        ),
    ]
    if budget is not None:
        verdict = "EXCEEDED" if summary.vcpu_share > budget else "ok"
        lines.append(f"Budget: {budget:.2%} of {summary.vcpu:g} vCPU: {verdict}")
    if not summary.routes:
        lines.append("No requests were recorded in this window.")
        return "\n".join(lines) + "\n"
    lines.append("")
    lines.append("Top routes by CPU:")
    rows = [("METHOD", "ROUTE", "REQ/MIN", "P50 MS", "P95 MS", "CPU S/MIN", "VCPU %")]
    for route in summary.routes[:top]:
        rows.append(
            (
                route.method,
                route.route,
                f"{route.requests_per_minute:.1f}",
                _optional_ms(route.wall_ms_p50),
                _optional_ms(route.wall_ms_p95),
                f"{route.cpu_seconds_per_minute:.3f}",
                f"{route.vcpu_share * 100:.2f}",
            )
        )
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    for row in rows:
        cells = [
            cell.ljust(widths[index]) if index < 2 else cell.rjust(widths[index])
            for index, cell in enumerate(row)
        ]
        lines.append("  ".join(cells).rstrip())
    hidden = summary.route_count - min(top, len(summary.routes))
    if hidden > 0:
        lines.append(f"({hidden} more routes; use --top or --json)")
    return "\n".join(lines) + "\n"


def _optional_ms(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"


def _duration(seconds: float) -> str:
    whole = round(seconds)
    if whole >= 60:
        minutes, remainder = divmod(whole, 60)
        return f"{minutes}m{remainder:02d}s" if remainder else f"{minutes}m"
    return f"{seconds:.0f}s"


def _is_loopback_host(hostname: str | None) -> bool:
    if hostname is None:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number.") from None
    if not 0 < number <= 1024:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1024.")
    return number


def _fraction(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number.") from None
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1.")
    return number


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer.") from None
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1.")
    return number
