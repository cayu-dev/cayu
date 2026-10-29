"""Where a read-only inspection command reads: a local store or a Cayu Cloud Agent.

`cayu task` takes ``--sqlite PATH`` or ``--postgres DSN`` for a store it opens itself
(resolved like `cayu session`: explicit, then ``CAYU_DATABASE_URL``, then the project's
configured or canonical store), or ``--application NAME`` for an Agent on Cayu Cloud,
read through its own Cayu API (`cayu.cli._cloud_agent_api`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path

from cayu.cli.store_targets import SessionStoreTarget, resolve_session_store_target


@dataclass(frozen=True)
class ReadTarget:
    """Exactly one of ``store``, ``application`` and ``server_url`` is set."""

    server_url: str | None = None
    authorization_env: str = "CAYU_API_AUTHORIZATION"
    store: SessionStoreTarget | None = None
    application: str | None = None
    context: Path | None = None
    timeout_seconds: float = 30.0


def add_read_target_options(parser: argparse.ArgumentParser) -> None:
    """Add ``--sqlite``/``--postgres``/``--application`` and the Cloud options."""

    target = parser.add_mutually_exclusive_group()
    target.add_argument("--server-url", help="Read the running Cayu server at this mount URL.")
    parser.add_argument(
        "--authorization-env",
        default="CAYU_API_AUTHORIZATION",
        help="Authorization header environment variable for --server-url.",
    )
    target.add_argument("--sqlite", metavar="PATH", help="Read this SQLite database.")
    target.add_argument("--postgres", metavar="DSN", help="Read this Postgres database.")
    target.add_argument(
        "--application",
        metavar="NAME",
        help=(
            "Read the Cayu Cloud Agent with this slug or name through its own Cayu API, "
            "signed in as `cayu cloud` is."
        ),
    )
    parser.add_argument(
        "--context",
        type=Path,
        metavar="PATH",
        help="With --application: the private Cayu Cloud context JSON (`cayu cloud --context`).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_seconds,
        default=30.0,
        metavar="SECONDS",
        help="With --application or --server-url: how long each request may take (default: 30).",
    )


def resolve_read_target(args: argparse.Namespace) -> ReadTarget:
    """Resolve the parsed options; raises ``ValueError`` for an invalid combination."""

    if args.server_url is not None:
        if args.context is not None:
            raise ValueError("--context applies only with --application.")
        from cayu.cli.diagnostics import request_cost_endpoint

        request_cost_endpoint(args.server_url)
        return ReadTarget(
            server_url=args.server_url,
            authorization_env=args.authorization_env,
            timeout_seconds=args.timeout_seconds,
        )
    if args.application is None:
        if args.context is not None:
            raise ValueError("--context applies only with --application.")
        return ReadTarget(
            store=resolve_session_store_target(sqlite=args.sqlite, postgres=args.postgres)
        )
    application = args.application.strip()
    if not application:
        raise ValueError("--application must name an Agent.")
    return ReadTarget(
        application=application,
        context=args.context,
        timeout_seconds=args.timeout_seconds,
    )


def run_read_command(
    args: argparse.Namespace,
    command: Callable[[argparse.Namespace, ReadTarget], Coroutine[object, object, int]],
    *,
    error_code: str,
    schema_version: str,
) -> int:
    """Resolve the target and run ``command``, rendering any failure the same way.

    JSON output gets ``{"schema_version", "error": {"code", "message", "category"?}}``;
    table output gets ``error: ...`` on stderr. Postgres DSNs never reach either.
    """

    from cayu.cli._cloud_agent_api import CayuApiError
    from cayu.cli._output import output_destination, terminal_safe
    from cayu.cli.session import _safe_error

    try:
        with output_destination(args.output):
            dsn: str | None = None
            try:
                target = resolve_read_target(args)
                dsn = None if target.store is None else target.store.postgres_dsn
                return asyncio.run(command(args, target))
            except Exception as exc:
                # Driver and SQLite failures have backend-specific types.
                message = _safe_error(str(exc), dsn)
                if args.output_format not in {"json", "jsonl"}:
                    print(f"error: {terminal_safe(message)}", file=sys.stderr)
                    return 1
                error: dict[str, str] = {"code": error_code, "message": message}
                if isinstance(exc, CayuApiError):
                    error["category"] = exc.category
                print(
                    json.dumps(
                        {"schema_version": schema_version, "error": error},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                return 1
    except OSError as exc:
        print(f"error: could not write output: {exc}", file=sys.stderr)
        return 1


def _positive_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive number of seconds") from None
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be a positive number of seconds")
    return seconds
