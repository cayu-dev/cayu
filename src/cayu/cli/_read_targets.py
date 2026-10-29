"""Where a read-only inspection command reads: a local store or a Cayu Cloud Agent.

`cayu task` takes ``--sqlite PATH`` or ``--postgres DSN`` for a store it opens itself
(resolved like `cayu session`: explicit, then ``CAYU_DATABASE_URL``, then the project's
configured or canonical store), or ``--application NAME`` for an Agent on Cayu Cloud,
read through its own Cayu API (`cayu.cli._cloud_agent_api`).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from cayu.cli.store_targets import SessionStoreTarget, resolve_session_store_target


@dataclass(frozen=True)
class ReadTarget:
    """Exactly one of ``store`` and ``application`` is set."""

    store: SessionStoreTarget | None = None
    application: str | None = None
    context: Path | None = None
    timeout_seconds: float = 30.0


def add_read_target_options(parser: argparse.ArgumentParser) -> None:
    """Add ``--sqlite``/``--postgres``/``--application`` and the Cloud options."""

    target = parser.add_mutually_exclusive_group()
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
        help="With --application: how long each request may take (default: 30).",
    )


def resolve_read_target(args: argparse.Namespace) -> ReadTarget:
    """Resolve the parsed options; raises ``ValueError`` for an invalid combination."""

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


def _positive_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive number of seconds") from None
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be a positive number of seconds")
    return seconds
