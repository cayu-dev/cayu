"""`cayu dev`: local development helpers for a project.

`cayu dev postgres up|url|down` starts, reports and removes a throwaway PostgreSQL server for
the project's tests (`cayu.cli.dev_postgres`). The server is what the generated CI gives the
tests, so `eval "$(cayu dev postgres up --export)"` before `pytest` runs locally what CI runs.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any

from cayu.cli import dev_postgres
from cayu.cli.dev_postgres import TEST_SERVER_VARIABLE, DevPostgres, DevPostgresError

_VARIABLE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def add_dev_parser(subparsers: Any) -> None:
    group = subparsers.add_parser(
        "dev",
        help="Local development helpers, such as a throwaway PostgreSQL for the tests.",
        description="Local development helpers for the project in the working directory.",
    )
    commands = group.add_subparsers(dest="dev_command", required=True)
    postgres = commands.add_parser(
        "postgres",
        help="Start, show or remove a throwaway PostgreSQL server for this project's tests.",
        description=(
            "A throwaway PostgreSQL server in Docker for this project's tests: the image and "
            "digest the generated CI uses, published on 127.0.0.1. Each project (the nearest "
            "folder with a pyproject.toml) gets its own container, and `down` removes only "
            "that one, with its data."
        ),
    )
    actions = postgres.add_subparsers(dest="postgres_command", required=True)
    up = actions.add_parser(
        "up",
        help="Start the server, unless it is running, and print its URL.",
        description=(
            "Start this project's PostgreSQL server unless it is already running, wait until "
            "it accepts connections, and print its URL. "
            f'`eval "$(cayu dev postgres up --export)"` also sets {TEST_SERVER_VARIABLE} for '
            "the tests. Messages go to stderr, so stdout holds only the URL."
        ),
    )
    up.add_argument(
        "--port",
        type=_port,
        help="Host port on 127.0.0.1 (default: a free one Docker picks).",
    )
    up.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for PostgreSQL to accept connections (default: 60).",
    )
    _add_common(up)
    url = actions.add_parser(
        "url",
        help="Print the running server's URL.",
        description=(
            "Print the URL of this project's running PostgreSQL server, for "
            f"{TEST_SERVER_VARIABLE} or CAYU_DATABASE_URL. Exits 3 when none is running."
        ),
    )
    _add_common(url)
    down = actions.add_parser(
        "down",
        help="Remove the server and its data.",
        description=(
            "Remove this project's PostgreSQL container and its data. Containers of other "
            "projects are left alone. Succeeds when none is running."
        ),
    )
    _add_project(down)


def _add_common(parser: argparse.ArgumentParser) -> None:
    _add_project(parser)
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--export",
        nargs="?",
        const=TEST_SERVER_VARIABLE,
        metavar="VARIABLE",
        type=_variable,
        help=(
            "Print a shell `export` line for eval instead of the bare URL. The variable is "
            f"{TEST_SERVER_VARIABLE} unless named, for example CAYU_DATABASE_URL."
        ),
    )
    output.add_argument("--json", action="store_true", help="Print a JSON object.")


def _add_project(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project",
        type=Path,
        help="The project folder (default: the nearest folder with a pyproject.toml).",
    )


def run_dev(args: argparse.Namespace) -> int:
    project = dev_postgres.project_root(args.project)
    try:
        if args.postgres_command == "up":
            return _up(args, project)
        if args.postgres_command == "url":
            return _url(args, project)
        return _down(project)
    except DevPostgresError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _up(args: argparse.Namespace, project: Path) -> int:
    server, started = dev_postgres.start(project, port=args.port, timeout=args.timeout)
    if started:
        print(
            f"Started PostgreSQL for {project} in container {server.container} on "
            f"127.0.0.1:{server.port}. Remove it with `cayu dev postgres down`.",
            file=sys.stderr,
        )
    else:
        note = (
            f" (not on --port {args.port}; run `cayu dev postgres down` first to move it)"
            if args.port is not None and args.port != server.port
            else ""
        )
        print(
            f"PostgreSQL for {project} is already running in container {server.container} on "
            f"127.0.0.1:{server.port}{note}.",
            file=sys.stderr,
        )
    _print_server(args, server, started=started)
    return 0


def _url(args: argparse.Namespace, project: Path) -> int:
    server = dev_postgres.find(project)
    if server is None:
        print(
            f"error: no PostgreSQL is running for {project}; start one with "
            "`cayu dev postgres up`.",
            file=sys.stderr,
        )
        return 3
    _print_server(args, server, started=None)
    return 0


def _down(project: Path) -> int:
    removed = dev_postgres.stop(project)
    if removed is None:
        print(f"No PostgreSQL is running for {project}; nothing to remove.", file=sys.stderr)
    else:
        print(f"Removed {removed} and its data.", file=sys.stderr)
    return 0


def _print_server(args: argparse.Namespace, server: DevPostgres, *, started: bool | None) -> None:
    if args.json:
        record: dict[str, object] = {
            "container": server.container,
            "project": str(server.project),
            "port": server.port,
            "url": server.url,
        }
        if started is not None:
            record["started"] = started
        print(json.dumps(record, indent=2))
    elif args.export:
        print(f"export {args.export}={shlex.quote(server.url)}")
    else:
        print(server.url)


def _port(value: str) -> int:
    try:
        port = int(value)
    except ValueError:
        port = 0
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("must be a port number from 1 to 65535")
    return port


def _variable(value: str) -> str:
    if not _VARIABLE.fullmatch(value):
        raise argparse.ArgumentTypeError("must be an environment variable name")
    return value
