"""`cayu storage` subcommands: schema status / migrate, and JSON/JSONL export.

Realizes ADR 0001 Phase 2 (explicit migrate + status) and Phase 3 (JSONL
export). Migrations are an explicit, operator-run step — never silent on import
(Decision 6) — so this CLI is the supported way to migrate a production database.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import unquote, urlsplit, urlunsplit

from cayu.cli._output import add_output_options
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import jsonl_export
from cayu.storage import migrations as schema

_SUBCOMMANDS = (
    ("status", "Show the database schema revision and any pending migrations."),
    ("migrate", "Apply pending forward migrations under the backend lock."),
    ("export", "Export sessions (or tasks) as JSON or JSONL for backup/replay."),
)


def add_storage_parser(subparsers: Any) -> None:
    """Register the ``storage`` command group on an argparse subparsers object."""
    storage = subparsers.add_parser(
        "storage",
        help="Inspect, migrate, and export Cayu storage.",
        description=(
            "Inspect, migrate, and export Cayu storage. Schema changes are explicit; "
            "start with `cayu storage status`."
        ),
    )
    inner = storage.add_subparsers(dest="storage_command", required=True)
    for name, help_text in _SUBCOMMANDS:
        sub = inner.add_parser(
            name,
            help=help_text,
            description=f"{help_text} Use `--output FILE` to write data to a destination.",
        )
        target = sub.add_mutually_exclusive_group(required=True)
        target.add_argument("--sqlite", metavar="PATH", help="Path to a SQLite database file.")
        target.add_argument("--postgres", metavar="DSN", help="Postgres connection string.")
        if name == "export":
            sub.add_argument(
                "--tasks", action="store_true", help="Export tasks instead of sessions."
            )
            add_output_options(sub, formats=("json", "jsonl"))
        else:
            add_output_options(sub)


def run_storage(args: argparse.Namespace) -> int:
    """Dispatch a parsed ``storage`` invocation; return a process exit code."""
    try:
        if args.storage_command == "status":
            return _status(args)
        if args.storage_command == "migrate":
            return _migrate(args)
        if args.storage_command == "export":
            return _export(args)
    except (schema.SchemaError, OSError) as exc:
        _render_error(args, str(exc))
        return 1
    return 1


def _redact_dsn(dsn: str) -> str:
    """Strip credentials from a DSN so it is safe to print.

    Handles both URL-style (``postgresql://user:pass@host/db``) and libpq
    key/value (``host=... password=...``) forms; the password never reaches
    stdout/stderr or logs.
    """
    parts = urlsplit(dsn)
    if parts.scheme and (parts.username or parts.password or parts.query):
        netloc = parts.hostname or ""
        if parts.port:
            netloc += f":{parts.port}"
        return urlunsplit(parts._replace(netloc=netloc, query=""))
    # libpq key=value form: redact password values including quoted strings and
    # backslash-escaped spaces. A simple \S+ regex leaks quoted passwords.
    return _redact_libpq_passwords(dsn)


def _redact_libpq_passwords(dsn: str) -> str:
    pattern = re.compile(r"(?i)(?<![A-Za-z0-9_])password\s*=")
    redacted: list[str] = []
    cursor = 0
    while True:
        match = pattern.search(dsn, cursor)
        if match is None:
            redacted.append(dsn[cursor:])
            return "".join(redacted)

        redacted.append(dsn[cursor : match.end()])
        value_start = match.end()
        value_end = _libpq_value_end(dsn, value_start)
        redacted.append("***")
        cursor = value_end


def _libpq_value_end(value: str, start: int) -> int:
    while start < len(value) and value[start].isspace():
        start += 1
    if start >= len(value):
        return start
    quote = value[start] if value[start] in {"'", '"'} else None
    index = start + 1 if quote is not None else start
    escaped = False
    while index < len(value):
        char = value[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif quote is not None:
            if char == quote:
                index += 1
                break
        elif char.isspace():
            break
        index += 1
    return index


def _libpq_password_values(dsn: str) -> list[str]:
    pattern = re.compile(r"(?i)(?<![A-Za-z0-9_])password\s*=")
    values: list[str] = []
    cursor = 0
    while True:
        match = pattern.search(dsn, cursor)
        if match is None:
            return values
        value_start = match.end()
        value_end = _libpq_value_end(dsn, value_start)
        raw_value = dsn[value_start:value_end].strip()
        values.append(_unquote_libpq_value(raw_value))
        cursor = value_end


def _unquote_libpq_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    output: list[str] = []
    escaped = False
    for char in value:
        if escaped:
            output.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        else:
            output.append(char)
    if escaped:
        output.append("\\")
    return "".join(output)


def _sanitize(message: str, dsn: str) -> str:
    """Scrub a DSN (and any embedded password) out of an error message."""
    out = message.replace(dsn, _redact_dsn(dsn))
    password = urlsplit(dsn).password
    if password:
        out = out.replace(password, "***")
        decoded_password = unquote(password)
        if decoded_password != password:
            out = out.replace(decoded_password, "***")
    for libpq_password in _libpq_password_values(dsn):
        if libpq_password:
            out = out.replace(libpq_password, "***")
    return out


def _status_payload(backend: str, target: str, state: schema.SchemaState) -> dict[str, Any]:
    pending = schema.pending(state.revision)
    return {
        "schema_version": "1",
        "backend": backend,
        "target": target,
        "database": {
            "revision": state.revision,
            "compatible_from": state.compatible_from,
            "initialized": state.revision != schema.UNINITIALIZED,
        },
        "application": {
            "latest_revision": schema.LATEST_REVISION,
            "min_supported_revision": schema.MIN_SUPPORTED_REVISION,
        },
        "pending_migrations": [item.revision for item in pending],
        "up_to_date": not pending,
    }


def _render_status(
    backend: str,
    target: str,
    state: schema.SchemaState,
    *,
    output_format: str,
    output: str | None,
) -> None:
    with _output_stream(output) as stream:
        if output_format == "json":
            print(json.dumps(_status_payload(backend, target, state), sort_keys=True), file=stream)
            return
        _print_status(backend, target, state, stream=stream)


def _print_status(
    backend: str,
    target: str,
    state: schema.SchemaState,
    *,
    stream: TextIO,
) -> None:
    if state.revision == schema.UNINITIALIZED:
        print(f"{backend} ({target}): uninitialized — no Cayu schema yet", file=stream)
    else:
        print(
            f"{backend} ({target}): revision {state.revision} "
            f"(compatible_from {state.compatible_from})",
            file=stream,
        )
    print(
        f"  app: latest revision {schema.LATEST_REVISION}, "
        f"min supported {schema.MIN_SUPPORTED_REVISION}",
        file=stream,
    )
    pending = schema.pending(state.revision)
    if pending:
        revs = ", ".join(str(rev.revision) for rev in pending)
        print(
            f"  pending migrations: {revs} (run `cayu storage migrate`)",
            file=stream,
        )
    else:
        print("  pending migrations: none (up to date)", file=stream)


def _status(args: argparse.Namespace) -> int:
    if args.sqlite is not None:
        connection = sqlite_support.connect(Path(args.sqlite))
        try:
            state = sqlite_support.read_schema_state(connection)
        finally:
            connection.close()
        _render_status(
            "sqlite",
            args.sqlite,
            state,
            output_format=args.output_format,
            output=args.output,
        )
        return 0

    async def run() -> schema.SchemaState:
        import psycopg

        from cayu.storage import postgres

        async with (
            await psycopg.AsyncConnection.connect(args.postgres) as conn,
            conn.cursor() as cur,
        ):
            return await postgres.read_schema_state(cur)

    state = _run_postgres(run, args)
    if state is None:
        return 1
    _render_status(
        "postgres",
        _redact_dsn(args.postgres),
        state,
        output_format=args.output_format,
        output=args.output,
    )
    return 0


def _run_postgres(run: Any, args: argparse.Namespace) -> schema.SchemaState | None:
    """Run a postgres coroutine, converting connection errors to a clean,
    DSN-redacted stderr message (``None`` signals failure). Schema-compatibility
    errors propagate to the top-level handler unchanged."""
    try:
        return asyncio.run(run())
    except schema.SchemaError:
        raise
    except Exception as exc:
        _render_error(args, _sanitize(str(exc), args.postgres))
        return None


def _migrate(args: argparse.Namespace) -> int:
    if args.sqlite is not None:
        from cayu import SQLiteSessionStore

        # Constructing in migrate mode applies pending forward revisions (the
        # baseline DDL creates every cayu_ table, so one store covers both stores).
        store = SQLiteSessionStore(args.sqlite, schema_mode=schema.SchemaMode.MIGRATE)

        async def close_sqlite() -> None:
            await store.close()

        asyncio.run(close_sqlite())
        connection = sqlite_support.connect(Path(args.sqlite))
        try:
            state = sqlite_support.read_schema_state(connection)
        finally:
            connection.close()
        _render_status(
            "sqlite",
            args.sqlite,
            state,
            output_format=args.output_format,
            output=args.output,
        )
        return 0

    async def run() -> schema.SchemaState:
        from cayu import PostgresSessionStore

        store = PostgresSessionStore(args.postgres, schema_mode=schema.SchemaMode.MIGRATE)
        try:
            await store.ensure_schema()
        finally:
            await store.close()
        import psycopg

        from cayu.storage import postgres

        async with (
            await psycopg.AsyncConnection.connect(args.postgres) as conn,
            conn.cursor() as cur,
        ):
            return await postgres.read_schema_state(cur)

    state = _run_postgres(run, args)
    if state is None:
        return 1
    _render_status(
        "postgres",
        _redact_dsn(args.postgres),
        state,
        output_format=args.output_format,
        output=args.output,
    )
    return 0


@contextlib.contextmanager
def _output_stream(path: str | None) -> Iterator[TextIO]:
    """Yield a writable stream: an opened file for ``path``, else stdout."""
    if path is None:
        yield sys.stdout
        return
    handle = open(path, "w", encoding="utf-8")  # noqa: SIM115 — closed in finally below
    try:
        yield handle
    finally:
        handle.close()


class _JsonArrayStream:
    """Adapt the JSONL exporter to a streaming JSON array without buffering records."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._buffer = ""
        self._first = True
        self._stream.write("[")

    def write(self, data: str, /) -> int:
        self._buffer += data
        while "\n" in self._buffer:
            record, self._buffer = self._buffer.split("\n", 1)
            self._write_record(record)
        return len(data)

    def finish(self) -> None:
        self._write_record(self._buffer)
        self._buffer = ""
        self._stream.write("]\n")

    def _write_record(self, record: str) -> None:
        if not record.strip():
            return
        if not self._first:
            self._stream.write(",")
        self._stream.write(record)
        self._first = False


def _export(args: argparse.Namespace) -> int:
    async def run() -> int:
        if args.tasks:
            store = _task_store(args)
            try:
                await _ensure_store_ready_for_export(store)
                with _output_stream(args.output) as stream:
                    count = await _export_tasks(
                        store, stream=stream, output_format=args.output_format
                    )
            finally:
                await store.close()
            noun = "task(s)"
        else:
            store = _session_store(args)
            try:
                await _ensure_store_ready_for_export(store)
                with _output_stream(args.output) as stream:
                    count = await _export_sessions(
                        store,
                        stream=stream,
                        output_format=args.output_format,
                    )
            finally:
                await store.close()
            noun = "session(s)"
        print(f"exported {count} {noun}", file=sys.stderr)
        return 0

    if args.postgres is None:
        return asyncio.run(run())
    # A Postgres connection/auth failure can embed the DSN password in its
    # message; route it through the same redaction as status/migrate so the
    # secret never reaches stderr. Schema errors carry no DSN and propagate to
    # the top-level handler unchanged.
    try:
        return asyncio.run(run())
    except schema.SchemaError:
        raise
    except Exception as exc:
        _render_error(args, _sanitize(str(exc), args.postgres))
        return 1


async def _export_sessions(store: Any, *, stream: TextIO, output_format: str) -> int:
    if output_format == "jsonl":
        return await jsonl_export.export_sessions(store, stream=stream)
    adapter = _JsonArrayStream(stream)
    count = await jsonl_export.export_sessions(store, stream=adapter)
    adapter.finish()
    return count


async def _export_tasks(store: Any, *, stream: TextIO, output_format: str) -> int:
    if output_format == "jsonl":
        return await jsonl_export.export_tasks(store, stream=stream)
    adapter = _JsonArrayStream(stream)
    count = await jsonl_export.export_tasks(store, stream=adapter)
    adapter.finish()
    return count


def _render_error(args: argparse.Namespace, message: str) -> None:
    if getattr(args, "output_format", None) in {"json", "jsonl"}:
        print(
            json.dumps(
                {
                    "schema_version": "1",
                    "error": {"code": "STORAGE_COMMAND_FAILED", "message": message},
                },
                sort_keys=True,
            )
        )
        return
    print(f"error: {message}", file=sys.stderr)


async def _ensure_store_ready_for_export(store: Any) -> None:
    ensure_schema = getattr(store, "ensure_schema", None)
    if ensure_schema is not None:
        await ensure_schema()


def _session_store(args: argparse.Namespace) -> Any:
    # Export reads; validate (don't create) so it never mutates the database.
    if args.sqlite is not None:
        from cayu import SQLiteSessionStore

        return SQLiteSessionStore(args.sqlite, schema_mode=schema.SchemaMode.VALIDATE)
    from cayu import PostgresSessionStore

    return PostgresSessionStore(args.postgres, schema_mode=schema.SchemaMode.VALIDATE)


def _task_store(args: argparse.Namespace) -> Any:
    if args.sqlite is not None:
        from cayu import SQLiteTaskStore

        return SQLiteTaskStore(args.sqlite, schema_mode=schema.SchemaMode.VALIDATE)
    from cayu import PostgresTaskStore

    return PostgresTaskStore(args.postgres, schema_mode=schema.SchemaMode.VALIDATE)
