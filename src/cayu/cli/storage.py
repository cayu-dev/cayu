"""`cayu storage` subcommands: schema status / migrate, and JSON/JSONL export.

Realizes ADR 0001 Phase 2 (explicit migrate + status) and Phase 3 (JSONL
export). Migrations are an explicit, operator-run step — never silent on import
(Decision 6) — so this CLI is the supported way to migrate a production database.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import hmac
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO, cast
from urllib.parse import unquote, urlsplit, urlunsplit

from cayu._filesystem_lock import cooperative_path_lock
from cayu._version import package_version
from cayu.build_provenance import current_runtime_build_provenance
from cayu.cli._output import add_output_options
from cayu.cli.store_targets import (
    SessionStoreBackend,
    SessionStoreTargetError,
    resolve_session_store_target,
)
from cayu.runtime.public_authority import public_authority_alias_codec_from_environment
from cayu.sessions.exports import SessionExportLimits
from cayu.storage import _sqlite_connection as sqlite_connection
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage import jsonl_export, migration_authority
from cayu.storage import migrations as schema

_SUBCOMMANDS = (
    ("status", "Show the database schema revision and any pending migrations."),
    ("migrate", "Apply pending forward migrations under the backend lock."),
    ("export", "Export sessions (or tasks) as JSON or JSONL for backup/replay."),
)

#: ``cayu storage status`` exit codes. 1 remains a general error and 2 an
#: argparse usage error, so deploy steps can branch on the migration need.
STATUS_EXIT_UP_TO_DATE = 0
STATUS_EXIT_MIGRATION_AVAILABLE = 3
STATUS_EXIT_INCOMPATIBLE = 4

#: Optional direct (unpooled) Postgres URL used by ``migrate`` and ``status``
#: when the target is resolved rather than passed explicitly. Migrations take
#: session-level advisory locks, which a transaction-pooling proxy such as
#: PgBouncer cannot preserve.
DIRECT_DATABASE_URL_ENV = "CAYU_DATABASE_DIRECT_URL"

_RDS_SNAPSHOT_IDENTIFIER = re.compile(r"(?:rds:)?[A-Za-z](?:-?[A-Za-z0-9])*")
_RFC3339_UTC_TIMESTAMP = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})[Tt](\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?(?:[Zz]|\+00:00)"
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
            description=(
                f"{help_text} Without --sqlite or --postgres, the target resolves from "
                "CAYU_DATABASE_URL, then [tool.cayu.session_store]. "
                "Use `--output FILE` to write data to a destination."
                + (
                    " Exits 0 when up to date, 3 when a forward migration is available, "
                    "and 4 when this build cannot migrate the database with the given inputs."
                    if name == "status"
                    else ""
                )
            ),
        )
        target = sub.add_mutually_exclusive_group()
        target.add_argument(
            "--sqlite",
            metavar="PATH",
            help="Path to a SQLite database file (overrides the resolved target).",
        )
        target.add_argument(
            "--postgres",
            metavar="DSN",
            help=(
                "Postgres connection string (overrides the resolved target). Prefer "
                "CAYU_DATABASE_URL to keep credentials off the command line."
            ),
        )
        if name == "export":
            sub.add_argument(
                "--tasks", action="store_true", help="Export tasks instead of sessions."
            )
            sub.add_argument(
                "--max-session-bytes",
                type=int,
                help="Session export byte ceiling (default: 67108864); raise for large histories.",
            )
            sub.add_argument(
                "--max-record-bytes",
                type=int,
                help="Session component byte ceiling (default: 8388608); does not change stored-value validation.",
            )
            add_output_options(sub, formats=("json", "jsonl"))
        else:
            if name == "migrate":
                backup = sub.add_mutually_exclusive_group()
                backup.add_argument(
                    "--backup",
                    metavar="PATH",
                    help=(
                        "SQLite backup destination (default: a unique sibling file). "
                        "Not valid for Postgres."
                    ),
                )
                backup.add_argument(
                    "--backup-sha256",
                    metavar="SHA256",
                    help=(
                        "Attest an application-consistent Postgres backup by its "
                        "lowercase SHA-256 digest."
                    ),
                )
                backup.add_argument(
                    "--backup-managed",
                    metavar="REF",
                    help=(
                        "Record a provider-managed Postgres backup: "
                        "rds-snapshot:<snapshot identifier> or "
                        "rds-pitr:<RFC 3339 UTC timestamp>."
                    ),
                )
                backup.add_argument(
                    "--waive-backup",
                    action="store_true",
                    help="Explicitly waive retained backup/restore authority.",
                )
            if name in {"migrate", "status"}:
                sub.add_argument(
                    "--acknowledge-breaking",
                    metavar="REVISION",
                    action="append",
                    type=int,
                    default=[],
                    help=(
                        "Acknowledge one exact pending breaking revision; repeat for every "
                        "breaking boundary in the planned path."
                    ),
                )
                sub.add_argument(
                    "--reset-empty-recall-state",
                    action="store_true",
                    help=(
                        "Authorize rebuilding the empty prerelease recall checkpoint/delivery "
                        "tables when crossing revision 73."
                    ),
                )
            add_output_options(sub)


def run_storage(args: argparse.Namespace) -> int:
    """Dispatch a parsed ``storage`` invocation; return a process exit code."""
    try:
        _resolve_storage_target(args)
        if args.storage_command == "status":
            return _status(args)
        if args.storage_command == "migrate":
            return _migrate(args)
        if args.storage_command == "export":
            return _export(args)
    except (schema.SchemaError, OSError, RuntimeError, ValueError) as exc:
        message = str(exc)
        if args.postgres is not None:
            message = _sanitize(message, args.postgres)
        _render_error(args, message)
        return 1
    return 1


def _resolve_storage_target(args: argparse.Namespace) -> None:
    """Fill ``args.sqlite``/``args.postgres`` from the shared session-store resolver.

    Explicit flags win and keep their historical meaning. Otherwise the target
    comes from ``CAYU_DATABASE_URL``, then ``[tool.cayu.session_store]``, so a
    deploy step can keep the credential-bearing DSN in the environment.
    ``migrate`` and ``status`` prefer ``CAYU_DATABASE_DIRECT_URL`` for a resolved
    Postgres target because migration holds session-level advisory locks.
    """

    if args.sqlite is not None or args.postgres is not None:
        args.target_source = "explicit"
        return
    target = resolve_session_store_target()
    source = target.source
    dsn = target.postgres_dsn
    direct = os.environ.get(DIRECT_DATABASE_URL_ENV)
    if args.storage_command in {"migrate", "status"} and direct is not None:
        if target.backend is not SessionStoreBackend.POSTGRES:
            raise SessionStoreTargetError(
                f"{DIRECT_DATABASE_URL_ENV} is set, but the resolved session store is "
                "SQLite; unset it or resolve a Postgres session store."
            )
        try:
            dsn = resolve_session_store_target(postgres=direct).postgres_dsn
        except SessionStoreTargetError:
            raise SessionStoreTargetError(
                f"{DIRECT_DATABASE_URL_ENV} must contain a Postgres URL."
            ) from None
        source = f"environment:{DIRECT_DATABASE_URL_ENV}"
    if target.backend is SessionStoreBackend.SQLITE:
        args.sqlite = str(target.sqlite_path)
    else:
        args.postgres = dsn
    args.target_source = source


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


def _status_payload(
    backend: str,
    target: str,
    state: schema.SchemaState,
    *,
    target_source: str,
    migration: dict[str, object] | None = None,
) -> dict[str, Any]:
    pending = schema.pending(state.revision)
    payload: dict[str, Any] = {
        "schema_version": "1",
        "backend": backend,
        "target": target,
        "target_source": target_source,
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
    if migration is not None:
        payload["migration"] = migration
        payload["up_to_date"] = migration["need"] == "up_to_date"
    return payload


def _assess_migration(
    state: schema.SchemaState,
    *,
    input_empty: bool,
    acknowledged: list[int],
    resume_state: schema.SchemaState | None = None,
) -> dict[str, object]:
    """Classify what ``cayu storage migrate`` would do, without preflight I/O.

    ``resume_state`` is the recorded input of an interrupted Postgres migration;
    its path, not the partially advanced live revision, decides which breaking
    acknowledgements the resuming invocation must repeat.
    """

    assessment: dict[str, object] = {
        "need": "up_to_date",
        "exit_code": STATUS_EXIT_UP_TO_DATE,
        "input_revision": state.revision,
        "input_empty": input_empty,
        "target_revision": schema.LATEST_REVISION,
        "breaking_boundary": False,
        "breaking_revisions": [],
        "missing_acknowledgements": [],
        "resuming": resume_state is not None,
        "reason": None,
        "detail": None,
    }
    try:
        planned = schema.validate_migration_input(state)
        plan_state = state if resume_state is None else resume_state
        plan = schema.validate_migration_input(plan_state)
    except schema.SchemaError as exc:
        newer = state.revision > schema.LATEST_REVISION or (
            state.compatible_from > schema.LATEST_REVISION
        )
        assessment.update(
            breaking_boundary=None,
            breaking_revisions=None,
            missing_acknowledgements=None,
        )
        return _incompatible_assessment(
            assessment,
            reason="database_newer_than_build" if newer else "invalid_schema_state",
            detail=str(exc),
        )
    if not planned:
        return assessment
    breaking = (
        []
        if plan_state.revision == schema.UNINITIALIZED
        else [item.revision for item in plan if item.kind is schema.RevisionKind.BREAKING]
    )
    supplied = set(acknowledged)
    assessment.update(
        need="forward_migration",
        exit_code=STATUS_EXIT_MIGRATION_AVAILABLE,
        breaking_boundary=bool(breaking),
        breaking_revisions=breaking,
        missing_acknowledgements=[item for item in breaking if item not in supplied],
    )
    return assessment


def _incompatible_assessment(
    assessment: dict[str, object],
    *,
    reason: str,
    detail: str,
) -> dict[str, object]:
    assessment.update(
        need="incompatible",
        exit_code=STATUS_EXIT_INCOMPATIBLE,
        reason=reason,
        detail=detail,
    )
    return assessment


def _rejected_assessment(
    assessment: dict[str, object],
    exc: Exception,
    *,
    dsn: str | None,
) -> dict[str, object]:
    detail = str(exc) if dsn is None else _sanitize(str(exc), dsn)
    missing = assessment.get("missing_acknowledgements")
    return _incompatible_assessment(
        assessment,
        reason=("breaking_acknowledgement_required" if missing else "migration_preflight_rejected"),
        detail=detail,
    )


def _render_status(
    backend: str,
    target: str,
    state: schema.SchemaState,
    *,
    target_source: str,
    migration: dict[str, object],
    output_format: str,
    output: str | None,
) -> None:
    with _output_stream(output) as stream:
        if output_format == "json":
            payload = _status_payload(
                backend,
                target,
                state,
                target_source=target_source,
                migration=migration,
            )
            print(json.dumps(payload, sort_keys=True), file=stream)
            return
        _print_status(backend, target, state, stream=stream)
        need = migration["need"]
        print(f"  migration need: {need} (exit {migration['exit_code']})", file=stream)
        if migration["breaking_revisions"]:
            revisions = cast("list[int]", migration["breaking_revisions"])
            print(
                "  breaking revisions: " + ", ".join(str(item) for item in revisions),
                file=stream,
            )
        if migration["reason"] is not None:
            print(f"  reason: {migration['reason']}: {migration['detail']}", file=stream)


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
        path = Path(args.sqlite)
        state = _read_sqlite_schema_state(path)
        migration = _assess_migration(
            state,
            input_empty=_sqlite_input_empty(path, state),
            acknowledged=args.acknowledge_breaking,
        )
        if migration["need"] == "forward_migration":
            try:
                codec = public_authority_alias_codec_from_environment()
                _preflight_sqlite_migration(path, codec, args)
            except (schema.SchemaError, RuntimeError, ValueError) as exc:
                migration = _rejected_assessment(migration, exc, dsn=None)
        _render_status(
            "sqlite",
            args.sqlite,
            state,
            target_source=args.target_source,
            migration=migration,
            output_format=args.output_format,
            output=args.output,
        )
        return cast("int", migration["exit_code"])

    async def run() -> tuple[schema.SchemaState, dict[str, object]]:
        import psycopg

        from cayu.storage import postgres

        async with await psycopg.AsyncConnection.connect(args.postgres) as conn:
            await conn.set_read_only(True)
            async with conn.cursor() as cur:
                state = await postgres.read_schema_state(cur)
                pending = await postgres.read_pending_migration_receipt(cur)
                resume_state = None if pending is None else _pending_receipt_input_state(pending[1])
                migration = _assess_migration(
                    state,
                    input_empty=await _postgres_input_empty(cur, state),
                    acknowledged=args.acknowledge_breaking,
                    resume_state=resume_state,
                )
                if migration["need"] != "forward_migration":
                    return state, migration
                if resume_state is not None:
                    # An interrupted migration resumes from its durable receipt
                    # without repeating preflight, but only for the same inputs.
                    if migration["missing_acknowledgements"]:
                        migration = _incompatible_assessment(
                            migration,
                            reason="breaking_acknowledgement_required",
                            detail=(
                                "Resume the interrupted migration with the same "
                                "--acknowledge-breaking revisions it recorded."
                            ),
                        )
                    return state, migration
                try:
                    codec = public_authority_alias_codec_from_environment()
                    await _preflight_postgres_migration(cur, state, codec, args)
                except (schema.SchemaError, RuntimeError, ValueError) as exc:
                    migration = _rejected_assessment(migration, exc, dsn=args.postgres)
                return state, migration

    result = _run_postgres(run, args)
    if result is None:
        return 1
    state, migration = result
    _render_status(
        "postgres",
        _redact_dsn(args.postgres),
        state,
        target_source=args.target_source,
        migration=migration,
        output_format=args.output_format,
        output=args.output,
    )
    return cast("int", migration["exit_code"])


def _pending_receipt_input_state(receipt: dict[str, object]) -> schema.SchemaState:
    revision = receipt.get("input_revision")
    compatible_from = receipt.get("input_compatible_from")
    if type(revision) is not int or type(compatible_from) is not int:
        raise RuntimeError("Postgres durable migration receipt is invalid.")
    return schema.SchemaState(revision=revision, compatible_from=compatible_from)


def _sqlite_input_empty(path: Path, state: schema.SchemaState) -> bool:
    """Whether a SQLite target has no Cayu schema at all (nothing to back up)."""

    if state.revision != schema.UNINITIALIZED:
        return False
    if not path.exists():
        return True
    connection = sqlite_connection.connect(path, read_only=True)
    try:
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name LIKE ? ESCAPE '\\' LIMIT 1",
            (_table_prefix_pattern(),),
        ).fetchone()
    finally:
        connection.close()
    return row is None


async def _postgres_input_empty(cur: Any, state: schema.SchemaState) -> bool:
    """Whether a Postgres target has no Cayu relation in its current schema."""

    if state.revision != schema.UNINITIALIZED:
        return False
    await cur.execute(
        "SELECT EXISTS ("
        "SELECT 1 FROM pg_catalog.pg_class AS c "
        "JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace "
        "WHERE n.nspname = current_schema() AND c.relname LIKE %s)",
        (_table_prefix_pattern(),),
    )
    row = await cur.fetchone()
    return row is not None and not bool(row[0])


def _table_prefix_pattern() -> str:
    escaped = schema.TABLE_PREFIX.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"{escaped}%"


def _run_postgres(run: Any, args: argparse.Namespace) -> Any:
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
        return _migrate_sqlite(args)

    return _migrate_postgres(args)


def _migrate_sqlite(args: argparse.Namespace) -> int:
    path = Path(args.sqlite)
    if str(path) == ":memory:":
        raise ValueError("`cayu storage migrate` requires a file-backed SQLite database.")
    if path.is_symlink():
        raise ValueError("SQLite migration target must not be a symbolic link.")
    if args.backup_sha256 is not None:
        raise ValueError("--backup-sha256 is only valid with --postgres.")
    if args.backup_managed is not None:
        raise ValueError("--backup-managed is only valid for a Postgres target.")

    path.parent.mkdir(parents=True, exist_ok=True)
    pending_receipt_path, durable_receipt_path = _sqlite_migration_receipt_paths(path)
    sqlite_sidecars = (Path(f"{path}-wal"), Path(f"{path}-shm"))
    reserved_paths = (path, pending_receipt_path, durable_receipt_path, *sqlite_sidecars)
    requested_backup_path = None if args.backup is None else Path(args.backup)
    if requested_backup_path is not None and any(
        requested_backup_path.resolve() == reserved.resolve() for reserved in reserved_paths
    ):
        raise ValueError(
            "SQLite backup path must differ from the migration target, receipt paths, "
            "and database sidecars."
        )
    _preflight_migration_output(
        args.output,
        forbidden=(
            *reserved_paths,
            *((requested_backup_path,) if requested_backup_path is not None else ()),
        ),
    )
    lock_target = path.resolve()
    with cooperative_path_lock(
        lock_target.parent,
        lock_target.name,
        lock_directory_name="cayu-storage-migration-locks-v1",
    ):
        return _migrate_sqlite_locked(
            args,
            path=path,
            pending_receipt_path=pending_receipt_path,
            durable_receipt_path=durable_receipt_path,
        )


def _migrate_sqlite_locked(
    args: argparse.Namespace,
    *,
    path: Path,
    pending_receipt_path: Path,
    durable_receipt_path: Path,
) -> int:
    from cayu import SQLiteSessionStore

    if path.is_symlink():
        raise ValueError("SQLite migration target must not be a symbolic link.")
    recovery = _recover_sqlite_migration_receipt(
        path,
        pending_receipt_path=pending_receipt_path,
        durable_receipt_path=durable_receipt_path,
        target_source=args.target_source,
        output_format=args.output_format,
        output=args.output,
    )
    if isinstance(recovery, int):
        return recovery

    codec = public_authority_alias_codec_from_environment()
    runtime_build = current_runtime_build_provenance().model_dump(mode="json")
    input_state = _preflight_sqlite_migration(path, codec, args)
    planned = schema.validate_migration_input(input_state)
    input_empty = _sqlite_input_empty(path, input_state)
    authority = migration_authority.authority_configuration_receipt(codec)
    migration_inputs: dict[str, object] = {
        "reset_empty_recall_state": bool(args.reset_empty_recall_state),
    }
    source_existed = path.exists()
    source_mode = path.stat().st_mode & 0o777 if source_existed else None

    retained_backup = not args.waive_backup
    resuming = recovery is not None
    backup_sha256: str | None = None
    if recovery is None:
        backup_path = (
            _sqlite_backup_path(path, args.backup)
            if retained_backup
            else _new_sqlite_path(
                path.parent,
                prefix=f".{path.name}.cayu-waived-snapshot-",
            )
        )
        receipt: dict[str, object] | None = None
        backup_ready = False
        pending_receipt_ready = False
    else:
        receipt = recovery
        backup_path, backup_sha256 = _resume_sqlite_migration_backup(
            receipt,
            args=args,
            path=path,
            input_state=input_state,
            planned=planned,
            authority=authority,
            runtime_build=runtime_build,
            migration_inputs=migration_inputs,
        )
        backup_ready = True
        pending_receipt_ready = True
    staging_path: Path | None = None
    published = False
    try:
        with _sqlite_migration_lock(path, source_existed):
            # Repeat every read-only decision while the writer reservation
            # excludes changes between the snapshot and publication.
            locked_state = _preflight_sqlite_migration(path, codec, args)
            if locked_state != input_state:
                raise RuntimeError(
                    "SQLite migration input changed after preflight; retry from the new revision."
                )
            if receipt is None:
                _write_sqlite_snapshot(path if source_existed else None, backup_path)
                backup_ready = True
                backup: dict[str, object]
                if retained_backup:
                    backup_sha256 = _sha256_file(backup_path)
                    backup = {
                        "mode": "retained",
                        "path": str(backup_path),
                        "sha256": backup_sha256,
                    }
                else:
                    backup = {"mode": "waived", "path": None, "sha256": None}
                operation_sha256 = _migration_operation_sha256(
                    backend="sqlite",
                    target=str(path),
                    input_state=input_state,
                    planned=planned,
                    acknowledged=args.acknowledge_breaking,
                    authority=authority,
                    runtime_build=runtime_build,
                    backup=backup,
                    migration_inputs=migration_inputs,
                )
                latest = schema.revision(schema.LATEST_REVISION)
                receipt = _migration_receipt(
                    backend="sqlite",
                    target=str(path),
                    input_state=input_state,
                    planned=planned,
                    acknowledged=args.acknowledge_breaking,
                    authority=authority,
                    runtime_build=runtime_build,
                    backup=backup,
                    output_state=schema.SchemaState(
                        revision=latest.revision,
                        compatible_from=latest.compatible_from,
                    ),
                    input_empty=input_empty,
                    checks={"integrity_check": "ok", "foreign_key_violations": 0},
                    execution_mode="atomic_publish",
                    migration_inputs=migration_inputs,
                    operation_sha256=operation_sha256,
                )
                if retained_backup:
                    _write_durable_sqlite_receipt(pending_receipt_path, receipt)
                    pending_receipt_ready = True
            elif backup_sha256 is None or not hmac.compare_digest(
                _sha256_file(backup_path), backup_sha256
            ):
                raise RuntimeError(
                    "SQLite retained migration backup changed after receipt recovery."
                )

            staging_path = _new_sqlite_path(
                path.parent,
                prefix=f".{path.name}.cayu-migration-",
            )
            if resuming:
                _write_sqlite_snapshot(path if source_existed else None, staging_path)
                if backup_sha256 is None or not hmac.compare_digest(
                    _sha256_file(staging_path), backup_sha256
                ):
                    raise RuntimeError(
                        "SQLite migration input changed after the retained backup was captured; "
                        "refusing to overwrite same-revision writes."
                    )
            else:
                shutil.copyfile(backup_path, staging_path)
            if source_mode is not None:
                os.chmod(staging_path, source_mode)
            if args.reset_empty_recall_state:
                reset_connection = sqlite_connection.connect(staging_path)
                try:
                    sqlite_support.reset_empty_recall_state(reset_connection)
                finally:
                    reset_connection.close()

            # The complete schema path and alias-key reconciliation run against
            # the snapshot. The live database path is still untouched.
            store = SQLiteSessionStore(
                staging_path,
                schema_mode=schema.SchemaMode.MIGRATE,
                public_authority_alias_codec=codec,
            )

            async def close_sqlite() -> None:
                await store.close()

            asyncio.run(close_sqlite())
            output_state, checks = _validate_sqlite_migration_result(staging_path)
            if receipt is None:
                raise RuntimeError("SQLite migration receipt was not prepared before staging.")
            expected_output = schema.SchemaState(
                revision=cast("int", receipt["output_revision"]),
                compatible_from=cast("int", receipt["output_compatible_from"]),
            )
            if output_state != expected_output or checks != receipt.get("checks"):
                raise RuntimeError(
                    "SQLite migration staging result does not match its pending receipt."
                )
            _checkpoint_and_sync_sqlite(staging_path)
            if not source_existed and path.exists():
                raise RuntimeError(
                    "SQLite migration target appeared after preflight; refusing to overwrite it."
                )
            if not pending_receipt_ready:
                _write_durable_sqlite_receipt(pending_receipt_path, receipt)
                pending_receipt_ready = True
            os.replace(staging_path, path)
            staging_path = None
            published = True
            _remove_sqlite_sidecars(path)
            _sync_directory(path.parent)
            os.replace(pending_receipt_path, durable_receipt_path)
            _sync_directory(path.parent)

        _render_migration(
            "sqlite",
            str(path),
            output_state,
            receipt,
            target_source=args.target_source,
            output_format=args.output_format,
            output=args.output,
        )
        _discard_durable_sqlite_receipt(durable_receipt_path)
        return 0
    except Exception as exc:
        if published:
            recovery_path = (
                durable_receipt_path if durable_receipt_path.exists() else pending_receipt_path
            )
            raise RuntimeError(
                "SQLite migration committed but receipt delivery did not complete; "
                f"rerun the same command to recover the durable receipt at {recovery_path}: {exc}"
            ) from exc
        raise
    finally:
        if staging_path is not None:
            _remove_sqlite_file_set(staging_path)
        if not retained_backup or not backup_ready or (not published and not pending_receipt_ready):
            _remove_sqlite_file_set(backup_path)
        if not published and (not retained_backup or not pending_receipt_ready):
            pending_receipt_path.unlink(missing_ok=True)
        # A failure before atomic publication either removes unbound artifacts
        # or retains the authenticated receipt plus backup for literal retry.
        if not published and not source_existed and path.exists() and path.stat().st_size == 0:
            path.unlink()


def _migrate_postgres(args: argparse.Namespace) -> int:
    if args.backup is not None:
        raise ValueError("--backup is only valid with --sqlite.")
    _preflight_migration_output(args.output, forbidden=())
    requested_backup = _requested_postgres_backup_authority(args)
    codec = public_authority_alias_codec_from_environment()
    runtime_build = current_runtime_build_provenance().model_dump(mode="json")
    target = _redact_dsn(args.postgres)
    authority = migration_authority.authority_configuration_receipt(codec)
    migration_inputs: dict[str, object] = {
        "reset_empty_recall_state": bool(args.reset_empty_recall_state),
    }

    async def load_pending() -> tuple[str, dict[str, object]] | None:
        import psycopg

        from cayu.storage import postgres

        async with (
            await psycopg.AsyncConnection.connect(args.postgres) as conn,
            conn.cursor() as cur,
        ):
            return await postgres.read_pending_migration_receipt(cur)

    async def preflight() -> tuple[schema.SchemaState, tuple[schema.Revision, ...], bool]:
        import psycopg

        from cayu.storage import postgres

        async with (
            await psycopg.AsyncConnection.connect(args.postgres) as conn,
            conn.cursor() as cur,
        ):
            state = await postgres.read_schema_state(cur)
            planned = await _preflight_postgres_migration(cur, state, codec, args)
            return state, planned, await _postgres_input_empty(cur, state)

    try:
        pending = asyncio.run(load_pending())
        if pending is None:
            input_state, planned, input_empty = asyncio.run(preflight())
            backup = _postgres_backup_authority(requested_backup, input_empty=input_empty)
            operation_sha256 = _migration_operation_sha256(
                backend="postgres",
                target=target,
                input_state=input_state,
                planned=planned,
                acknowledged=args.acknowledge_breaking,
                authority=authority,
                runtime_build=runtime_build,
                backup=backup,
                migration_inputs=migration_inputs,
            )
            latest = schema.revision(schema.LATEST_REVISION)
            receipt = _migration_receipt(
                backend="postgres",
                target=target,
                input_state=input_state,
                planned=planned,
                acknowledged=args.acknowledge_breaking,
                authority=authority,
                runtime_build=runtime_build,
                backup=backup,
                output_state=schema.SchemaState(
                    revision=latest.revision,
                    compatible_from=latest.compatible_from,
                ),
                input_empty=input_empty,
                checks={"schema_validation": "ok", "foreign_keys": "backend_enforced"},
                execution_mode="resumable_revisions",
                migration_inputs=migration_inputs,
                operation_sha256=operation_sha256,
            )
        else:
            operation_sha256, raw_receipt = pending
            receipt = _validate_postgres_migration_receipt(
                raw_receipt,
                target=target,
                operation_sha256=operation_sha256,
            )
            input_state = schema.SchemaState(
                revision=cast("int", receipt["input_revision"]),
                compatible_from=cast("int", receipt["input_compatible_from"]),
            )
            planned = schema.validate_migration_input(input_state)
            if receipt.get("migration_steps") != [revision.revision for revision in planned]:
                raise RuntimeError("Postgres durable migration receipt has a stale migration path.")
            backup = _postgres_backup_authority(
                requested_backup,
                input_empty=receipt.get("input_empty") is True,
            )
            retry_operation_sha256 = _migration_operation_sha256(
                backend="postgres",
                target=target,
                input_state=input_state,
                planned=planned,
                acknowledged=args.acknowledge_breaking,
                authority=authority,
                runtime_build=runtime_build,
                backup=backup,
                migration_inputs=migration_inputs,
            )
            if not hmac.compare_digest(retry_operation_sha256, operation_sha256):
                raise RuntimeError(
                    "Postgres has an undelivered receipt for a different migration "
                    "invocation; rerun the original command to recover it."
                )
    except schema.SchemaError:
        raise
    except Exception as exc:
        _render_error(args, _sanitize(str(exc), args.postgres))
        return 1

    receipt_json = json.dumps(receipt, sort_keys=True, separators=(",", ":"))

    async def run() -> schema.SchemaState:
        from cayu import PostgresSessionStore

        store = PostgresSessionStore(
            args.postgres,
            schema_mode=schema.SchemaMode.MIGRATE,
            public_authority_alias_codec=codec,
            migration_reset_empty_recall_state=args.reset_empty_recall_state,
            migration_expected_input_state=input_state,
            migration_operation_sha256=operation_sha256,
            migration_receipt_json=receipt_json,
        )
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
    expected_output = schema.SchemaState(
        revision=cast("int", receipt["output_revision"]),
        compatible_from=cast("int", receipt["output_compatible_from"]),
    )
    if state != expected_output:
        raise RuntimeError(
            "Postgres migration completed at a state that does not match its durable receipt."
        )
    _render_migration(
        "postgres",
        target,
        state,
        receipt,
        target_source=args.target_source,
        output_format=args.output_format,
        output=args.output,
    )

    async def discard_receipt() -> None:
        import psycopg

        from cayu.storage import postgres

        async with (
            await psycopg.AsyncConnection.connect(args.postgres) as conn,
            conn.cursor() as cur,
        ):
            await postgres.discard_pending_migration_receipt(cur, operation_sha256)
            await conn.commit()

    # A leftover receipt is safe: the next literal invocation validates it
    # against committed revision progress and re-delivers the same evidence.
    with contextlib.suppress(Exception):
        asyncio.run(discard_receipt())
    return 0


def _preflight_sqlite_migration(
    path: Path,
    codec: Any,
    args: argparse.Namespace,
) -> schema.SchemaState:
    if not path.exists():
        state = schema.SchemaState(
            revision=schema.UNINITIALIZED,
            compatible_from=0,
        )
        planned = schema.validate_migration_input(state)
        _validate_breaking_acknowledgements(state, planned, args.acknowledge_breaking)
        _validate_recall_reset_input(state, planned, args.reset_empty_recall_state)
        return state
    connection = sqlite_connection.connect(path, read_only=True)
    try:
        state = sqlite_support.read_schema_state(connection)
        planned = schema.validate_migration_input(state)
        _validate_breaking_acknowledgements(state, planned, args.acknowledge_breaking)
        _validate_recall_reset_input(state, planned, args.reset_empty_recall_state)
        sqlite_support.preflight_migration(
            connection,
            state,
            allow_empty_recall_reset=args.reset_empty_recall_state,
        )
        migration_authority.preflight_sqlite_public_authority(connection, codec)
        return state
    finally:
        connection.close()


def _validate_breaking_acknowledgements(
    state: schema.SchemaState,
    planned: tuple[schema.Revision, ...],
    acknowledged: list[int],
) -> None:
    required = (
        ()
        if state.revision == schema.UNINITIALIZED
        else tuple(
            revision.revision
            for revision in planned
            if revision.kind is schema.RevisionKind.BREAKING
        )
    )
    supplied = tuple(sorted(set(acknowledged)))
    if supplied == required:
        return
    missing = tuple(revision for revision in required if revision not in supplied)
    unexpected = tuple(revision for revision in supplied if revision not in required)
    details: list[str] = []
    if missing:
        details.append("missing " + ", ".join(str(revision) for revision in missing))
    if unexpected:
        details.append("unexpected " + ", ".join(str(revision) for revision in unexpected))
    raise schema.SchemaError(
        "Breaking migration acknowledgement does not match the exact pending path ("
        + "; ".join(details)
        + "). Repeat --acknowledge-breaking REVISION for every pending breaking revision."
    )


def _validate_recall_reset_input(
    state: schema.SchemaState,
    planned: tuple[schema.Revision, ...],
    requested: bool,
) -> None:
    if not requested:
        return
    if not (69 <= state.revision < 73 and any(revision.revision == 73 for revision in planned)):
        raise schema.SchemaError(
            "--reset-empty-recall-state is valid only for initialized revisions 69-72 "
            "that have revision 73 pending."
        )


async def _preflight_postgres_migration(
    cur: Any,
    state: schema.SchemaState,
    codec: Any,
    args: argparse.Namespace,
) -> tuple[schema.Revision, ...]:
    """Run every read-only migration check before the first schema write."""

    from cayu.storage import postgres

    planned = schema.validate_migration_input(state)
    _validate_breaking_acknowledgements(state, planned, args.acknowledge_breaking)
    _validate_recall_reset_input(state, planned, args.reset_empty_recall_state)
    await postgres.preflight_migration(
        cur,
        state,
        allow_empty_recall_reset=args.reset_empty_recall_state,
    )
    await migration_authority.preflight_postgres_public_authority(cur, codec)
    await cur.execute(
        "SELECT has_schema_privilege(current_user, current_schema(), 'USAGE'), "
        "has_schema_privilege(current_user, current_schema(), 'CREATE')"
    )
    privileges = await cur.fetchone()
    if privileges is None or not all(bool(value) for value in privileges):
        raise RuntimeError(
            "Postgres migration requires USAGE and CREATE authority on the current schema."
        )
    return planned


def _requested_postgres_backup_authority(args: argparse.Namespace) -> dict[str, object] | None:
    """Validate the operator's backup authority before connecting to Postgres."""

    if args.backup_managed is not None:
        return {
            "mode": "provider_managed",
            "path": None,
            "sha256": None,
            "reference": _managed_backup_reference(args.backup_managed),
        }
    digest = args.backup_sha256
    if digest is not None:
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("--backup-sha256 must be a lowercase SHA-256 digest.")
        return {"mode": "operator_attested", "path": None, "sha256": digest}
    if args.waive_backup:
        return {"mode": "waived", "path": None, "sha256": None}
    return None


def _postgres_backup_authority(
    requested: dict[str, object] | None,
    *,
    input_empty: bool,
) -> dict[str, object]:
    """Bind the requested backup authority, or none for a database with no Cayu schema."""

    if requested is not None:
        return requested
    if input_empty:
        return {"mode": "not_required", "path": None, "sha256": None}
    raise ValueError(
        "Postgres migration requires --backup-sha256 for an application-consistent "
        "backup, --backup-managed for a provider-managed backup, or the explicit "
        "--waive-backup operator waiver."
    )


def _managed_backup_reference(value: str) -> str:
    """Validate a provider-managed backup reference; return it unchanged."""

    kind, separator, detail = value.partition(":")
    if separator and kind == "rds-snapshot":
        if len(detail) <= 255 and _RDS_SNAPSHOT_IDENTIFIER.fullmatch(detail) is not None:
            return value
        raise ValueError(
            "--backup-managed rds-snapshot: requires an RDS snapshot identifier: a letter "
            "followed by letters, digits, or single hyphens, not ending in a hyphen, at "
            "most 255 characters (automated snapshots may keep their rds: prefix)."
        )
    if separator and kind == "rds-pitr":
        match = _RFC3339_UTC_TIMESTAMP.fullmatch(detail)
        if match is not None:
            try:
                datetime(*(int(part) for part in match.groups()[:6]), tzinfo=UTC)
            except ValueError:
                match = None
        if match is not None:
            return value
        raise ValueError(
            "--backup-managed rds-pitr: requires an RFC 3339 UTC timestamp such as "
            "2026-09-29T12:00:00Z."
        )
    raise ValueError(
        "--backup-managed must be rds-snapshot:<snapshot identifier> or "
        "rds-pitr:<RFC 3339 UTC timestamp>."
    )


def _sqlite_migration_receipt_paths(path: Path) -> tuple[Path, Path]:
    return (
        Path(f"{path}.cayu-migration-receipt.pending.json"),
        Path(f"{path}.cayu-migration-receipt.json"),
    )


def _preflight_migration_output(output: str | None, *, forbidden: tuple[Path, ...]) -> None:
    """Validate a deterministic receipt destination without truncating it."""

    if output is None:
        return
    destination = Path(output)
    if any(_paths_alias(destination, path) for path in forbidden):
        raise ValueError(
            "Migration output path must differ from database, backup, and migration sidecar paths."
        )
    if destination.exists():
        descriptor = os.open(destination, os.O_WRONLY)
        os.close(descriptor)
        return
    descriptor, probe = tempfile.mkstemp(
        prefix=f".{destination.name}.cayu-output-probe-",
        dir=destination.parent,
    )
    os.close(descriptor)
    Path(probe).unlink()


def _paths_alias(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


def _read_sqlite_migration_receipt(path: Path, *, target: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"SQLite durable migration receipt is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"SQLite durable migration receipt is invalid: {path}")
    receipt = dict(value)
    recorded_digest = receipt.pop("receipt_sha256", None)
    calculated_digest = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    receipt["receipt_sha256"] = recorded_digest
    recorded_target = receipt.get("target")
    if (
        receipt.get("schema_version") != "cayu.storage.migration-receipt.v1"
        or receipt.get("backend") != "sqlite"
        or not isinstance(recorded_target, str)
        or Path(recorded_target).resolve() != target.resolve()
        or not isinstance(recorded_digest, str)
        or not hmac.compare_digest(recorded_digest, calculated_digest)
        or type(receipt.get("input_revision")) is not int
        or type(receipt.get("input_compatible_from")) is not int
        or type(receipt.get("output_revision")) is not int
        or type(receipt.get("output_compatible_from")) is not int
    ):
        raise RuntimeError(f"SQLite durable migration receipt is invalid: {path}")
    return receipt


def _validate_postgres_migration_receipt(
    value: dict[str, object],
    *,
    target: str,
    operation_sha256: str,
) -> dict[str, object]:
    receipt = dict(value)
    recorded_digest = receipt.pop("receipt_sha256", None)
    calculated_digest = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    receipt["receipt_sha256"] = recorded_digest
    runtime = receipt.get("runtime")
    runtime_operation = (
        cast("dict[str, object]", runtime).get("operation_sha256")
        if isinstance(runtime, dict)
        else None
    )
    migration_steps = receipt.get("migration_steps")
    latest = schema.revision(schema.LATEST_REVISION)
    if (
        receipt.get("schema_version") != "cayu.storage.migration-receipt.v1"
        or receipt.get("backend") != "postgres"
        or receipt.get("target") != target
        or not isinstance(recorded_digest, str)
        or not hmac.compare_digest(recorded_digest, calculated_digest)
        or re.fullmatch(r"[0-9a-f]{64}", operation_sha256) is None
        or not isinstance(runtime_operation, str)
        or not hmac.compare_digest(runtime_operation, operation_sha256)
        or type(receipt.get("input_revision")) is not int
        or type(receipt.get("input_compatible_from")) is not int
        or receipt.get("output_revision") != latest.revision
        or receipt.get("output_compatible_from") != latest.compatible_from
        or not isinstance(migration_steps, list)
        or any(type(revision) is not int for revision in migration_steps)
    ):
        raise RuntimeError("Postgres durable migration receipt is invalid.")
    return receipt


def _read_sqlite_schema_state(path: Path) -> schema.SchemaState:
    if not path.exists():
        return schema.SchemaState(revision=schema.UNINITIALIZED, compatible_from=0)
    connection = sqlite_connection.connect(path, read_only=True)
    try:
        return sqlite_support.read_schema_state(connection)
    finally:
        connection.close()


def _recover_sqlite_migration_receipt(
    path: Path,
    *,
    pending_receipt_path: Path,
    durable_receipt_path: Path,
    target_source: str,
    output_format: str,
    output: str | None,
) -> int | dict[str, object] | None:
    existing = tuple(
        receipt_path
        for receipt_path in (pending_receipt_path, durable_receipt_path)
        if receipt_path.exists()
    )
    if not existing:
        return None
    if len(existing) != 1:
        raise RuntimeError(
            "SQLite migration has conflicting pending and final durable receipts; "
            "inspect both before retrying."
        )
    receipt_path = existing[0]
    receipt = _read_sqlite_migration_receipt(receipt_path, target=path)
    state = _read_sqlite_schema_state(path)
    output_state = schema.SchemaState(
        revision=cast("int", receipt["output_revision"]),
        compatible_from=cast("int", receipt["output_compatible_from"]),
    )
    input_state = schema.SchemaState(
        revision=cast("int", receipt["input_revision"]),
        compatible_from=cast("int", receipt["input_compatible_from"]),
    )
    if state == output_state:
        if receipt_path == pending_receipt_path:
            os.replace(pending_receipt_path, durable_receipt_path)
            _sync_directory(path.parent)
        try:
            _render_migration(
                "sqlite",
                str(path),
                state,
                receipt,
                target_source=target_source,
                output_format=output_format,
                output=output,
            )
        except Exception as exc:
            raise RuntimeError(
                "SQLite migration is already committed; durable receipt delivery failed "
                f"and can be retried from {durable_receipt_path}: {exc}"
            ) from exc
        _discard_durable_sqlite_receipt(durable_receipt_path)
        return 0

    if receipt_path == pending_receipt_path and state == input_state:
        return receipt
    raise RuntimeError(
        "SQLite database state does not match its durable migration receipt; "
        "inspect the database and receipt before retrying."
    )


def _resume_sqlite_migration_backup(
    receipt: dict[str, object],
    *,
    args: argparse.Namespace,
    path: Path,
    input_state: schema.SchemaState,
    planned: tuple[schema.Revision, ...],
    authority: dict[str, object],
    runtime_build: dict[str, object],
    migration_inputs: dict[str, object],
) -> tuple[Path, str]:
    backup_value = receipt.get("backup")
    runtime_value = receipt.get("runtime")
    backup = cast("dict[str, object]", backup_value) if isinstance(backup_value, dict) else None
    runtime = cast("dict[str, object]", runtime_value) if isinstance(runtime_value, dict) else None
    backup_path_value = backup.get("path") if backup is not None else None
    backup_sha256 = backup.get("sha256") if backup is not None else None
    operation_sha256 = runtime.get("operation_sha256") if runtime is not None else None
    if (
        args.waive_backup
        or backup is None
        or backup.get("mode") != "retained"
        or not isinstance(backup_path_value, str)
        or not isinstance(backup_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", backup_sha256) is None
        or not isinstance(operation_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", operation_sha256) is None
        or receipt.get("migration_steps") != [revision.revision for revision in planned]
    ):
        raise RuntimeError("SQLite pending migration receipt is not resumable.")
    backup_path = Path(backup_path_value)
    _preflight_migration_output(args.output, forbidden=(backup_path,))
    if args.backup is not None and Path(args.backup).resolve() != backup_path.resolve():
        raise RuntimeError(
            "SQLite pending migration receipt belongs to a different backup destination."
        )
    if (
        backup_path.is_symlink()
        or not backup_path.is_file()
        or not hmac.compare_digest(
            _sha256_file(backup_path),
            backup_sha256,
        )
    ):
        raise RuntimeError("SQLite retained migration backup does not match its receipt.")
    if _read_sqlite_schema_state(backup_path) != input_state:
        raise RuntimeError("SQLite retained migration backup has the wrong schema state.")
    retry_operation_sha256 = _migration_operation_sha256(
        backend="sqlite",
        target=str(path),
        input_state=input_state,
        planned=planned,
        acknowledged=args.acknowledge_breaking,
        authority=authority,
        runtime_build=runtime_build,
        backup=backup,
        migration_inputs=migration_inputs,
    )
    if not hmac.compare_digest(retry_operation_sha256, operation_sha256):
        raise RuntimeError(
            "SQLite pending migration receipt belongs to a different migration invocation."
        )
    return backup_path, backup_sha256


def _write_durable_sqlite_receipt(path: Path, receipt: dict[str, object]) -> None:
    descriptor, raw_temp_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temp_path = Path(raw_temp_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(receipt, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        _sync_directory(path.parent)
    finally:
        temp_path.unlink(missing_ok=True)


def _discard_durable_sqlite_receipt(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
        _sync_directory(path.parent)
    except OSError:
        # A leftover receipt is intentionally safe: the next invocation verifies
        # it against the live database and re-delivers the same authenticated result.
        pass


def _sqlite_backup_path(path: Path, requested: str | None) -> Path:
    if requested is None:
        return _new_sqlite_path(path.parent, prefix=f"{path.name}.cayu-backup-")
    backup_path = Path(requested)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    if backup_path.resolve() == path.resolve():
        raise ValueError("SQLite backup path must differ from the migration target.")
    try:
        descriptor = os.open(backup_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError(f"SQLite backup already exists: {backup_path}") from None
    os.close(descriptor)
    return backup_path


def _new_sqlite_path(parent: Path, *, prefix: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix=prefix, suffix=".sqlite3", dir=parent)
    os.close(descriptor)
    return Path(raw_path)


@contextlib.contextmanager
def _sqlite_migration_lock(path: Path, source_existed: bool) -> Iterator[None]:
    if not source_existed:
        yield
        return
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.execute("PRAGMA busy_timeout = 5000")
        checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint is not None and int(checkpoint[0]) != 0:
            raise RuntimeError(
                "SQLite WAL checkpoint is busy; stop database writers before migrating."
            )
        connection.execute("BEGIN IMMEDIATE")
        yield
    finally:
        if connection.in_transaction:
            connection.rollback()
        connection.close()


def _write_sqlite_snapshot(source: Path | None, destination: Path) -> None:
    target = sqlite3.connect(destination)
    try:
        if source is None:
            target.execute("PRAGMA application_id")
            target.commit()
        else:
            origin = sqlite_connection.connect(source, read_only=True)
            try:
                origin.backup(target)
            finally:
                origin.close()
    finally:
        target.close()
    _sync_file(destination)


def _validate_sqlite_migration_result(
    path: Path,
) -> tuple[schema.SchemaState, dict[str, object]]:
    connection = sqlite_connection.connect(path, read_only=True)
    try:
        state = sqlite_support.read_schema_state(connection)
        integrity_rows = tuple(str(row[0]) for row in connection.execute("PRAGMA integrity_check"))
        foreign_key_rows = tuple(connection.execute("PRAGMA foreign_key_check"))
    finally:
        connection.close()
    if integrity_rows != ("ok",):
        raise RuntimeError("SQLite migration staging database failed PRAGMA integrity_check.")
    if foreign_key_rows:
        raise RuntimeError("SQLite migration staging database failed PRAGMA foreign_key_check.")
    return state, {"integrity_check": "ok", "foreign_key_violations": 0}


def _checkpoint_and_sync_sqlite(path: Path) -> None:
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint is not None and int(checkpoint[0]) != 0:
            raise RuntimeError("SQLite migration staging WAL checkpoint did not complete.")
    finally:
        connection.close()
    _remove_sqlite_sidecars(path)
    _sync_file(path)


def _remove_sqlite_sidecars(path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


def _remove_sqlite_file_set(path: Path) -> None:
    path.unlink(missing_ok=True)
    _remove_sqlite_sidecars(path)


def _sync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _migration_receipt(
    *,
    backend: str,
    target: str,
    input_state: schema.SchemaState,
    planned: tuple[schema.Revision, ...],
    acknowledged: list[int],
    authority: dict[str, object],
    runtime_build: dict[str, object],
    backup: dict[str, object],
    output_state: schema.SchemaState,
    input_empty: bool,
    checks: dict[str, object],
    execution_mode: str,
    migration_inputs: dict[str, object],
    operation_sha256: str | None = None,
) -> dict[str, object]:
    migration_identity = {
        "backend": backend,
        "latest_revision": schema.LATEST_REVISION,
        "path": [
            {
                "revision": revision.revision,
                "kind": revision.kind.value,
                "compatible_from": revision.compatible_from,
            }
            for revision in planned
        ],
    }
    migration_identity_sha256 = hashlib.sha256(
        json.dumps(migration_identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    runtime: dict[str, object] = {
        "package_version": package_version(),
        "build_provenance": runtime_build,
        "migration_identity_sha256": migration_identity_sha256,
    }
    if operation_sha256 is not None:
        runtime["operation_sha256"] = operation_sha256
    receipt: dict[str, object] = {
        "schema_version": "cayu.storage.migration-receipt.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "backend": backend,
        "target": target,
        "input_revision": input_state.revision,
        "input_compatible_from": input_state.compatible_from,
        "input_empty": input_empty,
        "migration_steps": [revision.revision for revision in planned],
        "breaking_revisions": [
            revision.revision
            for revision in planned
            if revision.kind is schema.RevisionKind.BREAKING
        ],
        "acknowledged_breaking_revisions": sorted(set(acknowledged)),
        "migration_inputs": migration_inputs,
        "authority_configuration": authority,
        "backup": backup,
        "runtime": runtime,
        "execution_mode": execution_mode,
        "output_revision": output_state.revision,
        "output_compatible_from": output_state.compatible_from,
        "checks": checks,
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return receipt


def _migration_operation_sha256(
    *,
    backend: str,
    target: str,
    input_state: schema.SchemaState,
    planned: tuple[schema.Revision, ...],
    acknowledged: list[int],
    authority: dict[str, object],
    runtime_build: dict[str, object],
    backup: dict[str, object],
    migration_inputs: dict[str, object],
) -> str:
    """Bind resumable commits to one fully preflighted migration operation."""

    identity = {
        "schema_version": "cayu.storage.migration-operation.v1",
        "backend": backend,
        "target": target,
        "input_revision": input_state.revision,
        "input_compatible_from": input_state.compatible_from,
        "migration_path": [
            {
                "revision": revision.revision,
                "kind": revision.kind.value,
                "compatible_from": revision.compatible_from,
            }
            for revision in planned
        ],
        "acknowledged_breaking_revisions": sorted(set(acknowledged)),
        "migration_inputs": migration_inputs,
        "authority_configuration": authority,
        "backup": backup,
        "runtime": {
            "package_version": package_version(),
            "build_provenance": runtime_build,
        },
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _render_migration(
    backend: str,
    target: str,
    state: schema.SchemaState,
    receipt: dict[str, object],
    *,
    target_source: str,
    output_format: str,
    output: str | None,
) -> None:
    with _output_stream(output) as stream:
        if output_format == "json":
            payload = _status_payload(backend, target, state, target_source=target_source)
            payload["migration_receipt"] = receipt
            print(json.dumps(payload, sort_keys=True), file=stream)
            return
        _print_status(backend, target, state, stream=stream)
        print(f"  migration receipt: {receipt['receipt_sha256']}", file=stream)
        backup = receipt["backup"]
        backup_sha256: object | None = None
        backup_reference: object | None = None
        if isinstance(backup, dict):
            for key, value in backup.items():
                if key == "sha256":
                    backup_sha256 = value
                elif key == "reference":
                    backup_reference = value
        if backup_sha256 is not None:
            print(f"  backup sha256: {backup_sha256}", file=stream)
        if backup_reference is not None:
            print(f"  managed backup: {backup_reference}", file=stream)


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
    overrides = {
        field: value
        for field, value in (
            ("max_bytes", getattr(args, "max_session_bytes", None)),
            ("max_record_bytes", getattr(args, "max_record_bytes", None)),
        )
        if value is not None
    }
    if args.tasks and overrides:
        raise ValueError("Session byte limits do not apply to --tasks exports.")
    limits = SessionExportLimits(**overrides)

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
                        limits=limits,
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


async def _export_sessions(
    store: Any, *, stream: TextIO, output_format: str, limits: SessionExportLimits | None = None
) -> int:
    if output_format == "jsonl":
        return await jsonl_export.export_sessions(store, stream=stream, limits=limits)
    adapter = _JsonArrayStream(stream)
    count = await jsonl_export.export_sessions(store, stream=adapter, limits=limits)
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

        return SQLiteSessionStore(
            args.sqlite,
            schema_mode=schema.SchemaMode.VALIDATE,
            public_authority_alias_codec=public_authority_alias_codec_from_environment(),
        )
    from cayu import PostgresSessionStore

    return PostgresSessionStore(
        args.postgres,
        schema_mode=schema.SchemaMode.VALIDATE,
        public_authority_alias_codec=public_authority_alias_codec_from_environment(),
    )


def _task_store(args: argparse.Namespace) -> Any:
    if args.sqlite is not None:
        from cayu import SQLiteTaskStore

        return SQLiteTaskStore(args.sqlite, schema_mode=schema.SchemaMode.VALIDATE)
    from cayu import PostgresTaskStore

    return PostgresTaskStore(args.postgres, schema_mode=schema.SchemaMode.VALIDATE)
