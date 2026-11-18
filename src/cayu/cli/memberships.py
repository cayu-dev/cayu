"""`cayu memberships`: list, add and remove an application's members from the command line.

`cayu memberships grant-knowledge` grants organization knowledge to work admitted before the
policy granted any (:mod:`cayu.memberships.knowledge_grants`).

It reads and writes the application's own membership tables through
:class:`cayu.memberships.PostgresMemberships`, in the database Cayu uses
(`CAYU_DATABASE_URL`, then `[tool.cayu.session_store]`, or `--postgres`). Every change raises
the access revision, so Cayu rechecks access at each person's next step, and no change
leaves an organization that has an admin without one.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import io
import json
import sys
from functools import partial
from typing import Any

from cayu.cli._output import add_output_options, output_destination, print_table, terminal_safe

_COMMAND_FAILED = "MEMBERSHIPS_COMMAND_FAILED"


def add_memberships_parser(subparsers: Any) -> None:
    group = subparsers.add_parser(
        "memberships",
        help="List, add and remove the members of an application's organizations.",
        description=(
            "Read and change the organizations' members in the application's own tables "
            "(`cayu.memberships`, `cayu guide b2b`). The database is --postgres, else "
            "CAYU_DATABASE_URL, else [tool.cayu.session_store]; it must be PostgreSQL. "
            "Changes take effect at each person's next step."
        ),
    )
    commands = group.add_subparsers(dest="memberships_command", required=True)
    listing = commands.add_parser(
        "list",
        help="List an organization's members, or the organizations a person belongs to.",
        description="List an organization's members, or every membership of one person.",
    )
    which = listing.add_mutually_exclusive_group(required=True)
    which.add_argument("--organization", help="The organization whose members to list.")
    which.add_argument("--subject", help="The person whose memberships to list.")
    add = commands.add_parser(
        "add",
        help="Add a person to an organization, or change their role there.",
        description=(
            "Add SUBJECT to ORGANIZATION, or change their role if already a member. Refuses "
            "an organization that doesn't exist, and demoting the organization's last admin."
        ),
    )
    remove = commands.add_parser(
        "remove",
        help="Remove a person from an organization.",
        description=(
            "Remove SUBJECT from ORGANIZATION; their next protected step there is refused. "
            "Refuses to remove the organization's last admin."
        ),
    )
    for command in (add, remove):
        command.add_argument("subject", metavar="SUBJECT", help="The person's sign-in subject.")
        command.add_argument("organization", metavar="ORGANIZATION", help="The organization ID.")
    add.add_argument("--role", choices=("member", "admin"), default="member")
    grant = commands.add_parser(
        "grant-knowledge",
        help="Grant knowledge to conversations and tasks admitted before the policy had it.",
        description=(
            "Find the conversations and unfinished tasks admitted under AUTHORITY whose stored "
            "grant has no knowledge, which is all work admitted before "
            "MembershipAccessPolicy(knowledge=True) was turned on, and record that each of "
            "them may use the knowledge of the organizations it was admitted in. Stored work "
            "is left as it is; the policy reads the records, which are also the audit of "
            "what was granted. --dry-run only lists. Prints one JSON document."
        ),
    )
    grant.add_argument(
        "--dry-run", action="store_true", help="List what would be granted; record nothing."
    )
    grant.add_argument("--subject", help="Only this person's work.")
    grant.add_argument(
        "--authority",
        default=None,
        help="The policy's authority (default: MembershipAccessPolicy's default authority).",
    )
    grant.add_argument(
        "--postgres",
        metavar="DSN",
        help=(
            "The application's PostgreSQL database, where Cayu's sessions and tasks are too "
            "(default: CAYU_DATABASE_URL). Prefer CAYU_DATABASE_URL to keep credentials off "
            "the command line."
        ),
    )
    add_output_options(grant, formats=("json",), default="json")
    for command in (listing, add, remove):
        command.add_argument(
            "--postgres",
            metavar="DSN",
            help=(
                "The application's PostgreSQL database (default: CAYU_DATABASE_URL). Prefer "
                "CAYU_DATABASE_URL to keep credentials off the command line."
            ),
        )
        command.add_argument(
            "--schema", default="app", help="The schema of the membership tables (default: app)."
        )
        add_output_options(command, formats=("json", "table"), default="table")


def run_memberships(args: argparse.Namespace) -> int:
    from cayu.memberships import MembershipChangeRefused, MembershipTablesInvalid

    try:
        report = asyncio.run(_run(args))
        status = 0
    except (MembershipChangeRefused, MembershipTablesInvalid, ValueError, OSError) as exc:
        report, status = _failure(args, str(exc)), 1
    except Exception as exc:
        # psycopg's errors: a database that can't be reached or refuses the statement.
        if type(exc).__module__.split(".")[0] != "psycopg":
            raise
        report, status = _failure(args, f"The database refused it: {exc}"), 1
    try:
        with output_destination(args.output) as stream:
            stream.write(report)
    except OSError as exc:
        print(f"error: could not write output: {exc}", file=sys.stderr)
        return 2
    return status


async def _run(args: argparse.Namespace) -> str:
    from cayu.memberships import PostgresMemberships, check_membership_tables

    if args.memberships_command == "grant-knowledge":
        return _json(await _grant_knowledge(args, _database(args)))
    connect = partial(_connect(), _database(args))
    memberships = PostgresMemberships(connect, schema=args.schema)
    connection = connect()
    try:
        check_membership_tables(connection, schema=memberships.schema)
    finally:
        connection.close()
    if args.memberships_command == "list":
        members = await (
            memberships.of_organization(args.organization)
            if args.organization is not None
            else memberships.of_subject(args.subject)
        )
        rows = [dataclasses.asdict(item) for item in members]
        if args.output_format == "json":
            return _json({"members": rows})
        if not rows:
            return "No members.\n"
        return _table(rows)
    if args.memberships_command == "add":
        await memberships.add(args.subject, args.organization, role=args.role)
        result = {"subject": args.subject, "organization": args.organization, "role": args.role}
        if args.output_format == "json":
            return _json({"added": result})
        return f"{terminal_safe(args.subject)} is a{'n' if args.role == 'admin' else ''} {args.role} of {terminal_safe(args.organization)}.\n"
    removed = await memberships.remove(args.subject, args.organization)
    if args.output_format == "json":
        return _json(
            {"removed": removed, "subject": args.subject, "organization": args.organization}
        )
    subject, organization = terminal_safe(args.subject), terminal_safe(args.organization)
    if removed:
        return f"Removed {subject} from {organization}.\n"
    return f"{subject} wasn't a member of {organization}; nothing changed.\n"


async def _grant_knowledge(args: argparse.Namespace, dsn: str) -> dict[str, Any]:
    from datetime import UTC, datetime

    from cayu.memberships import KnowledgeGrant, PostgresKnowledgeGrants, admitted_organizations
    from cayu.memberships.policy import DEFAULT_AUTHORITY
    from cayu.resource_access import work_admitted_without_knowledge
    from cayu.storage import PostgresSessionStore, PostgresTaskStore
    from cayu.storage.migrations import SchemaMode

    authority = DEFAULT_AUTHORITY if args.authority is None else args.authority
    grants = PostgresKnowledgeGrants(partial(_connect(), dsn))
    sessions = PostgresSessionStore(dsn, schema_mode=SchemaMode.VALIDATE, read_only=True)
    try:
        tasks = PostgresTaskStore(dsn, schema_mode=SchemaMode.VALIDATE, read_only=True)
        try:
            work = await work_admitted_without_knowledge(
                sessions, tasks, authority=authority, subject=args.subject
            )
        finally:
            await tasks.close()
    finally:
        await sessions.close()
    now = datetime.now(UTC)
    rows: list[dict[str, Any]] = []
    pending: list[KnowledgeGrant] = []
    for item in work:
        organizations = admitted_organizations(item.binding)
        row: dict[str, Any] = {
            "binding_sha256": item.binding_sha256,
            "subject": item.binding.subject,
            "organizations": organizations,
            "session_ids": list(item.session_ids),
            "task_ids": list(item.task_ids),
        }
        if not organizations:
            # Admitted with no memberships: there is no organization's knowledge to grant.
            row["status"] = "no_organization"
        elif await grants.granted(item.binding_sha256):
            row["status"] = "already_granted"
        else:
            row["status"] = "would_grant" if args.dry_run else "granted"
            pending.append(
                KnowledgeGrant(
                    binding_sha256=item.binding_sha256,
                    authority=authority,
                    subject=item.binding.subject,
                    organizations=tuple(organizations),
                    sessions=len(item.session_ids),
                    tasks=len(item.task_ids),
                    granted_at=now,
                )
            )
        rows.append(row)
    if pending and not args.dry_run:
        recorded = {grant.binding_sha256 for grant in await grants.record(pending)}
        for row in rows:
            if row["status"] == "granted" and row["binding_sha256"] not in recorded:
                # Another run recorded it first.
                row["status"] = "already_granted"
    granting = "would_grant" if args.dry_run else "granted"
    counts = {granting: 0, "already_granted": 0, "no_organization": 0}
    for row in rows:
        counts[row["status"]] += 1
    return {
        "authority": authority,
        "dry_run": args.dry_run,
        "subject": args.subject,
        "counts": {
            **counts,
            "sessions": sum(len(row["session_ids"]) for row in rows),
            "tasks": sum(len(row["task_ids"]) for row in rows),
        },
        "admissions": rows,
        "granted_at": None if args.dry_run else now.isoformat(),
    }


def _database(args: argparse.Namespace) -> str:
    from cayu.cli.store_targets import SessionStoreTargetError, resolve_session_store_target
    from cayu.storage.targets import SessionStoreBackend

    try:
        target = resolve_session_store_target(postgres=args.postgres)
    except SessionStoreTargetError as exc:
        raise ValueError(str(exc)) from None
    if target.backend is not SessionStoreBackend.POSTGRES or target.postgres_dsn is None:
        raise ValueError(
            "Memberships live in the application's PostgreSQL database: set CAYU_DATABASE_URL "
            "or pass --postgres."
        )
    return target.postgres_dsn


def _connect():
    try:
        import psycopg
    except ModuleNotFoundError:
        raise ValueError(
            'cayu memberships needs psycopg: install it with `pip install "cayu[postgres]"`.'
        ) from None
    return psycopg.connect


def _table(rows: list[dict[str, Any]]) -> str:
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        print_table(
            ("Subject", "Organization", "Role"),
            [
                {
                    "Subject": row["subject"],
                    "Organization": row["organization"],
                    "Role": row["role"],
                }
                for row in rows
            ],
        )
    return text.getvalue()


def _failure(args: argparse.Namespace, message: str) -> str:
    if args.output_format == "json":
        return _json({"error": {"code": _COMMAND_FAILED, "message": message}})
    print(f"error: {terminal_safe(message, keep_lines=True)}", file=sys.stderr)
    return ""


def _json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
