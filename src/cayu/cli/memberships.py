"""`cayu memberships`: list, add and remove an application's members from the command line.

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
        help="Add a person to an organization, or move them to another department or role.",
        description=(
            "Add SUBJECT to ORGANIZATION, or move them if already a member. Refuses a "
            "department of another organization, and demoting the organization's last admin."
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
    add.add_argument("--department", help="A department of ORGANIZATION, such as acme/finance.")
    add.add_argument("--role", choices=("member", "admin"), default="member")
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
        await memberships.add(
            args.subject, args.organization, department=args.department, role=args.role
        )
        result = {
            "subject": args.subject,
            "organization": args.organization,
            "department": args.department,
            "role": args.role,
        }
        if args.output_format == "json":
            return _json({"added": result})
        where = args.department or args.organization
        return f"{terminal_safe(args.subject)} is a{'n' if args.role == 'admin' else ''} {args.role} of {terminal_safe(where)}.\n"
    removed = await memberships.remove(args.subject, args.organization)
    if args.output_format == "json":
        return _json(
            {"removed": removed, "subject": args.subject, "organization": args.organization}
        )
    subject, organization = terminal_safe(args.subject), terminal_safe(args.organization)
    if removed:
        return f"Removed {subject} from {organization}.\n"
    return f"{subject} wasn't a member of {organization}; nothing changed.\n"


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
            ("Subject", "Organization", "Department", "Role"),
            [
                {
                    "Subject": row["subject"],
                    "Organization": row["organization"],
                    "Department": row["department"],
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
