"""`cayu cloud admin runtime-record show|forget-migration`: an Agent's runtime record.

Cayu Cloud keeps one runtime record per deployed Agent, wherever its platform keeps it. Among
other things it notes which Release last ran the Agent's own ``[database] migrate``
(``app_database_migration``) and which release image last ran ``cayu storage migrate`` on its
database (``migrated_image_digest``), so a later publication doesn't run them again. After the
Agent's database is restored from a backup those notes describe tables that are gone, and a
platform administrator forgets them so the next publication migrates again:

    cayu cloud admin runtime-record show AGENT --organization ORG
    cayu cloud admin runtime-record forget-migration AGENT --organization ORG --revision REV

``show`` prints the record's revision, where Cloud keeps it and its migration notes (``--json``
prints the whole record). ``forget-migration`` drops the notes named with ``--note``
(``app_database_migration`` by default) only while the record is still at ``--revision``, no
publication of the Agent runs and no migration of it may still be running; otherwise Cloud
refuses it (``runtime_record_changed``, ``service_busy``, ``database_migration_running``) and
nothing changes. Both need a signed-in Cayu platform administrator and no platform access.
"""

from __future__ import annotations

import argparse
import shlex
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import quote

from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_releases import add_output_argument
from cayu.cli._cloud_settings import _cell

NOTES = ("app_database_migration", "migrated_image_digest")


def add_runtime_record_parser(
    actions: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    application: Callable[[str], str],
) -> None:
    record = actions.add_parser(
        "runtime-record",
        help="Show an Agent's runtime record, or forget a migration note in it.",
        description=(
            "Read the runtime record Cayu Cloud keeps for an Agent, and forget the notes that "
            "make its next publication skip a migration, such as after its database was "
            "restored from a backup."
        ),
    )
    commands = record.add_subparsers(dest="runtime_record_command", required=True)

    def agent(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("application", type=application, help="The exact Agent slug.")
        parser.add_argument(
            "--organization", required=True, help="The Organization that owns the Agent."
        )

    show = commands.add_parser(
        "show",
        help="Show the record's revision, where it is kept and its migration notes.",
        description=(
            "Show the Agent's runtime record: its revision (pass it to forget-migration), where "
            "Cayu Cloud keeps it and its migration notes. --json prints the whole record, which "
            "names secrets by reference and holds no secret values."
        ),
    )
    agent(show)
    add_output_argument(show, render_runtime_record)
    forget = commands.add_parser(
        "forget-migration",
        help="Forget a migration note, so the next publication runs that migration again.",
        description=(
            "Drop migration notes from the Agent's runtime record, only while it is still at "
            "--revision, no publication of the Agent runs and no migration of it may still be "
            "running; otherwise nothing changes. app_database_migration makes the next "
            "publication run the Agent's own [database] migrate again; migrated_image_digest "
            "makes it check and migrate its database with `cayu storage` again. Deploy the "
            "Agent afterwards."
        ),
    )
    agent(forget)
    forget.add_argument(
        "--revision", required=True, help="The record's revision, as `show` printed it."
    )
    forget.add_argument(
        "--note",
        dest="notes",
        action="append",
        choices=NOTES,
        help="A note to forget; repeat for both. Default: app_database_migration.",
    )
    add_output_argument(forget, render_forgotten)


def runtime_record_command(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    recovery_arguments: Sequence[str] = (),
) -> dict[str, Any]:
    """``recovery_arguments`` are the command's own ``--context`` and ``--api-key-file``, which
    the follow-up command `show` prints keeps, so it changes the same Cloud's record."""

    organization = str(arguments.organization)
    if not organization.strip():
        # An empty variable in a script must never name another Organization's Agent.
        raise CloudApiError("invalid_input", "--organization must name an Organization.")
    path = (
        f"/v1/admin/organizations/{quote(organization, safe='')}"
        f"/applications/{quote(str(arguments.application), safe='')}/runtime-record"
    )
    if arguments.runtime_record_command == "show":
        shown = _record(_request(client, "GET", path))
        forget = [
            "cayu",
            "cloud",
            *recovery_arguments,
            "admin",
            "runtime-record",
            "forget-migration",
            str(arguments.application),
            "--organization",
            organization,
            "--revision",
            shown["revision"],
        ]
        return {
            "operation": "admin.runtime-record.show",
            "result": {**shown, "commands": {"forget_migration": shlex.join(forget)}},
        }
    notes = sorted(set(arguments.notes or ("app_database_migration",)))
    result = _record(
        _request(
            client,
            "POST",
            f"{path}/forget-migration-notes",
            payload={"expected_revision": str(arguments.revision), "notes": notes},
        )
    )
    if not isinstance(result.get("forgotten"), list):
        raise CloudApiError("api_response_invalid", "The runtime record is invalid.")
    return {"operation": "admin.runtime-record.forget-migration", "result": result}


def _request(
    client: CloudApiClient, method: str, path: str, *, payload: dict[str, object] | None = None
) -> dict[str, Any]:
    try:
        return client.request(method, path, payload=payload)
    except CloudApiError as exc:
        # Every refusal of the routes carries one of Cloud's codes, so a 404 without one, or a
        # 405, means the routes themselves are missing.
        if exc.status_code == 405 or (exc.status_code == 404 and exc.code is None):
            raise CloudApiError(
                "runtime_records_unsupported",
                "This Cayu Cloud doesn't show Agents' runtime records yet.",
                status_code=exc.status_code,
            ) from None
        if exc.status_code == 403:
            raise CloudApiError(
                "platform_admin_required",
                "An Agent's runtime record needs a signed-in Cayu platform administrator "
                "(`cayu cloud login`).",
                status_code=403,
            ) from None
        raise


def _record(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value.get("revision"), str) or not isinstance(value.get("record"), dict):
        raise CloudApiError("api_response_invalid", "The runtime record is invalid.")
    return value


def _notes(record: Mapping[str, Any]) -> list[str]:
    migration = record.get("app_database_migration")
    database = record.get("database")
    digest = database.get("migrated_image_digest") if isinstance(database, Mapping) else None
    lines = []
    if isinstance(migration, Mapping):
        lines.append(
            f"  app_database_migration  Release {_cell(migration.get('deployment_id'))} "
            f"(created {_cell(migration.get('created_at'))}) on "
            f"{_cell(migration.get('database_name'))}"
        )
    if digest is not None:
        lines.append(f"  migrated_image_digest   {_cell(digest, limit=80)}")
    return lines


def render_runtime_record(result: Mapping[str, Any]) -> str:
    shown: Mapping[str, Any] = result["result"]
    record: Mapping[str, Any] = shown["record"]
    agent, organization = _cell(shown.get("application_id")), _cell(shown.get("organization_id"))
    notes = _notes(record)
    lines = [
        f"Agent:      {agent} (Organization {organization})",
        f"Kept in:    {_cell(shown.get('location'))}",
        f"Revision:   {_cell(shown.get('revision'))}",
        f"Release:    {_cell(record.get('deployment_id'))} ({_cell(record.get('status'))})",
        "Migration notes:",
        *(notes or ["  none"]),
    ]
    if notes:
        # Built from Cloud's revision and the caller's own arguments: cleaned like every cell.
        command = _cell(shown["commands"]["forget_migration"], limit=4096)
        lines += ["", f"To forget a note: {command} [--note NOTE]"]
    return "\n".join(lines)


def render_forgotten(result: Mapping[str, Any]) -> str:
    written: Mapping[str, Any] = result["result"]
    forgotten = [_cell(item) for item in written["forgotten"]]
    if not forgotten:
        return (
            "The record held none of those notes; nothing changed "
            f"(revision {_cell(written.get('revision'))})."
        )
    return (
        f"Forgot {', '.join(forgotten)}. The record is at revision "
        f"{_cell(written.get('revision'))}; deploy the Agent to run the migration again."
    )
