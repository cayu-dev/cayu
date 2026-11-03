"""`cayu cloud applications export|import`: copy an Agent's database and /data files.

``export AGENT`` asks Cayu Cloud to archive the Agent's PostgreSQL database and its ``/data``
files into one archive in the install's storage, and waits until it is stored. ``import AGENT
--from EXPORT_ID`` restores an export into a stopped Agent (``cayu cloud service destroy``
first), refusing one that already holds data unless ``--overwrite`` is given. Secret values
never leave Cloud: the export records the secret names, and the import reports the ones the
target doesn't set yet (``missing_secrets``).

Both run in Cloud's deployment worker; the CLI polls until they finish. ``--no-wait`` returns
once Cloud has queued one; ``exports``/``imports`` list an Agent's, and ``export-status``/
``import-status`` read one. An Organization admin works on its own Agents; a Cayu platform
administrator names any Organization's Agent with ``--organization`` and may import any
Organization's export, which is how an Agent's records follow it to another Organization.

Each command prints the JSON envelope the other ``cayu cloud`` commands print. A failed
export or import exits 2 with Cloud's ``code`` and message; a wait that runs out exits 2 with
``category`` ``agent_data_transfer_still_running`` and the command that reads it.
"""

from __future__ import annotations

import argparse
import shlex
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, cast
from urllib.parse import quote

from cayu.cli._cloud_api import CloudApiClient, CloudApiError

Kind = Literal["export", "import"]

COMMANDS = frozenset({"export", "exports", "export-status", "import", "imports", "import-status"})
_FINISHED = frozenset({"succeeded", "failed"})
_POLL_TRANSIENT_LIMIT = 5
_DEFAULT_WAIT_SECONDS = 3600.0
_DEFAULT_POLL_SECONDS = 5.0


class AgentDataTransferStillRunningError(CloudApiError):
    """The local wait ended while Cayu Cloud keeps running the export or import."""

    def __init__(self, *, kind: Kind, transfer: Mapping[str, Any], command: Sequence[str]) -> None:
        super().__init__(
            "agent_data_transfer_still_running",
            f"Cayu Cloud is still running this {kind}; it continues without the CLI.",
        )
        self.transfer_id = str(transfer.get("id"))
        self.status = str(transfer.get("status"))
        self.command = shlex.join(command)

    def public_details(self) -> dict[str, object]:
        return {"id": self.transfer_id, "status": self.status, "commands": {"status": self.command}}


class AgentDataTransferFailedError(CloudApiError):
    """Cayu Cloud finished the export or import as failed; ``transfer`` is its record."""

    def __init__(self, *, kind: Kind, transfer: Mapping[str, Any]) -> None:
        error = transfer.get("error")
        error = error if isinstance(error, Mapping) else {}
        code = error.get("code")
        message = error.get("message")
        super().__init__(
            f"agent_data_{kind}_failed",
            str(message) if isinstance(message, str) and message else f"The {kind} failed.",
            code=str(code) if isinstance(code, str) else None,
        )
        self.transfer = dict(transfer)

    def public_details(self) -> dict[str, object]:
        return {"id": self.transfer.get("id"), "transfer": self.transfer}


def add_data_transfer_parsers(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    application: Callable[[str], str],
    seconds: Callable[[str], float],
) -> None:
    """The six subcommands, under `cayu cloud applications`."""

    def agent(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("application", type=application, help="The exact Agent slug.")
        parser.add_argument(
            "--organization",
            help=(
                "The Agent's Organization, for a Cayu platform administrator working on "
                "another Organization's Agent. Without it, the selected Organization's."
            ),
        )

    def waits(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--no-wait", action="store_true", help="Return once queued.")
        parser.add_argument("--poll-seconds", type=seconds, default=_DEFAULT_POLL_SECONDS)
        parser.add_argument("--wait-seconds", type=seconds, default=_DEFAULT_WAIT_SECONDS)

    export = commands.add_parser(
        "export",
        help="Export an Agent's database and /data files into one archive in Cloud.",
        description=(
            "Archive the Agent's PostgreSQL database (dumped as its own database role) and its "
            "/data files into one archive in the install's storage, with the names (never the "
            "values) of its secrets, and wait until it is stored. Needs an organization "
            "administrator, or a platform administrator with --organization. An export of a "
            "running Agent may catch files newer or older than its database; destroy its "
            "service first (pausing leaves its web process writing) for an exact copy."
        ),
    )
    agent(export)
    waits(export)
    listed = commands.add_parser("exports", description="List an Agent's exports, newest first.")
    agent(listed)
    status = commands.add_parser("export-status", description="Show one export of an Agent.")
    agent(status)
    status.add_argument("transfer_id", metavar="EXPORT_ID")

    restore = commands.add_parser(
        "import",
        help="Restore an export into a stopped Agent.",
        description=(
            "Restore an export's database and /data files into the Agent, which must be stopped "
            "(`cayu cloud service destroy --application AGENT`). An Agent that already holds "
            "records or files is refused and left alone unless --overwrite is given. The "
            "result's missing_secrets names the export's secrets the Agent doesn't set yet; "
            "set them, then deploy or roll back to start it. Needs an organization "
            "administrator (its own Organization's export), or a platform administrator with "
            "--organization (any Organization's export)."
        ),
    )
    agent(restore)
    restore.add_argument(
        "--from", dest="export_id", required=True, metavar="EXPORT_ID", help="The export."
    )
    restore.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace whatever the Agent's database and /data hold.",
    )
    waits(restore)
    imports = commands.add_parser("imports", description="List an Agent's imports, newest first.")
    agent(imports)
    import_status = commands.add_parser("import-status", description="Show one import.")
    agent(import_status)
    import_status.add_argument("transfer_id", metavar="IMPORT_ID")


def data_transfer_command(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    validate_wait: Callable[[float, float], None],
    transient: Callable[[CloudApiError], bool],
    recovery_arguments: Sequence[str] = (),
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """``validate_wait`` and ``transient`` are the other waits' own checks (`cloud`)."""

    action = str(arguments.application_command)
    kind: Kind = "import" if action.startswith("import") else "export"
    base = _agent_path(arguments)
    collection = f"{base}/data-{kind}s"
    operation = f"applications.{action}"
    if action in {"exports", "imports"}:
        return {"operation": operation, "result": _items(client.request("GET", collection))}
    if action in {"export-status", "import-status"}:
        path = f"{collection}/{quote(str(arguments.transfer_id), safe='')}"
        return {"operation": operation, "result": _transfer(client.request("GET", path))}
    if not arguments.no_wait:
        validate_wait(arguments.poll_seconds, arguments.wait_seconds)
    payload = (
        {"export_id": arguments.export_id, "overwrite": bool(arguments.overwrite)}
        if kind == "import"
        else None
    )
    transfer = _transfer(client.request("POST", collection, payload=payload))
    if arguments.no_wait:
        return {"operation": operation, "result": transfer}
    status_command = [
        "cayu",
        "cloud",
        *recovery_arguments,
        "applications",
        f"{kind}-status",
        str(arguments.application),
        str(transfer["id"]),
        *(("--organization", arguments.organization) if arguments.organization else ()),
    ]
    path = f"{collection}/{quote(str(transfer['id']), safe='')}"
    deadline = monotonic() + arguments.wait_seconds
    transient_failures = 0
    while transfer.get("status") not in _FINISHED:
        if monotonic() >= deadline:
            raise AgentDataTransferStillRunningError(
                kind=kind, transfer=transfer, command=status_command
            )
        sleep(arguments.poll_seconds)
        try:
            transfer = _transfer(client.request("GET", path))
            transient_failures = 0
        except CloudApiError as exc:
            # The transfer keeps running in Cayu Cloud: a brief API outage is not a failure.
            if not transient(exc) or transient_failures >= _POLL_TRANSIENT_LIMIT:
                raise
            transient_failures += 1
    if transfer.get("status") == "failed":
        raise AgentDataTransferFailedError(kind=kind, transfer=transfer)
    return {"operation": operation, "result": transfer}


def _agent_path(arguments: argparse.Namespace) -> str:
    agent = quote(str(arguments.application), safe="")
    organization = arguments.organization
    if organization is not None and not str(organization).strip():
        # An empty variable in a script must never fall back to the selected Organization.
        raise CloudApiError("invalid_input", "--organization must name an Organization.")
    if organization:
        return f"/v1/admin/organizations/{quote(str(organization), safe='')}/applications/{agent}"
    return f"/v1/applications/{agent}"


def _transfer(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CloudApiError("api_response_invalid", "The export or import record is invalid.")
    record = cast("dict[str, Any]", value)
    if not isinstance(record.get("id"), str) or not isinstance(record.get("status"), str):
        raise CloudApiError("api_response_invalid", "The export or import record is invalid.")
    return record


def _items(value: object) -> dict[str, Any]:
    items = cast("dict[str, Any]", value).get("items") if isinstance(value, dict) else None
    if not isinstance(items, list):
        raise CloudApiError("api_response_invalid", "The export or import list is invalid.")
    return {"items": [_transfer(item) for item in items]}
