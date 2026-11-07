"""`cayu cloud connectors` and `cayu cloud connections`: an Organization's connections.

A connection is one of the Organization's authorized model credentials or app connectors
(`team-anthropic`, `warehouse`), which its Agents use through Cayu Cloud without ever holding
the credential. Developers and admins add connections and bind them to an Agent's slots; only an
admin changes which Agents may use one, rotates its credential or removes it.

Credentials are read from files, never from the command line: `--key-file PATH` for an API key,
`--credential FIELD=@PATH` for a field kept in a file (a private key) and `--credential
FIELD=VALUE` only for a field that isn't secret (`account`, `user`). No answer ever holds a
credential.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

if TYPE_CHECKING:
    from cayu.cli._cloud_api import CloudApiClient

_CONNECTIONS = "/v1/connections"
#: The `connections` actions on one Agent's slots, which name the Agent with --application.
AGENT_ACTIONS = frozenset({"bound", "bind", "unbind"})
#: The credential fields that aren't secret, which --credential may give as FIELD=VALUE; every
#: other field is read from a file (FIELD=@PATH).
PLAIN_FIELDS = frozenset({"account", "user"})
#: The credential fields Cayu Cloud's connectors take; an error names only these, since anything
#: else given as a field name may be a mistyped secret.
KNOWN_FIELDS = PLAIN_FIELDS | {"api_key", "private_key"}


class CredentialArgumentError(ValueError):
    """A credential argument names no field, or its file can't be read."""


def add_connections_parsers(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    commands.add_parser(
        "connectors",
        help="List the connectors this Cayu Cloud offers.",
        description="List the connectors this install offers: what a connection can be made to.",
    )
    connections = commands.add_parser(
        "connections",
        help="Add, list, bind, change or remove the Organization's connections.",
        description=(
            "Manage the Organization's connections: its model credentials and app connectors, "
            "which Agents use through Cayu Cloud without holding the credential."
        ),
    )
    actions = connections.add_subparsers(dest="connections_command", required=True)
    actions.add_parser(
        "list",
        help="List the Organization's connections.",
        description=(
            "List the Organization's connections, never their credentials. Bind one to an "
            "Agent's slot with `cayu cloud connections bind`."
        ),
    )
    show = actions.add_parser(
        "show",
        help="Show one connection.",
        description=(
            "Show one connection: its connector, status, scopes and the Agents that may use it. "
            "See its bindings with `cayu cloud connections bound --application AGENT`."
        ),
    )
    show.add_argument("name")

    add = actions.add_parser(
        "add",
        help="Add a connection with its credential.",
        description=(
            "Add a connection to CONNECTOR under --name. A developer names the Agents that may "
            "use it with --agent; an admin who names none makes it available to every Agent."
        ),
    )
    add.add_argument("connector", help="The connector, from `cayu cloud connectors`.")
    add.add_argument("--name", required=True, help="The connection's name, such as team-anthropic.")
    _add_credential_arguments(add)
    add.add_argument(
        "--scope", action="append", help="A scope it may hold; repeat. None: all it can hold."
    )
    add.add_argument("--agent", action="append", help="An Agent that may use it; repeat for each.")

    agents = actions.add_parser(
        "agents",
        help="Choose which Agents may use a connection (admins).",
        description="Make the connection available to exactly these Agents, or to every Agent.",
    )
    agents.add_argument("name")
    which = agents.add_mutually_exclusive_group(required=True)
    which.add_argument("--agent", action="append", help="An Agent that may use it; repeat.")
    which.add_argument("--all", action="store_true", help="Every Agent of the Organization.")

    rotate = actions.add_parser(
        "rotate",
        help="Replace a connection's credential (admins).",
        description="Replace the credential; the next call through the connection uses it.",
    )
    rotate.add_argument("name")
    _add_credential_arguments(rotate)

    remove = actions.add_parser(
        "remove",
        help="Remove a connection (admins).",
        description="Remove the connection and forget its credential; unbind every Agent first.",
    )
    remove.add_argument("name")

    bound = actions.add_parser(
        "bound",
        help="List an Agent's bound slots.",
        description=(
            "List the Agent's slots and the connection bound to each. Bind another with "
            "`cayu cloud connections bind`."
        ),
    )
    bound.add_argument("--application", required=True, help="The Agent.")
    bind = actions.add_parser(
        "bind",
        help="Bind one of an Agent's slots to a connection.",
        description="Bind the Agent's SLOT to CONNECTION, optionally to fewer of its scopes.",
    )
    bind.add_argument("slot")
    bind.add_argument("connection")
    bind.add_argument("--application", required=True, help="The Agent.")
    bind.add_argument("--scope", action="append", help="A scope the binding uses; repeat.")
    unbind = actions.add_parser(
        "unbind",
        help="Unbind one of an Agent's slots.",
        description=(
            "Unbind the Agent's SLOT: its calls through it stop at once. An admin can then "
            "remove the connection with `cayu cloud connections remove`."
        ),
    )
    unbind.add_argument("slot")
    unbind.add_argument("--application", required=True, help="The Agent.")


def _add_credential_arguments(parser: argparse.ArgumentParser) -> None:
    given = parser.add_mutually_exclusive_group(required=True)
    given.add_argument("--key-file", type=Path, help="A file holding the API key.")
    given.add_argument(
        "--credential",
        action="append",
        metavar="FIELD=@PATH|FIELD=VALUE",
        help="One credential field; @PATH reads it from a file. Repeat for each field.",
    )


def connectors_command(client: CloudApiClient) -> dict[str, Any]:
    return {"operation": "connectors.list", "result": client.request("GET", "/v1/connectors")}


def connections_command(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    application_id: str | None = None,
    agents: list[str] | None = None,
) -> dict[str, Any]:
    """Run one `connections` action. The caller resolves the Agents it names: the Agent actions'
    --application as ``application_id``, and `add` and `agents`' --agent values as ``agents``."""

    action = arguments.connections_command
    if action == "list":
        result = client.request("GET", _CONNECTIONS)
    elif action == "add":
        payload: dict[str, object] = {
            "connector": arguments.connector,
            "name": arguments.name,
            "credential": _credential(arguments),
        }
        if arguments.scope is not None:
            payload["scopes"] = arguments.scope
        if agents is not None:
            payload["agents"] = agents
        result = client.request("POST", _CONNECTIONS, payload=payload)
    elif action in AGENT_ACTIONS:
        result = _agent_action(arguments, client, application_id)
    else:
        path = f"{_CONNECTIONS}/{quote(arguments.name, safe='')}"
        if action == "show":
            result = client.request("GET", path)
        elif action == "agents":
            result = client.request(
                "PUT", f"{path}/agents", payload={"agents": None if arguments.all else agents}
            )
        elif action == "rotate":
            result = client.request(
                "PUT", f"{path}/credential", payload={"credential": _credential(arguments)}
            )
        else:
            client.request("DELETE", path)
            result = {"name": arguments.name, "removed": True}
    return {"operation": f"connections.{action}", "result": result}


def _agent_action(
    arguments: argparse.Namespace, client: CloudApiClient, application_id: str | None
) -> dict[str, Any]:
    if application_id is None:
        raise ValueError("An Agent's connections need its application id.")
    slots = f"/v1/applications/{quote(application_id, safe='')}/connections"
    if arguments.connections_command == "bound":
        return client.request("GET", slots)
    slot = f"{slots}/{quote(arguments.slot, safe='')}"
    if arguments.connections_command == "bind":
        payload: dict[str, object] = {"connection": arguments.connection}
        if arguments.scope is not None:
            payload["scopes"] = arguments.scope
        return client.request("PUT", slot, payload=payload)
    client.request("DELETE", slot)
    return {"application_id": application_id, "slot": arguments.slot, "unbound": True}


def _credential(arguments: argparse.Namespace) -> dict[str, str]:
    if arguments.key_file is not None:
        return {"api_key": _read(arguments.key_file)}
    fields: dict[str, str] = {}
    for given in arguments.credential:
        name, separator, value = given.partition("=")
        # Never repeat a malformed argument: it may be a mistyped secret.
        if not separator or not name.isidentifier() or not value:
            raise CredentialArgumentError("A credential field is FIELD=@PATH or FIELD=VALUE.")
        shown = name if name in KNOWN_FIELDS else "FIELD"
        if name in fields:
            raise CredentialArgumentError(f"The credential field {shown} is given twice.")
        if value.startswith("@"):
            fields[name] = _read(Path(value[1:]))
        elif name in PLAIN_FIELDS:
            fields[name] = value
        else:
            raise CredentialArgumentError(
                f"The credential field {shown} is secret: give it from a file, {shown}=@PATH."
            )
    return fields


def _read(path: Path) -> str:
    try:
        value = path.expanduser().read_text().strip()
    except OSError as exc:
        # Never repeat the path: it may be a secret given where its file belongs.
        raise CredentialArgumentError(
            f"Could not read a credential file: {exc.strerror}."
        ) from None
    if not value:
        raise CredentialArgumentError("A credential file is empty.")
    return value
