"""`cayu cloud connectors` and `cayu cloud connections`: an Organization's connections.

A connection is one of the Organization's authorized model credentials or app connectors
(`team-anthropic`, `warehouse`), which its Agents use through Cayu Cloud without ever holding
the credential. Developers and admins add connections and bind them to an Agent's slots; only an
admin changes which Agents may use one, rotates its credential or removes it.

Credentials are read from files, never from the command line: `--key-file PATH` for an API key,
`--credential FIELD=@PATH` for a field kept in a file (a private key) and `--credential
FIELD=VALUE` only for a field that isn't secret (`account`, `user`, and SAP's `base_url`,
`token_url`, `client_id`, `oauth_scope`, `sap_client`). No answer ever holds a credential.
`connections check` asks Cloud to test a connection that can be checked, such as SAP, and
prints each service's result.

A connector that signs in (Slack) takes no credential: an admin adds it without one, or starts
its sign-in again with `connections authorize`, and Cloud answers the link where the person
approves it, which these commands print.

`cayu cloud models list` shows an Agent's model slots; `connections bind SLOT --model MODEL` binds
one to an approved model and `connections unbind SLOT --model` unbinds it, so new sessions use
the slot's default.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from cayu.cli._cloud_settings import _table

if TYPE_CHECKING:
    from cayu.cli._cloud_api import CloudApiClient

_CONNECTIONS = "/v1/connections"
#: The `connections` actions on one Agent's slots, which name the Agent with --application.
AGENT_ACTIONS = frozenset({"bound", "bind", "unbind"})
#: The credential fields that aren't secret, which --credential may give as FIELD=VALUE; every
#: other field is read from a file (FIELD=@PATH).
PLAIN_FIELDS = frozenset(
    {"account", "user", "base_url", "token_url", "client_id", "oauth_scope", "sap_client"}
)
#: The credential fields Cayu Cloud's connectors take; an error names only these, since anything
#: else given as a field name may be a mistyped secret.
KNOWN_FIELDS = PLAIN_FIELDS | {
    "api_key",
    "private_key",
    "client_secret",
    "client_certificate",
    "client_key",
    "ca_bundle",
}


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
        help="Add a connection with its credential, or one that signs in.",
        description=(
            "Add a connection to CONNECTOR under --name. A developer names the Agents that may "
            "use it with --agent; an admin who names none makes it available to every Agent. "
            "A connector that signs in (slack) takes no credential and only an admin adds it: "
            "the command prints the link where you approve it."
        ),
    )
    add.add_argument("connector", help="The connector, from `cayu cloud connectors`.")
    add.add_argument("--name", required=True, help="The connection's name, such as team-anthropic.")
    _add_credential_arguments(add, required=False)
    add.add_argument(
        "--scope", action="append", help="A scope it may hold; repeat. None: all it can hold."
    )
    add.add_argument("--agent", action="append", help="An Agent that may use it; repeat for each.")
    _add_sign_in_output(add)

    authorize = actions.add_parser(
        "authorize",
        help="Start a connection's sign-in again (admins).",
        description=(
            "Start the sign-in of a connection that signs in again: to finish a pending one, "
            "retry one in error, or reconnect a ready one. Prints a new link where you approve it."
        ),
    )
    authorize.add_argument("name")
    _add_sign_in_output(authorize)

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

    check = actions.add_parser(
        "check",
        help="Test a connection and each service it reaches (admins).",
        description=(
            "Ask Cloud to test the connection now: that its credential signs in, and, for SAP, "
            "one small read of each service it holds. Prints each service's result. A rejected "
            "credential or certificate, or a host the install doesn't allow, sets the "
            "connection to error."
        ),
    )
    check.add_argument("name")
    check.add_argument(
        "--json", action="store_true", help="Print the JSON envelope instead of a table."
    )
    check.set_defaults(_cloud_render=render_check)

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
        help="Bind one of an Agent's slots to a connection, or its model slot to a model.",
        description=(
            "Bind the Agent's SLOT to CONNECTION, optionally to fewer of its scopes; or, with "
            "--model, bind its model slot SLOT to an approved model, which new sessions use."
        ),
    )
    bind.add_argument("slot")
    bind.add_argument("connection", nargs="?")
    bind.add_argument("--application", required=True, help="The Agent.")
    bind.add_argument("--scope", action="append", help="A scope the binding uses; repeat.")
    bind.add_argument(
        "--model", help="Bind the model slot SLOT to this approved model instead of a connection."
    )
    unbind = actions.add_parser(
        "unbind",
        help="Unbind one of an Agent's slots.",
        description=(
            "Unbind the Agent's SLOT: its calls through it stop at once. An admin can then "
            "remove the connection with `cayu cloud connections remove`. With --model, unbind "
            "its model slot SLOT: new sessions use the slot's default."
        ),
    )
    unbind.add_argument("slot")
    unbind.add_argument("--application", required=True, help="The Agent.")
    unbind.add_argument(
        "--model", action="store_true", help="Unbind the model slot SLOT, not a connection slot."
    )


def add_models_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    models = commands.add_parser(
        "models",
        help="List an Agent's model slots.",
        description=(
            "Read an Agent's model slots: the models it declares by name, and the approved "
            "model each one uses. Bind one with `cayu cloud connections bind SLOT --model MODEL`."
        ),
    )
    actions = models.add_subparsers(dest="models_command", required=True)
    listed = actions.add_parser(
        "list",
        help="List an Agent's model slots and approved models.",
        description=(
            "List the Agent's model slots: each one's model, whether it is bound or uses its "
            "default, whether the model is approved and whether the running Release declares "
            "the slot; then the approved models it can be bound to."
        ),
    )
    listed.add_argument("--application", required=True, help="The Agent.")
    listed.add_argument(
        "--json", action="store_true", help="Print the JSON envelope instead of a table."
    )
    listed.set_defaults(_cloud_render=render_models)


def _add_credential_arguments(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    given = parser.add_mutually_exclusive_group(required=required)
    given.add_argument("--key-file", type=Path, help="A file holding the API key.")
    given.add_argument(
        "--credential",
        action="append",
        metavar="FIELD=@PATH|FIELD=VALUE",
        help="One credential field; @PATH reads it from a file. Repeat for each field.",
    )


def _add_sign_in_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the JSON envelope, with the link as the result's authorization_url.",
    )
    parser.set_defaults(_cloud_render=render_sign_in)


def render_sign_in(outcome: Mapping[str, Any]) -> str | None:
    """A started sign-in as text: the link to open. Any other result has no text form."""

    result = outcome["result"]
    url = result.get("authorization_url")
    if not isinstance(url, str):
        return None
    opening = (
        f"{result['name']} keeps working until you sign in again. To reconnect it"
        if result.get("status") == "ready"
        else f"{result['name']} is waiting for its sign-in. To connect it"
    )
    return "\n".join(
        (
            f"{opening}, open this link in a browser where you're signed in to the Cayu Cloud "
            "portal as the same person, in the same Organization, that started it, and approve "
            "it there:",
            "",
            f"  {url}",
            "",
            "The link works once, within 10 minutes. For a new one: "
            f"cayu cloud connections authorize {result['name']}",
        )
    )


def render_models(outcome: Mapping[str, Any]) -> str:
    """An Agent's model slots as a table, then the models they can be bound to."""

    result = outcome["result"]
    lines = _table(
        ("SLOT", "MODEL", "SOURCE", "APPROVED", "DECLARED"),
        [
            (
                str(item.get("slot")),
                item.get("model") or "-",
                str(item.get("source")),
                _yes_no(item.get("approved")),
                _yes_no(item.get("declared")),
            )
            for item in result.get("items") or []
        ],
    )
    approved = result.get("approved_models")
    lines.append("")
    lines.append(
        "Approved models: " + ("unknown" if approved is None else ", ".join(approved) or "none")
    )
    return "\n".join(lines)


#: A connection check's own failures in words, for a Cayu Cloud that doesn't say them itself
#: (`check_message`).
_CHECK_ERRORS = {
    "host_unreachable": "The check couldn't reach the SAP host.",
    "token_url_unreachable": "The check couldn't reach the token URL.",
    "auth_failed": "SAP rejected the connection's credential; an admin should check or rotate it.",
    "certificate_rejected": (
        "SAP rejected the connection's client certificate; an admin should check or rotate it."
    ),
    "host_not_allowed": (
        "This Cayu Cloud doesn't allow the connection's host; an admin should fix base_url or "
        "ask the install's operator to allow it."
    ),
}
#: The check's failures that leave the connection's status as it was; every other one sets it
#: to error.
_TRANSIENT_CHECK_ERRORS = frozenset({"host_unreachable", "token_url_unreachable"})


def render_check(outcome: Mapping[str, Any]) -> str:
    """A connection check as text: the connection's status, why the check itself couldn't
    finish when it couldn't, then one row per service."""

    result = outcome["result"]
    error = result.get("error")
    if isinstance(error, Mapping):
        error = error.get("message") or error.get("code")
    heading = f"{result.get('name')}: {result.get('status') or 'unknown'}"
    lines = [heading + (f" ({error})" if error else "")]
    check_error = result.get("check_error")
    if check_error:
        said = result.get("check_message") or _CHECK_ERRORS.get(check_error, "The check failed.")
        then = (
            " The connection's status is unchanged; try again later."
            if check_error in _TRANSIENT_CHECK_ERRORS
            else ""
        )
        lines.append(f"{said} ({check_error}){then}")
    services = result.get("services") or []
    if services:
        lines.append("")
        lines.extend(
            _table(
                ("SERVICE", "RESULT", "DETAIL"),
                [
                    (
                        str(item.get("service")),
                        str(item.get("result")),
                        str(item.get("message") or item.get("status") or "-"),
                    )
                    for item in services
                ],
            )
        )
    return "\n".join(lines)


def _yes_no(value: object) -> str:
    return "unknown" if value is None else "yes" if value else "no"


def models_command(
    arguments: argparse.Namespace, *, client: CloudApiClient, application_id: str
) -> dict[str, Any]:
    """Run one `models` action on the Agent ``application_id``."""

    result = client.request("GET", _models(application_id))
    return {"operation": f"models.{arguments.models_command}", "result": result}


def _models(application_id: str) -> str:
    return f"/v1/applications/{quote(application_id, safe='')}/models"


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
        payload: dict[str, object] = {"connector": arguments.connector, "name": arguments.name}
        # None given: a connector that signs in, which Cloud answers with where to approve it.
        if arguments.key_file is not None or arguments.credential is not None:
            payload["credential"] = _credential(arguments)
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
        elif action == "authorize":
            result = client.request("POST", f"{path}/authorize")
        elif action == "check":
            result = client.request("POST", f"{path}/check")
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
    kind = "models" if action in {"bind", "unbind"} and arguments.model else "connections"
    return {"operation": f"{kind}.{action}", "result": result}


def _agent_action(
    arguments: argparse.Namespace, client: CloudApiClient, application_id: str | None
) -> dict[str, Any]:
    if application_id is None:
        raise ValueError("An Agent's connections need its application id.")
    if arguments.connections_command != "bound" and arguments.model:
        return _model_slot_action(arguments, client, application_id)
    if arguments.connections_command == "bind" and arguments.connection is None:
        raise ValueError("Bind SLOT to a CONNECTION, or its model slot with --model MODEL.")
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


def _model_slot_action(
    arguments: argparse.Namespace, client: CloudApiClient, application_id: str
) -> dict[str, Any]:
    slot = f"{_models(application_id)}/{quote(arguments.slot, safe='')}"
    if arguments.connections_command == "bind":
        if arguments.connection is not None or arguments.scope is not None:
            raise ValueError("A model slot binds to --model MODEL alone: no CONNECTION or --scope.")
        return client.request("PUT", slot, payload={"model": arguments.model})
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
