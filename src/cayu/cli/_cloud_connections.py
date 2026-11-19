"""`cayu cloud connectors` and `cayu cloud connections`: an Organization's connections.

A connection is one of the Organization's authorized model credentials or app connectors
(`team-anthropic`, `warehouse`), which its Agents use through Cayu Cloud without ever holding
the credential. Developers and admins add connections and bind them to an Agent's slots; only an
admin changes which Agents may use one, rotates its credential or removes it.

Credentials are read from files, never from the command line: `--key-file PATH` for an API key,
`--credential FIELD=@PATH` for a field kept in a file (a private key) and `--credential
FIELD=VALUE` only for a one-line field that isn't secret (`account`, `user`, a GitHub App's
`app_id`, SAP's `base_url`), as Cloud's `GET /v1/connectors` describes each connector's fields.
No answer ever holds a credential. `connections check` asks Cloud to test a connection that can
be checked, such as SAP or a GitHub App, and prints each service's result. A GitHub App's binding names
the repositories its slot may call with `bind --repository OWNER/NAME` (or `OWNER/*`).

A connector that signs in (Slack) takes no credential: an admin adds it without one, or starts
its sign-in again with `connections authorize`, and Cloud answers the link where the person
approves it, which these commands print.

`connections activity` lists the calls Agents made through a connection, newest first, since
`--since` (a duration such as `7d` or a time with its offset; Cloud's default is the last 24
hours), a page at a time: the answer's `next_cursor` goes on with `--cursor`, given with the
`--since` instant the answer names so every page reads the same window.

`cayu cloud models list` shows an Agent's model slots; `connections bind SLOT --model MODEL` binds
one to an approved model and `connections unbind SLOT --model` unbinds it, so new sessions use
the slot's default.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_reference import APPLICATION_HELP
from cayu.cli._cloud_settings import _table

_CONNECTIONS = "/v1/connections"
#: The `connections` actions on one Agent's slots, which name the Agent with --application.
AGENT_ACTIONS = frozenset({"bound", "bind", "unbind"})
#: For a Cayu Cloud whose `GET /v1/connectors` doesn't describe each connector's fields (one from
#: before it did): the credential fields that aren't secret, which --credential may give as
#: FIELD=VALUE; every other field is read from a file (FIELD=@PATH).
PLAIN_FIELDS = frozenset(
    {
        "account",
        "user",
        "app_id",
        "base_url",
        "token_url",
        "client_id",
        "oauth_scope",
        "sap_client",
    }
)
#: For such a Cayu Cloud, the credential fields its connectors take; an error names only these,
#: since anything else given as a field name may be a mistyped secret.
KNOWN_FIELDS = PLAIN_FIELDS | {
    "api_key",
    "private_key",
    "client_secret",
    "client_certificate",
    "client_key",
    "ca_bundle",
}


#: Cloud's always-mounted answer to whether its connections are on (`{"enabled": bool}`).
CONNECTIONS_STATUS = "/v1/connections-status"
CONNECTIONS_DISABLED = "connections_disabled"
CONNECTIONS_DISABLED_MESSAGE = "Connections are turned off in this Cayu Cloud installation."
CONNECTIONS_UNAVAILABLE = (
    "This Cayu Cloud doesn't serve connections: they're turned off in this installation, or "
    "it predates them. `cayu cloud platform` shows which revision it runs."
)
ACTIVITY_PAGING_UNAVAILABLE = (
    "This Cayu Cloud doesn't page a connection's activity, so --cursor can't go on: list it "
    "without --cursor, narrowing it with --since and --limit."
)

#: The most calls one `connections activity` page asks for, as Cloud allows.
MAX_ACTIVITY_LIMIT = 500
_DURATION = re.compile(r"([1-9][0-9]{0,5})([mhd])")
_DURATION_UNITS = {"m": "minutes", "h": "hours", "d": "days"}


class CredentialArgumentError(ValueError):
    """A credential argument names no field, or its file can't be read."""


@dataclass(frozen=True)
class CredentialFields:
    """The credential fields a connection's connector takes, and those of them --credential may
    give as FIELD=VALUE."""

    known: Collection[str]
    plain: Collection[str]
    #: The fields that aren't secret but span lines (a PEM certificate), read from a file too.
    multiline: Collection[str] = ()
    #: The connector, where Cloud described its fields; None for the lists above.
    connector: str | None = None


_OLDER_CLOUD_FIELDS = CredentialFields(known=KNOWN_FIELDS, plain=PLAIN_FIELDS)


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
        help="Add, list, bind, change or remove the Organization's connections; read their calls.",
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

    activity = actions.add_parser(
        "activity",
        help="List the calls Agents made through a connection.",
        description=(
            "List the calls the Organization's Agents made through the connection, newest "
            "first: when, which Agent and slot, the method and path, the connector's status or "
            "why Cloud refused the call, and how long it took. Never a query, body or "
            "credential. Cloud keeps each call for its install's retention, 30 days by default."
        ),
    )
    activity.add_argument("name")
    activity.add_argument(
        "--since",
        type=_since,
        metavar="DURATION|TIME",
        help=(
            "Only calls since then: a duration back from now (30m, 24h, 7d) or a time with "
            "its offset (2026-11-19T09:00:00Z). Cloud's default is the last 24 hours."
        ),
    )
    activity.add_argument(
        "--limit",
        type=_activity_limit,
        metavar="N",
        help=f"At most N calls, 1-{MAX_ACTIVITY_LIMIT}. Cloud's default is 100.",
    )
    activity.add_argument(
        "--cursor",
        help="Go on after an earlier listing, from its next_cursor, with the --since it printed.",
    )
    activity.add_argument(
        "--json", action="store_true", help="Print the JSON envelope instead of a table."
    )
    activity.set_defaults(_cloud_render=render_activity)

    check = actions.add_parser(
        "check",
        help="Test a connection and each service it reaches (admins).",
        description=(
            "Ask Cloud to test the connection now: that its credential signs in (a GitHub App's "
            "key reads the App), and, for SAP, one small read of each service it holds, "
            "printing each service's result. A rejected credential or certificate, or a host "
            "the install doesn't allow, sets the connection to error."
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
    bound.add_argument("--application", required=True, help=APPLICATION_HELP)
    bind = actions.add_parser(
        "bind",
        help="Bind one of an Agent's slots to a connection, or its model slot to a model.",
        description=(
            "Bind the Agent's SLOT to CONNECTION, optionally to fewer of its scopes, and for a "
            "GitHub App to the repositories it may call; or, with --model, bind its model slot "
            "SLOT to an approved model, which new sessions use."
        ),
    )
    bind.add_argument("slot")
    bind.add_argument("connection", nargs="?")
    bind.add_argument("--application", required=True, help=APPLICATION_HELP)
    bind.add_argument("--scope", action="append", help="A scope the binding uses; repeat.")
    bind.add_argument(
        "--repository",
        action="append",
        metavar="OWNER/NAME",
        help=(
            "A repository a GitHub App binding may call, or OWNER/* for every one of an owner "
            "the App is installed on; repeat. A GitHub App binding needs at least one."
        ),
    )
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
    unbind.add_argument("--application", required=True, help=APPLICATION_HELP)
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
    listed.add_argument("--application", required=True, help=APPLICATION_HELP)
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


def render_activity(outcome: Mapping[str, Any]) -> str:
    """A connection's calls as a table, newest first, then how to read on where more follow."""

    result = outcome["result"]
    items = result.get("items") or []
    since = result.get("since")
    window = f" since {since}" if since else ""
    if not items:
        return f"No calls{window}."
    lines = _table(
        ("TIME", "AGENT", "SLOT", "REQUEST", "RESULT", "DURATION"),
        [
            (
                str(item.get("at")),
                _agent_cell(item),
                str(item.get("slot")),
                f"{item.get('method')} {item.get('path')}",
                str(item.get("status") if item.get("status") is not None else item.get("error")),
                f"{item.get('duration_ms')} ms",
            )
            for item in items
        ],
    )
    lines.extend(["", f"{len(items)} {'call' if len(items) == 1 else 'calls'}{window}."])
    cursor = result.get("next_cursor")
    if isinstance(cursor, str):
        # The window this page read, as an instant: a relative --since would move with the clock.
        then = f" --since {since}" if since else ""
        lines.append(f"More calls follow: pass --cursor {cursor}{then}.")
    return "\n".join(lines)


def _agent_cell(item: Mapping[str, Any]) -> str:
    """The Agent's name, else its id; a connection check's calls (slot `check`) have none."""

    if not item.get("application_id"):
        return "connection check"
    return str(item.get("application_name") or item.get("application_id"))


def _since(value: str) -> str:
    """`--since`: a duration back from now, or a time with its offset, as Cloud's RFC 3339."""

    duration = _DURATION.fullmatch(value)
    if duration:
        back = timedelta(**{_DURATION_UNITS[duration.group(2)]: int(duration.group(1))})
        return (datetime.now(UTC) - back).isoformat()
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        moment = None
    if moment is None or moment.tzinfo is None:
        raise argparse.ArgumentTypeError(
            "give a duration such as 30m, 24h or 7d, or a time with its offset, such as "
            "2026-11-19T09:00:00Z"
        )
    return moment.isoformat()


def _activity_limit(value: str) -> int:
    if not value.isdecimal() or not 1 <= int(value) <= MAX_ACTIVITY_LIMIT:
        raise argparse.ArgumentTypeError(f"must be a whole number from 1 to {MAX_ACTIVITY_LIMIT}")
    return int(value)


def _yes_no(value: object) -> str:
    return "unknown" if value is None else "yes" if value else "no"


def models_command(
    arguments: argparse.Namespace, *, client: CloudApiClient, application_id: str
) -> dict[str, Any]:
    """Run one `models` action on the Agent ``application_id``."""

    return _connections_served(
        client,
        lambda: {
            "operation": f"models.{arguments.models_command}",
            "result": client.request("GET", _models(application_id)),
        },
    )


def _models(application_id: str) -> str:
    return f"/v1/applications/{quote(application_id, safe='')}/models"


def connectors_command(client: CloudApiClient) -> dict[str, Any]:
    return _connections_served(
        client,
        lambda: {"operation": "connectors.list", "result": client.request("GET", "/v1/connectors")},
    )


def connections_command(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    application_id: str | None = None,
    agents: list[str] | None = None,
) -> dict[str, Any]:
    """Run one `connections` action. The caller resolves the Agents it names: the Agent actions'
    --application as ``application_id``, and `add` and `agents`' --agent values as ``agents``."""

    return _connections_served(
        client,
        lambda: _connections_action(
            arguments, client=client, application_id=application_id, agents=agents
        ),
    )


def _connections_served(
    client: CloudApiClient, request: Callable[[], dict[str, Any]]
) -> dict[str, Any]:
    """Answer ``request``, saying in words when this Cayu Cloud serves no connections.

    An install with connections turned off doesn't mount their routes, so they answer a
    bare 404: no code, only FastAPI's "Not Found". The always-mounted
    ``GET /v1/connections-status`` then says whether connections are off; a Cloud without
    it is older, and either off or from before connections. A refusal with one of Cloud's
    own codes, such as ``connection_not_found`` for a name it doesn't have, is passed on.
    """

    try:
        return request()
    except CloudApiError as exc:
        if not _route_missing(exc):
            raise
        enabled = _connections_enabled(client)
        if enabled is None:
            raise _command_error("connections_unavailable", CONNECTIONS_UNAVAILABLE) from None
        if not enabled:
            raise _command_error(CONNECTIONS_DISABLED, CONNECTIONS_DISABLED_MESSAGE) from None
        raise


def _route_missing(exc: CloudApiError) -> bool:
    return exc.status_code == 405 or (
        exc.status_code == 404 and exc.code is None and exc.detail in {None, "Not Found"}
    )


def _connections_enabled(client: CloudApiClient) -> bool | None:
    """Whether this Cloud has connections on, or None when it doesn't say (an older one)."""

    try:
        status = client.request("GET", CONNECTIONS_STATUS)
    except CloudApiError as exc:
        if _route_missing(exc):
            return None
        raise
    enabled = status.get("enabled") if isinstance(status, dict) else None
    return enabled if isinstance(enabled, bool) else None


def _command_error(category: str, message: str) -> Exception:
    from cayu.cli.cloud import CloudCommandError

    return CloudCommandError(category, message)


def _connections_action(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    application_id: str | None,
    agents: list[str] | None,
) -> dict[str, Any]:
    action = arguments.connections_command
    if action == "list":
        result = client.request("GET", _CONNECTIONS)
    elif action == "add":
        payload: dict[str, object] = {"connector": arguments.connector, "name": arguments.name}
        # None given: a connector that signs in, which Cloud answers with where to approve it.
        if arguments.key_file is not None or arguments.credential is not None:
            payload["credential"] = _credential(
                arguments, lambda: _credential_fields(client, arguments.connector)
            )
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
        elif action == "activity":
            query = {
                key: str(value)
                for key, value in (
                    ("since", arguments.since),
                    ("limit", arguments.limit),
                    ("cursor", arguments.cursor),
                )
                if value is not None
            }
            result = client.request("GET", f"{path}/activity", query=query)
            if arguments.cursor is not None and "next_cursor" not in result:
                # A Cayu Cloud from before paging ignores the cursor and answers the first page.
                raise _command_error("activity_paging_unavailable", ACTIVITY_PAGING_UNAVAILABLE)
        elif action == "agents":
            result = client.request(
                "PUT", f"{path}/agents", payload={"agents": None if arguments.all else agents}
            )
        elif action == "rotate":
            credential = _credential(
                arguments,
                lambda: _credential_fields(client, client.request("GET", path).get("connector")),
            )
            result = client.request("PUT", f"{path}/credential", payload={"credential": credential})
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
        if arguments.repository is not None:
            payload["repositories"] = arguments.repository
        return client.request("PUT", slot, payload=payload)
    client.request("DELETE", slot)
    return {"application_id": application_id, "slot": arguments.slot, "unbound": True}


def _model_slot_action(
    arguments: argparse.Namespace, client: CloudApiClient, application_id: str
) -> dict[str, Any]:
    slot = f"{_models(application_id)}/{quote(arguments.slot, safe='')}"
    if arguments.connections_command == "bind":
        if (
            arguments.connection is not None
            or arguments.scope is not None
            or arguments.repository is not None
        ):
            raise ValueError(
                "A model slot binds to --model MODEL alone: no CONNECTION, --scope or --repository."
            )
        return client.request("PUT", slot, payload={"model": arguments.model})
    client.request("DELETE", slot)
    return {"application_id": application_id, "slot": arguments.slot, "unbound": True}


def _credential_fields(client: CloudApiClient, connector: object) -> CredentialFields:
    """The fields ``connector`` takes, as Cloud describes them; the lists above where it
    doesn't (an older Cloud, or a connector it doesn't offer, which it then refuses)."""

    listed = client.request("GET", "/v1/connectors")
    items = listed.get("items") if isinstance(listed, dict) else None
    spec = next(
        (item for item in items or () if isinstance(item, dict) and item.get("id") == connector),
        None,
    )
    fields = None if spec is None else spec.get("fields")
    if not isinstance(fields, list) or not all(
        isinstance(item, dict) and isinstance(item.get("name"), str) for item in fields
    ):
        return _OLDER_CLOUD_FIELDS
    unsecret = [item for item in fields if item.get("secret") is False]
    return CredentialFields(
        known=tuple(item["name"] for item in fields),
        plain=frozenset(item["name"] for item in unsecret if item.get("multiline") is False),
        multiline=frozenset(item["name"] for item in unsecret if item.get("multiline") is True),
        connector=str(connector),
    )


def _credential(
    arguments: argparse.Namespace, described: Callable[[], CredentialFields]
) -> dict[str, str]:
    """The credential the arguments give; ``described`` tells which fields it may have, asked
    only for --credential."""

    if arguments.key_file is not None:
        return {"api_key": _read(arguments.key_file)}
    given: list[tuple[str, str]] = []
    for item in arguments.credential:
        name, separator, value = item.partition("=")
        # Never repeat a malformed argument: it may be a mistyped secret.
        if not separator or not name.isidentifier() or not value:
            raise CredentialArgumentError("A credential field is FIELD=@PATH or FIELD=VALUE.")
        given.append((name, value))
    takes = described()
    fields: dict[str, str] = {}
    for name, value in given:
        # Never repeat a field name the connector doesn't take: it may be a mistyped secret.
        if name not in takes.known and takes.connector is not None:
            raise CredentialArgumentError(
                f"A credential field isn't one the {takes.connector} connector takes: "
                f"{', '.join(takes.known) or 'it takes none'}."
            )
        shown = name if name in takes.known else "FIELD"
        if name in fields:
            raise CredentialArgumentError(f"The credential field {shown} is given twice.")
        if value.startswith("@"):
            fields[name] = _read(Path(value[1:]))
        elif name in takes.plain:
            fields[name] = value
        else:
            why = "multi-line" if name in takes.multiline else "secret"
            raise CredentialArgumentError(
                f"The credential field {shown} is {why}: give it from a file, {shown}=@PATH."
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
