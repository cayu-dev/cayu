"""`cayu cloud ai-connections`: the Organization's own model accounts and its AI setting.

An Organization runs its Console coding agent either on Cayu's AI or on a model account of its
own, an AI connection. Connecting one is a device sign-in: `connect` starts it and prints the
address and code a person signs in with, and `complete` finishes it once they have. The first
connection to finish becomes the default. `use` chooses a ready connection and model, and
`use-cayu` goes back to Cayu's AI.

Each command calls Cayu Cloud's routes (`/v1/ai-connections`, `/v1/ai-settings`) with the
`cayu cloud login` (or a selected context's API key) and prints one JSON document. Reading
needs a developer or an admin; changing needs an Organization admin. No command takes or prints
a credential: the account's sign-in happens at the provider, and Cloud keeps what it returns.
"""

from __future__ import annotations

import argparse
from typing import Any
from urllib.parse import quote

from cayu.cli._cloud_api import CloudApiClient

_CONNECTIONS = "/v1/ai-connections"
_SETTINGS = "/v1/ai-settings"
#: The model `use` chooses when none is named, the one a subscription connection serves.
DEFAULT_MODEL = "gpt-5.4"
#: The commands that change the Organization's connections or setting.
WRITES = ("connect", "complete", "default", "disconnect", "use", "use-cayu")


def add_ai_connections_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Add `ai-connections` to `cayu cloud`."""

    group = commands.add_parser(
        "ai-connections",
        help="Connect the Organization's own model account, and choose what its AI runs on.",
        description=(
            "Manage the Organization's AI connections (its own model accounts) and which one "
            "its Console coding agent runs on, or Cayu's AI. Changes need an Organization "
            "admin. Every command prints one JSON document."
        ),
    )
    actions = group.add_subparsers(dest="ai_connections_command", required=True)
    actions.add_parser(
        "list",
        help="The Organization's AI connections.",
        description="The Organization's AI connections: provider, status and which is default.",
    )
    actions.add_parser(
        "connect",
        help="Start connecting an OpenAI subscription.",
        description=(
            "Start connecting an OpenAI (ChatGPT) subscription. Prints the new connection, "
            "pending, with the verification_url and user_code a person signs in with; then "
            "run `complete` with the connection's id."
        ),
    )
    actions.add_parser(
        "usage",
        help="The tokens the Organization's AI has used.",
        description="The input, output and total tokens the Organization's AI has used.",
    )
    actions.add_parser(
        "settings",
        help="What the Organization's AI runs on.",
        description=(
            "What the Organization's AI runs on: mode `cayu` (Cayu's AI) or `byoai` (one of "
            "its connections), with the connection and model."
        ),
    )
    actions.add_parser(
        "use-cayu",
        help="Run the Organization's AI on Cayu's AI.",
        description="Run the Organization's AI on Cayu's AI instead of a connection.",
    )
    complete = actions.add_parser(
        "complete",
        help="Finish connecting, once the person has signed in.",
        description=(
            "Finish a connection `connect` started. Prints it `ready` once the person has "
            "signed in, or still `pending` until they have; run it again then."
        ),
    )
    default = actions.add_parser(
        "default",
        help="Make a ready connection the default.",
        description="Make a ready connection the Organization's default.",
    )
    disconnect = actions.add_parser(
        "disconnect",
        help="Remove a connection.",
        description="Remove a connection and the credential Cloud keeps for it.",
    )
    use = actions.add_parser(
        "use",
        help="Run the Organization's AI on a connection.",
        description="Run the Organization's AI on a ready connection, with this model.",
    )
    for command in (complete, default, disconnect, use):
        command.add_argument("connection_id", help="The connection's id, from `list`.")
    use.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"The model to run (default {DEFAULT_MODEL}); the connection must serve it.",
    )


def ai_connections_command(
    arguments: argparse.Namespace, *, client: CloudApiClient
) -> dict[str, Any]:
    """Run one `cayu cloud ai-connections` command and return its JSON answer."""

    action = arguments.ai_connections_command
    if action == "list":
        result = client.request("GET", _CONNECTIONS)
    elif action == "connect":
        result = client.request("POST", f"{_CONNECTIONS}/openai-subscription")
    elif action == "usage":
        result = client.request("GET", f"{_CONNECTIONS}/usage")
    elif action == "settings":
        result = client.request("GET", _SETTINGS)
    elif action == "use-cayu":
        result = client.request(
            "PUT", _SETTINGS, payload={"mode": "cayu", "connection_id": None, "model": None}
        )
    elif action == "use":
        result = client.request(
            "PUT",
            _SETTINGS,
            payload={
                "mode": "byoai",
                "connection_id": arguments.connection_id,
                "model": arguments.model,
            },
        )
    else:
        connection = f"{_CONNECTIONS}/{quote(arguments.connection_id, safe='')}"
        if action == "complete":
            result = client.request("POST", f"{connection}/complete")
        elif action == "default":
            result = client.request("PUT", f"{connection}/default")
        else:
            client.request("DELETE", connection)
            result = {"id": arguments.connection_id, "disconnected": True}
    return {"operation": f"ai-connections.{action}", "result": result}
