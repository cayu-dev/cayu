"""`cayu cloud access-requests submit`: ask for access to Cayu Cloud.

It asks for access for someone who isn't in an Organization yet, so it sends no credential
(Cloud serves `POST /v1/access-requests` to anyone) and doesn't need `cayu cloud login`. Cloud
records the request; a Cayu platform administrator approves or rejects it.

It acts in no Organization, so `--organization` is refused and `CAYU_CLOUD_ORGANIZATION`
ignored for it (`_COMMANDS_OUTSIDE_AN_ORGANIZATION`).
"""

from __future__ import annotations

import argparse
from typing import Any

from cayu.cli._cloud_api import request_without_credentials

# Cloud's bounds on what a request carries.
_MAX_EMAIL = 320
_MAX_NAME = 200
_MAX_MESSAGE = 2000


def add_access_requests_parser(
    commands: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    access = commands.add_parser(
        "access-requests",
        help="Ask for access to Cayu Cloud.",
        description=(
            "Ask for access to Cayu Cloud for someone who isn't in an Organization yet. Prints "
            "one JSON document."
        ),
    )
    actions = access.add_subparsers(dest="access_requests_command", required=True)

    submit = actions.add_parser(
        "submit",
        help="Ask for access to Cayu Cloud.",
        description=(
            "Ask for access to Cayu Cloud. Needs no sign-in: the request goes to the Cayu "
            "Cloud the CLI is set up for (CAYU_CLOUD_API_URL or the selected context, else "
            "https://cloud.cayu.dev), and a platform administrator approves or rejects it."
        ),
    )
    submit.add_argument("--name", required=True, help="Your name.")
    submit.add_argument("--email", required=True, help="Your work email, where the invite goes.")
    submit.add_argument("--company", required=True, help="Your company or team.")
    submit.add_argument("--message", help="Anything whoever reviews the request should know.")


def submit_access_request(
    arguments: argparse.Namespace, *, api_url: str, timeout_seconds: float
) -> dict[str, Any]:
    """Send `access-requests submit` to the Cloud at ``api_url``, without a credential."""

    email = _bounded(arguments.email, "--email", _MAX_EMAIL)
    if "@" not in email or email.startswith("@") or email.endswith("@"):
        raise ValueError("--email must be a work email address, such as ada@example.com.")
    payload: dict[str, object] = {
        "email": email,
        "name": _bounded(arguments.name, "--name", _MAX_NAME),
        "organization_name": _bounded(arguments.company, "--company", _MAX_NAME),
    }
    if arguments.message is not None:
        payload["message"] = _bounded(arguments.message, "--message", _MAX_MESSAGE)
    result = request_without_credentials(
        api_url,
        "POST",
        "/v1/access-requests",
        payload=payload,
        timeout_seconds=timeout_seconds,
    )
    return {"operation": "access-requests.submit", "result": result}


def _bounded(value: str, flag: str, limit: int) -> str:
    text = value.strip()
    if not text:
        raise ValueError(f"{flag} must not be empty.")
    if len(text) > limit:
        raise ValueError(f"{flag} must be at most {limit} characters.")
    return text
