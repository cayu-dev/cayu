"""`cayu cloud inbox`: what Agents are waiting on a person for, across an Organization's Agents.

Each command calls Cayu Cloud's inbox routes with the `cayu cloud login` (or a selected
context's API key) and prints one JSON document. `list` reads every Agent the caller may see
(`GET /v1/inbox`), or one Agent's own inbox with `--application`; a decision is carried to the
Agent as the signed-in person (`POST /v1/applications/{id}/inbox/{item_id}/settle`), and
`decisions` lists Cloud's records of the decisions it carried and what each Agent answered.

Deciding needs a person: an Organization API key can list but not decide. Decisions take
`--idempotency-key` (`_cloud_writes`), and a refused one carries the item's current state when
the Agent gave it.

`cayu inbox --application NAME` (`cayu.cli.inbox`) works with one Agent's inbox in more depth:
it reads an item's protected review and decides with it, which an Agent with a human-review
policy needs, and it pages an inbox of any size. These commands send no review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_service_logs import log_time
from cayu.cli._cloud_writes import (
    CloudWriteError,
    add_idempotency_key_argument,
    send_write,
    write_key,
)
from cayu.cli._inbox_client import DECISION_FILTERS_UNSUPPORTED, DECISION_OUTCOMES

#: The decisions an item takes, each a command of its own.
DECISIONS = ("approve", "deny", "answer", "dismiss", "keep", "drop")
#: The most decision records Cloud answers with in one page.
_DECISIONS_PAGE_LIMIT = 100
_APPLICATION_HELP = "The Agent's name, slug or id."


def add_inbox_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `inbox` to `cayu cloud`."""

    inbox = commands.add_parser(
        "inbox",
        help="What your Agents are waiting on a person for, and deciding it.",
        description=(
            "What the Agents you may see are waiting on a person for: approvals, questions, "
            "recovery items and knowledge proposals, across your Organization. Decide an item "
            "as yourself, and read the decisions Cayu Cloud carried and what each Agent "
            "answered. To review an approval's arguments and decide with that review, which "
            "an Agent with a human-review policy needs, use `cayu inbox --application NAME`. "
            "Every command prints one JSON document."
        ),
    )
    actions = inbox.add_subparsers(dest="inbox_command", required=True)

    listing = actions.add_parser(
        "list",
        help="Everything waiting across the Agents you may see, oldest first.",
        description=(
            "Everything waiting across the active Agents you may see, oldest first, with "
            "`agents` saying which Agents were read and why any weren't. A sleeping Agent is "
            "read only when one of its sessions waits for a person. With --application, that "
            "one Agent's own inbox, read even while it sleeps (which wakes it)."
        ),
    )
    listing.add_argument("--application", help=_APPLICATION_HELP)

    show = actions.add_parser(
        "show",
        help="One item's state, wherever it was settled.",
        description=(
            "One item's state: pending with the item, settled with who decided and when, "
            "superseded, or unavailable."
        ),
    )
    show.add_argument("item_id")
    show.add_argument("--application", required=True, help=_APPLICATION_HELP)

    for name, what in (
        ("approve", "Approve an approval."),
        ("deny", "Deny an approval."),
        ("answer", "Answer a question."),
        ("dismiss", "Dismiss a recovery item whose session won't continue, abandoning it."),
        ("keep", "Keep a knowledge proposal."),
        ("drop", "Drop a knowledge proposal."),
    ):
        decide = actions.add_parser(
            name,
            help=what,
            description=(
                f"{what} Cayu Cloud carries the decision to the Agent as you, and the Agent "
                "records you as the person who decided. It needs a signed-in person, not an "
                "API key, and an Agent on Cayu Cloud sign-in."
            ),
        )
        decide.add_argument("item_id")
        decide.add_argument("--application", required=True, help=_APPLICATION_HELP)
        if name in {"approve", "deny"}:
            decide.add_argument("--reason", help="Why, recorded on the approval.")
        if name == "dismiss":
            decide.add_argument(
                "--reason", required=True, help="Why, recorded with the session's abandonment."
            )
        if name == "approve":
            decide.add_argument(
                "--arguments-file",
                type=Path,
                help=(
                    "Approve with changes: a file holding a JSON object of the arguments the "
                    "tool runs with instead."
                ),
            )
        if name == "answer":
            text = decide.add_mutually_exclusive_group(required=True)
            text.add_argument("--answer", help="The answer.")
            text.add_argument("--answer-file", type=Path, help="Read the answer from this file.")
        add_idempotency_key_argument(decide)

    decisions = actions.add_parser(
        "decisions",
        help="The decisions Cayu Cloud carried and what each Agent answered, newest first.",
        description=(
            "Cayu Cloud's records of the decisions it carried to the Agents you may see, newest "
            "first: who decided what on which item and session, and what the Agent answered "
            "(settled, replayed, refused with its code, or failed when Cloud couldn't tell). "
            "One page at a time: go on with --cursor and the same filters."
        ),
    )
    decisions.add_argument("--application", help=f"Only this Agent's. {_APPLICATION_HELP}")
    decisions.add_argument(
        "--since",
        type=log_time,
        help="Asked at or after this time: a span back from now (90m, 12h, 2d) or ISO 8601.",
    )
    decisions.add_argument(
        "--until",
        type=log_time,
        help="Asked before this time: a span back from now or an ISO 8601 time.",
    )
    decisions.add_argument(
        "--outcome",
        action="append",
        choices=DECISION_OUTCOMES,
        help="Only this outcome; repeat it for several.",
    )
    decisions.add_argument(
        "--item", dest="item_id", metavar="ITEM_ID", help="Only decisions on this item."
    )
    decisions.add_argument(
        "--session",
        dest="session_id",
        metavar="SESSION_ID",
        help="Only decisions on this session's items, including ones asked again after a Release.",
    )
    decisions.add_argument(
        "--limit",
        type=_page_limit,
        help=f"At most this many (1 to {_DECISIONS_PAGE_LIMIT}; Cloud's default otherwise).",
    )
    decisions.add_argument(
        "--cursor", help="Go on from a page: the next_cursor it printed, with the same filters."
    )


def inbox_command(
    arguments: argparse.Namespace, *, client: CloudApiClient, application_id: str | None
) -> dict[str, Any]:
    """Run one `cayu cloud inbox` command and return its JSON answer.

    ``application_id`` is the Agent `--application` names, already resolved, or None.
    """

    action = arguments.inbox_command
    if action == "list":
        path = "/v1/inbox" if application_id is None else _agent(application_id)
        return _answer(action, client.request("GET", path))
    if action == "decisions":
        return _answer(action, _decisions(arguments, client, application_id))
    assert application_id is not None
    item = f"{_agent(application_id)}/{quote(arguments.item_id, safe='')}"
    if action == "show":
        return _answer(action, client.request("GET", item))
    key = write_key(arguments, f"inbox.{action}")
    try:
        result = send_write(client, "POST", f"{item}/settle", key=key, payload=_decision(arguments))
    except CloudWriteError as exc:
        # The Agent's refusal names the item's current state, and Cloud its record of the try.
        body = exc.body or {}
        exc.details.update(
            {name: body[name] for name in ("item", "decision") if isinstance(body.get(name), dict)}
        )
        raise
    return {**_answer(action, result), "idempotency_key": key}


def _decision(arguments: argparse.Namespace) -> dict[str, object]:
    payload: dict[str, object] = {"decision": arguments.inbox_command}
    if getattr(arguments, "reason", None) is not None:
        payload["reason"] = arguments.reason
    if getattr(arguments, "arguments_file", None) is not None:
        path = arguments.arguments_file
        try:
            changed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CloudApiError(
                "invalid_input", f"Could not read the arguments JSON in {path}."
            ) from exc
        if not isinstance(changed, dict):
            raise CloudApiError("invalid_input", f"The arguments in {path} must be a JSON object.")
        payload["arguments"] = changed
    if arguments.inbox_command == "answer":
        if arguments.answer_file is None:
            payload["answer"] = arguments.answer
        else:
            try:
                payload["answer"] = arguments.answer_file.read_text(encoding="utf-8")
            except OSError as exc:
                raise CloudApiError(
                    "invalid_input", f"Could not read the answer file {arguments.answer_file}."
                ) from exc
    return payload


def _decisions(
    arguments: argparse.Namespace, client: CloudApiClient, application_id: str | None
) -> dict[str, Any]:
    filters: dict[str, str | list[str]] = {
        name: value
        for name, value in (
            ("since", None if arguments.since is None else arguments.since.isoformat()),
            ("until", None if arguments.until is None else arguments.until.isoformat()),
            ("outcome", arguments.outcome),
            ("item_id", arguments.item_id),
            ("session_id", arguments.session_id),
        )
        if value is not None
    }
    query: dict[str, str | list[str]] = dict(filters)
    for name, value in (
        ("application_id", application_id),
        ("limit", None if arguments.limit is None else str(arguments.limit)),
        ("cursor", arguments.cursor),
    ):
        if value is not None:
            query[name] = value
    result = client.request("GET", "/v1/inbox/decisions", query=query or None)
    if filters and not isinstance(result.get("filters"), dict):
        # A Cloud that doesn't echo the filters ignored them, and listed every record.
        raise CloudApiError("decision_filters_unsupported", DECISION_FILTERS_UNSUPPORTED)
    return result


def _agent(application_id: str) -> str:
    return f"/v1/applications/{quote(application_id, safe='')}/inbox"


def _answer(operation: str, result: dict[str, Any]) -> dict[str, Any]:
    return {"operation": f"inbox.{operation}", "result": result}


def _page_limit(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a whole number") from exc
    if not 1 <= number <= _DECISIONS_PAGE_LIMIT:
        raise argparse.ArgumentTypeError(f"must be 1 to {_DECISIONS_PAGE_LIMIT}")
    return number
