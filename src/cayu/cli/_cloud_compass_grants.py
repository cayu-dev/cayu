"""`cayu cloud compass grants`: the grants Compass proposes from accepted findings.

An accepted autonomy-grant finding says people approve every call of one gated tool as it
is proposed. These commands turn it into a grant proposal and record each decision on it,
through Cloud's `/v1/compass/agents/{agent}/grant-proposals` routes. Compass gives no grant
itself: an approved proposal carries the `grant_request` the Agent's owner approves the
tool's next pending call with (`cayu inbox approve --remember`).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

from cayu.cli._cloud_api import CloudApiClient, CloudApiError

STATUSES = ("proposed", "approved", "rejected", "live", "revoke_recommended", "revoked")
# The longest a proposed grant may last: 366 days.
_MAX_GRANT_SECONDS = 366 * 24 * 60 * 60
# Each decision, what it records, and whether it needs a reason.
_DECISIONS = (
    ("approve", "Approve a proposed grant; the answer carries its grant_request.", False),
    ("reject", "Reject a proposed grant, saying why.", True),
    (
        "revoked",
        "Record that the owner revoked the grant on the Agent, as Compass recommended.",
        False,
    ),
    ("keep", "Keep a grant Compass recommends revoking, saying why.", True),
)
WRITES = ("propose", *(name for name, _what, _reason in _DECISIONS))

# Sends one POST: the operation's name, its path and its body.
Write = Callable[[str, str, dict[str, object]], dict[str, Any]]


def _whole_number(minimum: int, maximum: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError("must be a whole number") from None
        if not minimum <= number <= maximum:
            raise argparse.ArgumentTypeError(f"must be from {minimum} to {maximum}")
        return number

    return parse


def add_grants_parser(
    actions: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    application: Callable[[str], str],
) -> None:
    """Add `grants` to `cayu cloud compass`."""

    grants = actions.add_parser(
        "grants",
        description=(
            "The grants Compass proposes from accepted autonomy-grant findings, and the "
            "decisions on them. Compass gives no grant itself: the Agent's owner approves "
            "the tool's next pending call with an approved proposal's grant_request."
        ),
    )
    grant_actions = grants.add_subparsers(dest="grants_command", required=True)

    listing = grant_actions.add_parser(
        "list", description="The Agent's grant proposals, newest first."
    )
    listing.add_argument("application", type=application)
    listing.add_argument(
        "--status",
        action="append",
        choices=STATUSES,
        dest="statuses",
        help="Only proposals in this status; repeat for several.",
    )
    listing.add_argument("--agent", help="Only this watched agent's proposals.")

    show = grant_actions.add_parser(
        "show", description="One grant proposal, by its id or by the finding it came from."
    )
    show.add_argument("application", type=application)
    show.add_argument("proposal_id", nargs="?")
    show.add_argument("--finding", metavar="FINDING_ID", help="The proposal of this finding.")

    propose = grant_actions.add_parser(
        "propose",
        description=(
            "Ask Compass to turn an accepted autonomy-grant finding into a grant proposal. "
            "Options left out take Compass's defaults."
        ),
    )
    propose.add_argument("application", type=application)
    propose.add_argument("finding_id")
    propose.add_argument(
        "--sample-rate-bps",
        type=_whole_number(0, 10_000),
        metavar="N",
        help=(
            "How many of the granted calls still go to a person, in hundredths of a percent "
            "(default: the rate the Agent's objectives declare)."
        ),
    )
    propose.add_argument(
        "--scope",
        choices=("workspace", "session"),
        help="Where the grant applies (default workspace).",
    )
    expiry = propose.add_mutually_exclusive_group()
    expiry.add_argument(
        "--expires-in-seconds",
        type=_whole_number(1, _MAX_GRANT_SECONDS),
        metavar="SECONDS",
        help="How long the grant lasts once given (default 30 days).",
    )
    expiry.add_argument(
        "--no-expiry", action="store_true", help="Propose a grant that doesn't expire."
    )
    propose.add_argument(
        "--revoke-threshold",
        metavar="SHARE",
        help=(
            "The share of sampled reviews people correct at which Compass recommends "
            'revoking, above 0 and at most 1 (default "0.05").'
        ),
    )
    propose.add_argument("--idempotency-key")

    for name, what, needs_reason in _DECISIONS:
        decision = grant_actions.add_parser(name, description=what)
        decision.add_argument("application", type=application)
        decision.add_argument("proposal_id")
        decision.add_argument("--note", required=needs_reason)
        decision.add_argument("--idempotency-key")


def unavailable(exc: CloudApiError) -> CloudApiError | None:
    """What a bare 404 means here: an older Cloud's missing route. Cloud's own codes say more."""

    if exc.status_code != 404 or exc.code is not None:
        return None
    return CloudApiError(
        "compass_grants_unavailable",
        "No such finding or grant proposal, or this Cayu Cloud has no Compass grant proposals yet.",
        status_code=404,
    )


def changed_proposal(exc: CloudApiError) -> dict[str, str] | None:
    """The status a proposal is in now, from a decision Cloud refused as `proposal_changed`."""

    proposal = (exc.body or {}).get("proposal") if exc.code == "proposal_changed" else None
    status = proposal.get("status") if isinstance(proposal, dict) else None
    return {"status": status} if status in STATUSES else None


def _read(
    client: CloudApiClient, path: str, query: dict[str, str | list[str]] | None = None
) -> dict[str, Any]:
    try:
        return client.request("GET", path, query=query)
    except CloudApiError as exc:
        raise unavailable(exc) or exc from None


def grants_command(
    client: CloudApiClient,
    arguments: argparse.Namespace,
    path: str,
    *,
    answer: Callable[[str, dict[str, Any]], dict[str, Any]],
    write: Write,
) -> dict[str, Any]:
    """Run one `grants` command and return its JSON answer.

    ``answer`` wraps what a read returned; ``write`` sends a POST under the command's
    idempotency key and returns the whole answer, key included.
    """

    action = arguments.grants_command
    operation = f"grants.{action}"
    proposals = f"{path}/grant-proposals"
    if action == "list":
        query: dict[str, str | list[str]] = {}
        if arguments.statuses:
            query["status"] = list(dict.fromkeys(arguments.statuses))
        if arguments.agent is not None:
            query["agent"] = arguments.agent
        return answer(operation, _read(client, proposals, query or None))
    if action == "show":
        if (arguments.proposal_id is None) == (arguments.finding is None):
            raise ValueError("Give PROPOSAL_ID or --finding, one of them.")
        if arguments.finding is not None:
            finding = quote(arguments.finding, safe="")
            return answer(operation, _read(client, f"{path}/findings/{finding}/grant-proposal"))
        return answer(
            operation, _read(client, f"{proposals}/{quote(arguments.proposal_id, safe='')}")
        )
    if action == "propose":
        proposal: dict[str, object] = {}
        if arguments.sample_rate_bps is not None:
            proposal["sampled_review_rate_bps"] = arguments.sample_rate_bps
        if arguments.scope is not None:
            proposal["scope"] = arguments.scope
        if arguments.no_expiry:
            proposal["expires_in_seconds"] = None
        elif arguments.expires_in_seconds is not None:
            proposal["expires_in_seconds"] = arguments.expires_in_seconds
        if arguments.revoke_threshold is not None:
            proposal["revoke_threshold"] = arguments.revoke_threshold
        finding = quote(arguments.finding_id, safe="")
        return write(operation, f"{path}/findings/{finding}/propose-grant", proposal)
    decision: dict[str, object] = {} if arguments.note is None else {"note": arguments.note}
    proposal_path = f"{proposals}/{quote(arguments.proposal_id, safe='')}"
    return write(operation, f"{proposal_path}/{action}", decision)
