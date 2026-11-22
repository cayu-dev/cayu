"""`cayu grants`: the tool authorization grants an application has given.

A grant is an approval a person asked to be remembered (`cayu inbox approve --remember`):
later calls of the same kind run without asking, except the ones sampled for review. These
commands list an application's grants, show one, list the calls it decided, and revoke it.
They read the project's own application in this process, or a Cayu Cloud Agent
(``--application``), through the same client as `cayu inbox`. Who may see or revoke a grant
is the application's decision. See ``docs/tool-authorization-grants.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from cayu.approvals.authorization_grants import TOOL_AUTHORIZATION_GRANT_MAX_LIST
from cayu.cli._inbox_client import InboxClient, InboxClientError
from cayu.cli._output import output_destination
from cayu.cli.inbox import (
    _add_target_options,
    _inbox,
    _limit,
    _plain,
    _print_json,
    _print_table,
)
from cayu.cli.project import ProjectError

_STATUSES = ("staged", "active", "abandoned", "revoked")
#: The revoke route's bound on a reason.
_REASON_MAX_CHARS = 2048
_DEFAULT_LIMIT = 100


def add_grants_parser(subparsers: Any) -> None:
    """Register ``cayu grants``."""

    grants = subparsers.add_parser(
        "grants",
        help="List, inspect and revoke remembered approvals.",
        description=(
            "A tool authorization grant is an approval a person asked to be remembered "
            "(`cayu inbox approve --remember`): later calls of the same kind run without "
            "asking. List an application's grants, show one, list the calls it decided, and "
            "revoke it. Without --application it opens the project's application in this "
            "process; with --application NAME it asks that Cayu Cloud Agent with your "
            "`cayu cloud login`. The application must host its grants "
            "(app.host_tool_authorization_grants), and decides who may see or revoke one."
        ),
    )
    commands = grants.add_subparsers(dest="grants_command", required=True)

    listing = commands.add_parser(
        "list",
        help="List the grants you may see, oldest first.",
        description=(
            "List the grants you may see, oldest first, one page at a time. When more are "
            "left it prints where the next page starts; pass those to --after-created-at "
            "and --after-grant-id with the same filters."
        ),
    )
    listing.add_argument(
        "--status",
        action="append",
        choices=_STATUSES,
        help="Only grants with this stored status. Repeat it for several.",
    )
    listing.add_argument("--tool", metavar="NAME", help="Only grants for this tool.")
    listing.add_argument(
        "--limit",
        type=_limit(TOOL_AUTHORIZATION_GRANT_MAX_LIST),
        default=_DEFAULT_LIMIT,
        help=(
            f"At most this many grants (1 to {TOOL_AUTHORIZATION_GRANT_MAX_LIST}, "
            f"default {_DEFAULT_LIMIT})."
        ),
    )
    listing.add_argument(
        "--after-created-at",
        metavar="TIME",
        help="Go on from a page: the after_created_at it printed.",
    )
    listing.add_argument(
        "--after-grant-id",
        metavar="GRANT_ID",
        help="Go on from a page: the after_grant_id it printed.",
    )
    _add_target_options(listing)

    show = commands.add_parser(
        "show",
        help="Show one grant.",
        description=(
            "Show one grant: the tool and gate it covers, what a call must match, where it "
            "applies, who created it and when it ends. Argument values the originating "
            "approval didn't allow to be published show as redacted."
        ),
    )
    show.add_argument("grant_id", metavar="GRANT_ID", help="The grant's id.")
    _add_target_options(show)

    uses = commands.add_parser(
        "uses",
        help="List the calls a grant decided.",
        description=(
            "List the calls a grant decided, oldest first: each call it allowed, and each it "
            "sent to a person for sampled review. When more are left it prints the cursor "
            "of the next page."
        ),
    )
    uses.add_argument("grant_id", metavar="GRANT_ID", help="The grant's id.")
    uses.add_argument(
        "--limit",
        type=_limit(TOOL_AUTHORIZATION_GRANT_MAX_LIST),
        default=_DEFAULT_LIMIT,
        help=(
            f"At most this many uses (1 to {TOOL_AUTHORIZATION_GRANT_MAX_LIST}, "
            f"default {_DEFAULT_LIMIT})."
        ),
    )
    uses.add_argument("--cursor", help="Go on from a page: the next_cursor it printed.")
    _add_target_options(uses)

    revoke = commands.add_parser(
        "revoke",
        help="Revoke a grant.",
        description=(
            "Revoke a grant: calls it covered ask for approval again from the next one. "
            "Revoking a grant that is already revoked answers the first revocation."
        ),
    )
    revoke.add_argument("grant_id", metavar="GRANT_ID", help="The grant's id.")
    revoke.add_argument(
        "--reason",
        help=f"Why, recorded on the grant (at most {_REASON_MAX_CHARS} characters).",
    )
    _add_target_options(revoke)


def run_grants(args: argparse.Namespace) -> int:
    try:
        with output_destination(args.output):
            return asyncio.run(_run(args))
    except OSError as exc:
        print(f"error: could not write output: {exc}", file=sys.stderr)
        return 1


async def _run(args: argparse.Namespace) -> int:
    from cayu.cli._cloud_api import CloudApiError
    from cayu.cli._cloud_auth import CloudAuthError
    from cayu.cli.cloud import CloudCommandError

    try:
        if args.application is not None and args.reviewer is not None:
            raise ValueError("--reviewer applies locally; a Cloud Agent records your login.")
        if args.as_subject is not None:
            raise ValueError(
                "--as reads a person's own inbox, which holds no grants. Name who you are "
                "with --reviewer instead."
            )
        if (args.grants_command == "list") and (
            (args.after_created_at is None) != (args.after_grant_id is None)
        ):
            raise ValueError("Give --after-created-at and --after-grant-id together.")
        reason = getattr(args, "reason", None)
        if reason is not None and not 0 < len(reason) <= _REASON_MAX_CHARS:
            raise ValueError(f"--reason must be 1 to {_REASON_MAX_CHARS} characters.")
        async with _inbox(args, command="cayu grants") as inbox:
            if args.grants_command == "list":
                return await _list(args, inbox)
            if args.grants_command == "show":
                return await _show(args, inbox)
            if args.grants_command == "uses":
                return await _uses(args, inbox)
            return await _revoke(args, inbox)
    except InboxClientError as exc:
        _error(args, exc.code, str(exc))
    except (CloudApiError, CloudAuthError, CloudCommandError) as exc:
        _error(args, exc.category, str(exc))
    except ProjectError as exc:
        _error(args, "project_unavailable", str(exc))
    except ValueError as exc:
        _error(args, "invalid_input", str(exc))
    return 1


def _error(args: argparse.Namespace, code: str, message: str) -> None:
    if args.output_format == "json":
        _print_json({"error": {"code": code, "message": message}})
        return
    print(f"error: {_plain(message)}", file=sys.stderr)


def _mapping(value: Any) -> dict[str, Any]:
    """A JSON object of an answer, or an empty one where the answer has something else."""

    return value if isinstance(value, dict) else {}


def _views(answer: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        view
        for view in answer.get("grants", [])
        if isinstance(view, dict) and isinstance(view.get("grant"), dict)
    ]


async def _list(args: argparse.Namespace, inbox: InboxClient) -> int:
    query: dict[str, Any] = {"limit": args.limit}
    if args.status:
        query["status"] = list(dict.fromkeys(args.status))
    if args.tool is not None:
        query["tool_name"] = args.tool
    if args.after_grant_id is not None:
        query["after_created_at"] = args.after_created_at
        query["after_grant_id"] = args.after_grant_id
    answer = await inbox.grants(query)
    views = _views(answer)
    following = answer.get("next")
    if not isinstance(following, dict):
        following = None
    if args.output_format == "json":
        _print_json({"where": inbox.where, "grants": views, "next": following})
        return 0
    if not views:
        print("No grants." if following is None else "No grants on this page.")
    else:
        _print_table(
            ("GRANT", "STATUS", "TOOL", "SCOPE", "CREATED BY", "CREATED", "EXPIRES", "SAMPLED"),
            [_row(view) for view in views],
        )
    if following is not None:
        print(
            "More grants are left. Go on with --after-created-at "
            f"{_plain(following.get('after_created_at'))} --after-grant-id "
            f"{_plain(following.get('after_grant_id'))}"
        )
    return 0


def _row(view: dict[str, Any]) -> dict[str, Any]:
    grant = _mapping(view.get("grant"))
    scope = _mapping(grant.get("scope"))
    return {
        "GRANT": grant.get("grant_id"),
        "STATUS": view.get("effective_status"),
        "TOOL": grant.get("tool_name"),
        "SCOPE": f"{scope.get('kind')} {scope.get('id')}",
        "CREATED BY": _subject(grant.get("created_by")),
        "CREATED": grant.get("created_at"),
        "EXPIRES": grant.get("expires_at") or "never",
        "SAMPLED": _sampled(grant.get("sampled_review_rate_bps")),
    }


def _subject(actor: Any) -> Any:
    return _mapping(actor).get("subject")


def _sampled(rate: object) -> str:
    """A sampled-review rate in basis points, as the share of calls it sends to a person."""

    if type(rate) is not int:
        return "-"
    return f"{rate / 100:g}%"


async def _show(args: argparse.Namespace, inbox: InboxClient) -> int:
    view = await inbox.grant(args.grant_id)
    if args.output_format == "json":
        _print_json({"where": inbox.where, **view})
        return 0
    _print_grant(view)
    return 0


def _print_grant(view: dict[str, Any]) -> None:
    grant = _mapping(view.get("grant"))
    scope = _mapping(grant.get("scope"))
    boundary = _mapping(grant.get("boundary"))
    matcher = _mapping(grant.get("matcher"))
    lines: list[tuple[str, object]] = [
        ("Grant", grant.get("grant_id")),
        ("Status", view.get("effective_status")),
        ("Tool", grant.get("tool_name")),
        ("Approval gate", grant.get("approval_gate")),
        ("Policy", grant.get("policy_id")),
        ("Scope", f"{scope.get('kind')} {scope.get('id')}"),
        ("Agent", boundary.get("agent_name")),
        ("Environment", boundary.get("environment_name")),
        ("Workspace", boundary.get("workspace_id")),
        ("Matches", _matcher(matcher)),
        ("Sampled review", _sampled(grant.get("sampled_review_rate_bps"))),
        ("Created", f"{grant.get('created_at')} by {_subject(grant.get('created_by'))}"),
        ("Expires", grant.get("expires_at") or "never"),
        ("Justification", grant.get("justification")),
    ]
    if grant.get("revoked_at") is not None:
        lines.append(
            ("Revoked", f"{grant.get('revoked_at')} by {_subject(grant.get('revoked_by'))}")
        )
        lines.append(("Revocation reason", grant.get("revocation_reason")))
    if grant.get("abandoned_at") is not None:
        lines.append(("Abandoned", grant.get("abandoned_at")))
        lines.append(("Abandonment reason", grant.get("abandonment_reason")))
    width = max(len(label) for label, _ in lines)
    for label, value in lines:
        print(f"{label + ':':<{width + 1}} {_plain(value)}")


def _matcher(matcher: dict[str, Any]) -> str:
    """What a call must match, in one line. Values are the application's public view."""

    if matcher.get("kind") == "process_command":
        argv = matcher.get("argv")
        command = " ".join(str(part) for part in argv) if isinstance(argv, list) else str(argv)
        where = matcher.get("cwd")
        return f"exactly this command: {command}" + (f" (in {where})" if where else "")
    if matcher.get("kind") == "tool_call":
        pinned = matcher.get("pinned_arguments")
        if isinstance(pinned, dict) and pinned:
            return "any call with " + ", ".join(
                f"{path} = {value!r}" for path, value in sorted(pinned.items())
            )
        return "any call to the tool under this gate"
    return str(matcher.get("kind"))


async def _uses(args: argparse.Namespace, inbox: InboxClient) -> int:
    answer = await inbox.grant_uses(args.grant_id, limit=args.limit, cursor=args.cursor)
    uses = [use for use in answer.get("uses", []) if isinstance(use, dict)]
    cursor = answer.get("next_cursor")
    if not isinstance(cursor, str):
        cursor = None
    if args.output_format == "json":
        _print_json(
            {"where": inbox.where, "grant_id": args.grant_id, "uses": uses, "next_cursor": cursor}
        )
        return 0
    if not uses:
        print("No uses." if cursor is None and args.cursor is None else "No uses on this page.")
    else:
        _print_table(
            ("RECORDED", "DECISION", "SESSION", "TOOL CALL"),
            [
                {
                    "RECORDED": use.get("recorded_at"),
                    "DECISION": use.get("decision"),
                    "SESSION": use.get("session_id"),
                    "TOOL CALL": use.get("tool_call_id"),
                }
                for use in uses
            ],
        )
    if cursor is not None:
        print(f"More uses are left. Go on with --cursor {_plain(cursor)}")
    return 0


async def _revoke(args: argparse.Namespace, inbox: InboxClient) -> int:
    view = await inbox.revoke_grant(args.grant_id, reason=args.reason)
    grant = _mapping(view.get("grant"))
    if view.get("effective_status") != "revoked":
        raise InboxClientError(
            "not_revoked",
            f"The {'Agent' if inbox.where == 'cloud' else 'application'} answered without "
            f"revoking this grant; it is {_plain(view.get('effective_status'))}.",
        )
    if args.output_format == "json":
        _print_json({"where": inbox.where, **view})
        return 0
    who = _subject(grant.get("revoked_by"))
    print(f"Revoked {_plain(grant.get('grant_id'))}" + (f" as {_plain(who)}." if who else "."))
    return 0
