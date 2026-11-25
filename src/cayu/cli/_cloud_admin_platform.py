"""`cayu cloud admin`'s commands on Cayu Cloud's platform administrator routes.

They act across Organizations, through `/v1/admin/...`, and Cloud answers them only for a
signed-in Cayu platform administrator; anyone else, an Organization admin or an Organization API
key included, gets `platform_admin_required`:

- `platform-admin list|grant|revoke`: who the install's platform administrators are;
- `access-requests list|get|approve|reject`: read and decide requests for access to the install;
- `organizations list|agents`: the install's Organizations and one Organization's Agents;
- `applications move`: move an Agent to another Organization;
- `agent-lease show|confirm-exit`: an Agent's lifecycle leases and the work started under them;
- `archive blocked|show|confirm-exit`: archives that wait on work Cloud can't see end;
- `connections move-model-credentials`: move model connections' credentials an earlier version
  kept with the app's into the model credential store.

Commands that name another Organization's Agent take its exact id and `--organization`, the
Organization that owns it, as `admin runtime-record` does. Each prints one JSON document.
"""

from __future__ import annotations

import argparse
import dataclasses
import shlex
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import quote

from cayu.cli._cloud_api import CloudApiClient, CloudApiError

# A move waits a bounded time for the Gateway's worker and restarts the Agent in one call.
_MOVE_TIMEOUT_SECONDS = 120.0
_ARCHIVE_BLOCKED_LIMIT = 100
_ACCESS_REQUEST_PAGE_LIMIT = 100
ACCESS_REQUEST_STATUSES = ("pending", "provisioning", "invited", "accepted", "rejected", "failed")


def add_platform_admin_parsers(
    actions: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    application: Callable[[str], str],
) -> None:
    """Add the platform administrator groups to `cayu cloud admin`; ``application`` checks an
    Agent slug argument."""

    _add_platform_admin(actions)
    _add_access_requests(actions)
    _add_organizations(actions)
    _add_applications(actions, application=application)
    _add_agent_lease(actions, application=application)
    _add_archive(actions, application=application)
    _add_connections(actions)


def _owned_agent(parser: argparse.ArgumentParser, application: Callable[[str], str]) -> None:
    parser.add_argument("application", type=application, help="The exact Agent slug.")
    parser.add_argument(
        "--organization", required=True, help="The Organization that owns the Agent."
    )


def _add_platform_admin(actions: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    group = actions.add_parser(
        "platform-admin",
        help="List, grant or revoke the install's platform administrators.",
        description=(
            "List the install's Cayu platform administrators, or grant or revoke a person's "
            "platform administrator access by their user id."
        ),
    )
    commands = group.add_subparsers(dest="platform_admin_command", required=True)
    commands.add_parser(
        "list",
        help="List the platform administrators.",
        description=(
            "List the platform administrators: each one's user id, name and email where Cloud "
            "can read them, who granted the access and when, and whether the install's "
            "configuration names them (`configured`), which only a change to it revokes."
        ),
    )
    grant = commands.add_parser(
        "grant",
        help="Make a person a platform administrator.",
        description="Make the person with this user id a platform administrator.",
    )
    grant.add_argument("user_id", help="The person's user id.")
    revoke = commands.add_parser(
        "revoke",
        help="Take a person's platform administrator access away.",
        description=(
            "Take platform administrator access from the person with this user id. Cloud "
            "refuses to revoke your own access or one the install's configuration names."
        ),
    )
    revoke.add_argument("user_id", help="The person's user id.")


def _add_access_requests(actions: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    group = actions.add_parser(
        "access-requests",
        help="List, read, approve or reject requests for access to the install.",
        description=(
            "Read the requests for access to this Cayu Cloud install and decide them: approve "
            "one into an existing Organization or a new one, or reject it."
        ),
    )
    commands = group.add_subparsers(dest="access_requests_command", required=True)
    listed = commands.add_parser(
        "list",
        help="List access requests, newest first.",
        description=(
            "List access requests, newest first: every one, or with --limit a page at a time "
            "(pass the printed next_cursor as --cursor for the next page)."
        ),
    )
    listed.add_argument("--status", choices=ACCESS_REQUEST_STATUSES, help="Only these.")
    listed.add_argument("--limit", type=_page_limit, help="At most this many (1-100).")
    listed.add_argument("--cursor", help="The next_cursor a previous page printed.")
    get = commands.add_parser(
        "get", help="Show one access request.", description="Show one access request."
    )
    get.add_argument("request_id", help="The access request's id.")
    approve = commands.add_parser(
        "approve",
        help="Approve a request, into an existing Organization or a new one.",
        description=(
            "Approve the request: invite the person into the Organization --organization "
            "names, with --role, or create an Organization named --new-organization for them."
        ),
    )
    approve.add_argument("request_id", help="The access request's id.")
    admission = approve.add_mutually_exclusive_group(required=True)
    admission.add_argument("--organization", help="Admit them to this existing Organization.")
    admission.add_argument(
        "--new-organization", metavar="NAME", help="Create an Organization with this name."
    )
    approve.add_argument(
        "--role",
        choices=("admin", "member"),
        default="member",
        help="Their role in the existing Organization (default: member).",
    )
    reject = commands.add_parser(
        "reject", help="Reject a request.", description="Reject the access request."
    )
    reject.add_argument("request_id", help="The access request's id.")


def _add_organizations(actions: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    group = actions.add_parser(
        "organizations",
        help="List the install's Organizations, or one Organization's Agents.",
        description=(
            "Read the install's Organizations across the whole install, such as to choose the "
            "one to admit an access request to."
        ),
    )
    commands = group.add_subparsers(dest="organizations_command", required=True)
    listed = commands.add_parser(
        "list",
        help="List the install's Organizations, with how many Agents each has.",
        description=(
            "List the install's Organizations, each with its id, name and Agent count; "
            "--query keeps those whose name matches."
        ),
    )
    listed.add_argument("--query", help="Only Organizations whose name matches this.")
    agents = commands.add_parser(
        "agents",
        help="List one Organization's Agents, archived ones included.",
        description="List the Organization's Agents, archived ones included, with each lifecycle.",
    )
    agents.add_argument("organization_id", help="The Organization's id.")


def _add_applications(
    actions: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    application: Callable[[str], str],
) -> None:
    group = actions.add_parser(
        "applications",
        help="Move an Agent to another Organization.",
        description="Work on Agents across Organizations.",
    )
    commands = group.add_subparsers(dest="applications_command", required=True)
    move = commands.add_parser(
        "move",
        help="Move an Agent to another Organization, keeping its slug, URL, data and settings.",
        description=(
            "Move the Agent to another Organization, keeping its slug, URL, data, settings, "
            "secrets and schedules. Cloud refuses while anything still runs that the move "
            "would cut off, and says what. Moving it again to the same Organization finishes "
            "a move whose runtime step is pending, and otherwise changes nothing."
        ),
    )
    move.add_argument("application", type=application, help="The exact Agent slug.")
    move.add_argument(
        "--to-organization", required=True, metavar="ORG_ID", help="The Organization to move to."
    )


def _add_agent_lease(
    actions: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    application: Callable[[str], str],
) -> None:
    group = actions.add_parser(
        "agent-lease",
        help="Show an Agent's lifecycle leases and their work, or confirm that work ended.",
        description=(
            "Read the lifecycle leases an Agent's publications and service changes hold, and "
            "the platform work (Jobs, tasks) started under them; confirm, with a reason Cloud "
            "records, that work Cloud can't see end has ended."
        ),
    )
    commands = group.add_subparsers(dest="agent_lease_command", required=True)
    show = commands.add_parser(
        "show",
        help="Show the Agent's live leases and the work started under them.",
        description="Show the Agent's live lifecycle leases and the work started under them.",
    )
    _owned_agent(show, application)
    confirm = commands.add_parser(
        "confirm-exit",
        help="Confirm that work the runtime no longer knows has ended.",
        description=(
            "Confirm, with a reason Cloud records, that the work --ref names has ended, so the "
            "lease it holds can be released. Cloud refuses while the platform still describes "
            "the work as running."
        ),
    )
    _owned_agent(confirm, application)
    confirm.add_argument("--ref", required=True, help="The work's ref, as `show` printed it.")
    confirm.add_argument("--reason", required=True, help="Why you know it ended.")
    confirm.add_argument(
        "--outside-runtime",
        action="store_true",
        help="The runtime can't answer for this work; you checked the platform directly.",
    )


def _add_archive(
    actions: argparse._SubParsersAction[argparse.ArgumentParser],
    *,
    application: Callable[[str], str],
) -> None:
    group = actions.add_parser(
        "archive",
        help="List blocked archives, show one, or confirm that what blocks it has ended.",
        description=(
            "Read the archives that wait on work Cloud can't see end, in every Organization, "
            "and confirm, with a reason Cloud records, that a task or runtime lease blocking "
            "one has ended."
        ),
    )
    commands = group.add_subparsers(dest="archive_command", required=True)
    blocked = commands.add_parser(
        "blocked",
        help="List blocked archives in every Organization.",
        description="List blocked archives in every Organization, most recently updated first.",
    )
    blocked.add_argument(
        "--limit",
        type=_archive_limit,
        default=_ARCHIVE_BLOCKED_LIMIT,
        help=f"At most this many (1-{_ARCHIVE_BLOCKED_LIMIT}; default {_ARCHIVE_BLOCKED_LIMIT}).",
    )
    show = commands.add_parser(
        "show",
        help="Show an Agent's archive and what blocks it.",
        description="Show the Agent's archive operation and what blocks it.",
    )
    _owned_agent(show, application)
    confirm = commands.add_parser(
        "confirm-exit",
        help="Confirm that a task or runtime lease blocking the archive has ended.",
        description=(
            "Confirm, with a reason Cloud records, that the task or runtime lease blocking the "
            "Agent's archive has ended. Cloud refuses when it doesn't block the archive or the "
            "platform still describes it as running."
        ),
    )
    _owned_agent(confirm, application)
    confirm.add_argument(
        "--kind",
        dest="subject_kind",
        required=True,
        choices=("agent_task", "runtime_lease"),
        help="What blocks the archive.",
    )
    confirm.add_argument("--subject", required=True, help="Its id, as `show` printed it.")
    confirm.add_argument("--reason", required=True, help="Why you know it ended.")


def _add_connections(actions: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    group = actions.add_parser(
        "connections",
        help="Move model connections' credentials to the model credential store.",
        description="Maintain the install's connections across Organizations.",
    )
    commands = group.add_subparsers(dest="connections_command", required=True)
    commands.add_parser(
        "move-model-credentials",
        help="Move model connections' credentials kept with the app's into the model store.",
        description=(
            "Move the credentials of model connections that an earlier Cloud kept with the "
            "app's credentials into the model credential store. Prints how many, never a "
            "credential; run again, it moves nothing."
        ),
    )


def _page_limit(value: str) -> int:
    try:
        limit = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a whole number") from None
    if not 1 <= limit <= _ACCESS_REQUEST_PAGE_LIMIT:
        raise argparse.ArgumentTypeError(f"must be from 1 to {_ACCESS_REQUEST_PAGE_LIMIT}")
    return limit


def _archive_limit(value: str) -> int:
    try:
        limit = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a whole number") from None
    if not 1 <= limit <= _ARCHIVE_BLOCKED_LIMIT:
        raise argparse.ArgumentTypeError(f"must be from 1 to {_ARCHIVE_BLOCKED_LIMIT}")
    return limit


PLATFORM_ADMIN_GROUPS = frozenset(
    {
        "platform-admin",
        "access-requests",
        "organizations",
        "applications",
        "agent-lease",
        "archive",
        "connections",
    }
)


def platform_admin_command(
    arguments: argparse.Namespace,
    *,
    client: CloudApiClient,
    recovery_arguments: Sequence[str] = (),
) -> dict[str, Any]:
    """Run one of `PLATFORM_ADMIN_GROUPS`; ``recovery_arguments`` are the command's own
    ``--context`` and ``--api-key-file``, which a follow-up command it prints keeps."""

    group = arguments.admin_command
    if group == "platform-admin":
        return _platform_admin(arguments, client)
    if group == "access-requests":
        return _access_requests(arguments, client)
    if group == "organizations":
        return _organizations(arguments, client)
    if group == "applications":
        return _move(arguments, client, recovery_arguments)
    if group == "connections":
        result = admin_request(
            client,
            "POST",
            "/v1/admin/connections/model-credentials/move",
            what="Moving model connections' credentials",
            route_is_fixed=True,
        )
        return {"operation": "admin.connections.move-model-credentials", "result": result}
    if group == "archive" and arguments.archive_command == "blocked":
        result = admin_request(
            client,
            "GET",
            "/v1/admin/archives",
            query={"limit": str(arguments.limit)},
            what="Listing blocked archives",
            route_is_fixed=True,
        )
        return {"operation": "admin.archive.blocked", "result": result}
    return _owned(arguments, client)


def _platform_admin(arguments: argparse.Namespace, client: CloudApiClient) -> dict[str, Any]:
    action = arguments.platform_admin_command
    what = "Changing who is a platform administrator"
    if action == "list":
        result = admin_request(
            client,
            "GET",
            "/v1/admin/platform-admins",
            what="Listing platform administrators",
            route_is_fixed=True,
        )
    else:
        admin_request(
            client,
            "PUT" if action == "grant" else "DELETE",
            f"/v1/admin/platform-admins/{_segment(arguments.user_id, 'user id')}",
            what=what,
        )
        result = {"user_id": arguments.user_id, "platform_admin": action == "grant"}
    return {"operation": f"admin.platform-admin.{action}", "result": result}


def _access_requests(arguments: argparse.Namespace, client: CloudApiClient) -> dict[str, Any]:
    action = arguments.access_requests_command
    if action == "list":
        query = {
            name: str(value)
            for name in ("status", "limit", "cursor")
            if (value := getattr(arguments, name)) is not None
        }
        result = admin_request(
            client,
            "GET",
            "/v1/admin/access-requests",
            query=query or None,
            what="Listing access requests",
            route_is_fixed=True,
        )
        return {"operation": "admin.access-requests.list", "result": result}
    if action == "get":
        result = admin_request(
            client,
            "GET",
            f"/v1/admin/access-requests/{_segment(arguments.request_id, 'request id')}",
            what="Reading an access request",
        )
        return {"operation": "admin.access-requests.get", "result": result}
    payload: dict[str, object]
    if action == "reject":
        payload = {"outcome": "rejected"}
    elif arguments.organization is not None:
        payload = {
            "outcome": "approved",
            "admission": {
                "kind": "existing",
                "organization_id": nonempty_value(arguments.organization, "--organization"),
                "role_slug": arguments.role,
            },
        }
    else:
        payload = {
            "outcome": "approved",
            "admission": {
                "kind": "new",
                "organization_name": nonempty_value(
                    arguments.new_organization, "--new-organization"
                ),
            },
        }
    result = admin_request(
        client,
        "PUT",
        f"/v1/admin/access-requests/{_segment(arguments.request_id, 'request id')}/decision",
        payload=payload,
        what="Deciding an access request",
    )
    return {"operation": f"admin.access-requests.{action}", "result": result}


def _organizations(arguments: argparse.Namespace, client: CloudApiClient) -> dict[str, Any]:
    action = arguments.organizations_command
    if action == "list":
        query = None if arguments.query is None else {"query": str(arguments.query)}
        result = admin_request(
            client,
            "GET",
            "/v1/admin/organizations",
            query=query,
            what="Listing the install's Organizations",
            route_is_fixed=True,
        )
    else:
        organization = _segment(arguments.organization_id, "Organization id")
        result = admin_request(
            client,
            "GET",
            f"/v1/admin/organizations/{organization}/applications",
            what="Listing an Organization's Agents",
        )
    return {"operation": f"admin.organizations.{action}", "result": result}


def _move(
    arguments: argparse.Namespace, client: CloudApiClient, recovery_arguments: Sequence[str]
) -> dict[str, Any]:
    organization = nonempty_value(arguments.to_organization, "--to-organization")
    # The move restarts the Agent in the same call, which takes longer than most requests.
    mover = dataclasses.replace(
        client, timeout_seconds=max(client.timeout_seconds, _MOVE_TIMEOUT_SECONDS)
    )
    result = admin_request(
        mover,
        "POST",
        f"/v1/admin/applications/{quote(str(arguments.application), safe='')}/move",
        payload={"organization_id": organization},
        what="Moving an Agent",
    )
    if result.get("status") == "runtime_pending":
        again = [
            "cayu",
            "cloud",
            *recovery_arguments,
            "admin",
            "applications",
            "move",
            str(arguments.application),
            "--to-organization",
            organization,
        ]
        result = {**result, "commands": {"finish_move": shlex.join(again)}}
    return {"operation": "admin.applications.move", "result": result}


def _owned(arguments: argparse.Namespace, client: CloudApiClient) -> dict[str, Any]:
    """`agent-lease` and `archive show|confirm-exit`, on another Organization's Agent."""

    group = arguments.admin_command
    resource = (
        f"/v1/admin/organizations/{_segment(arguments.organization, '--organization')}"
        f"/applications/{quote(str(arguments.application), safe='')}"
        f"/{'lease-work' if group == 'agent-lease' else 'archive'}"
    )
    action = arguments.agent_lease_command if group == "agent-lease" else arguments.archive_command
    if action == "show":
        result = admin_request(client, "GET", resource, what=f"Reading an Agent's {group}")
    elif group == "agent-lease":
        result = admin_request(
            client,
            "POST",
            f"{resource}/exit-confirmations",
            payload={
                "ref": nonempty_value(arguments.ref, "--ref"),
                "reason": nonempty_value(arguments.reason, "--reason"),
                "outside_runtime": bool(arguments.outside_runtime),
            },
            what="Confirming that an Agent's lease work ended",
        )
    else:
        result = admin_request(
            client,
            "POST",
            f"{resource}/exit-confirmations",
            payload={
                "subject_kind": arguments.subject_kind,
                "subject": nonempty_value(arguments.subject, "--subject"),
                "reason": nonempty_value(arguments.reason, "--reason"),
            },
            what="Confirming that what blocks an archive ended",
        )
    return {"operation": f"admin.{group}.{action}", "result": result}


def admin_request(
    client: CloudApiClient,
    method: str,
    path: str,
    *,
    what: str,
    payload: dict[str, object] | None = None,
    query: Mapping[str, str] | None = None,
    route_is_fixed: bool = False,
) -> dict[str, Any]:
    """Send one platform administrator request; ``what`` names it in a refusal.

    A ``route_is_fixed`` route takes no path parameters, so a 404 without one of Cloud's codes
    means the route itself is missing, as a 405 does on any route.
    """

    try:
        return client.request(method, path, payload=payload, query=query)
    except CloudApiError as exc:
        if exc.status_code == 405 or (
            route_is_fixed and exc.status_code == 404 and exc.code is None
        ):
            raise CloudApiError(
                "admin_command_unsupported",
                f"This Cayu Cloud doesn't serve this command yet ({what.lower()}).",
                status_code=exc.status_code,
            ) from None
        if exc.status_code == 403:
            raise CloudApiError(
                "platform_admin_required",
                f"{what} needs a signed-in Cayu platform administrator, and the credential used "
                "isn't one (an Organization admin or an Organization API key can't). Sign in as "
                "a platform administrator with `cayu cloud login` and run it without a key file.",
                status_code=403,
            ) from None
        raise


def nonempty_value(value: object, flag: str) -> str:
    """``value`` as text, refused when blank; ``flag`` names it in the refusal."""

    text = str(value)
    if not text.strip():
        # An empty variable in a script must never send an empty id or reason.
        raise CloudApiError("invalid_input", f"{flag} must not be empty.")
    return text


def _segment(value: object, what: str) -> str:
    return quote(nonempty_value(value, what), safe="")
