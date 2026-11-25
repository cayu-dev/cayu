"""`cayu cloud team` and `cayu cloud organizations create|rename|add-member`: who is in an
Organization, and who works with each of its Agents.

`team` acts in the Organization the command acts in (the login's, or `--organization`): its
members and their roles, its pending invitations, and each Agent's people, chosen from the
Organization's people. Reading the members and roles needs a signed-in member; invitations and
changes to members need a signed-in Organization admin, since Cloud refuses an API key there.

`organizations create|rename|add-member` are a Cayu platform administrator's: each names the
Organization it changes by its id, so it acts the same whichever Organization the login acts in.
`add-member` adds someone who already has a Cayu Cloud account, with no invitation email.

Every command prints one JSON document.
"""

from __future__ import annotations

import argparse
from typing import Any
from urllib.parse import quote

from cayu.cli._cloud_agent_homes import ORGANIZATION_ID
from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._output import terminal_safe

#: The `organizations` commands a platform administrator runs, which name their Organization.
ORGANIZATION_CHANGES = ("create", "rename", "add-member")
#: The `team` commands that change the Organization's members or invitations.
TEAM_CHANGES = ("invite", "revoke", "set-role", "remove")
#: The `team` commands that need an Organization admin; `members` and `roles` need a member.
_ADMIN_ACTIONS = frozenset({"invitations", *TEAM_CHANGES})
#: The `team` commands about the people who work with Agents, which an API key may run.
_AGENT_PEOPLE = frozenset({"people", "agent-people", "set-agent-people"})
_MEMBERS = "/v1/organization/members"
_INVITATIONS = "/v1/organization/invitations"
_MAX_MEMBERS_PAGE = 100


def add_organization_changes_parsers(
    actions: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Add `create`, `rename` and `add-member` beside `organizations use`."""

    create = actions.add_parser(
        "create",
        help="Create an Organization (Cayu platform administrators).",
        description=(
            "Create a new Cayu Cloud Organization named --name. Needs a signed-in Cayu "
            "platform administrator. Add its first admin with `organizations add-member`."
        ),
    )
    create.add_argument("--name", required=True, help="The Organization's name.")
    rename = actions.add_parser(
        "rename",
        help="Rename an Organization (Cayu platform administrators).",
        description=(
            "Give ORG_ID the name people see when they sign in. Needs a signed-in Cayu "
            "platform administrator, who needn't be a member of ORG_ID."
        ),
    )
    rename.add_argument("organization_id", metavar="ORG_ID", type=_organization_id)
    rename.add_argument("--name", required=True, help="The Organization's new name.")
    add_member = actions.add_parser(
        "add-member",
        help="Add someone with a Cayu Cloud account to an Organization (platform administrators).",
        description=(
            "Add the person whose verified Cayu Cloud account has --email to ORG_ID with "
            "--role, at once and with no invitation email; a pending invitation of theirs is "
            "revoked. Someone without an account is refused: invite them with `cayu cloud team "
            "invite` instead. Adding someone already a member with that role answers their "
            "membership. Needs a signed-in Cayu platform administrator, who needn't be a member "
            "of ORG_ID."
        ),
    )
    add_member.add_argument("organization_id", metavar="ORG_ID", type=_organization_id)
    add_member.add_argument("--email", required=True, help="Their Cayu Cloud account's email.")
    add_member.add_argument(
        "--role", required=True, help="Their role's slug, such as admin, developer or member."
    )


def organization_changes_command(
    arguments: argparse.Namespace, *, client: CloudApiClient
) -> dict[str, Any]:
    """Run `organizations create|rename|add-member` and return its JSON answer."""

    action = arguments.organization_command
    try:
        if action == "create":
            result = client.request(
                "POST", "/v1/admin/organizations", payload={"name": arguments.name}
            )
        else:
            path = f"/v1/admin/organizations/{quote(arguments.organization_id, safe='')}"
            if action == "rename":
                result = client.request("PATCH", path, payload={"name": arguments.name})
            else:
                result = client.request(
                    "POST",
                    f"{path}/members",
                    payload={"email": arguments.email, "role": arguments.role},
                )
    except CloudApiError as exc:
        if exc.status_code == 403 and exc.code is None:
            raise CloudApiError(
                "platform_admin_required",
                f"`cayu cloud organizations {action}` needs a signed-in Cayu platform "
                "administrator (`cayu cloud login`).",
                status_code=403,
            ) from None
        raise _with_reason(exc) from None
    return {"operation": f"organizations.{action}", "result": result}


def add_team_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    team = commands.add_parser(
        "team",
        help="The Organization's members, invitations and each Agent's people.",
        description=(
            "List and change who is in the Organization: its members and their roles, and its "
            "pending invitations. Choose each Agent's people, who see it, open it and get its "
            "approvals, from the Organization's people. Invitations and member changes need a "
            "signed-in Organization admin (`cayu cloud login`)."
        ),
    )
    actions = team.add_subparsers(dest="team_command", required=True)

    members = actions.add_parser(
        "members",
        help="The Organization's members and their roles.",
        description=(
            "One page of the Organization's members, with each one's role and membership id. "
            "Pass the answer's next_cursor as --cursor for the next page."
        ),
    )
    members.add_argument(
        "--limit",
        type=_page_size,
        default=_MAX_MEMBERS_PAGE,
        help=f"Members per page, 1 to {_MAX_MEMBERS_PAGE} (default {_MAX_MEMBERS_PAGE}).",
    )
    members.add_argument("--cursor", help="The next_cursor of the previous page.")
    actions.add_parser(
        "roles",
        help="The roles a member can have.",
        description="The Organization's roles, by the slug `invite` and `set-role` take.",
    )
    actions.add_parser(
        "invitations",
        help="The Organization's invitations.",
        description="The Organization's invitations, pending and past (Organization admins).",
    )
    invite = actions.add_parser(
        "invite",
        help="Invite someone by email.",
        description=(
            "Email an invitation to join the Organization with --role. Inviting an email that "
            "already has a pending invitation with that role answers that invitation."
        ),
    )
    invite.add_argument("--email", required=True, help="Who to invite.")
    invite.add_argument("--role", required=True, help="Their role's slug, from `team roles`.")
    actions.add_parser(
        "revoke",
        help="Revoke a pending invitation.",
        description="Revoke a pending invitation; revoking a revoked one again answers it.",
    ).add_argument("invitation_id", help="The invitation's id, from `team invitations`.")
    set_role = actions.add_parser(
        "set-role",
        help="Change a member's role.",
        description=(
            "Give a member another role. You can't change your own, and the Organization "
            "always keeps an admin."
        ),
    )
    set_role.add_argument("user_id", help="The member's user id, from `team members`.")
    set_role.add_argument("--role", required=True, help="The role's slug, from `team roles`.")
    remove = actions.add_parser(
        "remove",
        help="Remove a member from the Organization.",
        description=(
            "Remove a member from the Organization; they leave its Agents too. You can't "
            "remove yourself, and the Organization always keeps an admin. Where Cloud also "
            "retires the member's Gateway access it needs --membership-id, which makes sure "
            "the membership removed is the one you listed."
        ),
    )
    remove.add_argument("user_id", help="The member's user id, from `team members`.")
    remove.add_argument(
        "--membership-id", help="The membership's id from `team members`; refused once changed."
    )

    actions.add_parser(
        "people",
        help="Everyone who can work with the Organization's Agents.",
        description="Everyone in the Organization who can work with an Agent, with their role.",
    )
    actions.add_parser(
        "agent-people",
        help="Who works with an Agent.",
        description=(
            "The people who work with the Agent (they see it, open it and get its approvals), "
            "and whether you may change them (can_choose)."
        ),
    ).add_argument("--application", required=True, help="The Agent's name, slug or id.")
    set_people = actions.add_parser(
        "set-agent-people",
        help="Choose who works with an Agent.",
        description=(
            "Make exactly these people the Agent's people, replacing the ones it has. Each "
            "must be in the Organization (`team people`); anyone who isn't is refused with "
            "person_unknown and nothing changes."
        ),
    )
    set_people.add_argument("--application", required=True, help="The Agent's name, slug or id.")
    chosen = set_people.add_mutually_exclusive_group(required=True)
    chosen.add_argument(
        "--user-id",
        dest="user_ids",
        action="append",
        help="Someone who works with the Agent, from `team people`; repeat for each.",
    )
    chosen.add_argument(
        "--no-one",
        action="store_true",
        help="Leave the Agent with no people of its own.",
    )


def team_command(
    arguments: argparse.Namespace, *, client: CloudApiClient, application_id: str | None
) -> dict[str, Any]:
    """Run one `cayu cloud team` command and return its JSON answer.

    ``application_id`` is the Agent `--application` names, already resolved, or None.
    """

    action = arguments.team_command
    try:
        result = _team_request(arguments, client=client, application_id=application_id)
    except CloudApiError as exc:
        # Cloud's own refusals carry a code; a bare 403 is its sign-in or admin check.
        if exc.status_code == 403 and exc.code is None and action not in _AGENT_PEOPLE:
            needed = (
                "a signed-in Organization admin"
                if action in _ADMIN_ACTIONS
                else "a signed-in member of the Organization"
            )
            raise CloudApiError(
                "organization_admin_required" if action in _ADMIN_ACTIONS else "login_required",
                f"`cayu cloud team {action}` needs {needed} (`cayu cloud login`); an "
                "Organization API key can't.",
                status_code=403,
            ) from None
        raise _with_reason(exc) from None
    return {"operation": f"team.{action}", "result": result}


def _with_reason(exc: CloudApiError) -> CloudApiError:
    """``exc``, saying why Cloud refused: who `person_unknown` names, or Cloud's own reason
    for a missing member, invitation or Organization or a refused role or email."""

    if exc.code == "person_unknown" and exc.body is not None:
        unknown = exc.body.get("user_ids")
        if isinstance(unknown, list) and unknown and all(isinstance(u, str) for u in unknown):
            names = ", ".join(terminal_safe(user_id) for user_id in unknown[:20])
            return _reworded(exc, f"{exc} Not in the Organization: {names}.")
        return exc
    if exc.code is None and exc.status_code in {404, 422} and exc.detail is not None:
        return _reworded(exc, f"{exc} Cayu Cloud said: {exc.detail}")
    return exc


def _reworded(exc: CloudApiError, message: str) -> CloudApiError:
    return CloudApiError(
        exc.category,
        message,
        status_code=exc.status_code,
        detail=exc.detail,
        code=exc.code,
        rejected_fields=exc.rejected_fields,
        body=exc.body,
        request=exc.request,
        validation_messages=exc.validation_messages,
    )


def _team_request(
    arguments: argparse.Namespace, *, client: CloudApiClient, application_id: str | None
) -> dict[str, Any]:
    action = arguments.team_command
    if action == "members":
        query = {"limit": str(arguments.limit)}
        if arguments.cursor:
            query["cursor"] = arguments.cursor
        return client.request("GET", _MEMBERS, query=query)
    if action == "roles":
        return client.request("GET", "/v1/organization/roles")
    if action == "invitations":
        return client.request("GET", _INVITATIONS)
    if action == "invite":
        payload = {"email": arguments.email, "role": arguments.role}
        return client.request("POST", _INVITATIONS, payload=payload)
    if action == "revoke":
        path = f"{_INVITATIONS}/{quote(arguments.invitation_id, safe='')}"
        client.request("DELETE", path)
        return {"invitation_id": arguments.invitation_id, "status": "revoked"}
    if action == "set-role":
        member = f"{_MEMBERS}/{quote(arguments.user_id, safe='')}"
        return client.request("PUT", f"{member}/role", payload={"role": arguments.role})
    if action == "remove":
        member = f"{_MEMBERS}/{quote(arguments.user_id, safe='')}"
        query = (
            None if arguments.membership_id is None else {"membership_id": arguments.membership_id}
        )
        client.request("DELETE", member, query=query)
        return {"user_id": arguments.user_id, "status": "removed"}
    if action == "people":
        return client.request("GET", "/v1/organization/people")
    assert application_id is not None
    path = f"/v1/applications/{quote(application_id, safe='')}/people"
    if action == "agent-people":
        return client.request("GET", path)
    user_ids = [] if arguments.no_one else list(dict.fromkeys(arguments.user_ids))
    return client.request("PUT", path, payload={"user_ids": user_ids})


def _organization_id(value: str) -> str:
    if ORGANIZATION_ID.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be an Organization id such as org_01...")
    return value


def _page_size(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a whole number") from exc
    if not 1 <= number <= _MAX_MEMBERS_PAGE:
        raise argparse.ArgumentTypeError(f"must be 1 to {_MAX_MEMBERS_PAGE}")
    return number
