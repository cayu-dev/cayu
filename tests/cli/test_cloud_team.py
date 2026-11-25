"""`cayu cloud team` and `cayu cloud organizations create|rename|add-member` over Cloud's
Organization membership routes."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from tests.cli.test_cloud_api_keys import cloud as _cloud

from cayu.cli import main

cloud = _cloud

_APPS = "/v1/applications?lifecycle=all"
_AGENTS = {"items": [{"id": "month-end-close", "name": "Month-end close"}]}
_MEMBER = {"user_id": "user_ada", "role": "developer", "membership_id": "om_1"}
_INVITATION = {"id": "inv_1", "email": "ada@example.com", "role": "member", "state": "pending"}


def _run(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    code = main(["cloud", *arguments])
    return code, json.loads(capsys.readouterr().out)


def _sent(fake: Any) -> tuple[str, str, object]:
    request = fake.requests[-1]
    body = json.loads(request.content) if request.content else None
    return request.method, request.url.raw_path.decode(), body


def test_help_lists_every_team_and_organization_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(["cloud", "team", "--help"])
    team_help = capsys.readouterr().out
    for command in (
        "members",
        "roles",
        "invitations",
        "invite",
        "revoke",
        "set-role",
        "remove",
        "people",
        "agent-people",
        "set-agent-people",
    ):
        assert command in team_help
    with pytest.raises(SystemExit):
        main(["cloud", "organizations", "--help"])
    organizations_help = capsys.readouterr().out
    for command in ("use", "create", "rename", "add-member"):
        assert command in organizations_help


def test_team_reads_need_no_organization_check(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    fake = cloud(
        {
            ("GET", "/v1/organization/members?limit=20&cursor=next-1"): httpx.Response(
                200, json={"items": [_MEMBER], "next_cursor": None}
            ),
            ("GET", "/v1/organization/members?limit=100"): httpx.Response(
                200, json={"items": [_MEMBER], "next_cursor": "next-1"}
            ),
            ("GET", "/v1/organization/roles"): httpx.Response(
                200, json={"items": [{"slug": "admin", "name": "Admin", "description": None}]}
            ),
            ("GET", "/v1/organization/invitations"): httpx.Response(
                200, json={"items": [_INVITATION]}
            ),
            ("GET", "/v1/organization/people"): httpx.Response(
                200, json={"items": [{"user_id": "user_ada", "role": "developer"}]}
            ),
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("GET", "/v1/applications/month-end-close/people"): httpx.Response(
                200, json={"items": [], "can_choose": True}
            ),
        }
    )

    for arguments, operation in (
        (["members"], "team.members"),
        (["members", "--limit", "20", "--cursor", "next-1"], "team.members"),
        (["roles"], "team.roles"),
        (["invitations"], "team.invitations"),
        (["people"], "team.people"),
        (["agent-people", "--application", "Month-end close"], "team.agent-people"),
    ):
        code, output = _run(["team", *arguments], capsys)
        assert (code, output["operation"]) == (0, operation), output
        assert "organization" not in output
    assert all(request.url.path != "/v1/me" for request in fake.requests)


def test_members_refuses_a_page_size_cloud_would_refuse(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, output = _run(["team", "members", "--limit", "101"], capsys)
    assert code == 2
    assert "must be 1 to 100" in output["error"]["message"]


def test_team_changes_say_where_they_act_and_send_what_cloud_takes(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("POST", "/v1/organization/invitations"): httpx.Response(201, json=_INVITATION),
            ("DELETE", "/v1/organization/invitations/inv_1"): httpx.Response(204),
            ("PUT", "/v1/organization/members/user_ada/role"): httpx.Response(
                200, json={**_MEMBER, "role": "admin"}
            ),
            ("DELETE", "/v1/organization/members/user_ada?membership_id=om_1"): httpx.Response(204),
            ("DELETE", "/v1/organization/members/user_bob"): httpx.Response(204),
        }
    )

    expected = (
        (
            ["invite", "--email", "ada@example.com", "--role", "member"],
            "team.invite",
            (
                "POST",
                "/v1/organization/invitations",
                {"email": "ada@example.com", "role": "member"},
            ),
            _INVITATION,
        ),
        (
            ["revoke", "inv_1"],
            "team.revoke",
            ("DELETE", "/v1/organization/invitations/inv_1", None),
            {"invitation_id": "inv_1", "status": "revoked"},
        ),
        (
            ["set-role", "user_ada", "--role", "admin"],
            "team.set-role",
            ("PUT", "/v1/organization/members/user_ada/role", {"role": "admin"}),
            {**_MEMBER, "role": "admin"},
        ),
        (
            ["remove", "user_ada", "--membership-id", "om_1"],
            "team.remove",
            ("DELETE", "/v1/organization/members/user_ada?membership_id=om_1", None),
            {"user_id": "user_ada", "status": "removed"},
        ),
        (
            ["remove", "user_bob"],
            "team.remove",
            ("DELETE", "/v1/organization/members/user_bob", None),
            {"user_id": "user_bob", "status": "removed"},
        ),
    )
    for arguments, operation, request, result in expected:
        code, output = _run(["team", *arguments], capsys)
        assert code == 0, output
        assert output["operation"] == operation
        assert output["organization"]["id"] == "org_test"
        assert output["result"] == result
        assert _sent(fake) == request


def test_a_team_change_is_refused_in_another_organization_than_it_names(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud({})

    code, output = _run(
        [
            "team",
            "invite",
            "--email",
            "ada@example.com",
            "--role",
            "member",
            "--organization",
            "org_other",
        ],
        capsys,
    )

    assert code != 0
    assert output["ok"] is False
    assert "org_other" in output["error"]["message"]
    assert fake.requests == []


def test_set_agent_people_resolves_the_agent_and_sends_exactly_the_people(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    path = "/v1/applications/month-end-close/people"
    fake = cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("PUT", path): httpx.Response(200, json={"items": [], "can_choose": True}),
        }
    )

    code, output = _run(
        [
            "team",
            "set-agent-people",
            "--application",
            "Month-end close",
            "--user-id",
            "user_ada",
            "--user-id",
            "user_bob",
            "--user-id",
            "user_ada",
        ],
        capsys,
    )
    assert code == 0, output
    assert output["operation"] == "team.set-agent-people"
    assert output["organization"]["id"] == "org_test"
    assert _sent(fake) == ("PUT", path, {"user_ids": ["user_ada", "user_bob"]})

    code, output = _run(
        ["team", "set-agent-people", "--application", "month-end-close", "--no-one"], capsys
    )
    assert code == 0, output
    assert _sent(fake) == ("PUT", path, {"user_ids": []})


def test_set_agent_people_needs_people_or_no_one(capsys: pytest.CaptureFixture[str]) -> None:
    code, output = _run(["team", "set-agent-people", "--application", "month-end-close"], capsys)
    assert code == 2
    assert "--user-id" in output["error"]["message"]


def test_set_agent_people_names_who_isnt_in_the_organization(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("PUT", "/v1/applications/month-end-close/people"): httpx.Response(
                422,
                json={
                    "detail": {
                        "code": "person_unknown",
                        "message": "Choose people who are in the organization.",
                        "user_ids": ["user_eve"],
                    }
                },
            ),
        }
    )

    code, output = _run(
        ["team", "set-agent-people", "--application", "month-end-close", "--user-id", "user_eve"],
        capsys,
    )

    assert code != 0
    assert output["error"]["code"] == "person_unknown"
    assert "Choose people who are in the organization." in output["error"]["message"]
    assert "Not in the Organization: user_eve." in output["error"]["message"]


def test_team_refusals_say_who_may_run_them_and_why(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            ("POST", "/v1/organization/invitations"): httpx.Response(
                403, json={"detail": "Organization administrator access is required."}
            ),
            ("GET", "/v1/organization/members?limit=100"): httpx.Response(
                403, json={"detail": "A signed-in Cayu Cloud user is required."}
            ),
            ("PUT", "/v1/organization/members/user_ada/role"): httpx.Response(
                422, json={"detail": "Role owner does not exist in this organization."}
            ),
            ("DELETE", "/v1/organization/members/user_ada"): httpx.Response(
                409, json={"detail": "The organization must keep at least one administrator."}
            ),
        }
    )

    code, output = _run(
        ["team", "invite", "--email", "ada@example.com", "--role", "member"], capsys
    )
    assert code != 0
    assert output["error"]["category"] == "organization_admin_required"
    assert "signed-in Organization admin" in output["error"]["message"]

    code, output = _run(["team", "members"], capsys)
    assert output["error"]["category"] == "login_required"

    code, output = _run(["team", "set-role", "user_ada", "--role", "owner"], capsys)
    assert "Role owner does not exist in this organization." in output["error"]["message"]

    code, output = _run(["team", "remove", "user_ada"], capsys)
    assert "must keep at least one administrator" in output["error"]["message"]


def test_organizations_create_rename_and_add_member_name_their_organization(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("POST", "/v1/admin/organizations"): httpx.Response(
                201, json={"id": "org_new", "name": "Acme"}
            ),
            ("PATCH", "/v1/admin/organizations/org_new"): httpx.Response(
                200, json={"id": "org_new", "name": "Acme Labs"}
            ),
            ("POST", "/v1/admin/organizations/org_new/members"): httpx.Response(
                201, json={**_MEMBER, "role": "admin", "revoked_invitation_id": None}
            ),
        }
    )

    for arguments, operation, request in (
        (
            ["create", "--name", "Acme"],
            "organizations.create",
            ("POST", "/v1/admin/organizations", {"name": "Acme"}),
        ),
        (
            ["rename", "org_new", "--name", "Acme Labs"],
            "organizations.rename",
            ("PATCH", "/v1/admin/organizations/org_new", {"name": "Acme Labs"}),
        ),
        (
            ["add-member", "org_new", "--email", "ada@example.com", "--role", "admin"],
            "organizations.add-member",
            (
                "POST",
                "/v1/admin/organizations/org_new/members",
                {"email": "ada@example.com", "role": "admin"},
            ),
        ),
    ):
        code, output = _run(["organizations", *arguments], capsys)
        assert (code, output["operation"]) == (0, operation), output
        # They act on the Organization they name, whichever one the login acts in.
        assert "organization" not in output
        assert _sent(fake) == request
    assert all(request.url.path != "/v1/me" for request in fake.requests)


def test_organization_changes_refuse_organization_and_ignore_its_variable(
    cloud, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = cloud(
        {
            ("PATCH", "/v1/admin/organizations/org_new"): httpx.Response(
                200, json={"id": "org_new", "name": "Acme Labs"}
            ),
        }
    )

    code, output = _run(
        ["--organization", "org_other", "organizations", "rename", "org_new", "--name", "X"],
        capsys,
    )
    assert code != 0
    assert output["error"]["category"] == "invalid_input"
    assert "name the one they change as ORG_ID" in output["error"]["message"]
    assert fake.requests == []

    monkeypatch.setenv("CAYU_CLOUD_ORGANIZATION", "org_other")
    code, output = _run(["organizations", "rename", "org_new", "--name", "Acme Labs"], capsys)
    assert code == 0, output
    assert _sent(fake) == ("PATCH", "/v1/admin/organizations/org_new", {"name": "Acme Labs"})


def test_organization_changes_refuse_a_malformed_organization_id(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code, output = _run(["organizations", "rename", "org/../x", "--name", "X"], capsys)
    assert code == 2
    assert "Organization id" in output["error"]["message"]


def test_organization_changes_say_they_need_a_platform_administrator(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            ("POST", "/v1/admin/organizations"): httpx.Response(
                403, json={"detail": "Cayu Cloud platform administrator access is required."}
            ),
            ("POST", "/v1/admin/organizations/org_new/members"): httpx.Response(
                404,
                json={
                    "detail": "No one has a verified Cayu Cloud account with this email; invite "
                    "them to the Organization instead."
                },
            ),
        }
    )

    code, output = _run(["organizations", "create", "--name", "Acme"], capsys)
    assert code != 0
    assert output["error"]["category"] == "platform_admin_required"

    code, output = _run(
        ["organizations", "add-member", "org_new", "--email", "x@example.com", "--role", "member"],
        capsys,
    )
    assert "invite them to the Organization instead" in output["error"]["message"]


def test_an_empty_application_is_refused_as_json(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    cloud({("GET", _APPS): httpx.Response(200, json=_AGENTS)})

    code, output = _run(["team", "agent-people", "--application", ""], capsys)

    assert code == 2
    assert output["error"]["category"] == "application_not_found"
