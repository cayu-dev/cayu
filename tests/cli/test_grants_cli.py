"""`cayu inbox approve --remember` and `cayu grants`, locally and on Cayu Cloud."""

from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from tests.core.test_human_review import (
    DynamicFactory,
    Provider,
    RecordingTool,
    identity,
    pause,
)
from tests.core.test_human_review_long_fields import KEY
from tests.runtime.authorization_grant_support import GatedPolicy

import cayu.cli.inbox as inbox_cli
from cayu import AgentSpec, CayuApp
from cayu.approvals import (
    AllowedActorsGrantAuthority,
    InMemoryToolAuthorizationGrantStore,
    ToolArgumentReview,
    ToolAuthorizationGrantPolicy,
    ToolAuthorizationGrantQuery,
    ToolAuthorizationGrantService,
)
from cayu.cli import main
from cayu.cli._cloud_api import CloudApiError
from cayu.environments import EnvironmentSpec
from cayu.sessions.memory import InMemorySessionStore

SESSION = "review-session"


async def _owner_and_intruder_read(principal: Any, grant: Any) -> bool:
    return principal.subject in {"owner", "intruder"}


def _app(sessions, grants, *, hosted, human_review_policy, reopened=False):
    """An agent whose first step proposes one grantable, approval-gated call.

    Every part declares its execution-profile identity, so the application the CLI
    opens continues the session the first one paused. ``owner`` may create and revoke
    grants.
    """

    provider = Provider([("call-1", "side_effect", {"value": "morning"})])
    if reopened:
        # The reopened application continues after the step that proposed the call.
        provider.requests.append(None)
    app = CayuApp(
        session_store=sessions, human_review_policy=human_review_policy, enable_logging=False
    )
    app.register_provider(provider, default=True)
    tool = RecordingTool()
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=ToolAuthorizationGrantPolicy(
            policy_id="effects",
            store=grants,
            grantable={"side_effect": "tool_call"},
            then=GatedPolicy("effects"),
        ),
    )
    app.register_environment_factory(
        EnvironmentSpec(name="dynamic", execution_profile_identity=identity("review-environment")),
        DynamicFactory(),
        default=True,
    )
    if hosted:
        service = ToolAuthorizationGrantService(
            app, store=grants, authority=AllowedActorsGrantAuthority(creators=["owner"])
        )
        app.host_tool_authorization_grants(service, authorize_read=_owner_and_intruder_read)
    return app, tool


def _local(monkeypatch, tmp_path, *, hosted: bool = True, human_review_policy=None):
    """A paused, grantable approval, and the CLI opening a fresh application over it."""

    sessions = InMemorySessionStore()
    grants = InMemoryToolAuthorizationGrantStore()
    options = {"hosted": hosted, "human_review_policy": human_review_policy}

    async def paused() -> None:
        app, _tool = _app(sessions, grants, **options)
        await pause(app)

    asyncio.run(paused())
    opened: dict[str, Any] = {"grants": grants}

    def build(target, **_kwargs):
        app, tool = _app(sessions, grants, **options, reopened=True)
        opened.update(app=app, tool=tool)
        return app

    monkeypatch.setattr(
        inbox_cli,
        "resolve_project",
        lambda target, **_kwargs: SimpleNamespace(
            root=tmp_path, target=target or "app:build", service_target=None
        ),
    )
    monkeypatch.setattr(inbox_cli, "project_context", lambda _root: nullcontext())
    monkeypatch.setattr(inbox_cli, "build_project_app", build)
    return opened


def _json(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def _item_id(capsys) -> str:
    assert main(["inbox", "list", "--reviewer", "owner", "--json"]) == 0
    [item] = _json(capsys)["items"]
    return item["id"]


def test_a_local_approval_is_remembered_listed_and_revoked(monkeypatch, tmp_path, capsys) -> None:
    opened = _local(monkeypatch, tmp_path)
    item_id = _item_id(capsys)

    assert (
        main(
            [
                "inbox",
                "approve",
                item_id,
                "--remember",
                "--justification",
                "Approved unchanged every run.",
                "--expires-in-seconds",
                "3600",
                "--sample-rate-bps",
                "500",
                "--reviewer",
                "owner",
                "--json",
            ]
        )
        == 0
    )
    decided = _json(capsys)
    assert decided["decision"] == "approve"
    assert decided["item"]["settlement"]["decided_by"]["subject"] == "owner"
    # The approved call ran in this process, and the grant it made is active.
    assert decided["session_status"] == "completed"
    assert opened["tool"].calls == [{"value": "morning"}]
    view = decided["grant"]
    assert view["effective_status"] == "active"
    grant = view["grant"]
    assert grant["tool_name"] == "side_effect"
    assert grant["scope"] == {"kind": "session", "id": SESSION}
    assert grant["created_by"]["subject"] == "owner"
    assert grant["justification"] == "Approved unchanged every run."
    assert grant["lifetime_seconds"] == 3600
    assert grant["sampled_review_rate_bps"] == 500
    grant_id = grant["grant_id"]

    assert main(["grants", "list", "--reviewer", "owner", "--json"]) == 0
    listed = _json(capsys)
    assert listed["where"] == "local" and listed["next"] is None
    assert [row["grant"]["grant_id"] for row in listed["grants"]] == [grant_id]

    assert main(["grants", "list", "--status", "active", "--reviewer", "owner"]) == 0
    table = capsys.readouterr().out
    assert grant_id in table and "side_effect" in table and "5%" in table
    assert main(["grants", "list", "--status", "revoked", "--reviewer", "owner"]) == 0
    assert capsys.readouterr().out.strip() == "No grants."

    assert main(["grants", "show", grant_id, "--reviewer", "owner"]) == 0
    shown = capsys.readouterr().out
    assert "any call to the tool under this gate" in shown
    assert "Approved unchanged every run." in shown

    assert main(["grants", "uses", grant_id, "--reviewer", "owner"]) == 0
    assert capsys.readouterr().out.strip() == "No uses."

    # The application's grant authority decides who may revoke.
    assert main(["grants", "revoke", grant_id, "--reviewer", "intruder", "--json"]) == 1
    assert _json(capsys)["error"]["code"] == "grant_forbidden"
    assert (
        main(["grants", "revoke", grant_id, "--reason", "Sampled errors.", "--reviewer", "owner"])
        == 0
    )
    assert capsys.readouterr().out.strip() == f"Revoked {grant_id} as owner."
    assert main(["grants", "show", grant_id, "--reviewer", "owner", "--json"]) == 0
    revoked = _json(capsys)
    assert revoked["effective_status"] == "revoked"
    assert revoked["grant"]["revocation_reason"] == "Sampled errors."

    assert main(["grants", "show", "tag_missing", "--reviewer", "owner", "--json"]) == 1
    assert _json(capsys)["error"]["code"] == "grant_not_found"


def test_a_remembered_approval_carries_the_reviewers_protected_review(
    monkeypatch, tmp_path, capsys
) -> None:
    opened = _local(
        monkeypatch,
        tmp_path,
        human_review_policy=ToolArgumentReview(
            {"side_effect": ("value",)}, binding_key=KEY, attest_listed_arguments=True
        ),
    )
    item_id = _item_id(capsys)

    assert main(["inbox", "approve", item_id, "--remember", "--reviewer", "owner"]) == 0
    text = capsys.readouterr().out
    assert f"Approved {item_id} as owner." in text
    assert "Remembered as grant tag_" in text and f"side_effect in session {SESSION}" in text
    assert opened["tool"].calls == [{"value": "morning"}]
    [grant] = asyncio.run(opened["grants"].list(ToolAuthorizationGrantQuery()))
    assert grant.status.value == "active"


def test_someone_the_grant_authority_refuses_leaves_the_approval_waiting(
    monkeypatch, tmp_path, capsys
) -> None:
    opened = _local(monkeypatch, tmp_path)
    item_id = _item_id(capsys)

    assert (
        main(["inbox", "approve", item_id, "--remember", "--reviewer", "intruder", "--json"]) == 1
    )
    assert _json(capsys)["error"]["code"] == "grant_forbidden"
    assert asyncio.run(opened["grants"].list(ToolAuthorizationGrantQuery())) == ()
    assert opened["tool"].calls == []
    # Nothing was decided, so the same person can still approve it the ordinary way.
    assert main(["inbox", "show", item_id, "--reviewer", "intruder", "--json"]) == 0
    assert _json(capsys)["item"]["status"] == "pending"


def test_an_application_that_hosts_no_grants_says_so(monkeypatch, tmp_path, capsys) -> None:
    opened = _local(monkeypatch, tmp_path, hosted=False)
    item_id = _item_id(capsys)

    assert main(["grants", "list", "--reviewer", "owner", "--json"]) == 1
    refused = _json(capsys)["error"]
    assert refused["code"] == "grants_not_served"
    assert "app.host_tool_authorization_grants" in refused["message"]

    assert main(["inbox", "approve", item_id, "--remember", "--reviewer", "owner", "--json"]) == 1
    assert _json(capsys)["error"]["code"] == "grants_not_served"
    assert opened["tool"].calls == []
    assert main(["inbox", "show", item_id, "--reviewer", "owner", "--json"]) == 0
    assert _json(capsys)["item"]["status"] == "pending"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            ["inbox", "approve", "inbox_a", "--scope", "workspace"],
            "--scope applies with --remember",
        ),
        (
            ["inbox", "approve", "inbox_a", "--pin", "repo", "--justification", "x"],
            "--justification, --pin applies with --remember",
        ),
        (
            ["inbox", "approve", "inbox_a", "--remember", "--claim-task"],
            "--remember can't be combined with --claim-task or --adopt-profile",
        ),
        (
            ["grants", "list", "--after-grant-id", "tag_1"],
            "Give --after-created-at and --after-grant-id together",
        ),
        (["grants", "list", "--as", "ana"], "--as reads a person's own inbox"),
        (
            ["grants", "revoke", "tag_1", "--reason", "x" * 2049],
            "--reason must be 1 to 2048 characters",
        ),
        (
            ["inbox", "approve", "inbox_a", "--remember", "--as", "ana"],
            "--remember can't be combined with --as",
        ),
    ],
)
def test_options_that_dont_go_together_are_refused_before_anything_opens(
    monkeypatch, capsys, arguments, message
) -> None:
    def unopened(*_args, **_kwargs):
        raise AssertionError("the application was opened")

    monkeypatch.setattr(inbox_cli, "resolve_project", unopened)

    assert main(arguments) == 1
    assert message in capsys.readouterr().err


class _Cloud:
    """Cayu Cloud's routes for one Agent, answering from canned documents."""

    def __init__(self, routes):
        self.routes = routes
        self.calls: list[tuple[str, str, dict]] = []

    def request(self, method, path, **options):
        self.calls.append((method, path, options))
        answer = self.routes[(method, path.removeprefix("/v1/applications/bugs"))]
        if isinstance(answer, Exception):
            raise answer
        return answer


def _cloud(monkeypatch, routes) -> _Cloud:
    import cayu.cli.cloud as cloud_cli

    cloud = _Cloud(routes)
    monkeypatch.setattr(cloud_cli, "cloud_client", lambda **_kwargs: cloud)
    monkeypatch.setattr(cloud_cli, "resolve_application_id", lambda _client, name: name)
    return cloud


_PENDING = {
    "id": "inbox_a",
    "kind": "approval",
    "session_id": "s1",
    "agent_name": "bugs",
    "tool_name": "file_issue",
    "decisions": ["approve", "deny"],
    "arguments": None,
    "arguments_unavailable_reason": "quarantined",
}
_STATE = {"id": "inbox_a", "kind": "approval", "status": "pending", "item": _PENDING}
_REVIEW = {
    "status": "permitted",
    "session_id": "s1",
    "kind": "tool_approval",
    "fields": [],
    "reference": {"context": {"recipient": "user_1", "purpose": "approvals"}},
}
_VIEW = {
    "effective_status": "active",
    "grant": {
        "grant_id": "tag_1",
        "tool_name": "file_issue",
        "scope": {"kind": "workspace", "id": "repo-1"},
        "created_by": {"subject": "user_1"},
        "created_at": "2026-10-01T00:00:00Z",
        "expires_at": "2026-10-08T00:00:00Z",
        "sampled_review_rate_bps": 0,
    },
}
_SETTLED = {
    "id": "inbox_a",
    "kind": "approval",
    "status": "settled",
    "settlement": {"decision": "approve", "decided_by": {"subject": "user_1"}},
}


def test_cloud_carries_a_remembered_approval_with_the_settle(monkeypatch, capsys) -> None:
    cloud = _cloud(
        monkeypatch,
        {
            ("GET", "/inbox/inbox_a"): _STATE,
            ("GET", "/inbox/inbox_a/review"): {"item": _STATE, "review": _REVIEW},
            ("POST", "/inbox/inbox_a/settle"): {
                "replayed": False,
                "item": _SETTLED,
                "grant": _VIEW,
            },
        },
    )

    assert (
        main(
            [
                "inbox",
                "approve",
                "inbox_a",
                "--application",
                "bugs",
                "--remember",
                "--scope",
                "workspace",
                "--expires-in-seconds",
                "604800",
                "--pin",
                "repo",
                "--reason",
                "ok",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.splitlines() == [
        "Approved inbox_a as user_1.",
        "Remembered as grant tag_1 (active): file_issue in workspace repo-1, "
        "until 2026-10-08T00:00:00Z.",
    ]
    method, path, options = cloud.calls[-1]
    assert (method, path) == ("POST", "/v1/applications/bugs/inbox/inbox_a/settle")
    assert options["payload"] == {
        "decision": "approve",
        "reason": "ok",
        "review_reference": _REVIEW["reference"],
        "grant": {
            "scope": "workspace",
            "expires_in_seconds": 604800,
            "pinned_argument_paths": ["repo"],
        },
    }
    assert options["idempotency_key"]


def test_a_cloud_that_doesnt_know_grants_says_so(monkeypatch, capsys) -> None:
    older = CloudApiError(
        "api_request_rejected",
        "Cayu Cloud API returned HTTP 422.",
        status_code=422,
        rejected_fields=(("body.grant", "extra_forbidden"),),
    )
    missing = CloudApiError(
        "api_request_rejected", "Cayu Cloud API returned HTTP 404.", status_code=404
    )
    _cloud(
        monkeypatch,
        {
            ("GET", "/inbox/inbox_a"): _STATE,
            ("GET", "/inbox/inbox_a/review"): {"item": _STATE, "review": _REVIEW},
            ("POST", "/inbox/inbox_a/settle"): older,
            ("GET", "/tool-authorization-grants"): missing,
        },
    )

    assert (
        main(["inbox", "approve", "inbox_a", "--application", "bugs", "--remember", "--json"]) == 1
    )
    assert _json(capsys)["error"]["code"] == "grants_unsupported"
    assert main(["grants", "list", "--application", "bugs", "--json"]) == 1
    assert _json(capsys)["error"]["code"] == "grants_unsupported"


def test_grants_are_read_and_revoked_on_a_cloud_agent(monkeypatch, capsys) -> None:
    revoked = {
        "effective_status": "revoked",
        "grant": {**_VIEW["grant"], "revoked_by": {"subject": "user_1"}},
    }
    cloud = _cloud(
        monkeypatch,
        {
            ("GET", "/tool-authorization-grants"): {
                "served": True,
                "grants": [_VIEW],
                "next": {"after_created_at": "2026-10-01T00:00:00Z", "after_grant_id": "tag_1"},
            },
            ("GET", "/tool-authorization-grants/tag_1"): _VIEW,
            ("GET", "/tool-authorization-grants/tag_1/uses"): {
                "uses": [
                    {
                        "recorded_at": "2026-10-02T00:00:00Z",
                        "decision": "allowed",
                        "session_id": "s2",
                        "tool_call_id": "call_9",
                    }
                ],
                "next_cursor": "c2",
            },
            ("POST", "/tool-authorization-grants/tag_1/revoke"): revoked,
        },
    )

    assert (
        main(
            [
                "grants",
                "list",
                "--application",
                "bugs",
                "--status",
                "active",
                "--status",
                "staged",
                "--tool",
                "file_issue",
                "--limit",
                "1",
            ]
        )
        == 0
    )
    listed = capsys.readouterr().out
    assert "tag_1" in listed and "workspace repo-1" in listed
    assert (
        "More grants are left. Go on with --after-created-at 2026-10-01T00:00:00Z "
        "--after-grant-id tag_1" in listed
    )
    assert cloud.calls[-1] == (
        "GET",
        "/v1/applications/bugs/tool-authorization-grants",
        {"query": {"limit": "1", "status": ["active", "staged"], "tool_name": "file_issue"}},
    )

    assert main(["grants", "uses", "tag_1", "--application", "bugs", "--cursor", "c1"]) == 0
    uses = capsys.readouterr().out
    assert "allowed" in uses and "More uses are left. Go on with --cursor c2" in uses
    assert cloud.calls[-1][2] == {"query": {"limit": "100", "cursor": "c1"}}

    assert main(["grants", "revoke", "tag_1", "--application", "bugs", "--reason", "Stale."]) == 0
    assert capsys.readouterr().out.strip() == "Revoked tag_1 as user_1."
    assert cloud.calls[-1] == (
        "POST",
        "/v1/applications/bugs/tool-authorization-grants/tag_1/revoke",
        {"payload": {"reason": "Stale."}},
    )


def test_an_agent_whose_release_serves_no_grants_says_so_through_cloud(monkeypatch, capsys) -> None:
    refused = CloudApiError(
        "api_request_rejected",
        "Cayu Cloud API returned HTTP 409: The Agent's Release doesn't serve them.",
        status_code=409,
        code="grants_not_served",
    )
    _cloud(
        monkeypatch,
        {
            ("GET", "/tool-authorization-grants"): {"served": False, "grants": [], "next": None},
            ("POST", "/tool-authorization-grants/tag_1/revoke"): refused,
        },
    )

    assert main(["grants", "list", "--application", "bugs", "--json"]) == 1
    assert _json(capsys)["error"]["code"] == "grants_not_served"
    assert main(["grants", "revoke", "tag_1", "--application", "bugs", "--json"]) == 1
    assert _json(capsys)["error"]["code"] == "grants_not_served"
