"""`cayu inbox`: list, show and decide an agent's approvals, locally and on Cayu Cloud."""

from __future__ import annotations

import asyncio
import getpass
import json
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("fastapi")

from tests.core.test_human_review import (
    ApprovalPolicy,
    DynamicFactory,
    Provider,
    RecordingTool,
    identity,
    pause,
)
from tests.core.test_human_review_long_fields import KEY

import cayu.cli.inbox as inbox_cli
from cayu import AgentSpec, CayuApp
from cayu.approvals import ToolArgumentReview
from cayu.cli import main
from cayu.cli._cloud_api import CloudApiError
from cayu.cli._inbox_client import CloudInbox, InboxClientError
from cayu.environments import EnvironmentSpec
from cayu.storage import SQLiteSessionStore

LONG = "".join(f"line {n} <&> of a long value\n" for n in range(400))


def _app(store, policy, value, *, reopened=False):
    """An agent whose first step proposes one approval-gated call, then finishes."""

    provider = Provider([("call-1", "side_effect", {"value": value})])
    if reopened:
        # The reopened app continues after the step that proposed the call.
        provider.requests.append(None)
    app = CayuApp(session_store=store, human_review_policy=policy, enable_logging=False)
    app.register_provider(provider, default=True)
    tool = RecordingTool()
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"), tools=[tool], tool_policy=ApprovalPolicy()
    )
    app.register_environment_factory(
        EnvironmentSpec(name="dynamic", execution_profile_identity=identity("review-environment")),
        DynamicFactory(),
        default=True,
    )
    return app, tool


def _local(monkeypatch, tmp_path, *, policy, value="hello <b>\x1b[31m"):
    """A paused approval in a SQLite store, and the CLI opening a fresh app over it."""

    path = tmp_path / "sessions.sqlite"

    async def paused() -> None:
        store = SQLiteSessionStore(path)
        app, _tool = _app(store, policy, value)
        await pause(app)
        await app.aclose()
        await store.close()

    asyncio.run(paused())
    opened: dict[str, object] = {}

    def build(target, **_kwargs):
        store = SQLiteSessionStore(path)
        app, tool = _app(store, policy, value, reopened=True)
        opened.update(app=app, tool=tool, store=store)
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


def _only_item(capsys) -> dict:
    assert main(["inbox", "list", "--json"]) == 0
    listing = _json(capsys)
    assert listing["where"] == "local" and listing["complete"] is True
    [item] = listing["items"]
    return item


def test_a_local_approval_shows_the_reviewed_arguments_and_settles(
    monkeypatch, tmp_path, capsys
) -> None:
    policy = ToolArgumentReview(
        {"side_effect": ("value",)}, binding_key=KEY, attest_listed_arguments=True
    )
    opened = _local(monkeypatch, tmp_path, policy=policy)
    item = _only_item(capsys)
    assert item["kind"] == "approval" and item["tool_name"] == "side_effect"
    assert item["arguments"] is None
    assert item["arguments_unavailable_reason"] == "quarantined"

    assert main(["inbox", "show", item["id"], "--json"]) == 0
    shown = _json(capsys)
    assert shown["item"]["status"] == "pending"
    review = shown["review"]
    assert review["status"] == "permitted"
    assert review["fields"] == [{"label": "value", "text": "hello &lt;b&gt;\x1b[31m"}]
    # The opaque reference never leaves the CLI.
    assert "reference" not in review

    assert main(["inbox", "show", item["id"]]) == 0
    text = capsys.readouterr().out
    assert "Arguments the policy shows:" in text
    # HTML escaping is undone for the terminal, and the escape sequence is neutralized.
    assert "value: hello <b> [31m" in text
    assert "\x1b" not in text

    assert main(["inbox", "approve", item["id"], "--reason", "Checked.", "--json"]) == 0
    decided = _json(capsys)
    assert decided["decision"] == "approve" and decided["replayed"] is False
    settlement = decided["item"]["settlement"]
    assert settlement["decision"] == "approve"
    assert settlement["decided_by"]["subject"] == f"local:{getpass.getuser()}"
    # The approved call ran in this process before the command returned.
    assert decided["session_status"] == "completed"
    assert opened["tool"].calls == [{"value": "hello <b>\x1b[31m"}]

    assert main(["inbox", "list", "--json"]) == 0
    assert _json(capsys)["items"] == []
    assert main(["inbox", "deny", item["id"], "--json"]) == 1
    refused = _json(capsys)
    assert refused["error"]["code"] == "not_pending"
    assert refused["item"]["settlement"]["decision"] == "approve"


def test_a_long_argument_is_read_whole_only_with_full(monkeypatch, tmp_path, capsys) -> None:
    policy = ToolArgumentReview(
        {"side_effect": ("value",)}, binding_key=KEY, attest_listed_arguments=True
    )
    _local(monkeypatch, tmp_path, policy=policy, value=LONG)
    item = _only_item(capsys)

    assert main(["inbox", "show", item["id"]]) == 0
    assert "run with --full to read it whole" in capsys.readouterr().out

    assert main(["inbox", "show", item["id"], "--full", "--json"]) == 0
    shown = _json(capsys)
    [read] = shown["long_fields"].values()
    assert read["status"] == "permitted" and "reference" not in read
    assert main(["inbox", "show", item["id"], "--full"]) == 0
    marked = "\n".join(f"    | {line}" for line in LONG.splitlines())
    assert marked in capsys.readouterr().out


def test_a_withheld_review_can_only_be_denied(monkeypatch, tmp_path, capsys) -> None:
    # The round's environment is dynamic and the policy doesn't attest the argument.
    policy = ToolArgumentReview({"side_effect": ("value",)}, binding_key=KEY)
    opened = _local(monkeypatch, tmp_path, policy=policy)
    item = _only_item(capsys)

    assert main(["inbox", "show", item["id"], "--json"]) == 0
    review = _json(capsys)["review"]
    assert review["status"] != "permitted" and review["fields"] == []

    assert main(["inbox", "approve", item["id"], "--json"]) == 1
    assert _json(capsys)["error"]["code"] == "review_withheld"
    assert main(["inbox", "deny", item["id"], "--reviewer", "ops@example.com"]) == 0
    assert f"Denied {item['id']} as ops@example.com." in capsys.readouterr().out
    assert opened["tool"].calls == []


def test_without_a_policy_the_inbox_says_why_no_arguments_show(
    monkeypatch, tmp_path, capsys
) -> None:
    opened = _local(monkeypatch, tmp_path, policy=None)
    item = _only_item(capsys)
    assert item["arguments_unavailable_reason"] == "no_review_policy"

    assert main(["inbox", "show", item["id"]]) == 0
    assert "the application has no human-review policy" in capsys.readouterr().out

    assert main(["inbox", "list"]) == 0
    table = capsys.readouterr().out.splitlines()
    assert table[0].split() == ["KIND", "AGENT", "SESSION", "WHAT", "CREATED", "ID"]
    assert table[1].startswith("approval") and item["id"] in table[1]

    assert main(["inbox", "deny", item["id"], "--reason", "Not now.", "--json"]) == 0
    assert _json(capsys)["item"]["settlement"]["decision"] == "deny"
    assert opened["tool"].calls == []


def test_a_blank_reason_and_a_cloud_reviewer_are_refused(capsys) -> None:
    assert main(["inbox", "deny", "inbox_x", "--reason", " ", "--json"]) == 1
    assert _json(capsys)["error"]["code"] == "invalid_input"
    assert main(["inbox", "list", "--application", "a", "--reviewer", "me", "--json"]) == 1
    assert _json(capsys)["error"]["code"] == "invalid_input"


class _Cloud:
    """Cayu Cloud's inbox routes, answering from canned documents."""

    def __init__(self, routes):
        self.routes = routes
        self.calls: list[tuple[str, str, dict]] = []

    def request(self, method, path, **options):
        self.calls.append((method, path, options))
        answer = self.routes[(method, path.split("/inbox", 1)[1])]
        if isinstance(answer, Exception):
            raise answer
        return answer


_PENDING = {
    "id": "inbox_a",
    "kind": "approval",
    "session_id": "s1",
    "agent_name": "bugs",
    "tool_name": "file_issue",
    "title": "Tool approval required",
    "decisions": ["approve", "deny"],
    "arguments": None,
    "arguments_unavailable_reason": "quarantined",
}
_STATE = {"id": "inbox_a", "kind": "approval", "status": "pending", "item": _PENDING}
_REVIEW = {
    "status": "permitted",
    "guidance": "Review all calls before deciding.",
    "session_id": "s1",
    "kind": "tool_approval",
    "fields": [{"label": "title", "text": "Crash on start"}],
    "reference": {"context": {"recipient": "user_1", "purpose": "approvals"}},
}
_SETTLED = {
    "replayed": False,
    "settled": True,
    "item": {
        "id": "inbox_a",
        "kind": "approval",
        "status": "settled",
        "settlement": {"decision": "approve", "decided_by": {"subject": "user_1"}},
    },
}


def _run(coroutine):
    return asyncio.run(coroutine)


def test_the_cloud_inbox_filters_lists_and_binds_a_decision_to_the_review() -> None:
    other = {**_PENDING, "id": "inbox_b", "kind": "question", "agent_name": "triage"}
    cloud = _Cloud(
        {
            ("GET", ""): {
                "items": [_PENDING, other],
                "agents": [{"state": "read", "issues": []}],
                "complete": True,
            },
            ("GET", "/inbox_a/review"): {"item": _STATE, "review": _REVIEW},
            ("POST", "/inbox_a/settle"): _SETTLED,
        }
    )
    inbox = CloudInbox(cloud, "bug-hunter", idempotency_key=lambda: "key-1")

    listing = _run(inbox.list_items(kind="approval", session_id=None, agent_name=None))
    assert [item["id"] for item in listing.items] == ["inbox_a"] and listing.complete
    assert _run(inbox.review("inbox_a", "s1", purpose="approvals")) == _REVIEW
    body = {"decision": "approve", "review_reference": _REVIEW["reference"]}
    assert _run(inbox.settle("inbox_a", body)) == {"replayed": False, "item": _SETTLED["item"]}
    assert cloud.calls[1] == (
        "GET",
        "/v1/applications/bug-hunter/inbox/inbox_a/review",
        {"query": {"purpose": "approvals"}},
    )
    assert cloud.calls[2] == (
        "POST",
        "/v1/applications/bug-hunter/inbox/inbox_a/settle",
        {"payload": body, "idempotency_key": "key-1"},
    )


def test_an_unread_cloud_agent_makes_the_listing_incomplete() -> None:
    cloud = _Cloud(
        {
            ("GET", ""): {
                "items": [],
                "agents": [{"state": "unsupported", "detail": "Cloud sign-in is off."}],
                "complete": False,
            }
        }
    )
    listing = _run(CloudInbox(cloud, "a").list_items(kind=None, session_id=None, agent_name=None))
    assert listing.complete is False and listing.notes == ["Cloud sign-in is off."]


def test_a_cloud_without_the_review_route_says_so() -> None:
    missing = CloudApiError("api_request_rejected", "HTTP 404.", status_code=404)
    unknown_field = CloudApiError(
        "api_request_rejected",
        "HTTP 422.",
        status_code=422,
        rejected_fields=(("body.review_reference", "extra_forbidden"),),
    )
    cloud = _Cloud(
        {("GET", "/inbox_a/review"): missing, ("POST", "/inbox_a/settle"): unknown_field}
    )
    inbox = CloudInbox(cloud, "a")
    with pytest.raises(InboxClientError) as read:
        _run(inbox.review("inbox_a", "s1", purpose="approvals"))
    assert read.value.code == "review_unsupported"
    with pytest.raises(InboxClientError) as settled:
        _run(inbox.settle("inbox_a", {"decision": "deny", "review_reference": {}}))
    assert settled.value.code == "review_unsupported"


def test_a_refused_cloud_settle_carries_the_items_state() -> None:
    state = {**_SETTLED["item"], "settlement": {"decision": "deny"}}
    conflict = CloudApiError(
        "api_request_rejected",
        "Cayu Cloud API returned HTTP 409: Someone settled it first.",
        status_code=409,
        code="already_settled",
        body={"code": "already_settled", "message": "Someone settled it first.", "item": state},
    )
    inbox = CloudInbox(_Cloud({("POST", "/inbox_a/settle"): conflict}), "a")
    with pytest.raises(InboxClientError) as refused:
        _run(inbox.settle("inbox_a", {"decision": "approve"}))
    assert refused.value.code == "already_settled" and refused.value.item == state


def test_a_long_refusal_from_the_agent_shows_its_first_line_through_cloud(monkeypatch) -> None:
    """An Agent's multi-line refusal through Cloud is shown by its first line."""

    import httpx

    import cayu.cli._cloud_api as cloud_api
    from cayu.execution_profiles import ExecutionProfileComponentClass
    from cayu.runtime.execution_profiles import ExecutionProfileMismatchError

    refusal = ExecutionProfileMismatchError(
        session_id="scout-1-file",
        expected_profile_fingerprint="a" * 64,
        candidate_profile_fingerprint="b" * 64,
        changed_component_classes=(ExecutionProfileComponentClass.RUNTIME,),
        boundary="continuation",
    )
    state = {**_SETTLED["item"], "status": "pending", "settlement": None}
    # What Cloud answers for an Agent's refused settle: its code and message, cut at
    # 1000 characters, with the item's state.
    detail = {"code": "conflict", "message": str(refusal)[:1000], "item": state}
    real_client = httpx.Client

    def client(**options):
        return real_client(
            transport=httpx.MockTransport(lambda _: httpx.Response(409, json={"detail": detail})),
            **options,
        )

    monkeypatch.setattr(cloud_api.httpx, "Client", client)
    inbox = CloudInbox(cloud_api.CloudApiClient(api_url="https://cloud.example", api_key="k"), "a")
    with pytest.raises(InboxClientError) as refused:
        _run(inbox.settle("inbox_a", {"decision": "deny"}))
    assert str(refused.value) == (
        "Cayu Cloud API returned HTTP 409: Session scout-1-file execution profile changed "
        "in: runtime. No resumed work was admitted."
    )
    assert refused.value.code == "conflict" and refused.value.item == state


def test_the_cli_decides_on_a_cloud_agent_through_the_same_commands(monkeypatch, capsys) -> None:
    cloud = _Cloud(
        {
            ("GET", "/inbox_a"): _STATE,
            ("GET", "/inbox_a/review"): {"item": _STATE, "review": _REVIEW},
            ("POST", "/inbox_a/settle"): _SETTLED,
        }
    )
    import cayu.cli.cloud as cloud_cli

    monkeypatch.setattr(cloud_cli, "cloud_client", lambda **_kwargs: cloud)
    monkeypatch.setattr(cloud_cli, "resolve_application_id", lambda _client, name: f"id-{name}")

    assert main(["inbox", "show", "inbox_a", "--application", "bugs"]) == 0
    assert "title: Crash on start" in capsys.readouterr().out
    assert main(["inbox", "approve", "inbox_a", "--application", "bugs", "--reason", "ok"]) == 0
    assert capsys.readouterr().out.strip() == "Approved inbox_a as user_1."
    method, path, options = cloud.calls[-1]
    assert (method, path) == ("POST", "/v1/applications/id-bugs/inbox/inbox_a/settle")
    assert options["payload"] == {
        "decision": "approve",
        "reason": "ok",
        "review_reference": _REVIEW["reference"],
    }


def test_the_cli_dismisses_recovery_items_and_approvals(monkeypatch, capsys) -> None:
    recovery = {
        "id": "inbox_r",
        "kind": "recovery",
        "status": "pending",
        "session_id": "s2",
        "item": {
            **_PENDING,
            "id": "inbox_r",
            "kind": "recovery",
            "title": "Manual recovery required",
            "decisions": ["dismiss"],
        },
    }
    dismissed = {
        "replayed": False,
        "settled": True,
        "item": {
            "id": "inbox_r",
            "kind": "recovery",
            "status": "settled",
            "settlement": {"decision": "dismiss", "decided_by": {"subject": "user_1"}},
        },
    }
    cloud = _Cloud(
        {
            ("GET", "/inbox_a"): _STATE,
            ("GET", "/inbox_r"): recovery,
            ("POST", "/inbox_r/settle"): dismissed,
            ("POST", "/inbox_a/settle"): {
                "replayed": False,
                "item": {
                    "id": "inbox_a",
                    "kind": "approval",
                    "status": "settled",
                    "settlement": {"decision": "dismiss", "decided_by": {"subject": "user_1"}},
                },
            },
        }
    )
    import cayu.cli.cloud as cloud_cli

    monkeypatch.setattr(cloud_cli, "cloud_client", lambda **_kwargs: cloud)
    monkeypatch.setattr(cloud_cli, "resolve_application_id", lambda _client, name: f"id-{name}")

    with pytest.raises(SystemExit):
        main(["inbox", "dismiss", "inbox_r", "--application", "bugs"])
    assert "--reason" in capsys.readouterr().err
    # An approval whose session can't resume is dismissed by closing that session.
    assert main(["inbox", "dismiss", "inbox_a", "--application", "bugs", "--reason", "x"]) == 0
    assert capsys.readouterr().out.strip() == "Dismissed inbox_a as user_1."
    method, path, options = cloud.calls[-1]
    assert (method, path) == ("POST", "/v1/applications/id-bugs/inbox/inbox_a/settle")
    assert options["payload"] == {"decision": "dismiss", "reason": "x"}
    assert main(["inbox", "deny", "inbox_r", "--application", "bugs"]) == 1
    assert "settles tool approvals; this item is a recovery" in capsys.readouterr().err
    assert (
        main(["inbox", "dismiss", "inbox_r", "--application", "bugs", "--reason", "Old run"]) == 0
    )
    assert capsys.readouterr().out.strip() == "Dismissed inbox_r as user_1."
    method, path, options = cloud.calls[-1]
    assert (method, path) == ("POST", "/v1/applications/id-bugs/inbox/inbox_r/settle")
    assert options["payload"] == {"decision": "dismiss", "reason": "Old run"}


def test_an_older_cloud_settles_without_a_reference_when_it_cant_say_about_a_policy(
    monkeypatch, capsys
) -> None:
    older = {key: value for key, value in _PENDING.items() if not key.startswith("arguments")}
    cloud = _cloud_cli(
        monkeypatch,
        {
            ("GET", "/inbox_a"): {**_STATE, "item": older},
            ("GET", "/inbox_a/review"): CloudApiError("x", "HTTP 404.", status_code=404),
            ("POST", "/inbox_a/settle"): _SETTLED,
        },
    )
    assert main(["inbox", "deny", "inbox_a", "--application", "bugs", "--json"]) == 0
    assert cloud.calls[-1][2]["payload"] == {"decision": "deny"}
    assert _json(capsys)["session_status"] is None


def _cloud_cli(monkeypatch, routes) -> _Cloud:
    cloud = _Cloud(routes)
    import cayu.cli.cloud as cloud_cli

    monkeypatch.setattr(cloud_cli, "cloud_client", lambda **_kwargs: cloud)
    monkeypatch.setattr(cloud_cli, "resolve_application_id", lambda _client, name: name)
    return cloud


def test_an_agent_without_reviews_is_decided_without_a_reference_unless_it_has_a_policy(
    monkeypatch, capsys
) -> None:
    not_served = CloudApiError(
        "api_request_rejected",
        "Cayu Cloud API returned HTTP 409: The Agent's Cayu doesn't serve protected review.",
        status_code=409,
        code="review_not_served",
        body={"code": "review_not_served"},
    )
    unsaid = {**_PENDING, "arguments_unavailable_reason": None}
    cloud = _cloud_cli(
        monkeypatch,
        {
            ("GET", "/inbox_a"): {**_STATE, "item": unsaid},
            ("GET", "/inbox_a/review"): not_served,
            ("POST", "/inbox_a/settle"): _SETTLED,
        },
    )
    assert main(["inbox", "deny", "inbox_a", "--application", "bugs", "--json"]) == 0
    assert cloud.calls[-1][2]["payload"] == {"decision": "deny"}
    capsys.readouterr()

    # An item that says its application has a policy can't be decided without a review.
    cloud = _cloud_cli(
        monkeypatch,
        {("GET", "/inbox_a"): _STATE, ("GET", "/inbox_a/review"): not_served},
    )
    assert main(["inbox", "deny", "inbox_a", "--application", "bugs", "--json"]) == 1
    assert _json(capsys)["error"]["code"] == "review_not_served"
    assert [call[0] for call in cloud.calls] == ["GET", "GET"]


def test_a_project_services_sessions_continue_with_its_loop_policies(
    monkeypatch, tmp_path, capsys
) -> None:
    opened = _local(monkeypatch, tmp_path, policy=None)
    asked: list[str] = []

    async def continuation_loop_policies(session_id: str) -> tuple:
        asked.append(session_id)
        return ()

    def build_service(target, *, mode, command):
        assert (target, mode) == ("app:service", "production")
        return SimpleNamespace(
            cayu_app=inbox_cli.build_project_app("app:build"),
            _continuation_loop_policies=continuation_loop_policies,
        )

    monkeypatch.setattr(
        inbox_cli,
        "resolve_project",
        lambda target, **_kwargs: SimpleNamespace(
            root=tmp_path, target="app:build", service_target="app:service"
        ),
    )
    monkeypatch.setattr(inbox_cli, "build_project_service", build_service)
    item = _only_item(capsys)

    assert main(["inbox", "approve", item["id"], "--json"]) == 0
    assert _json(capsys)["session_status"] == "completed"
    # The route asked the service for the session's continuation policies before resuming it.
    assert asked and opened["tool"].calls == [{"value": "hello <b>\x1b[31m"}]


def test_the_local_listing_is_oldest_first_across_sessions() -> None:
    from cayu.cli._inbox_client import LocalInbox

    pages = [
        {"items": [{"id": "inbox_new", "created_at": "2026-10-28T10:00:00Z"}], "next_cursor": "c"},
        {"items": [{"id": "inbox_old", "created_at": "2026-10-27T10:00:00Z"}], "next_cursor": None},
    ]
    inbox = LocalInbox.__new__(LocalInbox)

    async def answer(method, path, **_options):
        return pages.pop(0)

    inbox._json = answer  # type: ignore[method-assign]
    listing = _run(inbox.list_items(kind=None, session_id=None, agent_name=None))
    assert [item["id"] for item in listing.items] == ["inbox_old", "inbox_new"]


def test_a_task_continuation_claim_is_sent_with_the_decision(monkeypatch, capsys) -> None:
    unsaid = {**_PENDING, "arguments_unavailable_reason": "no_review_policy"}
    cloud = _cloud_cli(
        monkeypatch,
        {("GET", "/inbox_a"): {**_STATE, "item": unsaid}, ("POST", "/inbox_a/settle"): _SETTLED},
    )
    assert main(["inbox", "deny", "inbox_a", "--application", "bugs", "--claim-task"]) == 0
    assert cloud.calls[-1][2]["payload"] == {"decision": "deny", "task_continuation": "claim"}


def test_a_reviewed_value_cant_print_lines_that_look_like_other_arguments(capsys) -> None:
    review = {
        "status": "permitted",
        "guidance": "Review all calls before deciding.",
        "fields": [{"label": "memo", "text": "Invoice 42\n  amount: 10.00"}],
    }
    inbox_cli._print_arguments({"kind": "approval"}, review, None, {})
    assert "  memo: Invoice 42\n    |   amount: 10.00\n" in capsys.readouterr().out


def test_an_unknown_user_must_name_a_reviewer(monkeypatch, tmp_path, capsys) -> None:
    _local(monkeypatch, tmp_path, policy=None)

    def no_user() -> str:
        raise KeyError("getpwuid(): uid not found: 1000690000")

    monkeypatch.setattr(inbox_cli.getpass, "getuser", no_user)
    assert main(["inbox", "list", "--json"]) == 1
    assert "--reviewer" in _json(capsys)["error"]["message"]
    assert main(["inbox", "list", "--reviewer", "ops", "--json"]) == 0


def test_a_local_decision_starts_the_model_policy_and_waits_for_continued_work(
    monkeypatch, tmp_path, capsys
) -> None:
    opened = _local(monkeypatch, tmp_path, policy=None)
    item = _only_item(capsys)
    started: list[bool] = []
    closed: list[float | None] = []
    close = inbox_cli.close_project_app

    async def start_model_policy() -> None:
        started.append(True)

    def build(target, **_kwargs):
        app = build_before(target)
        app.start_model_policy = start_model_policy
        return app

    async def recording_close(app, *, timeout_s=None):
        closed.append(timeout_s)
        await (close(app) if timeout_s is None else close(app, timeout_s=timeout_s))

    build_before = inbox_cli.build_project_app
    monkeypatch.setattr(inbox_cli, "build_project_app", build)
    monkeypatch.setattr(inbox_cli, "close_project_app", recording_close)

    assert main(["inbox", "deny", item["id"], "--wait-seconds", "120", "--json"]) == 0
    # Shutdown gets what the wait for the continued session left of the 120 seconds.
    assert started == [True] and len(closed) == 1 and 0 < closed[0] <= 120.0
    assert opened["tool"].calls == []


def test_the_local_continuation_wait_stops_at_wait_seconds(monkeypatch) -> None:
    import asyncio

    from cayu.cli import _inbox_client
    from cayu.cli._inbox_client import LocalInbox

    clock = [0.0]
    inbox = LocalInbox.__new__(LocalInbox)
    inbox._wait_seconds = 2.0
    inbox._monotonic = lambda: clock[0]
    inbox._wait_deadline = None
    polls: list[str] = []

    async def still_running(method: str, path: str, **_: object) -> dict:
        polls.append(path)
        clock[0] += 1.5
        return {"status": "running"}

    async def no_sleep(_: float) -> None:
        return None

    inbox._json = still_running  # type: ignore[method-assign]
    monkeypatch.setattr(_inbox_client.asyncio, "sleep", no_sleep)

    assert asyncio.run(inbox.after_settle("sess_1")) == "running"
    assert len(polls) == 2
    # Shutdown gets nothing more: the continuation wait used the whole budget.
    assert inbox.remaining_wait_seconds() == 0.0


@pytest.mark.parametrize(
    "message", ["\ud800 refused. more\nline two", "x" * 600 + "\udfff refused. more\nline two"]
)
def test_unencodable_cloud_refusal_preserves_conflict_and_item(monkeypatch, message):
    import httpx

    import cayu.cli._cloud_api as cloud_api

    state = {**_SETTLED["item"], "status": "pending", "settlement": None}
    detail = {"code": "conflict", "message": message, "item": state}
    real_client = httpx.Client
    monkeypatch.setattr(
        cloud_api.httpx,
        "Client",
        lambda **options: real_client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(409, content=json.dumps({"detail": detail}).encode())
            ),
            **options,
        ),
    )
    inbox = CloudInbox(cloud_api.CloudApiClient(api_url="https://cloud.example", api_key="k"), "a")
    with pytest.raises(InboxClientError) as refused:
        _run(inbox.settle("inbox_a", {"decision": "deny"}))
    assert str(refused.value) == "Cayu Cloud API returned HTTP 409."
    assert refused.value.code == "conflict"
    assert refused.value.item == state


def _earlier(item_id: str, session: str) -> dict:
    item = {**_PENDING, "id": item_id, "session_id": session, "created_at": "2026-11-01T23:44Z"}
    return {"id": item_id, "kind": "approval", "status": "pending", "item": item}


def test_items_an_older_cloud_left_out_as_earlier_are_read_by_id_and_listed(
    monkeypatch, capsys
) -> None:
    """An older Cloud names items asked before the Agent's Release only by id."""

    unreadable = CloudApiError("api_request_failed", "HTTP 502.", status_code=502)
    cloud = _cloud_cli(
        monkeypatch,
        {
            ("GET", ""): {
                "items": [],
                "agents": [
                    {
                        "state": "read",
                        "issues": [],
                        "earlier_release_items": ["inbox_e1", "inbox_e2", "inbox_gone"],
                    }
                ],
                "complete": True,
            },
            ("GET", "/inbox_e1"): _earlier("inbox_e1", "s1"),
            ("GET", "/inbox_e2"): _earlier("inbox_e2", "s2"),
            ("GET", "/inbox_gone"): {
                "id": "inbox_gone",
                "kind": "approval",
                "status": "settled",
                "settlement": {"decision": "deny"},
            },
        },
    )
    assert main(["inbox", "list", "--application", "bugs", "--json"]) == 0
    listing = _json(capsys)
    assert [item["id"] for item in listing["items"]] == ["inbox_e1", "inbox_e2"]
    assert all(item["asked_before_release"] is True for item in listing["items"])
    assert listing["complete"] is True

    assert main(["inbox", "list", "--application", "bugs"]) == 0
    out = capsys.readouterr().out
    assert "Nothing is waiting." not in out
    assert out.startswith("Asked before the Agent's current Release (2).")
    assert "inbox_e1" in out and "inbox_e2" in out

    # The filters go to Cloud, and are applied here too for a Cloud that ignores them.
    assert main(["inbox", "list", "--application", "bugs", "--session", "s2", "--json"]) == 0
    assert [item["id"] for item in _json(capsys)["items"]] == ["inbox_e2"]
    assert (
        "GET",
        "/v1/applications/bugs/inbox",
        {"query": {"session_id": "s2", "limit": 50}},
    ) in cloud.calls

    # One that can't be read makes the listing incomplete and says so.
    cloud.routes[("GET", "/inbox_e2")] = unreadable
    assert main(["inbox", "list", "--application", "bugs", "--json"]) == 0
    listing = _json(capsys)
    assert [item["id"] for item in listing["items"]] == ["inbox_e1"]
    assert listing["complete"] is False
    assert "1 item(s) asked before the Agent's current Release" in listing["notes"][0]


def test_a_newer_clouds_earlier_release_items_are_listed_in_their_own_section(
    monkeypatch, capsys
) -> None:
    earlier = {**_PENDING, "id": "inbox_e", "created_at": "1", "asked_before_release": True}
    current = {**_PENDING, "id": "inbox_c", "created_at": "2", "asked_before_release": False}
    _cloud_cli(
        monkeypatch,
        {
            ("GET", ""): {
                "items": [earlier, current],
                "agents": [{"state": "read", "issues": [], "earlier_release_items": []}],
                "complete": True,
            }
        },
    )
    assert main(["inbox", "list", "--application", "bugs"]) == 0
    current_part, earlier_part = capsys.readouterr().out.split("Asked before", 1)
    assert "inbox_c" in current_part and "inbox_e" not in current_part
    assert "inbox_e" in earlier_part


def test_a_sessions_item_is_shown_and_settled_by_session(monkeypatch, capsys) -> None:
    """A waiting session leads to its item without the item id."""

    question = {**_PENDING, "id": "inbox_q", "kind": "question", "session_id": "s1"}
    cloud = _cloud_cli(
        monkeypatch,
        {
            ("GET", ""): {"items": [_PENDING, question], "agents": [], "complete": True},
            ("GET", "/inbox_a"): {
                **_STATE,
                "item": {**_PENDING, "arguments_unavailable_reason": "no_review_policy"},
            },
            ("POST", "/inbox_a/settle"): _SETTLED,
        },
    )
    # Denying picks the session's one approval; the question isn't one.
    assert main(["inbox", "deny", "--session", "s1", "--application", "bugs", "--json"]) == 0
    assert _json(capsys)["item"]["id"] == "inbox_a"
    assert cloud.calls[-1][1] == "/v1/applications/bugs/inbox/inbox_a/settle"
    # Show takes any kind, so two items need the id.
    assert main(["inbox", "show", "--session", "s1", "--application", "bugs", "--json"]) == 1
    error = _json(capsys)["error"]
    assert error["code"] == "several_pending_items"
    assert "inbox_a (approval)" in error["message"] and "inbox_q (question)" in error["message"]
    assert main(["inbox", "show", "--session", "s9", "--application", "bugs", "--json"]) == 1
    assert _json(capsys)["error"]["code"] == "no_pending_item"
    # One of ITEM_ID or --session.
    assert main(["inbox", "show", "--application", "bugs", "--json"]) == 1
    assert _json(capsys)["error"]["code"] == "invalid_input"
    assert main(["inbox", "show", "inbox_a", "--session", "s1", "--json"]) == 1
    assert _json(capsys)["error"]["code"] == "invalid_input"


def test_an_approval_is_settled_with_a_changed_profile_adopted(monkeypatch, capsys) -> None:
    unsaid = {**_STATE, "item": {**_PENDING, "arguments_unavailable_reason": "no_review_policy"}}
    changed = CloudApiError(
        "api_request_rejected",
        "Cayu Cloud API returned HTTP 409: The execution profile changed.",
        status_code=409,
        code="profile_changed",
    )
    cloud = _cloud_cli(
        monkeypatch, {("GET", "/inbox_a"): unsaid, ("POST", "/inbox_a/settle"): changed}
    )
    assert main(["inbox", "approve", "inbox_a", "--application", "bugs"]) == 1
    assert "--adopt-profile --reason" in capsys.readouterr().err
    assert main(["inbox", "approve", "inbox_a", "--application", "bugs", "--adopt-profile"]) == 1
    assert "--adopt-profile needs --reason" in capsys.readouterr().err

    cloud.routes[("POST", "/inbox_a/settle")] = _SETTLED
    command = ["inbox", "approve", "inbox_a", "--application", "bugs"]
    assert main([*command, "--adopt-profile", "--reason", "same tools"]) == 0
    capsys.readouterr()
    payload = cloud.calls[-1][2]["payload"]
    adoption = payload.pop("profile_adoption")
    assert payload == {"decision": "approve", "reason": "same tools"}
    assert adoption["reason"] == "same tools" and len(adoption["idempotency_key"]) >= 24

    # A Cloud whose settle route doesn't take the adoption says so.
    cloud.routes[("POST", "/inbox_a/settle")] = CloudApiError(
        "api_request_rejected",
        "HTTP 422.",
        status_code=422,
        rejected_fields=(("body.profile_adoption", "extra_forbidden"),),
    )
    assert main([*command, "--adopt-profile", "--reason", "x", "--json"]) == 1
    assert _json(capsys)["error"]["code"] == "profile_adoption_unsupported"


def test_session_show_names_the_item_the_local_inbox_lists(monkeypatch, tmp_path, capsys) -> None:
    _local(monkeypatch, tmp_path, policy=None)
    item = _only_item(capsys)
    database = str(tmp_path / "sessions.sqlite")
    assert main(["session", "show", item["session_id"], "--sqlite", database, "--json"]) == 0
    pending = _json(capsys)["pending_action"]
    assert pending["inbox_items"] == [{"id": item["id"], "kind": "approval"}]


def test_a_session_is_not_settled_by_its_only_listed_item_when_the_listing_is_incomplete(
    monkeypatch, capsys
) -> None:
    unreadable = CloudApiError("api_request_failed", "HTTP 502.", status_code=502)
    _cloud_cli(
        monkeypatch,
        {
            ("GET", ""): {
                "items": [_PENDING],
                "agents": [{"state": "read", "issues": [], "earlier_release_items": ["inbox_e"]}],
                "complete": True,
            },
            ("GET", "/inbox_e"): unreadable,
        },
    )
    assert main(["inbox", "deny", "--session", "s1", "--application", "bugs", "--json"]) == 1
    assert _json(capsys)["error"]["code"] == "listing_incomplete"


def test_an_earlier_item_whose_state_cant_be_read_makes_the_listing_incomplete(
    monkeypatch, capsys
) -> None:
    _cloud_cli(
        monkeypatch,
        {
            ("GET", ""): {
                "items": [],
                "agents": [{"state": "read", "issues": [], "earlier_release_items": ["inbox_u"]}],
                "complete": True,
            },
            ("GET", "/inbox_u"): {"id": "inbox_u", "kind": "approval", "status": "unavailable"},
        },
    )
    assert main(["inbox", "list", "--application", "bugs", "--json"]) == 0
    listing = _json(capsys)
    assert (listing["items"], listing["complete"]) == ([], False)


def test_clouds_profile_changed_refusal_keeps_its_code(monkeypatch) -> None:
    import httpx

    import cayu.cli._cloud_api as cloud_api

    state = {**_SETTLED["item"], "status": "pending", "settlement": None}
    detail = {"code": "profile_changed", "message": "The profile changed.", "item": state}
    real_client = httpx.Client
    monkeypatch.setattr(
        cloud_api.httpx,
        "Client",
        lambda **options: real_client(
            transport=httpx.MockTransport(lambda _: httpx.Response(409, json={"detail": detail})),
            **options,
        ),
    )
    inbox = CloudInbox(cloud_api.CloudApiClient(api_url="https://cloud.example", api_key="k"), "a")
    with pytest.raises(InboxClientError) as refused:
        _run(inbox.settle("inbox_a", {"decision": "approve"}))
    assert refused.value.code == "profile_changed" and refused.value.item == state


def test_session_show_names_no_item_for_a_session_id_output_would_redact(tmp_path) -> None:
    from cayu.cli._session_reader import StoreSessionReader

    reader = StoreSessionReader.__new__(StoreSessionReader)
    secret = "sk-ant-api03-" + "A" * 40
    assert _run(reader._inbox_items(f"session-{secret}")) is None


class _PagedCloud(_Cloud):
    """A Cayu Cloud that pages an Agent's inbox: one page per request, and its counts."""

    def __init__(self, items, *, counts=True):
        super().__init__({})
        self.items = items
        self.counted = counts

    def request(self, method, path, **options):
        self.calls.append((method, path, options))
        query = options.get("query") or {}
        if path.endswith("/inbox/counts"):
            if not self.counted:
                raise CloudApiError("api_request_rejected", "HTTP 404.", status_code=404)
            counts = dict.fromkeys(("approval", "question", "recovery", "knowledge"), 0)
            for item in self.items:
                counts[item["kind"]] += 1
            return {"counts": counts, "total_count": len(self.items), "complete": True}
        wanted = [item for item in self.items if query.get("kind") in (None, item["kind"])]
        start = int(query.get("cursor", 0))
        end = start + int(query["limit"])
        more = end < len(wanted)
        return {
            "items": wanted[start:end],
            "agents": [{"state": "read", "issues": [], "earlier_release_items": []}],
            "complete": not more,
            "next_cursor": str(end) if more else None,
            "has_more": more,
        }


def _waiting() -> list[dict]:
    """Seven items in the Agent's own order, which isn't oldest first."""

    kinds = ["approval", "recovery", "recovery", "approval", "recovery", "knowledge", "knowledge"]
    return [
        {
            **_PENDING,
            "id": f"inbox_{index}",
            "kind": kind,
            "session_id": f"s{index}",
            "created_at": f"2026-11-0{7 - index}T10:00:00Z",
            "asked_before_release": index == 2,
        }
        for index, kind in enumerate(kinds)
    ]


def test_a_large_cloud_inbox_is_listed_whole_page_by_page(monkeypatch, capsys) -> None:
    import cayu.cli._inbox_client as inbox_client

    monkeypatch.setattr(inbox_client, "_CLOUD_PAGE_LIMIT", 2)
    items = _waiting()
    cloud = _PagedCloud(items)
    _cloud_cli(monkeypatch, {})
    monkeypatch.setattr("cayu.cli.cloud.cloud_client", lambda **_kwargs: cloud)

    assert main(["inbox", "list", "--application", "bugs", "--json"]) == 0
    listing = _json(capsys)
    # Every item, oldest first, from four pages, each one request to Cloud.
    assert [item["id"] for item in listing["items"]] == [f"inbox_{n}" for n in range(6, -1, -1)]
    assert (listing["complete"], listing["next_cursor"]) == (True, None)
    assert [call[2]["query"].get("cursor") for call in cloud.calls] == [None, "2", "4", "6"]
    # Counted from the whole listing, without another request.
    assert listing["counts"] == {"approval": 2, "question": 0, "recovery": 3, "knowledge": 2}
    assert not any(path.endswith("/counts") for _method, path, _options in cloud.calls)

    # Each kind is listed whole the same way.
    cloud.calls.clear()
    assert main(["inbox", "list", "--application", "bugs", "--kind", "recovery", "--json"]) == 0
    recovery = _json(capsys)
    assert [item["id"] for item in recovery["items"]] == ["inbox_4", "inbox_2", "inbox_1"]
    assert recovery["complete"] is True and recovery["counts"] == {"recovery": 3}
    assert {call[2]["query"]["kind"] for call in cloud.calls} == {"recovery"}
    assert len(cloud.calls) == 2

    # The table keeps the items asked before the Release in their own section.
    assert main(["inbox", "list", "--application", "bugs"]) == 0
    out = capsys.readouterr().out
    current, earlier = out.split("Asked before the Agent's current Release (1).")
    assert "inbox_2" in earlier and "inbox_2" not in current
    assert "Waiting: approval 2, question 0, recovery 3, knowledge 2." in out


def test_a_cloud_inbox_is_paged_with_a_limit_and_cursor_in_a_stable_order(
    monkeypatch, capsys
) -> None:
    items = _waiting()
    cloud = _PagedCloud(items)
    _cloud_cli(monkeypatch, {})
    monkeypatch.setattr("cayu.cli.cloud.cloud_client", lambda **_kwargs: cloud)

    pages: list[list[str]] = []
    cursor = None
    for _ in range(10):
        command = ["inbox", "list", "--application", "bugs", "--limit", "3", "--json"]
        assert main(command + (["--cursor", cursor] if cursor else [])) == 0
        page = _json(capsys)
        pages.append([item["id"] for item in page["items"]])
        # A page can't count what waits, so the inbox counts it.
        assert page["counts"] == {"approval": 2, "question": 0, "recovery": 3, "knowledge": 2}
        cursor = page["next_cursor"]
        assert page["complete"] is (cursor is None)
        if cursor is None:
            break
    # Pages in the Agent's own order, each item once.
    assert pages == [
        ["inbox_0", "inbox_1", "inbox_2"],
        ["inbox_3", "inbox_4", "inbox_5"],
        ["inbox_6"],
    ]

    # A kind pages the same way, the filter going to Cloud with each page.
    assert (
        main(["inbox", "list", "--application", "bugs", "--kind", "recovery", "--limit", "2"]) == 0
    )
    captured = capsys.readouterr()
    assert "inbox_1" in captured.out and "inbox_4" not in captured.out
    assert "more items follow; list them with --cursor 2" in captured.err

    # A Cloud that can't count says so, and the page is listed anyway.
    cloud.counted = False
    assert main(["inbox", "list", "--application", "bugs", "--limit", "3", "--json"]) == 0
    page = _json(capsys)
    assert (page["counts"], page["counts_unavailable_reason"]) == (None, "not_served")
    assert len(page["items"]) == 3


def test_an_older_cloud_without_paging_still_lists_its_bounded_read(monkeypatch, capsys) -> None:
    # It reads `counts` as an item id, which isn't one.
    not_an_item = CloudApiError("api_request_rejected", "HTTP 422.", status_code=422)
    _cloud_cli(
        monkeypatch,
        {
            ("GET", ""): {"items": [_PENDING], "agents": [{"state": "read"}], "complete": False},
            ("GET", "/counts"): not_an_item,
        },
    )
    assert main(["inbox", "list", "--application", "bugs", "--json"]) == 0
    listing = _json(capsys)
    assert [item["id"] for item in listing["items"]] == ["inbox_a"]
    assert (listing["complete"], listing["next_cursor"], listing["counts"]) == (False, None, None)
    # Asked for a page, it says the answer is that bounded read.
    assert main(["inbox", "list", "--application", "bugs", "--limit", "5", "--json"]) == 0
    page = _json(capsys)
    assert [item["id"] for item in page["items"]] == ["inbox_a"]
    assert "doesn't page an Agent's inbox yet" in page["notes"][-1]


def test_the_local_inbox_pages_every_kind_and_counts_it(tmp_path) -> None:
    from tests.server.test_inbox_paging import APPROVALS, PROPOSALS, RECOVERIES, large_inbox

    from cayu.cli._inbox_client import LocalInbox

    app = large_inbox(tmp_path)

    async def scenario() -> None:
        inbox = LocalInbox(app, reviewer="local:me")
        try:
            whole = await inbox.list_items(kind=None, session_id=None, agent_name=None)
            assert whole.complete and whole.next_cursor is None
            assert len(whole.items) == APPROVALS + RECOVERIES + PROPOSALS
            created = [str(item["created_at"]) for item in whole.items]
            assert created == sorted(created)
            for kind in (None, "approval", "recovery", "knowledge"):
                paged: list[str] = []
                cursor = None
                while True:
                    page = await inbox.list_items(
                        kind=kind, session_id=None, agent_name=None, limit=4, cursor=cursor
                    )
                    assert len(page.items) <= 4
                    paged.extend(item["id"] for item in page.items)
                    cursor = page.next_cursor
                    if cursor is None:
                        assert page.complete
                        break
                expected = {item["id"] for item in whole.items if kind in (None, item["kind"])}
                assert len(paged) == len(set(paged)) and set(paged) == expected
            counts = await inbox.counts(session_id=None, agent_name=None)
            assert counts.counts == {
                "approval": APPROVALS,
                "question": 0,
                "recovery": RECOVERIES,
                "knowledge": PROPOSALS,
            }
        finally:
            await inbox.aclose()
            await app.aclose()

    asyncio.run(scenario())


def _record(record_id: str, outcome: str, **fields: Any) -> dict[str, Any]:
    return {
        "id": record_id,
        "application_id": "bugs",
        "item_id": "inbox_a",
        "item_kind": "approval",
        "item_title": "Tool approval required",
        "session_id": "s1",
        "attention_id": None,
        "decision": "approve",
        "with_changes": False,
        "decided_by": "user_1",
        "role": "member",
        "requested_at": "2026-11-06T06:00:00+00:00",
        "outcome": outcome,
        "outcome_code": None,
        "completed_at": "2026-11-06T06:00:31+00:00",
        "agent_decided_by": None,
        "agent_decided_at": None,
        **fields,
    }


class _DecisionsCloud:
    """Cloud's decision records route, answering ``pages`` in turn."""

    def __init__(self, *pages: dict[str, Any]) -> None:
        self.pages = list(pages)
        self.calls: list[tuple[str, str, dict]] = []

    def request(self, method, path, **options):
        self.calls.append((method, path, options))
        return self.pages.pop(0)


def test_decisions_lists_clouds_records_chosen_by_time_outcome_and_session(
    monkeypatch, capsys
) -> None:
    filters = {
        "since": "2026-11-05T17:00:00+00:00",
        "until": None,
        "outcome": ["failed", "refused"],
        "item_id": None,
        "session_id": "s1",
    }
    cloud = _DecisionsCloud(
        {
            "items": [_record("decision_2", "failed", outcome_code="timeout")],
            "next_cursor": "c1",
            "filters": filters,
        },
        {
            "items": [_record("decision_1", "refused", outcome_code="already_settled")],
            "next_cursor": None,
            "filters": filters,
        },
    )
    import cayu.cli.cloud as cloud_cli

    monkeypatch.setattr(cloud_cli, "cloud_client", lambda **_kwargs: cloud)
    monkeypatch.setattr(cloud_cli, "resolve_application_id", lambda _client, name: f"id-{name}")

    assert (
        main(
            [
                "inbox",
                "decisions",
                "--application",
                "bugs",
                "--since",
                "2026-11-05T17:00:00Z",
                "--outcome",
                "failed",
                "--outcome",
                "refused",
                "--session",
                "s1",
                "--json",
            ]
        )
        == 0
    )

    printed = _json(capsys)
    assert [row["id"] for row in printed["items"]] == ["decision_2", "decision_1"]
    assert printed["filters"] == filters
    assert printed["next_cursor"] is None
    # The same filters go with each page, and the second goes on from the first's cursor.
    asked = [call[2]["query"] for call in cloud.calls]
    assert [call[1] for call in cloud.calls] == ["/v1/inbox/decisions"] * 2
    assert asked[0] == {
        "since": "2026-11-05T17:00:00+00:00",
        "outcome": ["failed", "refused"],
        "session_id": "s1",
        "application_id": "id-bugs",
        "limit": "50",
    }
    assert asked[1] == {**asked[0], "limit": "49", "cursor": "c1"}


def test_decisions_prints_what_the_agent_answered_and_where_to_go_on(monkeypatch, capsys) -> None:
    cloud = _DecisionsCloud(
        {
            "items": [_record("decision_2", "failed", outcome_code="timeout")],
            "next_cursor": "c1",
            "filters": {"outcome": [], "since": None, "until": None, "item_id": None},
        }
    )
    import cayu.cli.cloud as cloud_cli

    monkeypatch.setattr(cloud_cli, "cloud_client", lambda **_kwargs: cloud)
    monkeypatch.setattr(cloud_cli, "resolve_application_id", lambda _client, name: name)

    assert main(["inbox", "decisions", "--application", "bugs", "--limit", "1"]) == 0

    out, err = capsys.readouterr()
    header, row = out.splitlines()
    assert header.split() == [
        "REQUESTED",
        "DECISION",
        "OUTCOME",
        "CODE",
        "KIND",
        "ITEM",
        "SESSION",
        "BY",
    ]
    assert row.split() == [
        "2026-11-06T06:00:00+00:00",
        "approve",
        "failed",
        "timeout",
        "approval",
        "inbox_a",
        "s1",
        "user_1",
    ]
    assert "--cursor c1" in err
    assert cloud.calls[0][2]["query"] == {"application_id": "bugs", "limit": "1"}


def test_decisions_refuses_filters_an_older_cloud_would_ignore(monkeypatch, capsys) -> None:
    cloud = _DecisionsCloud({"items": [_record("decision_1", "settled")], "next_cursor": None})
    import cayu.cli.cloud as cloud_cli

    monkeypatch.setattr(cloud_cli, "cloud_client", lambda **_kwargs: cloud)
    monkeypatch.setattr(cloud_cli, "resolve_application_id", lambda _client, name: name)

    assert (
        main(["inbox", "decisions", "--application", "bugs", "--outcome", "failed", "--json"]) == 1
    )
    assert _json(capsys)["error"]["code"] == "decision_filters_unsupported"

    # Without filters, it lists what that Cloud records.
    cloud.pages.append({"items": [_record("decision_1", "settled")], "next_cursor": None})
    assert main(["inbox", "decisions", "--application", "bugs", "--json"]) == 0
    assert [row["id"] for row in _json(capsys)["items"]] == ["decision_1"]


def test_decisions_needs_a_cloud_agent(capsys) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["inbox", "decisions"])
    assert exited.value.code == 2
    assert "--application" in capsys.readouterr().err


def test_decisions_refuse_pages_that_repeat_a_cursor() -> None:
    cloud = _DecisionsCloud(
        {"items": [_record("decision_2", "settled")], "next_cursor": "c1", "filters": {}},
    )

    with pytest.raises(InboxClientError) as refused:
        _run(CloudInbox(cloud, "a").decisions(filters={}, limit=50, cursor="c1"))

    assert refused.value.code == "api_response_invalid"
    assert len(cloud.calls) == 1
