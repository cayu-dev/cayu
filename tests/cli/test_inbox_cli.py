"""`cayu inbox`: list, show and decide an agent's approvals, locally and on Cayu Cloud."""

from __future__ import annotations

import asyncio
import getpass
import json
from contextlib import nullcontext
from types import SimpleNamespace

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
