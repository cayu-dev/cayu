"""`cayu cloud inbox` lists what Agents wait on a person for and decides it through Cloud."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from tests.cli.test_cloud_api_keys import cloud as _cloud

from cayu.cli import main
from cayu.cli._cloud_inbox import DECISIONS

cloud = _cloud

_ME = "/v1/me"
_ORGANIZATIONS = "/v1/me/organizations"
_APPS = "/v1/applications?lifecycle=all"
_AGENTS = {"items": [{"id": "month-end-close", "name": "Month-end close"}]}
_INBOX = "/v1/applications/month-end-close/inbox"
_IDENTITY = httpx.Response(200, json={"organization_id": "org_1"})
_NAMED = httpx.Response(200, json={"items": [{"id": "org_1", "name": "Acme"}]})
_SETTLED = {"settled": True, "replayed": False, "item": {"status": "settled"}}


def _run(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, dict[str, Any]]:
    code = main(["cloud", "inbox", *arguments])
    return code, json.loads(capsys.readouterr().out)


def _writes(fake: Any) -> list[tuple[str, object, str | None]]:
    return [
        (
            request.url.raw_path.decode(),
            json.loads(request.content) if request.content else None,
            request.headers.get("Idempotency-Key"),
        )
        for request in fake.requests
        if request.method == "POST"
    ]


class _AnyQuery(dict):
    """Routes that answer a path whatever its query, beside the exact ones."""

    def __init__(self, exact: dict[Any, Any], by_path: dict[tuple[str, str], Any]) -> None:
        super().__init__(exact)
        self.by_path = by_path

    def __missing__(self, key: tuple[str, str]) -> Any:
        method, raw_path = key
        return self.by_path[(method, raw_path.partition("?")[0])]


def test_help_lists_every_inbox_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["cloud", "inbox", "--help"])

    assert raised.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    for command in ("list", "show", *DECISIONS, "decisions"):
        assert command in help_text
    assert "cayu inbox --application NAME" in help_text


def test_list_reads_every_agent_or_one_agents_own_inbox_without_an_organization_check(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = cloud(
        {
            ("GET", "/v1/inbox"): httpx.Response(
                200, json={"items": [{"id": "inbox_1"}], "agents": [], "complete": True}
            ),
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("GET", _INBOX): httpx.Response(200, json={"items": [], "complete": True}),
            ("GET", f"{_INBOX}/inbox%2F2"): httpx.Response(
                200, json={"id": "inbox/2", "status": "pending"}
            ),
        }
    )

    code, everything = _run(["list"], capsys)
    assert (code, everything["operation"]) == (0, "inbox.list"), everything
    assert everything["result"]["items"] == [{"id": "inbox_1"}]

    code, one = _run(["list", "--application", "Month-end close"], capsys)
    assert (code, one["result"]["complete"]) == (0, True), one

    code, shown = _run(["show", "inbox/2", "--application", "month-end-close"], capsys)
    assert (code, shown["operation"], shown["result"]["status"]) == (0, "inbox.show", "pending")

    assert "organization" not in everything and "organization" not in shown
    assert all(request.url.path not in {_ME, _ORGANIZATIONS} for request in fake.requests)


def test_each_decision_is_settled_on_the_agent_under_a_key(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    changed = tmp_path / "arguments.json"
    changed.write_text(json.dumps({"vendor": "Example Vendor"}))
    answer = tmp_path / "answer.txt"
    answer.write_text("Yes, hold it.")
    routes = {
        ("GET", _ME): _IDENTITY,
        ("GET", _ORGANIZATIONS): _NAMED,
        ("GET", _APPS): httpx.Response(200, json=_AGENTS),
    }
    for item in ("a", "b", "c", "d", "e", "f"):
        routes[("POST", f"{_INBOX}/{item}/settle")] = httpx.Response(200, json=_SETTLED)
    fake = cloud(routes)
    agent = ["--application", "Month-end close"]

    for arguments in (
        ["approve", "a", *agent, "--reason", "Recoded.", "--arguments-file", str(changed)],
        ["deny", "b", *agent, "--idempotency-key", "deny-1"],
        ["answer", "c", *agent, "--answer-file", str(answer)],
        ["dismiss", "d", *agent, "--reason", "The run ended long ago."],
        ["keep", "e", *agent],
        ["drop", "f", *agent],
    ):
        code, output = _run(arguments, capsys)
        assert code == 0, output
        assert output["operation"] == f"inbox.{arguments[0]}"
        assert output["organization"]["id"] == "org_test"
        assert output["result"]["settled"] is True

    writes = _writes(fake)
    assert [(path, body) for path, body, _key in writes] == [
        (
            f"{_INBOX}/a/settle",
            {
                "decision": "approve",
                "reason": "Recoded.",
                "arguments": {"vendor": "Example Vendor"},
            },
        ),
        (f"{_INBOX}/b/settle", {"decision": "deny"}),
        (f"{_INBOX}/c/settle", {"decision": "answer", "answer": "Yes, hold it."}),
        (f"{_INBOX}/d/settle", {"decision": "dismiss", "reason": "The run ended long ago."}),
        (f"{_INBOX}/e/settle", {"decision": "keep"}),
        (f"{_INBOX}/f/settle", {"decision": "drop"}),
    ]
    keys = [key for _path, _body, key in writes]
    assert keys[1] == "deny-1"
    for key, decision in zip(keys, DECISIONS, strict=True):
        assert key is not None
        if decision != "deny":
            assert key.startswith(f"inbox-{decision}-")


def test_a_refused_decision_names_its_code_key_and_the_items_state(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            ("GET", _ME): _IDENTITY,
            ("GET", _ORGANIZATIONS): _NAMED,
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
            ("POST", f"{_INBOX}/a/settle"): httpx.Response(
                409,
                json={
                    "detail": {
                        "code": "already_settled",
                        "message": "Someone decided this first.",
                        "item": {"status": "settled", "settlement": {"decision": "deny"}},
                    }
                },
            ),
        }
    )

    code, output = _run(["approve", "a", "--application", "month-end-close"], capsys)

    assert code == 2
    assert output["error"]["code"] == "already_settled"
    assert output["error"]["idempotency_key"].startswith("inbox-approve-")
    assert output["error"]["item"]["settlement"]["decision"] == "deny"


def test_approve_refuses_arguments_that_are_not_a_json_object_before_sending(
    cloud, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    changed = tmp_path / "arguments.json"
    changed.write_text("[1]")
    fake = cloud(
        {
            ("GET", _ME): _IDENTITY,
            ("GET", _ORGANIZATIONS): _NAMED,
            ("GET", _APPS): httpx.Response(200, json=_AGENTS),
        }
    )

    code, output = _run(
        ["approve", "a", "--application", "month-end-close", "--arguments-file", str(changed)],
        capsys,
    )

    assert (code, output["error"]["category"]) == (2, "invalid_input")
    assert _writes(fake) == []


def test_a_decision_needs_its_agent(capsys: pytest.CaptureFixture[str]) -> None:
    code, output = _run(["keep", "a"], capsys)

    assert (code, output["error"]["category"]) == (2, "invalid_input")
    assert "--application" in output["error"]["message"]


def test_decisions_sends_its_filters_and_paging(cloud, capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[httpx.QueryParams] = []

    def records(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.params)
        return httpx.Response(
            200, json={"items": [], "next_cursor": None, "filters": {"outcome": ["failed"]}}
        )

    fake = cloud({("GET", _APPS): httpx.Response(200, json=_AGENTS)})
    # Every query reaches the same records; the test reads what each request asked for.
    fake.routes = _AnyQuery(fake.routes, {("GET", "/v1/inbox/decisions"): records})

    code, everything = _run(["decisions"], capsys)
    assert (code, everything["operation"]) == (0, "inbox.decisions"), everything

    code, output = _run(
        [
            "decisions",
            "--application",
            "month-end-close",
            "--since",
            "2026-11-05T17:00:00Z",
            "--outcome",
            "failed",
            "--outcome",
            "refused",
            "--session",
            "close-september",
            "--limit",
            "10",
            "--cursor",
            "c1",
        ],
        capsys,
    )
    assert code == 0, output
    assert seen[0] == httpx.QueryParams()
    assert seen[1].get_list("outcome") == ["failed", "refused"]
    assert (seen[1]["since"], seen[1]["application_id"], seen[1]["limit"]) == (
        "2026-11-05T17:00:00+00:00",
        "month-end-close",
        "10",
    )


def test_decisions_refuses_filters_a_cloud_did_not_apply(
    cloud, capsys: pytest.CaptureFixture[str]
) -> None:
    cloud(
        {
            ("GET", "/v1/inbox/decisions?item_id=a"): httpx.Response(
                200, json={"items": [{"id": "dec_1"}], "next_cursor": None}
            )
        }
    )

    code, output = _run(["decisions", "--item", "a"], capsys)

    assert code == 2
    assert output["error"]["category"] == "decision_filters_unsupported"


def test_decisions_limit_is_one_page_of_cloud(capsys: pytest.CaptureFixture[str]) -> None:
    code, output = _run(["decisions", "--limit", "101"], capsys)

    assert (code, output["error"]["category"]) == (2, "invalid_input")
    assert "1 to 100" in output["error"]["message"]
