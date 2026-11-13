"""`cayu session start` and `cayu inbox answer`: demonstration data from the terminal."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from cayu.cli import main

_FACTORY = """from cayu import AgentSpec, CayuApp, Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.evals import ScriptedModelProvider
from cayu.providers import ProviderNotConfigured
from cayu.resource_access import ResourceAccessPolicy
from cayu.runtime import ExecutionProfileBehaviorIdentity
from cayu.sessions._access_rules import (
    SessionAccessRule,
    SessionAccessScope,
    SessionAccessSelector,
)
from cayu.storage import SQLiteSessionStore
from cayu.tools import ToolPolicy, ToolPolicyDecision, ToolPolicyResult
from cayu.tools.user_input import UserInputTool


def identity(name):
    return ExecutionProfileBehaviorIdentity(
        name=name, behavior_version="1", implementation_version="1"
    )


class Unselected(ScriptedModelProvider):
    def __init__(self):
        super().__init__([], name="unconfigured")

    def preflight_model_target(self, *, model):
        raise ProviderNotConfigured("No model provider is set. Set CAYU_PROVIDER.")


class Organizations(ResourceAccessPolicy):
    authority = "organizations-v1"

    async def resolve(self, subject):
        organization = {"alice": "acme", "bob": "globex"}[subject]
        rule = SessionAccessRule(
            selectors=(SessionAccessSelector(key="organization", values=(organization,)),)
        )
        return SessionAccessScope(read=(rule,), create=(rule,), execute=(rule,), inspect_state=(rule,))


class Record(Tool):
    spec = ToolSpec(
        name="record",
        description="Record a note.",
        input_schema={"type": "object", "properties": {"note": {"type": "string"}}},
        effect=ToolEffect.IDEMPOTENT,
        max_terminal_payload_bytes=1024,
        execution_profile_identity=identity("record"),
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content="Recorded " + args["note"] + ".")


class ApproveRecords(ToolPolicy):
    execution_profile_identity = identity("approve-records")

    async def authorize(self, request):
        if request.tool_name == "record":
            return ToolPolicyResult(decision=ToolPolicyDecision.REQUIRE_APPROVAL, reason="Check it.")
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


def build_app():
    app = CayuApp(
        session_store=SQLiteSessionStore("runtime.db"),
        resource_access_policy=Organizations(),
        enable_logging=False,
    )
    app.register_provider(Unselected(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="unset"),
        tools=[UserInputTool(), Record()],
        tool_policy=ApproveRecords(),
    )
    return app
"""

_QUESTION = '{"question": "Which week suits the auditors?", "options": ["17 Nov", "24 Nov"]}'


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "start_project:build_app"\n', encoding="utf-8"
    )
    (project / "start_project.py").write_text(_FACTORY, encoding="utf-8")
    sys.modules.pop("start_project", None)
    monkeypatch.chdir(project)
    return project


def _json(capsys: pytest.CaptureFixture[str]) -> dict:
    return json.loads(capsys.readouterr().out)


def _start(*arguments: str) -> int:
    return main(
        [
            "session",
            "start",
            "--agent",
            "assistant",
            "--subject",
            "alice",
            "--label",
            "organization=acme",
            "--json",
            *arguments,
        ]
    )


def test_a_scripted_reply_starts_a_finished_demo_conversation_as_the_person(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        _start(
            "--session-id",
            "close-checklist",
            "--message",
            "What is still open for the Q3 close?",
            "--scripted-reply",
            "Three things are open.",
        )
        == 0
    )

    started = _json(capsys)
    assert started["scripted"] is True and started["failure"] is None
    assert started["session"]["id"] == "close-checklist"
    assert started["session"]["status"] == "completed"
    assert started["session"]["provider_name"] == "demo"
    assert started["session"]["labels"] == {"organization": "acme"}

    assert main(["session", "transcript", "close-checklist", "--sqlite", "runtime.db"]) == 0
    assert "Three things are open." in capsys.readouterr().out


def test_the_resource_access_policy_decides_who_may_start_it(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "session",
                "start",
                "--agent",
                "assistant",
                "--subject",
                "bob",
                "--label",
                "organization=acme",
                "--message",
                "hi",
                "--scripted-reply",
                "hello",
                "--json",
            ]
        )
        == 1
    )
    assert _json(capsys)["error"]["code"] == "SESSION_START_FAILED"


def test_without_a_script_the_projects_provider_answers_and_says_what_to_set(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _start("--message", "hi") == 1

    assert "No model provider is set. Set CAYU_PROVIDER." in _json(capsys)["error"]["message"]


def test_a_scripted_question_is_listed_shown_with_numbers_and_answered_from_the_terminal(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        _start(
            "--session-id",
            "auditor-visit",
            "--message",
            "Set up the auditor's visit.",
            "--scripted-tool-call",
            f"ask_user={_QUESTION}",
        )
        == 0
    )
    started = _json(capsys)
    assert [action["kind"] for action in started["waiting_on"]] == ["user_input"], started

    assert main(["inbox", "list", "--json"]) == 0
    [item] = _json(capsys)["items"]
    assert item["kind"] == "question"

    assert main(["inbox", "show", item["id"]]) == 0
    shown = capsys.readouterr().out
    assert "1. 17 Nov" in shown and "2. 24 Nov" in shown

    assert main(["inbox", "answer", item["id"], "--option", "3", "--reviewer", "carol"]) == 1
    assert "--option takes 1 to 2" in capsys.readouterr().err

    assert (
        main(
            [
                "inbox",
                "answer",
                "--session",
                "auditor-visit",
                "--option",
                "1",
                "--reviewer",
                "carol",
                "--json",
            ]
        )
        == 0
    )
    answered = _json(capsys)
    assert answered["decision"] == "answer"
    settlement = answered["item"]["settlement"]
    assert settlement["decision"] == "answer"
    assert settlement["decided_by"]["subject"] == "carol"
    assert answered["session_status"] == "completed"

    assert main(["session", "transcript", "auditor-visit", "--sqlite", "runtime.db"]) == 0
    assert "Demonstration reply, no model was called. Noted: 17 Nov" in capsys.readouterr().out


def test_a_scripted_call_needing_approval_waits_in_the_inbox(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        _start(
            "--message",
            "Note the decision.",
            "--scripted-tool-call",
            'record={"note": "week of 17 Nov"}',
        )
        == 0
    )
    started = _json(capsys)
    assert [action["tool_name"] for action in started["waiting_on"]] == ["record"], started

    assert main(["inbox", "list", "--json"]) == 0
    [item] = _json(capsys)["items"]
    assert item["kind"] == "approval" and item["tool_name"] == "record"

    assert main(["inbox", "answer", item["id"], "yes", "--reviewer", "carol"]) == 1
    assert "answers questions; this item is a approval" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--scripted-tool-call", "ask_user"], "is TOOL=JSON"),
        (["--scripted-tool-call", "ask_user=[1]"], "must be a JSON object"),
        (["--scripted-tool-call", "ask_user={nope"], "aren't JSON"),
        (["--label", "organization"], "a label is KEY=VALUE"),
        (["--scripted-reply", "one", "--scripted-reply", "two"], "nothing can follow"),
        (
            ["--scripted-tool-call", f"ask_user={_QUESTION}", "--scripted-reply", "Booked."],
            "nothing can follow",
        ),
    ],
)
def test_malformed_scripted_turns_are_refused(
    project: Path, capsys: pytest.CaptureFixture[str], arguments: list[str], message: str
) -> None:
    with pytest.raises(SystemExit):
        _start("--message", "hi", *arguments)
    assert message in capsys.readouterr().err


def test_answer_takes_text_or_an_option_not_both(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["inbox", "answer", "inbox_x", "yes", "--option", "1"]) == 1
    assert "TEXT or with --option N, not both" in capsys.readouterr().err


def test_turns_after_a_call_that_waits_for_approval_are_reported_unplayed(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        _start(
            "--message",
            "Note it.",
            "--scripted-tool-call",
            'record={"note": "x"}',
            "--scripted-reply",
            "Noted.",
        )
        == 1
    )
    started = _json(capsys)
    assert "stopped after 1 of 2 scripted turns" in started["failure"]


def test_an_unknown_agent_is_an_error_not_a_traceback(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["session", "start", "--agent", "nope", "--message", "hi", "--json"]) == 1
    assert "nope" in _json(capsys)["error"]["message"]


def test_a_cloud_question_without_a_review_policy_is_answered_without_a_reference() -> None:
    import asyncio
    from types import SimpleNamespace

    import cayu.cli.inbox as inbox_cli
    from cayu.cli._inbox_client import InboxClientError

    class Cloud:
        where = "cloud"
        purpose = None
        reviews_configured = None

        async def review(self, item_id, session_id, *, purpose):
            raise InboxClientError("http_403", "Human-review access is not authorized.")

    args = SimpleNamespace(item_id="inbox_q", purpose=None)
    item = {"kind": "question", "session_id": "s", "arguments_unavailable_reason": None}
    assert asyncio.run(inbox_cli._decision_reference(args, Cloud(), item, "answer")) is None
    approval = {**item, "kind": "approval"}
    with pytest.raises(InboxClientError):
        asyncio.run(inbox_cli._decision_reference(args, Cloud(), approval, "approve"))
