from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from cayu.cli import main
from cayu.execution_profiles import ExecutionProfileComponentClass

_PROJECT = """import os
from pathlib import Path

from cayu import AgentSpec, CayuApp, Tool, ToolEffect, ToolResult, ToolSpec
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.evals import ScriptedModelProvider
from cayu.storage import SQLiteSessionStore


class LookupTool(Tool):
    def __init__(self):
        self.spec = ToolSpec(
            name="lookup",
            description=os.environ.get("TOOL_DESCRIPTION", "Look something up."),
            effect=ToolEffect.NONE,
            input_schema={"type": "object", "additionalProperties": False},
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="tests:lookup", behavior_version="1", implementation_version="1"
            ),
        )
        super().__init__()

    async def run(self, ctx, args):
        return ToolResult(content="found")


def build_app():
    app = CayuApp(session_store=SQLiteSessionStore(Path("sessions.db")), enable_logging=False)
    app.register_provider(ScriptedModelProvider([], name="scripted"), default=True)
    app.register_agent(
        AgentSpec(
            name="reviewer",
            model="test-model",
            system_prompt=os.environ.get("SYSTEM_PROMPT", "Review carefully."),
        ),
        tools=[LookupTool()],
    )
    return app
"""


def _session_count(path: Path) -> int:
    # The factory's store may initialize its schema; inspection never admits.
    with sqlite3.connect(path) as connection:
        return connection.execute("SELECT COUNT(*) FROM cayu_sessions").fetchone()[0]


def _project(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "profile_project:build_app"\n',
        encoding="utf-8",
    )
    (tmp_path / "profile_project.py").write_text(_PROJECT, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("profile_project", None)


def test_candidates_and_predict_report_release_profiles_without_admitting_sessions(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _project(tmp_path, monkeypatch)

    assert main(["execution-profile", "candidates", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == "1"
    assert report["execution_profile_policy"] == {"configured": False, "identity": None}
    assert report["errors"] == []
    (candidate,) = report["candidates"]
    assert (candidate["agent_name"], candidate["provider_name"], candidate["model"]) == (
        "reviewer",
        "scripted",
        "test-model",
    )
    profile = candidate["execution_profile"]
    assert len(profile["fingerprint"]) == 64
    assert "Review carefully." not in json.dumps(report)

    # The command resolves a stored session's recorded values exactly.
    assert (
        main(
            [
                "execution-profile",
                "candidates",
                "--no-environment",
                "--causal-budget-id",
                "session-resume",
                "--json",
            ]
        )
        == 0
    )
    (for_session,) = json.loads(capsys.readouterr().out)["candidates"]
    assert (for_session["environment_name"], for_session["causal_budget_id"]) == (
        None,
        "session-resume",
    )

    entries = [
        {
            "id": f"session-{boundary}",
            "boundary": boundary,
            "agent_name": "reviewer",
            "environment_name": None,
            "provider_name": "scripted",
            "model": "test-model",
            "causal_budget_id": f"session-{boundary}",
            "expected_profile": profile,
        }
        for boundary in ("continuation", "resume")
    ]
    stored = tmp_path / "stored.json"
    stored.write_text(json.dumps({"sessions": entries}), encoding="utf-8")

    # A changed system prompt keeps the stored projection: both resume.
    monkeypatch.setenv("SYSTEM_PROMPT", "Review very carefully.")
    sys.modules.pop("profile_project", None)
    assert main(["execution-profile", "predict", "--input", str(stored), "--json"]) == 0
    prompt_change = json.loads(capsys.readouterr().out)
    assert prompt_change["summary"] == {
        "admits": 2,
        "does_not_admit": 0,
        "total": 2,
        "undetermined": 0,
    }
    assert [item["prediction"]["outcome"] for item in prompt_change["predictions"]] == [
        "exact_reuse",
        "exact_reuse",
    ]

    # A changed tool declaration is authority-bearing: neither resumes.
    monkeypatch.setenv("TOOL_DESCRIPTION", "Look up something else.")
    sys.modules.pop("profile_project", None)
    assert main(["execution-profile", "predict", "--input", str(stored), "--json"]) == 3
    tool_change = json.loads(capsys.readouterr().out)
    assert tool_change["summary"]["does_not_admit"] == 2
    for item in tool_change["predictions"]:
        assert item["prediction"]["outcome"] == "rejected"
        assert item["prediction"]["changed_component_classes"] == ["direct_tools"]
        assert item["candidate_fingerprint"] != profile["fingerprint"]
    assert _session_count(tmp_path / "sessions.db") == 0


def test_candidate_and_input_failures_use_the_structured_contract(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _project(tmp_path, monkeypatch)

    assert main(["execution-profile", "candidates", "--agent", "missing", "--json"]) == 4
    report = json.loads(capsys.readouterr().out)
    assert report["candidates"] == []
    assert report["errors"][0]["agent_name"] == "missing"
    assert report["errors"][0]["code"] == "CANDIDATE_UNAVAILABLE"

    assert main(["execution-profile", "candidates", "--provider", "scripted", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "INVALID_ARGUMENTS"

    malformed = tmp_path / "malformed.json"
    malformed.write_text(json.dumps({"sessions": [{"id": "s"}]}), encoding="utf-8")
    assert main(["execution-profile", "predict", "--input", str(malformed), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "INVALID_INPUT"


def test_a_close_failure_does_not_replace_resolved_candidates(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _project(tmp_path, monkeypatch)

    async def failing_close(app) -> None:
        raise RuntimeError("drain failed")

    monkeypatch.setattr("cayu.cli.execution_profile.close_project_app", failing_close)
    assert main(["execution-profile", "candidates", "--json"]) == 0
    captured = capsys.readouterr()
    assert len(json.loads(captured.out)["candidates"]) == 1
    assert "closing the application failed: RuntimeError: drain failed" in captured.err


_COMPARE_PROJECT = """import json
import os
from pathlib import Path

from cayu import AgentSpec, CayuApp, Tool, ToolEffect, ToolResult, ToolSpec
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.runtime import LoopPolicy
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.storage import SQLiteSessionStore
from cayu.tools import AlwaysRequireApprovalToolPolicy

# Tests fill this before creating sessions; positional batches keep the same
# provider identity in the CLI's process, where it stays empty.
BATCHES = []


class CompletionCheck(LoopPolicy):
    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="tests:completion-check", behavior_version="1", implementation_version="1"
        )


# A request loop policy without a declared identity.
class OpaqueCheck(LoopPolicy):
    pass


class LookupTool(Tool):
    def __init__(self):
        version = os.environ.get("TOOL_VERSION", "1")
        self.spec = ToolSpec(
            name="lookup",
            description="Look something up.",
            effect=ToolEffect.NONE,
            input_schema={"type": "object", "additionalProperties": False},
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="tests:lookup", behavior_version=version, implementation_version=version
            ),
        )
        super().__init__()

    async def run(self, ctx, args):
        return ToolResult(content="found")


def build_app():
    app = CayuApp(session_store=SQLiteSessionStore(Path("sessions.db")), enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(list(BATCHES), name=os.environ.get("PROVIDER_NAME", "scripted")),
        default=True,
    )
    app.register_agent(
        AgentSpec(
            name=os.environ.get("AGENT_NAME", "reviewer"),
            model="test-model",
            provider_options=json.loads(os.environ.get("PROVIDER_OPTIONS", "{}")),
        ),
        tools=[LookupTool()],
        tool_policy=AlwaysRequireApprovalToolPolicy(tools=["lookup"]),
    )
    return app
"""


def _compare_project(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "compare_project:build_app"\n',
        encoding="utf-8",
    )
    (tmp_path / "compare_project.py").write_text(_COMPARE_PROJECT, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("compare_project", None)


def _create_session(
    session_id: str,
    *,
    pause_for_approval: bool,
    thinking=None,
    loop_policies: bool = False,
    opaque_loop_policy: bool = False,
) -> None:
    import compare_project

    from cayu import Message, RunRequest
    from cayu.providers import ModelStreamEvent

    first_step = (
        [
            ModelStreamEvent.tool_call(id="call-1", name="lookup", arguments={}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]
        if pause_for_approval
        else [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]
    )
    compare_project.BATCHES[:] = [first_step]

    async def run() -> None:
        app = compare_project.build_app()
        try:
            request = RunRequest(
                agent_name="reviewer",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
                thinking=thinking,
                loop_policies=(
                    (compare_project.OpaqueCheck(),)
                    if opaque_loop_policy
                    else (compare_project.CompletionCheck(),)
                    if loop_policies
                    else ()
                ),
            )
            _ = [event async for event in app.run(request)]
        finally:
            await app.aclose()

    asyncio.run(run())
    sys.modules.pop("compare_project", None)


def _approve_pending(
    session_id: str, *, loop_policies: bool = False, opaque_loop_policy: bool = False
) -> None:
    """Resolve the pending approval in a fresh app, as a real continuation would."""

    import compare_project

    from cayu.approvals import ToolApprovalDecision, ToolApprovalRequest
    from cayu.providers import ModelStreamEvent
    from cayu.sessions import PendingActionQuery

    compare_project.BATCHES[:] = [
        [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]
    ]

    async def resolve() -> None:
        app = compare_project.build_app()
        try:
            pending = await app.session_store.query_pending_actions(
                PendingActionQuery(session_id=session_id)
            )
            (action,) = pending.actions
            request = ToolApprovalRequest(
                session_id=session_id,
                approval_id=action.approval_id,
                tool_round_id=action.round_id,
                tool_call_id=action.tool_call_id,
                decision=ToolApprovalDecision.APPROVE,
                reason="test operator",
                loop_policies=(
                    (compare_project.OpaqueCheck(),)
                    if opaque_loop_policy
                    else (compare_project.CompletionCheck(),)
                    if loop_policies
                    else ()
                ),
            )
            _ = [event async for event in app.resolve_tool_approval(request)]
        finally:
            await app.aclose()

    asyncio.run(resolve())
    sys.modules.pop("compare_project", None)


# Runs the first step in a child process that exits at the named event, with no
# pending approval, user input or tool round.
_CRASH_AT_EVENT = """import asyncio
import os
import sys

import compare_project
from cayu import Message, RunRequest
from cayu.providers import ModelStreamEvent

compare_project.BATCHES[:] = [[ModelStreamEvent.text_delta("first"), ModelStreamEvent.completed()]]


async def main():
    app = compare_project.build_app()
    request = RunRequest(
        agent_name="reviewer", session_id=sys.argv[1], messages=[Message.text("user", "hello")]
    )
    async for event in app.run(request):
        if event.type == sys.argv[2]:
            os._exit(42)


asyncio.run(main())
raise SystemExit("the run never reached the crash event")
"""


def _crash_at_event(tmp_path: Path, session_id: str, event_type: str) -> None:
    import os
    import subprocess

    env = {**os.environ, "PYTHONPATH": os.pathsep.join(path for path in sys.path if path)}
    result = subprocess.run(
        [sys.executable, "-c", _CRASH_AT_EVENT, session_id, event_type],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 42, result.stderr


def _resume(session_id: str) -> str:
    """Continue the session in a fresh app and return its last event type."""

    import compare_project

    from cayu import Message, ResumeRequest
    from cayu.providers import ModelStreamEvent

    compare_project.BATCHES[:] = [
        [ModelStreamEvent.text_delta("second"), ModelStreamEvent.completed()]
    ]

    async def resume() -> str:
        app = compare_project.build_app()
        try:
            request = ResumeRequest(
                session_id=session_id, messages=[Message.text("user", "continue")]
            )
            events = [event async for event in app.resume(request)]
            return str(events[-1].type)
        finally:
            await app.aclose()

    try:
        return asyncio.run(resume())
    finally:
        sys.modules.pop("compare_project", None)


def _event_count(path: Path) -> int:
    with sqlite3.connect(path) as connection:
        return connection.execute("SELECT COUNT(*) FROM cayu_events").fetchone()[0]


def test_compare_shows_stored_and_current_profiles_without_touching_the_session(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _compare_project(tmp_path, monkeypatch)
    _create_session("finished", pause_for_approval=False)
    events = _event_count(tmp_path / "sessions.db")

    assert main(["execution-profile", "compare", "finished", "--json"]) == 0
    unchanged = json.loads(capsys.readouterr().out)
    assert unchanged["boundary"] == "resume"
    assert unchanged["prediction"]["outcome"] == "exact_reuse"
    assert unchanged["guidance"] == []
    assert not any(item["changed"] for item in unchanged["components"])
    assert unchanged["stored_fingerprint"] == unchanged["candidate_fingerprint"]

    # Editing the tool's declared version is a real change, explained class by class.
    monkeypatch.setenv("TOOL_VERSION", "2")
    sys.modules.pop("compare_project", None)
    assert main(["execution-profile", "compare", "finished", "--json"]) == 3
    edited = json.loads(capsys.readouterr().out)
    assert edited["prediction"]["admits"] is False
    changed = [item for item in edited["components"] if item["changed"]]
    assert changed == [
        {
            "component_class": "tool_implementations",
            "stored": "application_versioned",
            "current": "application_versioned",
            "compared": True,
            "changed": True,
        }
    ]
    guidance = "\n".join(edited["guidance"])
    assert "tool_implementations (tool implementations): declared -> declared" in guidance
    assert 'app.resume(ResumeRequest(session_id="finished"' in guidance
    assert "profile_adoption=ExecutionProfileAdoptionIntent(" in guidance
    assert "has no ExecutionProfilePolicy" in guidance

    assert main(["execution-profile", "compare", "finished", "--table"]) == 3
    table = capsys.readouterr().out
    assert "tool_implementations" in table and "changed" in table

    # Comparison is read-only: no admission, decision or rejection evidence.
    assert _event_count(tmp_path / "sessions.db") == events
    assert _session_count(tmp_path / "sessions.db") == 1


def test_compare_withholds_adoption_while_work_is_pending(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    _compare_project(tmp_path, monkeypatch)
    _create_session("paused", pause_for_approval=True)

    monkeypatch.setenv("TOOL_VERSION", "2")
    assert main(["execution-profile", "compare", "paused", "--json"]) == 3
    report = json.loads(capsys.readouterr().out)
    assert report["boundary"] == "continuation"
    guidance = "\n".join(report["guidance"])
    assert "adoption is not available here" in guidance
    assert "profile_adoption=" not in guidance


def test_compare_reports_a_missing_session(tmp_path: Path, monkeypatch, capsys) -> None:
    _compare_project(tmp_path, monkeypatch)

    assert main(["execution-profile", "compare", "missing", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "SESSION_NOT_FOUND"


@pytest.mark.parametrize(
    ("renamed", "expected_error"),
    [
        ("AGENT_NAME", "Agent not registered: reviewer"),
        ("PROVIDER_NAME", "Provider not registered: scripted"),
    ],
    ids=["agent", "provider"],
)
def test_compare_reports_a_missing_registration_as_the_continuation_does(
    tmp_path: Path,
    monkeypatch,
    capsys,
    renamed: str,
    expected_error: str,
) -> None:
    _compare_project(tmp_path, monkeypatch)
    _create_session("paused", pause_for_approval=True)
    monkeypatch.setenv(renamed, "renamed")

    assert main(["execution-profile", "compare", "paused", "--json"]) == 1
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["code"] == "REGISTRATION_MISSING"
    assert error["message"].startswith(f"{expected_error}. The session is still stored")
    assert "register the" in error["message"]
    assert _session_count(tmp_path / "sessions.db") == 1

    # The real continuation fails on the same registry lookup.
    with pytest.raises(KeyError) as raised:
        _approve_pending("paused")
    assert raised.value.args == (expected_error,)


@pytest.mark.parametrize("fresh_policy", [False, True])
def test_compare_predicts_the_rejection_of_an_opaque_request_loop_policy(
    tmp_path: Path,
    monkeypatch,
    capsys,
    fresh_policy: bool,
) -> None:
    from cayu.runtime.execution_profiles import ExecutionProfileMismatchError

    _compare_project(tmp_path, monkeypatch)
    _create_session("paused", pause_for_approval=True, opaque_loop_policy=True)

    assert main(["execution-profile", "compare", "paused", "--json"]) == 3
    report = json.loads(capsys.readouterr().out)
    assert report["boundary"] == "continuation"
    assert report["prediction"]["outcome"] == "rejected"
    assert report["prediction"]["admits"] is False
    assert report["prediction"]["changed_component_classes"] == ["invocation_policies"]
    assert report["not_compared"] == []
    rows = {item["component_class"]: item for item in report["components"]}
    assert rows["invocation_policies"]["stored"] == "process_local"
    assert rows["invocation_policies"]["compared"] is True
    assert "no later process can match it" in "\n".join(report["guidance"])

    # Runtime rejects the continuation whether or not the caller passes a new
    # instance of the same undeclared policy.
    with pytest.raises(ExecutionProfileMismatchError) as raised:
        _approve_pending("paused", opaque_loop_policy=fresh_policy)
    assert raised.value.changed_component_classes == (
        ExecutionProfileComponentClass.INVOCATION_POLICIES,
    )


@pytest.mark.parametrize("loop_policies", [False, True])
def test_compare_reuses_recorded_request_settings_for_a_pending_continuation(
    tmp_path: Path,
    monkeypatch,
    capsys,
    loop_policies: bool,
) -> None:
    from cayu.context.thinking import ThinkingConfig

    _compare_project(tmp_path, monkeypatch)
    _create_session(
        "paused",
        pause_for_approval=True,
        thinking=ThinkingConfig(effort="low"),
        loop_policies=loop_policies,
    )

    assert main(["execution-profile", "compare", "paused", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["boundary"] == "continuation"
    assert report["prediction"]["changed_component_classes"] == []
    assert report["guidance"] == []
    not_compared = {item["component_class"] for item in report["not_compared"]}
    assert not_compared == ({"invocation_policies"} if loop_policies else set())
    rows = {item["component_class"]: item for item in report["components"]}
    assert rows["provider_request_policy"]["compared"] is True
    assert rows["provider_request_policy"]["changed"] is False
    assert rows["invocation_policies"]["compared"] is not loop_policies

    # The prediction matches Runtime: the unchanged continuation is admitted.
    _approve_pending("paused", loop_policies=loop_policies)


def test_compare_reuses_recorded_request_settings_at_a_resume_boundary(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    from cayu.context.thinking import ThinkingConfig

    _compare_project(tmp_path, monkeypatch)
    _create_session(
        "finished",
        pause_for_approval=False,
        thinking=ThinkingConfig(effort="low"),
        loop_policies=True,
    )

    assert main(["execution-profile", "compare", "finished", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["boundary"] == "resume"
    assert report["prediction"]["outcome"] == "exact_reuse"
    # Each ResumeRequest sets thinking again and a finished interaction keeps no
    # value for it, so the class is reported as not compared rather than changed.
    not_compared = {item["component_class"]: item["reason"] for item in report["not_compared"]}
    assert set(not_compared) == {"provider_request_policy", "invocation_policies"}
    assert "thinking" in not_compared["provider_request_policy"]
    assert report["guidance"] == []

    # A real change is still reported next to the request settings it reused.
    monkeypatch.setenv("TOOL_VERSION", "2")
    assert main(["execution-profile", "compare", "finished", "--json"]) == 3
    changed = json.loads(capsys.readouterr().out)["prediction"]["changed_component_classes"]
    assert changed == ["tool_implementations"]


@pytest.mark.parametrize(
    ("crash_event", "boundary"),
    [
        # Nothing is pending, so the continuation first closes the abandoned run
        # and then admits the new request at the resume boundary.
        ("model.completed", "resume"),
        # The lost model call keeps the invocation open; Runtime rebuilds the
        # recorded request classes from the next request.
        ("model.text.delta", "continuation"),
    ],
)
@pytest.mark.parametrize("opaque_options", [False, True])
def test_compare_agrees_with_the_continuation_of_a_crashed_run(
    tmp_path: Path,
    monkeypatch,
    capsys,
    crash_event: str,
    boundary: str,
    opaque_options: bool,
) -> None:
    from cayu.runtime.execution_profiles import ExecutionProfileMismatchError

    # A top-level option has no privacy-safe projection, so its identity is opaque.
    monkeypatch.setenv("PROVIDER_OPTIONS", '{"temperature": 0.2}' if opaque_options else "{}")
    _compare_project(tmp_path, monkeypatch)
    _crash_at_event(tmp_path, "crashed", crash_event)
    events = _event_count(tmp_path / "sessions.db")

    code = main(["execution-profile", "compare", "crashed", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert _event_count(tmp_path / "sessions.db") == events
    rows = {item["component_class"]: item for item in report["components"]}
    if not opaque_options:
        assert code == 0
        assert report["prediction"]["outcome"] == "exact_reuse"
        assert report["boundary"] == boundary
        assert _resume("crashed") == "session.completed"
        return
    assert code == 3
    assert report["prediction"]["outcome"] == "rejected"
    assert report["prediction"]["admits"] is False
    assert report["prediction"]["changed_component_classes"] == ["provider_request_policy"]
    assert report["boundary"] == boundary
    assert rows["provider_request_policy"]["stored"] == "process_local"
    assert rows["provider_request_policy"]["compared"] is True
    with pytest.raises(ExecutionProfileMismatchError) as raised:
        _resume("crashed")
    assert raised.value.changed_component_classes == (
        ExecutionProfileComponentClass.PROVIDER_REQUEST_POLICY,
    )
