from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

from cayu.cli import main

_PROJECT = """import os
from pathlib import Path

from cayu import (
    AgentSpec,
    CayuApp,
    ExecutionProfileBehaviorIdentity,
    ScriptedModelProvider,
    SQLiteSessionStore,
    Tool,
    ToolEffect,
    ToolResult,
    ToolSpec,
)


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
