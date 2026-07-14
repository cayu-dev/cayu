from __future__ import annotations

import json
import sys
from pathlib import Path

from cayu.cli import main


def test_check_json_reports_actionable_provider_and_policy_failures(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "broken_project:build_app"\n',
        encoding="utf-8",
    )
    (tmp_path / "broken_project.py").write_text(
        """from cayu import AgentSpec, CayuApp, Tool, ToolEffect, ToolResult, ToolSpec


class SendTool(Tool):
    spec = ToolSpec(
        name="send",
        effect=ToolEffect.EXTERNAL,
        input_schema={"type": "object", "additionalProperties": False},
    )

    async def run(self, ctx, args):
        return ToolResult(content="sent")


def build_app():
    app = CayuApp(enable_logging=False)
    app.register_agent(
        AgentSpec(name="sender", model="missing-model", provider_name="missing"),
        tools=[SendTool()],
    )
    return app
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("broken_project", None)

    assert main(["check", "--json"]) == 1

    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == "1"
    assert report["manifest_fingerprint"]
    assert [item["code"] for item in report["diagnostics"]] == [
        "AGENT_PROVIDER_NOT_FOUND",
        "EXTERNAL_TOOL_UNGUARDED",
    ]
    provider_finding = report["diagnostics"][0]
    assert provider_finding["path"] == "agents.sender.configured_provider"
    assert provider_finding["parameters"] == {"agent": "sender", "provider": "missing"}
    assert provider_finding["hint"] == "Register provider 'missing' or change sender.provider_name."
    assert provider_finding["documentation_anchor"].endswith("#agent-provider-not-found")


def test_check_json_distinguishes_factory_failure_from_findings(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.cayu]\nfactory = "failed_project:build_app"\n',
        encoding="utf-8",
    )
    (tmp_path / "failed_project.py").write_text(
        'def build_app():\n    raise RuntimeError("boot exploded")\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("failed_project", None)

    assert main(["check", "--json"]) == 2

    error = json.loads(capsys.readouterr().out)["error"]
    assert error == {
        "code": "PROJECT_CHECK_FAILED",
        "message": "Application factory failed (RuntimeError): boot exploded",
    }
