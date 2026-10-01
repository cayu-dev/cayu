from __future__ import annotations

import json

import pytest
from tests.cli.test_scaffold_capabilities import close_constructed_sqlite_stores  # noqa: F401

from cayu.cli import main


@pytest.mark.parametrize("approvals", [False, True])
def test_generated_starter_accepts_effective_composite_coverage(
    tmp_path, monkeypatch, capsys, approvals
):
    args = ["new", "profile", "--dir", str(tmp_path)]
    if not approvals:
        args += ["--without", "approvals"]
    assert main(args) == 0
    capsys.readouterr()
    project = tmp_path / "profile"
    policy = project / "policies/tools.py"
    source = policy.read_text()
    downstream = (
        'AlwaysRequireApprovalToolPolicy(tools=("remember_knowledge",))'
        if approvals
        else 'StaticToolPolicy(deny=("remember_knowledge",))'
    )
    policy.write_text(
        source
        + "\nfrom cayu import GuardedToolPolicy, RequiredArguments, AlwaysRequireApprovalToolPolicy\n"
        "\ndef build_tool_policy(external_tool_names):\n"
        '    return GuardedToolPolicy(guards=(RequiredArguments({"ask_user": ("question",)}),), '
        f"then={downstream})\n"
    )
    monkeypatch.chdir(project)
    assert main(["inspect", "--json"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    proposal = next(
        tool for tool in manifest["agents"][0]["tools"] if tool["name"] == "remember_knowledge"
    )
    assert proposal["policy_coverage"] == ("approval_required" if approvals else "denied")
    assert proposal["parameter_policy_decision"] is None
    assert main(["check", "--fail-on", "warning", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["diagnostics"] == []


def test_inspect_retains_manifest_when_coverage_drifts(tmp_path, monkeypatch, capsys):
    assert main(["new", "profile", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "profile"
    policy = project / "policies/tools.py"
    policy.write_text(
        policy.read_text().replace("ToolPolicyDecision.REQUIRE_APPROVAL", "ToolPolicyDecision.DENY")
    )
    monkeypatch.chdir(project)
    assert main(["inspect", "--json", "--tool", "remember_knowledge"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["error"]["code"] == "SCAFFOLD_CAPABILITY_DRIFT"
    assert report["manifest"]["fingerprint"]
    assert report["manifest"]["agents"]
    assert report["manifest"]["providers"] == []
    assert [tool["name"] for agent in report["manifest"]["agents"] for tool in agent["tools"]] == [
        "remember_knowledge"
    ]
    assert any(d["path"].endswith("policy_coverage") for d in report["diagnostics"])
