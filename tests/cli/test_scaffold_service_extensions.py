"""Public CLI regressions for explicit application-owned service extensions."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cayu.cli import main


def _edit(project, relative, old, new):
    path = project / relative
    source = path.read_text()
    assert old in source
    path.write_text(source.replace(old, new))


def _wire_extension(project, capability):
    if capability == "artifacts":
        _edit(
            project,
            "tools/registration.py",
            "    return tuple(tools)",
            "    tools.append(ListArtifactsTool())\n    return tuple(tools)",
        )
        _edit(
            project,
            "environments/local.py",
            "selected_artifacts is None and False",
            "selected_artifacts is None and True",
        )
        _edit(
            project,
            "environments/local.py",
            "if not False and artifact_store",
            "if not True and artifact_store",
        )
        _edit(project, "environments/local.py", "    if False\n", "    if True\n")
    elif capability == "knowledge":
        _edit(project, "policies/tools.py", "    if False\n", "    if True\n")
        _edit(
            project,
            "tools/registration.py",
            "    if False:\n        tools.extend(",
            "    if True:\n        tools.extend(",
        )
        _edit(
            project,
            "tools/registration.py",
            '("remember_knowledge",) if False',
            '("remember_knowledge",) if True',
        )
        _edit(project, "tools/registration.py", "    if False\n", "    if True\n")
        _edit(
            project,
            "configuration/storage.py",
            "if not False and knowledge_scope",
            "if not True and knowledge_scope",
        )
        _edit(project, "knowledge/retrieval.py", "if not False:", "if not True:")
        _edit(
            project,
            "environments/local.py",
            "if not False and (knowledge_store",
            "if not True and (knowledge_store",
        )
        _edit(project, "environments/local.py", "    if False\n", "    if True\n")
        _edit(
            project,
            "configuration/runtime.py",
            "knowledge_review_namespace=None",
            'knowledge_review_namespace="project:profile:agent:profile"',
        )
    else:
        _edit(
            project,
            "agents/registration.py",
            "from cayu import ContextPolicy",
            "from cayu import ContextPolicy, SubagentTool, SubagentResultTool, "
            "ExecutionProfileBehaviorIdentity",
        )
        _edit(
            project,
            "agents/registration.py",
            "    starter_tools = list(build_agent_tools())",
            """    helper = _agent_for_provider_override(
        AGENT.model_copy(update={"name": "helper"}), provider_override
    )
    app.register_agent(helper, tools=())
    identity = ExecutionProfileBehaviorIdentity(
        name="profile.delegation", behavior_version="1", implementation_version="1"
    )
    starter_tools = list(build_agent_tools())
    starter_tools.extend((
        SubagentTool(app, agents={"helper": "helper"}, execution_profile_identity=identity),
        SubagentResultTool(app.session_store, execution_profile_identity=identity),
    ))""",
        )
        _edit(
            project,
            "agents/registration.py",
            "starter_external_tool_names = list(external_effect_tool_names())",
            'starter_external_tool_names = [*external_effect_tool_names(), "subagent"]',
        )


def _check(capsys, expected):
    assert main(["check", "--deploy", "--fail-on", "warning", "--json"]) == expected
    payload = json.loads(capsys.readouterr().out)
    if expected == 0:
        assert payload["diagnostics"] == []
        assert payload["service_evidence"]["service_contract"] == "verified_maintained"
        assert payload["service_evidence"]["control_plane_access"] == "verified_authenticated"
    return payload


@pytest.mark.parametrize("capability", ("artifacts", "knowledge", "delegation"))
def test_explicit_service_extension_strict_cli(tmp_path, monkeypatch, capsys, capability):
    monkeypatch.setenv(
        "PRODUCT_AUTH_TOKENS_JSON",
        '{"test-customer":{"tenant_id":"tenant-a","subject_id":"test-user"}}',
    )
    monkeypatch.setenv("CAYU_OPERATOR_BEARER_TOKEN", "test-operator")
    assert main(["new", "profile", "--preset", "service", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "profile"
    monkeypatch.chdir(project)
    protected = {
        name: (project / name).read_bytes() for name in ("app.py", "service.py", "product_store.py")
    }
    _check(capsys, 0)
    config = project / "pyproject.toml"
    baseline = config.read_text()
    config.write_text(baseline.replace("extensions = []\n", ""))
    _check(capsys, 0)  # Existing convention-1 declarations remain valid.
    config.write_text(baseline)
    declaration = f'extensions = ["{capability}"]'
    # The old attempted declaration gives an actionable migration hint.
    old_selection = sorted(["approvals", "evals", "observability", "tasks", capability])
    config.write_text(
        baseline.replace(
            'capabilities = ["approvals", "evals", "observability", "tasks"]',
            "capabilities = " + json.dumps(old_selection),
        )
    )
    report = _check(capsys, 1)
    invalid = next(
        item for item in report["diagnostics"] if item["code"] == "SCAFFOLD_CONTRACT_INVALID"
    )
    assert "[tool.cayu.scaffold].extensions" in invalid["hint"]
    config.write_text(baseline.replace("extensions = []", declaration))
    # Metadata alone constructs nothing and must fail the live comparison.
    report = _check(capsys, 1)
    assert all(item["code"] == "SCAFFOLD_CAPABILITY_DRIFT" for item in report["diagnostics"])
    assert any(
        item["parameters"] == {"capability": capability, "expected": True, "observed": False}
        for item in report["diagnostics"]
    )
    config.write_text(baseline)
    _wire_extension(project, capability)
    report = _check(capsys, 1)
    assert all(item["code"] == "SCAFFOLD_CAPABILITY_DRIFT" for item in report["diagnostics"])
    assert any(
        item["parameters"] == {"capability": capability, "expected": False, "observed": True}
        for item in report["diagnostics"]
    )
    config.write_text(baseline.replace("extensions = []", declaration))
    _check(capsys, 0)
    assert main(["inspect", "--json"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    if capability in {"knowledge", "delegation"}:
        name = "remember_knowledge" if capability == "knowledge" else "subagent"
        tool = next(
            tool for agent in manifest["agents"] for tool in agent["tools"] if tool["name"] == name
        )
        assert tool["policy_coverage"] == (
            "approval_required" if capability == "knowledge" else "denied"
        )
    for name, source in protected.items():
        assert (project / name).read_bytes() == source
    # Exercise the generated authentication and tenant-isolation suite against
    # the extended source, using this checkout's exact Runtime implementation.
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_public_service_security.py"],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # Removing one tool from a declared family remains actual drift.
    if capability == "delegation":
        _edit(
            project,
            "agents/registration.py",
            "        tools=starter_tools,",
            '        tools=[tool for tool in starter_tools if tool.name != "subagent_result"],',
        )
    else:
        missing = "list_artifacts" if capability == "artifacts" else "read_knowledge"
        _edit(
            project,
            "tools/registration.py",
            "    return tuple(tools)",
            f'    return tuple(tool for tool in tools if tool.name != "{missing}")',
        )
    report = _check(capsys, 1)
    assert any(item["path"] == "agents.tools." + capability for item in report["diagnostics"])


@pytest.mark.parametrize("capability", ("artifacts", "knowledge", "delegation"))
def test_service_extension_discovery_and_generator_guidance(tmp_path, capsys, capability):
    assert main(["new", "--explain", capability, "--json"]) == 0
    spec = json.loads(capsys.readouterr().out)["capability"]
    assert spec["extension_presets"] == ["service"]
    assert spec["extension_declaration"] == "tool.cayu.scaffold.extensions"
    assert "service" not in spec["supported_presets"]
    assert (
        main(
            [
                "new",
                "profile",
                "--preset",
                "service",
                "--with",
                capability,
                "--dir",
                str(tmp_path),
                "--dry-run",
                "--json",
            ]
        )
        == 1
    )
    report = json.loads(capsys.readouterr().out)
    assert report["error"]["code"] == "CAPABILITY_NOT_SELECTABLE"
    assert "[tool.cayu.scaffold].extensions" in report["error"]["message"]
    assert not (tmp_path / "profile").exists()


@pytest.mark.parametrize(
    "value,reason",
    (
        ('"artifacts"', "invalid_extensions"),
        ("[1]", "invalid_extensions"),
        ('["unknown"]', "unknown_capability"),
        ('["workers"]', "unsupported_extension"),
        ('["artifacts", "artifacts"]', "extensions_not_normalized"),
        ('["knowledge", "artifacts"]', "extensions_not_normalized"),
    ),
)
def test_invalid_service_extension_declarations(tmp_path, monkeypatch, capsys, value, reason):
    assert main(["new", "profile", "--preset", "service", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "profile"
    _edit(project, "pyproject.toml", "extensions = []", "extensions = " + value)
    monkeypatch.chdir(project)
    report = _check(capsys, 1)
    assert any(
        item["code"] == "SCAFFOLD_CONTRACT_INVALID" and item["parameters"]["reason"] == reason
        for item in report["diagnostics"]
    )


@pytest.mark.parametrize("preset", ("agent", "coding"))
def test_service_extension_declarations_do_not_enable_other_presets(
    tmp_path, monkeypatch, capsys, preset
):
    assert main(["new", "profile", "--preset", preset, "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "profile"
    _edit(project, "pyproject.toml", "extensions = []", 'extensions = ["artifacts"]')
    monkeypatch.chdir(project)
    assert main(["check", "--fail-on", "warning", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert any(
        item["code"] == "SCAFFOLD_CONTRACT_INVALID"
        and item["parameters"]["reason"] == "unsupported_extension"
        for item in report["diagnostics"]
    )
