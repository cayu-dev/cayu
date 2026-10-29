from __future__ import annotations

import json
import sys
from pathlib import Path

from cayu.cli import main

PROJECT = """from cayu import AgentSpec, CayuApp, Tool, ToolSpec, ToolResult, ToolEffect
from cayu.evals import ScriptedModelProvider
from cayu.runtime import ExecutionProfileBehaviorIdentity

class CustomProvider(ScriptedModelProvider):
    PROVIDER_IDENTITY

class Lookup(Tool):
    spec = ToolSpec(name="lookup", description="Look up", effect=ToolEffect.NONE, max_terminal_payload_bytes=1024, input_schema={"type": "object"}IDENTITY)
    async def run(self, ctx, args):
        raise AssertionError("check must never execute tools")

def build_app():
    app = CayuApp(enable_logging=False)
    app.register_provider(CustomProvider([], name="custom"), default=True)
    app.register_agent(AgentSpec(name="agent", model="fixture"), tools=[Lookup()])
    return app
"""
IDENTITY = ', execution_profile_identity=ExecutionProfileBehaviorIdentity(name="stable", behavior_version="1", implementation_version="1")'


def test_check_names_undeclared_execution_identities_and_keeps_warning_exit(
    tmp_path: Path, monkeypatch, capsys
):
    path = tmp_path / "identity_check.py"
    path.write_text(PROJECT.replace("PROVIDER_IDENTITY", "pass").replace("IDENTITY", ""))
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("identity_check", None)
    assert main(["check", "identity_check:build_app", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    warnings = [
        item
        for item in report["diagnostics"]
        if item["code"] == "EVAL_EXECUTION_IDENTITY_PROCESS_LOCAL"
    ]
    assert {item["subject"] for item in warnings} == {"tool:agent/lookup", "provider:custom"}
    assert all(item["severity"] == "warning" for item in warnings)
    assert all(
        "execution_profile_identity=ExecutionProfileBehaviorIdentity" in item["hint"]
        for item in warnings
    )
    assert main(["check", "identity_check:build_app", "--fail-on", "warning", "--json"]) == 1
    capsys.readouterr()


def test_check_declared_identities_and_builtin_provider_do_not_warn(
    tmp_path: Path, monkeypatch, capsys
):
    path = tmp_path / "identity_check_declared.py"
    path.write_text(
        PROJECT.replace(
            "PROVIDER_IDENTITY", "execution_profile_identity = " + IDENTITY.split("=", 1)[1]
        ).replace("IDENTITY", IDENTITY)
    )
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("identity_check_declared", None)
    assert main(["check", "identity_check_declared:build_app", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert not any(
        item["code"] == "EVAL_EXECUTION_IDENTITY_PROCESS_LOCAL" for item in report["diagnostics"]
    )


def test_identity_checks_do_not_bypass_source_import_guard(tmp_path, monkeypatch, capsys):
    assert main(["new", "project", "--dir", str(tmp_path)]) == 0
    capsys.readouterr()
    project = tmp_path / "project"
    (project / "tools" / "probe.py").write_text(
        'from pathlib import Path\nPath("should-not-exist").write_text("unsafe")\n'
    )
    monkeypatch.chdir(project)
    assert main(["check", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["manifest_fingerprint"] == "unavailable"
    assert not (project / "should-not-exist").exists()
    assert not any(
        item["code"] == "EVAL_EXECUTION_IDENTITY_PROCESS_LOCAL" for item in report["diagnostics"]
    )


def test_generated_projects_declare_inspection_only_provider_and_optional_tool_identities(tmp_path):
    import os
    import subprocess

    source_root = str(Path(__file__).resolve().parents[2] / "src")
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "CAYU_PROVIDER",
            "CAYU_MODEL",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENROUTER_API_KEY",
        }
    }
    env["PYTHONPATH"] = source_root
    env["PRODUCT_AUTH_TOKENS_JSON"] = (
        '{"local-test-token":{"tenant_id":"local-test","subject_id":"test"}}'
    )
    env["CAYU_OPERATOR_BEARER_TOKEN"] = "local-test-operator"
    for name, options in (
        ("starter", []),
        ("coding", ["--preset", "coding"]),
        ("service", ["--preset", "service"]),
        ("reduced", ["--without", "recovery"]),
    ):
        created = subprocess.run(
            [sys.executable, "-m", "cayu", "new", name, "--dir", str(tmp_path), *options],
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
        )
        assert created.returncode == 0, created.stderr
        checked = subprocess.run(
            [
                sys.executable,
                "-m",
                "cayu",
                "check",
                "--fail-on",
                "warning",
                "--json",
                *(["--deploy"] if name == "service" else []),
            ],
            cwd=tmp_path / name,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr
        assert json.loads(checked.stdout)["diagnostics"] == []
        if name == "starter":
            rejected = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from configuration.providers import configured_provider; "
                    "configured_provider().preflight_model_target(model='not-configured')",
                ],
                cwd=tmp_path / name,
                env=env,
                text=True,
                capture_output=True,
                timeout=60,
            )
            assert rejected.returncode != 0
            assert "no provider is selected" in rejected.stderr


def assert_subscription_identity_warning(report):
    # The built-in subscription adapter currently has process-local identity.
    # Keep that warning visible while checking unrelated scaffold guarantees.
    assert [
        (item["code"], item["subject"], item["severity"]) for item in report["diagnostics"]
    ] == [("EVAL_EXECUTION_IDENTITY_PROCESS_LOCAL", "provider:openai_subscription", "warning")]
