from __future__ import annotations

from cayu.cli import main


def test_package_shipped_authoring_and_diagnostic_guides_are_discoverable(capsys) -> None:
    assert main(["guide", "authoring"]) == 0
    authoring = capsys.readouterr().out
    assert "# Building applications with Cayu" in authoring
    assert "understand -> clarify -> inspect -> check" in authoring
    assert "Model-controlled command selectors are untrusted argv input" in authoring
    assert "An executable allowlist does not authorize its argument protocol" in authoring
    assert "do not replace container or microVM isolation" in authoring
    assert "workflow_tool_names" in authoring
    assert "registered for that same agent" in authoring
    assert "cannot prove prompt comprehension" in authoring
    assert "cayu guide tool-effects" in authoring

    assert main(["guide", "diagnostics"]) == 0
    diagnostics = capsys.readouterr().out
    assert "# Cayu project diagnostics" in diagnostics
    assert "## agent-generated-tracer-bullet-unfinished" in diagnostics
    assert "## agent-provider-not-found" in diagnostics
    assert "## agent-workflow-tool-not-registered" in diagnostics


def test_package_shipped_application_anatomy_guide_is_discoverable(capsys) -> None:
    assert main(["guide", "anatomy"]) == 0
    anatomy = capsys.readouterr().out

    assert "# Cayu application anatomy" in anatomy
    assert "## Application lifecycle boundaries" in anatomy
    assert "SQLite store constructors open their files" in anatomy
    assert "## Process roles" in anatomy
    for role in (
        "One-off script",
        "Interactive console",
        "Server integration",
        "Worker integration",
        "Test",
    ):
        assert role in anatomy


def test_package_shipped_tool_effect_guide_renders_canonical_decisions(capsys) -> None:
    assert main(["guide", "tool-effects"]) == 0
    guidance = capsys.readouterr().out
    normalized = guidance.casefold()

    assert "# Choosing a ToolEffect" in guidance
    assert "public http read" in normalized
    assert "`NONE`" in guidance
    assert "paid or logged read" in normalized
    assert "stable downstream idempotency key" in normalized
    assert "stable operation identity or equivalent idempotency contract" in normalized
    assert "durable snapshot or artifact" in normalized
    assert "outcome is unknown after a timeout" in normalized
    assert "does not authorize execution" in guidance
    assert "verify_tool_effect" in guidance
    assert "bounded temporary Cayu workspace" in guidance
    assert "`cayu check` remains structural" in guidance
