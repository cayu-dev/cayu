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
    assert "prompt_tool_alignment" in authoring
    assert "registered_tool_names" in authoring
    assert "cannot prove prompt comprehension" in authoring

    assert main(["guide", "diagnostics"]) == 0
    diagnostics = capsys.readouterr().out
    assert "# Cayu project diagnostics" in diagnostics
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
