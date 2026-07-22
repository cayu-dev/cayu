from __future__ import annotations

import re

import pytest

from cayu.cli import main


def test_bare_guide_lists_topics_and_help_describes_them(capsys) -> None:
    assert main(["guide"]) == 0
    listing = capsys.readouterr().out
    assert "Package-shipped Cayu guides:" in listing
    assert "structured-output" in listing
    assert "Credential-free structured-output runtime proof." in listing
    assert "providers" in listing

    with pytest.raises(SystemExit) as excinfo:
        main(["guide", "--help"])

    assert excinfo.value.code == 0
    help_output = capsys.readouterr().out
    assert "TOPIC[#SECTION]" in help_output
    assert "Explicit provider and compatible model selection." in help_output


def test_guide_accepts_emitted_section_anchors(capsys) -> None:
    assert main(["guide", "diagnostics#app-no-agents"]) == 0
    section = capsys.readouterr().out

    assert section.startswith("## app-no-agents")
    assert "APP_NO_AGENTS" in section
    assert "agent-provider-not-found" not in section


def test_package_shipped_authoring_and_diagnostic_guides_are_discoverable(capsys) -> None:
    assert main(["guide", "authoring"]) == 0
    authoring = capsys.readouterr().out
    assert "# Building applications with Cayu" in authoring
    assert "Start with one model-only agent" in authoring
    assert "## Cayu Map" in authoring
    assert "A tool-backed slice is optional" in authoring
    assert "Clarify users, jobs, triggers" not in authoring
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


def test_every_cayu_map_row_routes_to_a_package_shipped_local_guide(capsys) -> None:
    assert main(["guide", "authoring"]) == 0
    authoring = capsys.readouterr().out
    rows = [line for line in authoring.splitlines() if line.startswith("|")][2:]

    assert len(rows) >= 20
    for row in rows:
        commands = re.findall(r"`(cayu guide [^`]+)`", row)
        assert commands, row
        for command in commands:
            assert main(command.split()[1:]) == 0, command
            assert capsys.readouterr().out


def test_structured_output_and_provider_guides_are_credential_free_and_public(capsys) -> None:
    assert main(["guide", "structured-output"]) == 0
    structured = capsys.readouterr().out
    assert "scripted_structured_output" in structured
    assert "invalid first" in structured
    assert "outcome.structured_output.output" in structured
    assert "cayu.runtime" not in structured

    assert main(["guide", "providers#anthropic"]) == 0
    providers = capsys.readouterr().out
    assert "AnthropicProvider" in providers
    assert "ANTHROPIC_API_KEY" in providers
