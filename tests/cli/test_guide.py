from __future__ import annotations

from cayu.cli import main


def test_package_shipped_authoring_and_diagnostic_guides_are_discoverable(capsys) -> None:
    assert main(["guide", "authoring"]) == 0
    authoring = capsys.readouterr().out
    assert "# Building applications with Cayu" in authoring
    assert "understand -> clarify -> inspect -> check" in authoring

    assert main(["guide", "diagnostics"]) == 0
    diagnostics = capsys.readouterr().out
    assert "# Cayu project diagnostics" in diagnostics
    assert "## agent-provider-not-found" in diagnostics
