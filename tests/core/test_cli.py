from __future__ import annotations

from cayu.cli import main


def test_cli_version(capsys):
    assert main(["version"]) == 0

    output = capsys.readouterr().out.strip()

    assert output.startswith("cayu ")
