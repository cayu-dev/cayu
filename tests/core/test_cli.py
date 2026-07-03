from __future__ import annotations

import pytest

from cayu.cli import main


def test_cli_version(capsys):
    assert main(["version"]) == 0

    output = capsys.readouterr().out.strip()

    assert output.startswith("cayu ")


@pytest.mark.parametrize("command", ["serve", "validate"])
def test_cli_unimplemented_stubs_removed(command, capsys):
    # The serve/validate stubs were removed rather than left as misleading
    # "not implemented yet" placeholders; argparse now rejects them.
    with pytest.raises(SystemExit) as excinfo:
        main([command])
    assert excinfo.value.code != 0
    assert "invalid choice" in capsys.readouterr().err
