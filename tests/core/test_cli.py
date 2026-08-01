from __future__ import annotations

import pytest

from cayu.cli import main


def test_cli_version(capsys):
    assert main(["version"]) == 0

    output = capsys.readouterr().out.strip()

    assert output.startswith("cayu ")


def test_cli_unimplemented_validate_stub_removed(capsys):
    # The validate stub was removed rather than left as a misleading
    # "not implemented yet" placeholder; argparse now rejects it.
    with pytest.raises(SystemExit) as excinfo:
        main(["validate"])
    assert excinfo.value.code != 0
    assert "invalid choice" in capsys.readouterr().err
