from __future__ import annotations

import pytest

from cayu.cli import main


def test_core_cli_owns_cloud_namespace(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--help"])

    assert raised.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "cloud" in help_text
    assert "Manage Cayu Cloud." in help_text


def test_cloud_namespace_explains_its_future_core_owned_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["cloud"]) == 0

    help_text = " ".join(capsys.readouterr().out.split())
    assert "Deployment commands are not available in this release." in help_text


def test_cloud_deploy_remains_unimplemented(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["cloud", "deploy"])

    assert raised.value.code == 2
    assert "unrecognized arguments: deploy" in capsys.readouterr().err
