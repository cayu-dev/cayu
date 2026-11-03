from __future__ import annotations

import pytest

from cayu.cli import _build_parser


@pytest.fixture(scope="session", autouse=True)
def _cli_command_modules_imported_unpatched() -> None:
    """Import every command module before any CLI test patches what they import.

    `cayu` builds its parser on the first `main()`, which imports the command modules, and
    many of them bind `CayuApi` or `open_cloud_agent_api` by name at import. Tests patch
    `cayu.cli._cloud_agent_api.CayuApi` with a fake bound to their own server; if one is the
    first in a process to call `main()`, those modules keep the fake after the patch is undone,
    and later tests in the same worker reach that test's server instead of their own.
    """

    _build_parser()


@pytest.fixture(autouse=True)
def _cloud_changes_kept_local(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Answer which Organization a `cayu cloud` change acts in without asking the test's fake
    client, and keep the record of where each Agent deploys out of the home directory.

    Every command that changes an Agent first asks Cloud (`GET /v1/me`) which Organization it
    acts in; the fakes most tests give `_cloud_client` answer only the requests the test is
    about. Tests of that lookup and of the records (`test_cloud_agent_homes.py`) undo this.
    """

    from cayu.cli import cloud
    from cayu.cli._cloud_agent_homes import Organization

    def acting(client: object) -> Organization:
        api_url = str(getattr(client, "api_url", "https://cloud.cayu.dev")).rstrip("/")
        return Organization(api_url=api_url, id="org_test", name="Test Organization")

    homes = tmp_path_factory.mktemp("cayu-cloud-agent-homes")
    monkeypatch.setattr(cloud, "_acting_organization", acting)
    monkeypatch.setattr(cloud, "_agent_homes_directory", lambda: homes)
