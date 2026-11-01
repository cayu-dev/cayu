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
