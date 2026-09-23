from __future__ import annotations

from contextlib import ExitStack
from tempfile import TemporaryDirectory

import pytest


@pytest.fixture(autouse=True)
def browser_acceptance_directories(monkeypatch):
    """Own builder directories through test teardown, including failed builds.

    Plans can remain reachable through cycles after a test returns. Retaining
    and explicitly closing their directories avoids unrelated tests receiving
    TemporaryDirectory finalizer warnings when those cycles are collected.
    Patch only this builder, not tempfile or the warning machinery globally.
    """
    from cayu.evals.internal import browser_acceptance

    with ExitStack() as ownership:

        def temporary_directory(*args, **kwargs):
            directory = TemporaryDirectory(*args, **kwargs)
            ownership.callback(directory.cleanup)
            return directory

        monkeypatch.setattr(browser_acceptance, "TemporaryDirectory", temporary_directory)
        yield ownership
