from __future__ import annotations

import asyncio
import gc
import warnings
from pathlib import Path

import pytest

from cayu.evals.browser_acceptance_fixture import BrowserAcceptanceFixtureV1
from cayu.evals.internal import browser_acceptance


@pytest.mark.parametrize("fail_build", [False, True])
def test_builder_directories_are_cleaned_before_later_warning_observers(
    browser_acceptance_directories, monkeypatch, fail_build
):
    # Do not attribute unreachable resources left by earlier tests to this builder.
    gc.collect()
    directories = []
    allocate = browser_acceptance.TemporaryDirectory

    def record_directory(*args, **kwargs):
        directory = allocate(*args, **kwargs)
        directories.append(directory)
        return directory

    monkeypatch.setattr(browser_acceptance, "TemporaryDirectory", record_directory)
    if fail_build:

        def reject_runtime(**kwargs):
            raise RuntimeError("injected build failure")

        monkeypatch.setattr(browser_acceptance, "_build_runtime", reject_runtime)

    async def scenario(fixture):
        if fail_build:
            with pytest.raises(RuntimeError, match="injected build failure"):
                await browser_acceptance.build(fixture)
            return
        plan = await browser_acceptance.build(fixture)
        try:
            assert plan.eval_plan.app is not None
        finally:
            await plan.eval_plan.app.session_store.close()

    with BrowserAcceptanceFixtureV1() as fixture:
        asyncio.run(scenario(fixture))
    assert directories
    paths = [Path(directory.name) for directory in directories]
    assert all(path.is_dir() for path in paths)
    # This is the same ExitStack.close executed by the fixture on teardown.
    # Keep the TemporaryDirectory objects alive to prove cleanup is explicit.
    browser_acceptance_directories.close()
    assert all(not path.exists() for path in paths)
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always", ResourceWarning)
        directories.clear()
        gc.collect()
    directory_warnings = []
    for item in observed:
        if issubclass(item.category, ResourceWarning) and any(
            str(path) in str(item.message) for path in paths
        ):
            directory_warnings.append(item)
        else:
            # Preserve unrelated diagnostics; this regression owns only its directories.
            warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)
    assert not directory_warnings
