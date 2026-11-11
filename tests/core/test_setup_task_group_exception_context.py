"""Setup owner return must not depend on a caller's handled exception."""

import pytest
from tests.core import test_setup_task_group_quiescence as setup_regression

from cayu import CayuApp


@pytest.mark.parametrize("close_on_failure", [False, True])
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "failure", [None, "factory", "transcript"], ids=["success", "factory", "transcript"]
)
def test_setup_group_owner_returns_inside_caller_exception_handler(
    failure, backend, tmp_path, monkeypatch, close_on_failure
):
    original_run = CayuApp.run

    async def run_from_exception_handler(self, *args, **kwargs):
        try:
            raise ValueError("unrelated caller failure")
        except ValueError:
            stream = original_run(self, *args, **kwargs)
            try:
                async for event in stream:
                    yield event
            finally:
                await stream.aclose()

    monkeypatch.setattr(CayuApp, "run", run_from_exception_handler)
    setup_regression.test_setup_failure_releases_task_group_after_cleanup(
        failure, backend, tmp_path, close_on_failure
    )
