from __future__ import annotations

import inspect
from importlib.metadata import version

import pytest
from packaging.version import Version


def test_reviewed_e2b_sdk_exposes_running_sandbox_network_replacement() -> None:
    e2b = pytest.importorskip("e2b")

    assert Version(version("e2b")) >= Version("2.45.1")
    assert Version(version("e2b")) < Version("3")
    assert inspect.iscoroutinefunction(inspect.unwrap(e2b.AsyncSandbox.update_network))
    parameters = inspect.signature(e2b.AsyncSandbox.update_network).parameters
    assert "network" in parameters
