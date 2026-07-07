"""Importing ``cayu.server`` without the server extra must fail helpfully.

Consumer #1 installed cayu without the ``server`` extra (their own app already
provided fastapi) and got a raw ``ModuleNotFoundError: No module named
'sse_starlette'`` with no hint that ``pip install "cayu[server]"`` exists —
every other extra-backed module (postgres, otel, e2b, vertex, files) already
raises the friendly install hint. These tests pin the same contract for the
server package, including the nuance that an unrelated missing module must
surface raw instead of being masked by the hint.
"""

from __future__ import annotations

import importlib
import sys

import pytest


def _purge_cached_modules(monkeypatch: pytest.MonkeyPatch, *prefixes: str) -> None:
    for module_name in list(sys.modules):
        if module_name in prefixes or module_name.startswith(tuple(f"{p}." for p in prefixes)):
            monkeypatch.delitem(sys.modules, module_name, raising=False)


@pytest.mark.parametrize("blocked", ["sse_starlette", "fastapi"])
def test_missing_server_dependency_raises_helpful_error(
    monkeypatch: pytest.MonkeyPatch, blocked: str
) -> None:
    _purge_cached_modules(monkeypatch, "cayu.server", blocked)
    # A None entry in sys.modules makes the import machinery raise
    # ModuleNotFoundError for that name, simulating the missing package.
    monkeypatch.setitem(sys.modules, blocked, None)

    with pytest.raises(RuntimeError, match=r"cayu\[server\]"):
        importlib.import_module("cayu.server")


def test_unrelated_missing_module_is_not_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    _purge_cached_modules(monkeypatch, "cayu.server", "pydantic")
    monkeypatch.setitem(sys.modules, "pydantic", None)

    with pytest.raises(ModuleNotFoundError, match="pydantic"):
        importlib.import_module("cayu.server")


def test_server_imports_cleanly_with_extra_installed() -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("sse_starlette")
    # Leaves a sane module cache for later tests in the same process, and
    # proves the guard is a no-op when the dependencies are present.
    module = importlib.import_module("cayu.server")
    assert hasattr(module, "create_server")
