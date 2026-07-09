from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_core_conftest() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "core_conftest_under_test",
        Path(__file__).with_name("conftest.py"),
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CORE_CONFTEST = _load_core_conftest()


def test_postgres_fixture_uses_pgvector_container_image() -> None:
    assert CORE_CONFTEST._POSTGRES_CONTAINER_IMAGE == "pgvector/pgvector:pg16"


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_postgres_required_accepts_truthy_env_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("CAYU_REQUIRE_POSTGRES", value)

    assert CORE_CONFTEST._postgres_required() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
def test_postgres_required_rejects_non_truthy_env_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("CAYU_REQUIRE_POSTGRES", value)

    assert CORE_CONFTEST._postgres_required() is False


def test_postgres_unavailable_skips_when_not_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CAYU_REQUIRE_POSTGRES", raising=False)

    with pytest.raises(pytest.skip.Exception, match="missing postgres"):
        CORE_CONFTEST._skip_or_fail_postgres_unavailable("missing postgres")


def test_postgres_unavailable_fails_when_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAYU_REQUIRE_POSTGRES", "1")

    with pytest.raises(pytest.fail.Exception, match="missing postgres"):
        CORE_CONFTEST._skip_or_fail_postgres_unavailable("missing postgres")
