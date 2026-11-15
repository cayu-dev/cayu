from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


def _load_test_conftest() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_conftest_under_test",
        Path(__file__).parents[1] / "conftest.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TEST_CONFTEST = _load_test_conftest()


def test_postgres_fixture_uses_pgvector_container_image() -> None:
    assert TEST_CONFTEST._POSTGRES_CONTAINER_IMAGE == "pgvector/pgvector:pg16"


def test_postgres_version_selects_a_digest_pinned_pgvector_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CAYU_TEST_POSTGRES_VERSION", raising=False)
    assert TEST_CONFTEST._postgres_container_image() == "pgvector/pgvector:pg16"

    monkeypatch.setenv("CAYU_TEST_POSTGRES_VERSION", " 18 ")
    image = TEST_CONFTEST._postgres_container_image()
    assert image.startswith("pgvector/pgvector:pg18@sha256:")
    assert len(image.rsplit("@sha256:", 1)[1]) == 64

    monkeypatch.setenv("CAYU_TEST_POSTGRES_VERSION", "15")
    with pytest.raises(pytest.UsageError, match="CAYU_TEST_POSTGRES_VERSION=15"):
        TEST_CONFTEST._postgres_container_image()


@pytest.mark.parametrize(("server_version_num", "fails"), [("180007", False), ("160010", True)])
def test_postgres_version_fails_tests_on_another_server(
    monkeypatch: pytest.MonkeyPatch, server_version_num: str, fails: bool
) -> None:
    import psycopg

    class Connection:
        def __enter__(self) -> Connection:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def execute(self, query: str) -> Connection:
            assert query == "SHOW server_version_num"
            return self

        def fetchone(self) -> tuple[str]:
            return (server_version_num,)

    monkeypatch.setattr(psycopg, "connect", lambda dsn: Connection())
    monkeypatch.setenv("CAYU_TEST_POSTGRES_VERSION", "18")
    if fails:
        with pytest.raises(pytest.fail.Exception, match="runs PostgreSQL 16"):
            TEST_CONFTEST._check_postgres_version("postgresql://test")
    else:
        TEST_CONFTEST._check_postgres_version("postgresql://test")


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_postgres_required_accepts_truthy_env_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("CAYU_REQUIRE_POSTGRES_TESTS", value)

    assert TEST_CONFTEST._postgres_required() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
def test_postgres_required_rejects_non_truthy_env_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("CAYU_REQUIRE_POSTGRES_TESTS", value)

    assert TEST_CONFTEST._postgres_required() is False


def test_postgres_unavailable_skips_when_not_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CAYU_REQUIRE_POSTGRES_TESTS", raising=False)

    with pytest.raises(pytest.skip.Exception, match="missing postgres"):
        TEST_CONFTEST._skip_or_fail_postgres_unavailable("missing postgres")


def test_postgres_unavailable_fails_when_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAYU_REQUIRE_POSTGRES_TESTS", "1")

    with pytest.raises(pytest.fail.Exception, match="missing postgres"):
        TEST_CONFTEST._skip_or_fail_postgres_unavailable("missing postgres")
