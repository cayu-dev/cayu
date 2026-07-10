from __future__ import annotations

import os

import pytest

_DOCKER_SKIP_REASON = "Docker is unavailable; skipping Postgres store tests."
_DSN_ENV_VAR = "CAYU_TEST_POSTGRES_DSN"
_REQUIRE_POSTGRES_ENV_VAR = "CAYU_REQUIRE_POSTGRES"
_POSTGRES_CONTAINER_IMAGE = "pgvector/pgvector:pg16"
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def _docker_available() -> bool:
    try:
        import docker
    except Exception:
        return False
    try:
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


def _postgres_required() -> bool:
    return os.environ.get(_REQUIRE_POSTGRES_ENV_VAR, "").strip().lower() in _TRUTHY_ENV_VALUES


def _skip_or_fail_postgres_unavailable(reason: str) -> None:
    if _postgres_required():
        pytest.fail(reason)
    pytest.skip(reason)


@pytest.fixture(scope="session")
def postgres_dsn() -> str:
    """Session-scoped Postgres DSN for the store parity tests.

    Resolution order:

    1. ``CAYU_TEST_POSTGRES_DSN`` — point the tests at an already-running Postgres
       (a CI service container, or a local instance). Used as-is.
    2. A Dockerized pgvector-capable Postgres via testcontainers.

    Skips the whole module when neither is available, unless
    ``CAYU_REQUIRE_POSTGRES`` is set. CI sets that flag so a lost Postgres tier
    fails loudly instead of disappearing behind a green check.
    Tests own their schema and ``DROP TABLE`` between runs, so the target database
    must be disposable — never point this at a database with data you care about.
    """
    env_dsn = os.environ.get(_DSN_ENV_VAR)
    if env_dsn and env_dsn.strip():
        yield env_dsn.strip()
        return

    if not _docker_available():
        _skip_or_fail_postgres_unavailable(_DOCKER_SKIP_REASON)

    try:
        from testcontainers.postgres import PostgresContainer
    except Exception as exc:  # pragma: no cover - dependency guard
        _skip_or_fail_postgres_unavailable(f"testcontainers unavailable: {exc}")

    container = PostgresContainer(_POSTGRES_CONTAINER_IMAGE)
    container.start()
    try:
        url = container.get_connection_url()
        # testcontainers returns a SQLAlchemy-style URL; normalize to a psycopg DSN.
        dsn = url.replace("postgresql+psycopg2://", "postgresql://").replace(
            "postgresql+psycopg://", "postgresql://"
        )
        yield dsn
    finally:
        container.stop()
