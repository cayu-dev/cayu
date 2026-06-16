from __future__ import annotations

import os

import pytest

_DOCKER_SKIP_REASON = "Docker is unavailable; skipping Postgres store tests."
_DSN_ENV_VAR = "CAYU_TEST_POSTGRES_DSN"


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


@pytest.fixture(scope="session")
def postgres_dsn() -> str:
    """Session-scoped Postgres DSN for the store parity tests.

    Resolution order:

    1. ``CAYU_TEST_POSTGRES_DSN`` — point the tests at an already-running Postgres
       (a CI service container, or a local instance). Used as-is.
    2. A Dockerized ``postgres:16-alpine`` via testcontainers.

    Skips the whole module when neither is available so Docker-less CI stays green.
    Tests own their schema and ``DROP TABLE`` between runs, so the target database
    must be disposable — never point this at a database with data you care about.
    """
    env_dsn = os.environ.get(_DSN_ENV_VAR)
    if env_dsn and env_dsn.strip():
        yield env_dsn.strip()
        return

    if not _docker_available():
        pytest.skip(_DOCKER_SKIP_REASON)

    try:
        from testcontainers.postgres import PostgresContainer
    except Exception as exc:  # pragma: no cover - dependency guard
        pytest.skip(f"testcontainers unavailable: {exc}")

    container = PostgresContainer("postgres:16-alpine")
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
