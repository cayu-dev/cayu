from __future__ import annotations

import json
import os
import secrets
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.sqlite_resources import SQLiteResourceScope


@pytest.fixture
def sqlite_resources(
    tmp_path: Path, request: pytest.FixtureRequest
) -> Iterator[SQLiteResourceScope]:
    """Enter inside asyncio.run so work and teardown share the same loop."""
    scope = SQLiteResourceScope(tmp_path, request.node.nodeid)
    yield scope
    scope.assert_finished()


_DOCKER_SKIP_REASON = "Docker is unavailable; skipping Postgres store tests."
_DSN_ENV_VAR = "CAYU_TEST_POSTGRES_DSN"
_REQUIRE_POSTGRES_ENV_VAR = "CAYU_REQUIRE_POSTGRES"
_REQUIRE_CURRENT_TEST_DURATIONS_ENV_VAR = "CAYU_REQUIRE_CURRENT_TEST_DURATIONS"
_POSTGRES_CONTAINER_IMAGE = "pgvector/pgvector:pg16"
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
_MAX_UNKNOWN_DURATION_FRACTION = 0.05
_CI_REPORT_PATH: Path | None = None


def pytest_configure(config: pytest.Config) -> None:
    global _CI_REPORT_PATH
    if os.environ.get("GITHUB_ACTIONS") != "true" or not config.getoption("splits", None):
        return
    owner = os.environ.setdefault("CAYU_CI_REPORT_OWNER_PID", str(os.getpid()))
    if owner != str(os.getpid()):
        return
    _CI_REPORT_PATH = Path(config.rootpath) / ".ci-test-reports.jsonl"
    _CI_REPORT_PATH.write_text("", encoding="utf-8")


def _append_ci_report(record: dict[str, object]) -> None:
    if _CI_REPORT_PATH is not None:
        with _CI_REPORT_PATH.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=True) + "\n")


def pytest_runtest_logstart(nodeid: str, location: tuple[str, int | None, str]) -> None:
    _append_ci_report({"nodeid": nodeid, "phase": "start"})


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if _CI_REPORT_PATH is None:
        return
    record: dict[str, object] = {
        "nodeid": report.nodeid,
        "phase": report.when,
        "duration": report.duration,
        "outcome": report.outcome,
    }
    if report.failed:
        failure = report.longreprtext
        record["failure"] = failure[:16384]
        record["failure_truncated"] = len(failure) > 16384
    _append_ci_report(record)


class ProviderCredentialCanaries:
    """Generated host-only values with a deliberately content-free repr."""

    __slots__ = ("auth_home", "host_env", "positive_env", "values")

    def __init__(
        self,
        *,
        values: dict[str, str],
        host_env: dict[str, str],
        auth_home: Path,
        positive_env: dict[str, str],
    ) -> None:
        self.values = values
        self.host_env = host_env
        self.auth_home = auth_home
        self.positive_env = positive_env

    def __repr__(self) -> str:
        return "ProviderCredentialCanaries(configured=True)"


@pytest.fixture
def provider_credential_canaries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> ProviderCredentialCanaries:
    """Install generated provider-authority canaries in the trusted host only."""

    def canary(label: str) -> str:
        return f"cayu-{label}-{secrets.token_urlsafe(24)}"

    values = {
        "openai_api_key": canary("openai"),
        "anthropic_api_key": canary("anthropic"),
        "gemini_api_key": canary("gemini"),
        "aws_access_key_id": canary("aws-access"),
        "aws_secret_access_key": canary("aws-secret"),
        "aws_session_token": canary("aws-session"),
        "aws_profile": canary("aws-profile"),
        "oauth_access_token": canary("access"),
        "oauth_refresh_token": canary("refresh"),
        "account_id": canary("account"),
        "authorization_header": f"Bearer {canary('header')}",
    }
    auth_home = tmp_path / "provider-auth-home"
    auth_home.mkdir()
    values["auth_store_path"] = str(auth_home)
    aws_credentials_path = auth_home / "aws-credentials"
    aws_config_path = auth_home / "aws-config"
    google_credentials_path = auth_home / "google-application-credentials.json"
    values["aws_credentials_path"] = str(aws_credentials_path)
    values["aws_config_path"] = str(aws_config_path)
    values["google_application_credentials_path"] = str(google_credentials_path)
    values["google_private_key"] = canary("google-private-key")
    (auth_home / "auth.json").write_text(
        json.dumps(
            {
                "version": 1,
                "providers": {
                    "openai_subscription": {
                        "access_token": values["oauth_access_token"],
                        "refresh_token": values["oauth_refresh_token"],
                        "account_id": values["account_id"],
                        "expires_at": 2_000_000_000,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (auth_home / "auth.json").chmod(0o600)
    aws_credentials_path.write_text(
        "[default]\n"
        f"aws_access_key_id={values['aws_access_key_id']}\n"
        f"aws_secret_access_key={values['aws_secret_access_key']}\n"
        f"aws_session_token={values['aws_session_token']}\n",
        encoding="utf-8",
    )
    aws_config_path.write_text("[default]\nregion=us-east-1\n", encoding="utf-8")
    google_credentials_path.write_text(
        json.dumps(
            {
                "type": "service_account",
                "private_key": values["google_private_key"],
                "client_email": "credential-canary@example.invalid",
            }
        ),
        encoding="utf-8",
    )
    host_env = {
        "OPENAI_API_KEY": values["openai_api_key"],
        "ANTHROPIC_API_KEY": values["anthropic_api_key"],
        "GEMINI_API_KEY": values["gemini_api_key"],
        "AWS_ACCESS_KEY_ID": values["aws_access_key_id"],
        "AWS_SECRET_ACCESS_KEY": values["aws_secret_access_key"],
        "AWS_SESSION_TOKEN": values["aws_session_token"],
        "AWS_PROFILE": values["aws_profile"],
        "AWS_CONFIG_FILE": values["aws_config_path"],
        "AWS_SHARED_CREDENTIALS_FILE": values["aws_credentials_path"],
        "GOOGLE_APPLICATION_CREDENTIALS": values["google_application_credentials_path"],
        "CAYU_HOME": str(auth_home),
        "OPENAI_AUTHORIZATION": values["authorization_header"],
    }
    for name, value in host_env.items():
        monkeypatch.setenv(name, value)
    positive_env = {
        "CAYU_PROBE_VISIBLE": canary("operational-control"),
        "CAYU_WORKLOAD_TOKEN": canary("workload-control"),
    }
    return ProviderCredentialCanaries(
        values=values,
        host_env=host_env,
        auth_home=auth_home,
        positive_env=positive_env,
    )


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


def _requests_postgres(item: pytest.Item) -> bool:
    if "postgres_dsn" in item.fixturenames:
        return True
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return False
    return any(isinstance(value, str) and value == "postgres" for value in callspec.params.values())


def _require_current_test_durations(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    required = (
        os.environ.get(_REQUIRE_CURRENT_TEST_DURATIONS_ENV_VAR, "").strip().lower()
        in _TRUTHY_ENV_VALUES
    )
    if not required or not items:
        return
    snapshot_path = Path(config.rootpath) / ".test_durations"
    try:
        stored = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise pytest.UsageError(f"invalid test-duration snapshot: {exc}") from exc
    if not isinstance(stored, dict):
        raise pytest.UsageError("test-duration snapshot must be a JSON object")
    unknown = sorted(item.nodeid for item in items if item.nodeid not in stored)
    allowed = max(1, int(len(items) * _MAX_UNKNOWN_DURATION_FRACTION))
    if len(unknown) <= allowed:
        return
    examples = ", ".join(unknown[:5])
    raise pytest.UsageError(
        ".test_durations is stale: "
        f"{len(unknown)} of {len(items)} collected tests lack timings "
        f"(maximum {allowed}); run "
        "`uv run pytest --store-durations --clean-durations`. "
        f"First missing tests: {examples}"
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Keep every Postgres consumer in its isolated CI lane."""

    for item in items:
        if item.nodeid.startswith("tests/qualification/"):
            item.add_marker(pytest.mark.qualification)
        if _requests_postgres(item):
            item.add_marker(pytest.mark.postgres)
    _require_current_test_durations(config, items)


@pytest.fixture(scope="session")
def _postgres_server_dsn() -> str:
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


@pytest.fixture(scope="module")
def postgres_dsn(_postgres_server_dsn: str) -> str:
    """Give each test module its own database on the shared disposable server.

    Module-specific schema revisions and alias keyrings must not become the
    deployment configuration observed by another module. An externally supplied
    test DSN therefore needs permission to create disposable databases.
    """
    from uuid import uuid4

    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    database = f"cayu_test_{uuid4().hex}"
    with psycopg.connect(_postgres_server_dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    try:
        yield make_conninfo(_postgres_server_dsn, dbname=database)
    finally:
        with psycopg.connect(_postgres_server_dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database))
            )
