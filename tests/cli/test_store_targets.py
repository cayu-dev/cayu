from __future__ import annotations

from pathlib import Path

import pytest

from cayu.cli.store_targets import (
    SessionStoreBackend,
    SessionStoreTargetError,
    resolve_session_store_target,
)


def _write_project(root: Path, config: str) -> Path:
    pyproject = root / "pyproject.toml"
    pyproject.write_text(config, encoding="utf-8")
    return pyproject


def test_explicit_store_target_overrides_environment_and_project(tmp_path: Path) -> None:
    _write_project(
        tmp_path,
        """
[tool.cayu.session_store]
backend = "sqlite"
path = "data/configured.db"
""",
    )

    target = resolve_session_store_target(
        sqlite=tmp_path / "explicit.db",
        environ={"CAYU_DATABASE_URL": "postgresql://user:secret@db.example/cayu"},
        start=tmp_path,
    )

    assert target.backend is SessionStoreBackend.SQLITE
    assert target.sqlite_path == (tmp_path / "explicit.db").resolve()
    assert target.postgres_dsn is None
    assert target.source == "explicit"


def test_explicit_relative_sqlite_path_resolves_from_start(tmp_path: Path) -> None:
    start = tmp_path / "project" / "src"
    start.mkdir(parents=True)

    target = resolve_session_store_target(
        sqlite="../data/cayu.db",
        environ={},
        start=start,
    )

    assert target.sqlite_path == (start / "../data/cayu.db").resolve()


def test_store_target_rejects_mutually_exclusive_explicit_selectors(tmp_path: Path) -> None:
    with pytest.raises(SessionStoreTargetError, match="mutually exclusive"):
        resolve_session_store_target(
            sqlite=tmp_path / "cayu.db",
            postgres="postgresql://db.example/cayu",
            environ={},
            start=tmp_path,
        )


def test_explicit_sqlite_target_rejects_blank_value(tmp_path: Path) -> None:
    with pytest.raises(SessionStoreTargetError, match="--sqlite must be a non-empty path"):
        resolve_session_store_target(sqlite="", environ={}, start=tmp_path)


def test_explicit_postgres_target_is_validated(tmp_path: Path) -> None:
    target = resolve_session_store_target(
        postgres="postgresql://db.example/cayu",
        environ={},
        start=tmp_path,
    )

    assert target.backend is SessionStoreBackend.POSTGRES
    assert target.postgres_dsn == "postgresql://db.example/cayu"
    assert target.source == "explicit"


@pytest.mark.parametrize(
    "url",
    [
        "postgresql:///cayu",
        "postgresql://app@/cayu?host=/var/run/postgresql",
    ],
)
def test_postgres_target_accepts_libpq_socket_urls(tmp_path: Path, url: str) -> None:
    target = resolve_session_store_target(postgres=url, environ={}, start=tmp_path)

    assert target.backend is SessionStoreBackend.POSTGRES
    assert target.postgres_dsn == url


@pytest.mark.parametrize(
    "value",
    [
        "not-a-dsn",
        "postgresql://",
        "mysql://admin:do-not-print@db.example/cayu",
        "sqlite:///data/cayu.db",
        "postgresql://admin:do-not-print@[invalid/cayu",
    ],
)
def test_explicit_postgres_target_rejects_invalid_urls_without_credentials(
    tmp_path: Path,
    value: str,
) -> None:
    with pytest.raises(SessionStoreTargetError) as excinfo:
        resolve_session_store_target(postgres=value, environ={}, start=tmp_path)

    assert "do-not-print" not in str(excinfo.value)


def test_database_url_precedes_project_configuration(tmp_path: Path) -> None:
    _write_project(
        tmp_path,
        """
[tool.cayu.session_store]
backend = "sqlite"
path = "data/configured.db"
""",
    )

    target = resolve_session_store_target(
        environ={"CAYU_DATABASE_URL": "postgresql://user:secret@db.example/cayu"},
        start=tmp_path,
    )

    assert target.backend is SessionStoreBackend.POSTGRES
    assert target.postgres_dsn == "postgresql://user:secret@db.example/cayu"
    assert target.source == "environment:CAYU_DATABASE_URL"


def test_database_url_accepts_absolute_sqlite_url(tmp_path: Path) -> None:
    database = tmp_path / "data" / "cayu.db"

    target = resolve_session_store_target(
        environ={"CAYU_DATABASE_URL": f"sqlite://{database}"},
        start=tmp_path / "ignored",
    )

    assert target.backend is SessionStoreBackend.SQLITE
    assert target.sqlite_path == database
    assert target.source == "environment:CAYU_DATABASE_URL"


def test_database_url_rejects_relative_sqlite_url(tmp_path: Path) -> None:
    with pytest.raises(SessionStoreTargetError, match="absolute SQLite URL"):
        resolve_session_store_target(
            environ={"CAYU_DATABASE_URL": "sqlite:data/cayu.db"},
            start=tmp_path,
        )


def test_database_url_rejects_sqlite_url_with_hostname(tmp_path: Path) -> None:
    with pytest.raises(SessionStoreTargetError, match="malformed SQLite URL"):
        resolve_session_store_target(
            environ={"CAYU_DATABASE_URL": "sqlite://database.example/data/cayu.db"},
            start=tmp_path,
        )


def test_blank_database_url_is_an_explicit_configuration_error(tmp_path: Path) -> None:
    _write_project(
        tmp_path,
        '[tool.cayu.session_store]\nbackend = "sqlite"\npath = "data/cayu.db"\n',
    )

    with pytest.raises(SessionStoreTargetError, match="must be a non-empty string"):
        resolve_session_store_target(
            environ={"CAYU_DATABASE_URL": ""},
            start=tmp_path,
        )


def test_configured_relative_sqlite_path_resolves_from_pyproject(tmp_path: Path) -> None:
    project = tmp_path / "project"
    nested = project / "src" / "package"
    nested.mkdir(parents=True)
    pyproject = _write_project(
        project,
        """
[tool.cayu]
factory = "app:build_app"

[tool.cayu.session_store]
backend = "sqlite"
path = "data/cayu.db"
""",
    )

    target = resolve_session_store_target(environ={}, start=nested)

    assert target.backend is SessionStoreBackend.SQLITE
    assert target.sqlite_path == project / "data" / "cayu.db"
    assert target.config_path == pyproject
    assert target.source == "project"


def test_explicitly_configured_custom_sqlite_path_is_used(tmp_path: Path) -> None:
    custom = tmp_path / "var" / "runtime.db"
    _write_project(
        tmp_path,
        """
[tool.cayu.session_store]
backend = "sqlite"
path = "var/runtime.db"
""",
    )

    target = resolve_session_store_target(environ={}, start=tmp_path)

    assert target.sqlite_path == custom
    assert target.source == "project"


def test_configured_postgres_target_reads_named_environment_variable(tmp_path: Path) -> None:
    _write_project(
        tmp_path,
        """
[tool.cayu.session_store]
backend = "postgres"
env = "PROJECT_DATABASE_URL"
""",
    )

    target = resolve_session_store_target(
        environ={"PROJECT_DATABASE_URL": "postgres://db.example/cayu"},
        start=tmp_path,
    )

    assert target.backend is SessionStoreBackend.POSTGRES
    assert target.postgres_dsn == "postgres://db.example/cayu"
    assert target.source == "project:PROJECT_DATABASE_URL"


def test_configured_postgres_target_requires_named_environment_variable(tmp_path: Path) -> None:
    pyproject = _write_project(
        tmp_path,
        '[tool.cayu.session_store]\nbackend = "postgres"\nenv = "PROJECT_DATABASE_URL"\n',
    )

    with pytest.raises(SessionStoreTargetError) as excinfo:
        resolve_session_store_target(environ={}, start=tmp_path)

    assert str(pyproject) in str(excinfo.value)
    assert "PROJECT_DATABASE_URL is not set" in str(excinfo.value)


def test_configured_postgres_target_rejects_non_postgres_url_without_secret(
    tmp_path: Path,
) -> None:
    secret = "do-not-print"
    pyproject = _write_project(
        tmp_path,
        '[tool.cayu.session_store]\nbackend = "postgres"\nenv = "PROJECT_DATABASE_URL"\n',
    )

    with pytest.raises(SessionStoreTargetError) as excinfo:
        resolve_session_store_target(
            environ={"PROJECT_DATABASE_URL": f"sqlite:///{secret}.db"},
            start=tmp_path,
        )

    assert str(pyproject) in str(excinfo.value)
    assert "must contain a Postgres URL" in str(excinfo.value)
    assert secret not in str(excinfo.value)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ('[tool.cayu.session_store]\nbackend = "sqlite"\n', "requires path"),
        (
            '[tool.cayu.session_store]\nbackend = "postgres"\ndsn = "postgresql://secret"\n',
            "requires env",
        ),
        ('[tool.cayu.session_store]\nbackend = "mysql"\n', "backend must be"),
        ('[tool.cayu]\nsession_store = "data/cayu.db"\n', "must be a table"),
        (
            '[tool.cayu.session_store]\nbackend = "sqlite"\npath = "data/cayu.db"\nextra = true\n',
            "unsupported keys: extra",
        ),
        (
            '[tool.cayu.session_store]\nbackend = "postgres"\nenv = "DATABASE_URL"\nextra = true\n',
            "unsupported keys: extra",
        ),
    ],
)
def test_malformed_store_configuration_fails_actionably(
    tmp_path: Path,
    config: str,
    message: str,
) -> None:
    pyproject = _write_project(tmp_path, config)

    with pytest.raises(SessionStoreTargetError) as excinfo:
        resolve_session_store_target(environ={}, start=tmp_path)

    assert str(pyproject) in str(excinfo.value)
    assert message in str(excinfo.value)


def test_malformed_database_url_error_redacts_credentials(tmp_path: Path) -> None:
    secret = "do-not-print"

    with pytest.raises(SessionStoreTargetError) as excinfo:
        resolve_session_store_target(
            environ={"CAYU_DATABASE_URL": f"mysql://admin:{secret}@db.example/cayu"},
            start=tmp_path,
        )

    assert secret not in str(excinfo.value)
    assert "unsupported database URL scheme" in str(excinfo.value)


def test_unsupported_database_url_error_does_not_echo_parsed_scheme(tmp_path: Path) -> None:
    secret = "do-not-print"

    with pytest.raises(SessionStoreTargetError) as excinfo:
        resolve_session_store_target(
            environ={"CAYU_DATABASE_URL": f"{secret}:rest@db.example/cayu"},
            start=tmp_path,
        )

    assert secret not in str(excinfo.value)
    assert "unsupported database URL scheme" in str(excinfo.value)


def test_resolved_postgres_target_repr_omits_dsn(tmp_path: Path) -> None:
    secret = "do-not-print"
    target = resolve_session_store_target(
        postgres=f"postgresql://admin:{secret}@db.example/cayu",
        environ={},
        start=tmp_path,
    )

    assert secret not in repr(target)
    assert "postgres_dsn" not in repr(target)


def test_structurally_invalid_database_url_error_redacts_credentials(tmp_path: Path) -> None:
    secret = "do-not-print"

    with pytest.raises(SessionStoreTargetError) as excinfo:
        resolve_session_store_target(
            environ={"CAYU_DATABASE_URL": f"postgresql://admin:{secret}@[invalid/cayu"},
            start=tmp_path,
        )

    assert secret not in str(excinfo.value)
    assert "malformed database URL" in str(excinfo.value)


def test_missing_store_target_has_zero_flag_and_override_guidance(tmp_path: Path) -> None:
    with pytest.raises(SessionStoreTargetError) as excinfo:
        resolve_session_store_target(environ={}, start=tmp_path)

    message = str(excinfo.value)
    assert "[tool.cayu.session_store]" in message
    assert "CAYU_DATABASE_URL" in message
    assert "--sqlite" in message
    assert "--postgres" in message


def test_factory_only_project_stops_discovery_and_reports_its_configuration(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    nested = project / "src" / "package"
    nested.mkdir(parents=True)
    pyproject = _write_project(project, '[tool.cayu]\nfactory = "app:build_app"\n')

    with pytest.raises(SessionStoreTargetError) as excinfo:
        resolve_session_store_target(environ={}, start=nested)

    assert str(pyproject) in str(excinfo.value)


def test_malformed_pyproject_is_reported_as_store_target_error(tmp_path: Path) -> None:
    _write_project(tmp_path, "[tool.cayu\n")

    with pytest.raises(SessionStoreTargetError, match="Could not read"):
        resolve_session_store_target(environ={}, start=tmp_path)
