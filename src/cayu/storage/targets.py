"""Database target parsing and environment selection shared by apps and the CLI.

Application factories (through :func:`cayu.open_application_stores`) and the
storage-aware CLI commands read the same variables through this module, so the
two cannot select different stores from the same configuration:

- ``CAYU_DATABASE_URL`` selects PostgreSQL (``postgres://`` or ``postgresql://``)
  or an absolute SQLite file (``sqlite:///abs/path.db``). Unset means the
  project's local SQLite default.
- ``CAYU_DATABASE_DIRECT_URL`` optionally gives the task-admission ``LISTEN``
  connection a direct server address when ``CAYU_DATABASE_URL`` points at a
  transaction-pooling proxy such as PgBouncer.
- ``CAYU_DATABASE_POOL_MAX`` bounds each shared PostgreSQL connection pool.
- ``CAYU_REQUIRE_POSTGRES=1`` makes every Cayu SQLite store refuse to open.

This module imports no database driver and no CLI code.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import cast
from urllib.parse import unquote, urlsplit

DATABASE_URL_ENV = "CAYU_DATABASE_URL"
DATABASE_DIRECT_URL_ENV = "CAYU_DATABASE_DIRECT_URL"
DATABASE_POOL_MAX_ENV = "CAYU_DATABASE_POOL_MAX"
REQUIRE_POSTGRES_ENV = "CAYU_REQUIRE_POSTGRES"
DEFAULT_DATABASE_POOL_MAX = 5
CANONICAL_SQLITE_PATH = Path("data/cayu.db")


class SessionStoreBackend(StrEnum):
    SQLITE = "sqlite"
    POSTGRES = "postgres"


@dataclass(frozen=True)
class SessionStoreTarget:
    """A typed durable-store selection safe to pass to a backend constructor."""

    backend: SessionStoreBackend
    sqlite_path: Path | None = None
    postgres_dsn: str | None = field(default=None, repr=False)
    source: str = "explicit"
    config_path: Path | None = None

    def __post_init__(self) -> None:
        if self.backend is SessionStoreBackend.SQLITE:
            if self.sqlite_path is None or self.postgres_dsn is not None:
                raise ValueError("A SQLite target requires only sqlite_path.")
        elif self.postgres_dsn is None or self.sqlite_path is not None:
            raise ValueError("A Postgres target requires only postgres_dsn.")


class SessionStoreTargetError(ValueError):
    """A store target could not be resolved without guessing."""


class PostgresRequiredError(RuntimeError):
    """A SQLite store was opened while ``CAYU_REQUIRE_POSTGRES=1`` is set."""


def configured_database_url(environ: Mapping[str, str] | None = None) -> str | None:
    """Return ``CAYU_DATABASE_URL`` exactly as configured, or ``None`` when unset.

    A set but blank value is returned unchanged so the store selection rejects
    it instead of silently falling back to local SQLite.
    """

    return (os.environ if environ is None else environ).get(DATABASE_URL_ENV)


def configured_database_direct_url(environ: Mapping[str, str] | None = None) -> str | None:
    """Return ``CAYU_DATABASE_DIRECT_URL``, or ``None`` when unset."""

    return (os.environ if environ is None else environ).get(DATABASE_DIRECT_URL_ENV)


def configured_database_pool_max(environ: Mapping[str, str] | None = None) -> int:
    """Return the per-pool PostgreSQL connection bound (default 5)."""

    value = (os.environ if environ is None else environ).get(DATABASE_POOL_MAX_ENV)
    if value is None:
        return DEFAULT_DATABASE_POOL_MAX
    text = value.strip()
    if not text.isascii() or not text.isdigit() or int(text) < 1:
        raise SessionStoreTargetError(f"{DATABASE_POOL_MAX_ENV} must be a positive integer.")
    return int(text)


def postgres_required(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether ``CAYU_REQUIRE_POSTGRES`` forbids Cayu SQLite stores."""

    value = (os.environ if environ is None else environ).get(REQUIRE_POSTGRES_ENV)
    if value is None or value.strip() in {"", "0"}:
        return False
    if value.strip() == "1":
        return True
    raise PostgresRequiredError(
        f"{REQUIRE_POSTGRES_ENV} must be 1 (require PostgreSQL) or 0 when set; "
        "Cayu SQLite stores refuse to open until it is corrected."
    )


def require_sqlite_store_allowed(store_name: str) -> None:
    """Refuse to construct a SQLite-backed store when PostgreSQL is required.

    Every Cayu SQLite store calls this at construction. Deployments set
    ``CAYU_REQUIRE_POSTGRES=1`` so a missing ``CAYU_DATABASE_URL`` fails at
    startup instead of writing durable state to a local or network filesystem.
    """

    if not postgres_required():
        return
    raise PostgresRequiredError(
        f"{store_name} cannot open because {REQUIRE_POSTGRES_ENV}=1 requires PostgreSQL. "
        f"Set {DATABASE_URL_ENV} to the postgresql:// URL of a migrated database "
        "(run `cayu storage migrate` as a deploy step), or unset "
        f"{REQUIRE_POSTGRES_ENV} for local development."
    )


def parse_database_url(value: str, *, source: str) -> SessionStoreTarget:
    """Parse a PostgreSQL or absolute SQLite URL without echoing credentials."""

    url = _require_nonblank(value, source)
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise SessionStoreTargetError(f"{source} contains a malformed database URL.") from exc
    scheme = parts.scheme.lower()
    if scheme in {"postgres", "postgresql"}:
        if not (parts.netloc or parts.path.strip("/") or parts.query):
            raise SessionStoreTargetError(f"{source} contains a malformed Postgres URL.")
        return SessionStoreTarget(
            backend=SessionStoreBackend.POSTGRES,
            postgres_dsn=url,
            source=source,
        )
    if scheme == "sqlite":
        if parts.netloc not in {"", "localhost"} or not parts.path or parts.query or parts.fragment:
            raise SessionStoreTargetError(f"{source} contains a malformed SQLite URL.")
        path = Path(unquote(parts.path))
        if not path.is_absolute():
            raise SessionStoreTargetError(f"{source} must contain an absolute SQLite URL.")
        return SessionStoreTarget(
            backend=SessionStoreBackend.SQLITE,
            sqlite_path=path.resolve(),
            source=source,
        )
    raise SessionStoreTargetError(
        f"{source} uses an unsupported database URL scheme; use postgresql:// "
        "or an absolute sqlite:/// URL."
    )


def session_store_target_from_config(
    value: object,
    *,
    pyproject: Path,
    environ: Mapping[str, str],
) -> SessionStoreTarget:
    """Parse one ``[tool.cayu.session_store]`` table from ``pyproject``."""

    if not isinstance(value, dict):
        raise SessionStoreTargetError(f"{pyproject}: [tool.cayu].session_store must be a table.")
    configured = cast("dict[str, object]", value)
    backend = configured.get("backend")
    if backend == SessionStoreBackend.SQLITE:
        path_value = configured.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            raise SessionStoreTargetError(
                f"{pyproject}: SQLite [tool.cayu.session_store] requires path."
            )
        unexpected = set(configured) - {"backend", "path"}
        if unexpected:
            raise SessionStoreTargetError(
                f"{pyproject}: SQLite [tool.cayu.session_store] has unsupported keys: "
                f"{', '.join(sorted(unexpected))}."
            )
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = pyproject.parent / path
        return SessionStoreTarget(
            backend=SessionStoreBackend.SQLITE,
            sqlite_path=path.resolve(),
            source="project",
            config_path=pyproject,
        )
    if backend == SessionStoreBackend.POSTGRES:
        env_value = configured.get("env")
        if not isinstance(env_value, str) or not env_value.strip():
            raise SessionStoreTargetError(
                f"{pyproject}: Postgres [tool.cayu.session_store] requires env; "
                "do not commit a DSN."
            )
        unexpected = set(configured) - {"backend", "env"}
        if unexpected:
            raise SessionStoreTargetError(
                f"{pyproject}: Postgres [tool.cayu.session_store] has unsupported keys: "
                f"{', '.join(sorted(unexpected))}."
            )
        env_name = env_value.strip()
        dsn = environ.get(env_name)
        if dsn is None or not dsn.strip():
            raise SessionStoreTargetError(
                f"{pyproject}: environment variable {env_name} is not set."
            )
        target = parse_database_url(dsn, source=f"project:{env_name}")
        if target.backend is not SessionStoreBackend.POSTGRES:
            raise SessionStoreTargetError(
                f"{pyproject}: environment variable {env_name} must contain a Postgres URL."
            )
        return SessionStoreTarget(
            backend=target.backend,
            postgres_dsn=target.postgres_dsn,
            source=target.source,
            config_path=pyproject,
        )
    raise SessionStoreTargetError(
        f'{pyproject}: [tool.cayu.session_store].backend must be "sqlite" or "postgres".'
    )


def application_store_target(
    database_url: str | None,
    *,
    sqlite_path: str | os.PathLike[str],
) -> SessionStoreTarget:
    """Select the application's durable store from a URL or the local SQLite path.

    ``database_url`` is normally :func:`configured_database_url`. ``None`` selects
    ``sqlite_path``, which must be absolute so the store never depends on the
    process working directory.
    """

    if database_url is not None:
        return parse_database_url(database_url, source=f"environment:{DATABASE_URL_ENV}")
    path = Path(sqlite_path)
    if not path.is_absolute():
        raise SessionStoreTargetError(
            "sqlite_path must be absolute; derive it from the project root, not the "
            "working directory."
        )
    return SessionStoreTarget(
        backend=SessionStoreBackend.SQLITE,
        sqlite_path=path.resolve(),
        source="application-default",
    )


def _require_nonblank(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SessionStoreTargetError(f"{label} must be a non-empty string.")
    return value.strip()


__all__ = [
    "CANONICAL_SQLITE_PATH",
    "DATABASE_DIRECT_URL_ENV",
    "DATABASE_POOL_MAX_ENV",
    "DATABASE_URL_ENV",
    "DEFAULT_DATABASE_POOL_MAX",
    "REQUIRE_POSTGRES_ENV",
    "PostgresRequiredError",
    "SessionStoreBackend",
    "SessionStoreTarget",
    "SessionStoreTargetError",
    "application_store_target",
    "configured_database_direct_url",
    "configured_database_pool_max",
    "configured_database_url",
    "parse_database_url",
    "postgres_required",
    "require_sqlite_store_allowed",
    "session_store_target_from_config",
]
