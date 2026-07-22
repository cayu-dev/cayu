"""Typed, non-booting durable-session store discovery for Cayu CLI commands."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import cast
from urllib.parse import unquote, urlsplit

from cayu.cli.project import ProjectError, discover_cayu_project_configuration


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
    """A CLI store target could not be resolved without guessing."""


def resolve_session_store_target(
    *,
    sqlite: str | Path | None = None,
    postgres: str | None = None,
    environ: Mapping[str, str] | None = None,
    start: Path | None = None,
) -> SessionStoreTarget:
    """Resolve explicit, environment, then project-configured targets."""

    if sqlite is not None and postgres is not None:
        raise SessionStoreTargetError("--sqlite and --postgres are mutually exclusive.")
    if sqlite is not None:
        if isinstance(sqlite, str) and not sqlite.strip():
            raise SessionStoreTargetError("--sqlite must be a non-empty path.")
        path = Path(sqlite).expanduser()
        if not path.is_absolute():
            path = ((Path.cwd() if start is None else start) / path).resolve()
        else:
            path = path.resolve()
        return SessionStoreTarget(
            backend=SessionStoreBackend.SQLITE,
            sqlite_path=path,
            source="explicit",
        )
    if postgres is not None:
        target = _target_from_database_url(postgres, source="--postgres")
        if target.backend is not SessionStoreBackend.POSTGRES:
            raise SessionStoreTargetError("--postgres must contain a Postgres URL.")
        return SessionStoreTarget(
            backend=target.backend,
            postgres_dsn=target.postgres_dsn,
            source="explicit",
        )

    environment = os.environ if environ is None else environ
    database_url = environment.get("CAYU_DATABASE_URL")
    if database_url is not None:
        return _target_from_database_url(
            database_url,
            source="environment:CAYU_DATABASE_URL",
        )

    try:
        project = discover_cayu_project_configuration(
            discovery_keys=("session_store", "factory"),
            start=start,
        )
    except ProjectError as exc:
        raise SessionStoreTargetError(str(exc)) from exc
    if project is None:
        raise _missing_target_error()

    configured = project.config.get("session_store")
    if configured is not None:
        return _target_from_project_config(
            configured,
            pyproject=project.pyproject,
            environ=environment,
        )

    raise _missing_target_error(project.pyproject)


def _target_from_project_config(
    value: object,
    *,
    pyproject: Path,
    environ: Mapping[str, str],
) -> SessionStoreTarget:
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
        target = _target_from_database_url(dsn, source=f"project:{env_name}")
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


def _target_from_database_url(value: str, *, source: str) -> SessionStoreTarget:
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
    raise SessionStoreTargetError(f"{source} uses an unsupported database URL scheme.")


def _missing_target_error(pyproject: Path | None = None) -> SessionStoreTargetError:
    location = "" if pyproject is None else f" in {pyproject}"
    return SessionStoreTargetError(
        "No Cayu session store is configured"
        f"{location}. Add [tool.cayu.session_store], set CAYU_DATABASE_URL, "
        "or pass --sqlite PATH or --postgres DSN."
    )


def _require_nonblank(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SessionStoreTargetError(f"{label} must be a non-empty string.")
    return value.strip()
