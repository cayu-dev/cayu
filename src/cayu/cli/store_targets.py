"""Typed, non-booting durable-session store discovery for Cayu CLI commands.

URL and ``[tool.cayu.session_store]`` parsing lives in :mod:`cayu.storage.targets`
so application factories and the CLI share one implementation. This module adds
project discovery and re-exports the shared names for existing importers.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from cayu.cli.project import ProjectError, discover_cayu_project_configuration
from cayu.storage.targets import (
    CANONICAL_SQLITE_PATH,
    SessionStoreBackend,
    SessionStoreTarget,
    SessionStoreTargetError,
    configured_database_url,
    parse_database_url,
    session_store_target_from_config,
)
from cayu.storage.targets import DATABASE_URL_ENV as _DATABASE_URL_ENV


def resolve_session_store_target(
    *,
    sqlite: str | Path | None = None,
    postgres: str | None = None,
    environ: Mapping[str, str] | None = None,
    start: Path | None = None,
) -> SessionStoreTarget:
    """Resolve explicit, environment, project-configured, then canonical targets."""

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
        target = parse_database_url(postgres, source="--postgres")
        if target.backend is not SessionStoreBackend.POSTGRES:
            raise SessionStoreTargetError("--postgres must contain a Postgres URL.")
        return SessionStoreTarget(
            backend=target.backend,
            postgres_dsn=target.postgres_dsn,
            source="explicit",
        )

    target, pyproject = _resolve_project_session_store_target(
        environ=environ,
        start=start,
        local_development=False,
    )
    if target is not None:
        return target
    raise _missing_target_error(pyproject)


def resolve_project_session_store_target(
    *,
    environ: Mapping[str, str] | None = None,
    start: Path | None = None,
    local_development: bool,
) -> SessionStoreTarget | None:
    """Resolve project storage for automatic Control Plane assembly.

    Unlike the ordinary CLI resolver, production may return ``None`` so the
    server can retain preview-only behavior. Explicit malformed configuration
    still fails. Trusted local development receives the documented project-local
    SQLite default even before that file exists.
    """

    if type(local_development) is not bool:
        raise TypeError("local_development must be a bool.")
    target, _ = _resolve_project_session_store_target(
        environ=environ,
        start=start,
        local_development=local_development,
    )
    return target


def _resolve_project_session_store_target(
    *,
    environ: Mapping[str, str] | None,
    start: Path | None,
    local_development: bool,
) -> tuple[SessionStoreTarget | None, Path | None]:
    environment = os.environ if environ is None else environ
    database_url = configured_database_url(environment)
    if database_url is not None:
        return (
            parse_database_url(
                database_url,
                source=f"environment:{_DATABASE_URL_ENV}",
            ),
            None,
        )

    try:
        project = discover_cayu_project_configuration(
            discovery_keys=("session_store", "factory"),
            start=start,
        )
    except ProjectError as exc:
        raise SessionStoreTargetError(str(exc)) from exc
    if project is None:
        return None, None

    configured = project.config.get("session_store")
    if configured is not None:
        return (
            session_store_target_from_config(
                configured,
                pyproject=project.pyproject,
                environ=environment,
            ),
            project.pyproject,
        )

    canonical = project.root / CANONICAL_SQLITE_PATH
    if canonical.is_file() or local_development:
        return (
            SessionStoreTarget(
                backend=SessionStoreBackend.SQLITE,
                sqlite_path=canonical,
                source=(
                    "canonical-discovery" if canonical.is_file() else "local-development-default"
                ),
                config_path=project.pyproject,
            ),
            project.pyproject,
        )
    return None, project.pyproject


def _missing_target_error(pyproject: Path | None = None) -> SessionStoreTargetError:
    location = "" if pyproject is None else f" in {pyproject}"
    return SessionStoreTargetError(
        "No Cayu session store is configured"
        f"{location}. Add [tool.cayu.session_store], set CAYU_DATABASE_URL, "
        "or pass --sqlite PATH or --postgres DSN."
    )


__all__ = [
    "CANONICAL_SQLITE_PATH",
    "SessionStoreBackend",
    "SessionStoreTarget",
    "SessionStoreTargetError",
    "configured_database_url",
    "parse_database_url",
    "resolve_project_session_store_target",
    "resolve_session_store_target",
    "session_store_target_from_config",
]
