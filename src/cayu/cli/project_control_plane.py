"""Project-owned Control Plane identity and EvalStore assembly."""

from __future__ import annotations

import asyncio
import os
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path

from cayu._validation import require_clean_nonblank, require_unicode_scalar_text
from cayu.cli.project import ProjectError
from cayu.cli.store_targets import (
    SessionStoreBackend,
    SessionStoreTarget,
    SessionStoreTargetError,
    resolve_project_session_store_target,
)
from cayu.project_control_plane import (
    ProjectControlPlaneAccess,
    ProjectControlPlaneContext,
    _create_project_control_plane_context,
)

_DISTRIBUTION_NAME_RE = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?\Z",
    re.ASCII,
)
_DISTRIBUTION_NAME_SEPARATOR_RE = re.compile(r"[-_.]+")
_MAX_PROJECT_ID_CHARS = 256
_MAX_RELEASE_ID_CHARS = 256


def build_project_control_plane_context(
    root: Path,
    *,
    mode: str,
    environ: Mapping[str, str] | None = None,
) -> ProjectControlPlaneContext:
    """Resolve one framework-owned context without importing project code."""

    root = root.resolve()
    if mode == "development":
        access = ProjectControlPlaneAccess.TRUSTED_LOCAL_DEVELOPMENT
        local_development = True
    elif mode == "production":
        access = ProjectControlPlaneAccess.AUTHENTICATED_PRODUCTION
        local_development = False
    else:
        raise ValueError("mode must be development or production.")
    environment = os.environ if environ is None else environ
    project_id = _project_id(root)
    configured_release_id = _configured_release_id(environment)
    try:
        target = resolve_project_session_store_target(
            environ=environment,
            start=root,
            local_development=local_development,
        )
    except SessionStoreTargetError as exc:
        raise ProjectError(str(exc)) from exc
    try:
        eval_store = _create_eval_store(target)
    except ProjectError:
        raise
    except Exception as exc:
        raise ProjectError(
            f"Could not initialize project Evals storage ({type(exc).__name__})."
        ) from exc
    return _create_project_control_plane_context(
        project_root=root,
        project_id=project_id,
        configured_release_id=configured_release_id,
        eval_store=eval_store,
        store_backend=None if target is None else target.backend.value,
        store_source=None if target is None else target.source,
        access=access,
    )


def close_project_control_plane_context(context: ProjectControlPlaneContext | None) -> None:
    """Synchronously close a CLI-owned context after serving or inspection."""

    if context is not None:
        try:
            asyncio.run(context.close())
        except Exception as exc:
            raise ProjectError(
                f"Could not close project Control Plane storage ({type(exc).__name__})."
            ) from exc


def _project_id(root: Path) -> str | None:
    pyproject = root / "pyproject.toml"
    try:
        document = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except FileNotFoundError:
        # Explicit ``cayu check module:factory`` targets do not require a
        # project declaration. They retain the normal missing-identity finding.
        return None
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ProjectError(f"Could not read {pyproject}: {exc}") from exc
    project = document.get("project")
    if project is None:
        return None
    if not isinstance(project, dict):
        raise ProjectError(f"{pyproject}: [project] must be a table.")
    name = project.get("name")
    if name is None:
        return None
    if not isinstance(name, str) or _DISTRIBUTION_NAME_RE.fullmatch(name) is None:
        raise ProjectError(f"{pyproject}: [project].name must be a valid distribution name.")
    normalized = _DISTRIBUTION_NAME_SEPARATOR_RE.sub("-", name).lower()
    if len(normalized) > _MAX_PROJECT_ID_CHARS:
        raise ProjectError(
            f"{pyproject}: normalized [project].name cannot exceed "
            f"{_MAX_PROJECT_ID_CHARS} characters."
        )
    return normalized


def _configured_release_id(environ: Mapping[str, str]) -> str | None:
    value = environ.get("CAYU_RELEASE_ID")
    if value is None:
        return None
    try:
        value = require_clean_nonblank(value, "CAYU_RELEASE_ID")
        require_unicode_scalar_text(value, "CAYU_RELEASE_ID")
    except (TypeError, ValueError) as exc:
        raise ProjectError("CAYU_RELEASE_ID must be clean nonblank text.") from exc
    if len(value) > _MAX_RELEASE_ID_CHARS:
        raise ProjectError(f"CAYU_RELEASE_ID cannot exceed {_MAX_RELEASE_ID_CHARS} characters.")
    return value


def _create_eval_store(target: SessionStoreTarget | None):
    if target is None:
        return None
    if target.backend is SessionStoreBackend.SQLITE:
        assert target.sqlite_path is not None
        from cayu.storage.evals_sqlite import SQLiteEvalStore

        return SQLiteEvalStore(target.sqlite_path)
    assert target.postgres_dsn is not None
    try:
        from cayu.storage.evals_postgres import PostgresEvalStore
    except ModuleNotFoundError as exc:
        if (exc.name or "").partition(".")[0] not in {"psycopg", "psycopg_pool"}:
            raise
        raise ProjectError(
            'PostgreSQL Evals storage requires the postgres extra. Install "cayu[postgres]".'
        ) from exc
    return PostgresEvalStore(target.postgres_dsn)
