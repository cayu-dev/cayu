"""Project-owned Control Plane identity and EvalStore assembly."""

from __future__ import annotations

import asyncio
import os
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

from cayu.budgets.pricing import PriceBook, default_price_book
from cayu.cli.project import ProjectError
from cayu.cli.store_targets import (
    SessionStoreBackend,
    SessionStoreTarget,
    SessionStoreTargetError,
    resolve_project_session_store_target,
)
from cayu.project_control_plane import (
    MAX_PROJECT_EVAL_TARGETS,
    ProjectControlPlaneAccess,
    ProjectControlPlaneContext,
    ProjectEvalJudgeConfiguration,
    ProjectEvalTarget,
    _create_project_control_plane_context,
)
from cayu.storage.targets import configured_database_pool_max

_DISTRIBUTION_NAME_RE = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?\Z",
    re.ASCII,
)
_DISTRIBUTION_NAME_SEPARATOR_RE = re.compile(r"[-_.]+")
_MAX_PROJECT_ID_CHARS = 256
_DEFAULT_JUDGE_KEYS = {
    "allow_same_model",
    "cost_currency",
    "max_input_tokens",
    "max_output_tokens",
    "max_total_tokens",
    "max_estimated_cost",
    "model",
    "privacy_policy",
    "provider",
    "timeout_seconds",
}
_EVALS_KEYS = {"default_judge", "price_book", "targets", "worker"}
_EVAL_TARGET_REFERENCE_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:[A-Za-z_][A-Za-z0-9_]*"
    r"(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z",
    re.ASCII,
)


def build_project_control_plane_context(
    root: Path,
    *,
    mode: str,
    environ: Mapping[str, str] | None = None,
    eval_targets: tuple[ProjectEvalTarget, ...] = (),
) -> ProjectControlPlaneContext:
    """Resolve one framework-owned context without importing project code.

    ``eval_targets`` are the declared eval targets the caller loaded from the
    project (see ``cayu.cli._project_evals``); this function imports none of them.
    """

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
    project_id, eval_judge_configuration, eval_price_book = _project_metadata(root)
    eval_worker = project_eval_worker(root)
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
        eval_judge_configuration=eval_judge_configuration,
        eval_price_book=eval_price_book,
        eval_store=eval_store,
        store_backend=None if target is None else target.backend.value,
        store_source=None if target is None else target.source,
        access=access,
        eval_targets=eval_targets,
        eval_worker=eval_worker,
    )


def project_eval_worker(root: Path) -> str | None:
    """The ``[tool.cayu.workers]`` name in ``[tool.cayu.evals].worker``, if any.

    That worker runs the eval runs of the project's declared eval targets, so the
    web process doesn't.
    """

    pyproject = root.resolve() / "pyproject.toml"
    try:
        document = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ProjectError(f"Could not read {pyproject}: {exc}") from exc
    evals = _evals_table(document, pyproject=pyproject)
    return None if evals is None else _eval_worker(document, evals, pyproject=pyproject)


def project_eval_target_references(root: Path) -> tuple[str, ...]:
    """The ``module:function`` references in ``[tool.cayu.evals].targets``, in order."""

    pyproject = root.resolve() / "pyproject.toml"
    try:
        document = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ()
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ProjectError(f"Could not read {pyproject}: {exc}") from exc
    evals = _evals_table(document, pyproject=pyproject)
    return () if evals is None else _eval_target_references(evals, pyproject=pyproject)


def close_project_control_plane_context(context: ProjectControlPlaneContext | None) -> None:
    """Synchronously close a CLI-owned context after serving or inspection."""

    if context is not None:
        try:
            asyncio.run(context.close())
        except Exception as exc:
            raise ProjectError(
                f"Could not close project Developer dashboard storage ({type(exc).__name__})."
            ) from exc


def _project_metadata(
    root: Path,
) -> tuple[str | None, ProjectEvalJudgeConfiguration | None, PriceBook | None]:
    pyproject = root / "pyproject.toml"
    try:
        document = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except FileNotFoundError:
        # Explicit ``cayu check module:factory`` targets do not require a
        # project declaration. They retain the normal missing-identity finding.
        return None, None, None
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ProjectError(f"Could not read {pyproject}: {exc}") from exc
    project = document.get("project")
    if project is None:
        project_id = None
    elif not isinstance(project, Mapping):
        raise ProjectError(f"{pyproject}: [project] must be a table.")
    else:
        project = cast("Mapping[str, object]", project)
        name = project.get("name")
        if name is None:
            project_id = None
        else:
            if not isinstance(name, str) or _DISTRIBUTION_NAME_RE.fullmatch(name) is None:
                raise ProjectError(
                    f"{pyproject}: [project].name must be a valid distribution name."
                )
            project_id = _DISTRIBUTION_NAME_SEPARATOR_RE.sub("-", name).lower()
    if project_id is not None and len(project_id) > _MAX_PROJECT_ID_CHARS:
        raise ProjectError(
            f"{pyproject}: normalized [project].name cannot exceed "
            f"{_MAX_PROJECT_ID_CHARS} characters."
        )
    eval_judge_configuration, eval_price_book = _project_evals_configuration(
        document,
        pyproject=pyproject,
    )
    return project_id, eval_judge_configuration, eval_price_book


def _evals_table(
    document: Mapping[str, object],
    *,
    pyproject: Path,
) -> Mapping[str, object] | None:
    tool = document.get("tool")
    if tool is None:
        return None
    if not isinstance(tool, Mapping):
        raise ProjectError(f"{pyproject}: [tool] must be a table.")
    tool = cast("Mapping[str, object]", tool)
    cayu = tool.get("cayu")
    if cayu is None:
        return None
    if not isinstance(cayu, Mapping):
        raise ProjectError(f"{pyproject}: [tool.cayu] must be a table.")
    cayu = cast("Mapping[str, object]", cayu)
    evals = cayu.get("evals")
    if evals is None:
        return None
    if not isinstance(evals, Mapping):
        raise ProjectError(f"{pyproject}: [tool.cayu.evals] must be a table.")
    evals = cast("Mapping[str, object]", evals)
    unexpected = set(evals) - _EVALS_KEYS
    if unexpected:
        raise ProjectError(
            f"{pyproject}: [tool.cayu.evals] has unsupported keys: {', '.join(sorted(unexpected))}."
        )
    return evals


def _eval_target_references(
    evals: Mapping[str, object],
    *,
    pyproject: Path,
) -> tuple[str, ...]:
    configured = evals.get("targets")
    if configured is None:
        return ()
    if not isinstance(configured, list) or not all(isinstance(item, str) for item in configured):
        raise ProjectError(
            f'{pyproject}: [tool.cayu.evals].targets must be a list of "module:function" text.'
        )
    references = cast("list[str]", configured)
    if len(references) > MAX_PROJECT_EVAL_TARGETS:
        raise ProjectError(
            f"{pyproject}: [tool.cayu.evals].targets lists at most "
            f"{MAX_PROJECT_EVAL_TARGETS} eval targets."
        )
    for reference in references:
        if _EVAL_TARGET_REFERENCE_RE.fullmatch(reference) is None:
            raise ProjectError(
                f"{pyproject}: [tool.cayu.evals].targets entry {reference!r} must be "
                '"module:function".'
            )
    if len(set(references)) != len(references):
        raise ProjectError(f"{pyproject}: [tool.cayu.evals].targets lists an entry twice.")
    return tuple(references)


def _eval_worker(
    document: Mapping[str, object],
    evals: Mapping[str, object],
    *,
    pyproject: Path,
) -> str | None:
    worker = evals.get("worker")
    if worker is None:
        return None
    if not isinstance(worker, str) or not worker.strip():
        raise ProjectError(f"{pyproject}: [tool.cayu.evals].worker must be a worker name.")
    if not _eval_target_references(evals, pyproject=pyproject):
        raise ProjectError(
            f"{pyproject}: [tool.cayu.evals].worker runs the declared eval targets; "
            "declare them in [tool.cayu.evals].targets."
        )
    tool = cast("Mapping[str, object]", document["tool"])
    cayu = cast("Mapping[str, object]", tool["cayu"])
    workers = cayu.get("workers")
    if not isinstance(workers, Mapping) or worker not in workers:
        raise ProjectError(
            f"{pyproject}: [tool.cayu.evals].worker names {worker!r}, which "
            "[tool.cayu.workers] doesn't declare."
        )
    return worker


def _project_evals_configuration(
    document: Mapping[str, object],
    *,
    pyproject: Path,
) -> tuple[ProjectEvalJudgeConfiguration | None, PriceBook | None]:
    evals = _evals_table(document, pyproject=pyproject)
    if evals is None:
        return None, None
    _eval_target_references(evals, pyproject=pyproject)
    _eval_worker(document, evals, pyproject=pyproject)
    price_book_source = evals.get("price_book")
    if price_book_source is None:
        eval_price_book = None
    elif price_book_source == "bundled-public":
        eval_price_book = default_price_book()
    else:
        raise ProjectError(f'{pyproject}: [tool.cayu.evals].price_book must be "bundled-public".')
    configured = evals.get("default_judge")
    if configured is None:
        return None, eval_price_book
    if not isinstance(configured, Mapping):
        raise ProjectError(f"{pyproject}: [tool.cayu.evals.default_judge] must be a table.")
    configured = cast("Mapping[str, object]", configured)
    unexpected = set(configured) - _DEFAULT_JUDGE_KEYS
    if unexpected:
        raise ProjectError(
            f"{pyproject}: [tool.cayu.evals.default_judge] has unsupported keys: "
            f"{', '.join(sorted(unexpected))}."
        )
    required = {"provider", "model", "privacy_policy", "allow_same_model"}
    missing = required - set(configured)
    if missing:
        raise ProjectError(
            f"{pyproject}: [tool.cayu.evals.default_judge] is missing required keys: "
            f"{', '.join(sorted(missing))}."
        )
    provider = configured["provider"]
    model = configured["model"]
    privacy_policy = configured["privacy_policy"]
    allow_same_model = configured["allow_same_model"]
    if not isinstance(provider, str) or not provider.strip():
        raise ProjectError(
            f"{pyproject}: [tool.cayu.evals.default_judge].provider must be non-empty text."
        )
    if not isinstance(model, str) or not model.strip():
        raise ProjectError(
            f"{pyproject}: [tool.cayu.evals.default_judge].model must be non-empty text."
        )
    if not isinstance(privacy_policy, str) or privacy_policy not in {
        "public-only",
        "public-and-transcript",
    }:
        raise ProjectError(
            f"{pyproject}: [tool.cayu.evals.default_judge].privacy_policy must be "
            '"public-only" or "public-and-transcript".'
        )
    if type(allow_same_model) is not bool:
        raise ProjectError(
            f"{pyproject}: [tool.cayu.evals.default_judge].allow_same_model must be true or false."
        )
    integer_defaults = {
        "timeout_seconds": 120,
        "max_input_tokens": 32_768,
        "max_output_tokens": 4_096,
        "max_total_tokens": 36_864,
    }
    integers: dict[str, int] = {}
    for key, default in integer_defaults.items():
        value = configured.get(key, default)
        if type(value) is not int:
            raise ProjectError(
                f"{pyproject}: [tool.cayu.evals.default_judge].{key} must be an integer."
            )
        integers[key] = value
    max_estimated_cost = configured.get("max_estimated_cost")
    cost_currency = configured.get("cost_currency")
    if max_estimated_cost is not None and not isinstance(max_estimated_cost, str):
        raise ProjectError(
            f"{pyproject}: [tool.cayu.evals.default_judge].max_estimated_cost must be text."
        )
    if cost_currency is not None and not isinstance(cost_currency, str):
        raise ProjectError(
            f"{pyproject}: [tool.cayu.evals.default_judge].cost_currency must be text."
        )
    try:
        judge = ProjectEvalJudgeConfiguration(
            provider_name=provider,
            model=model,
            privacy_policy=cast(
                'Literal["public-only", "public-and-transcript"]',
                privacy_policy,
            ),
            allow_same_model=allow_same_model,
            max_estimated_cost=max_estimated_cost,
            cost_currency=cost_currency,
            **integers,
        )
    except (TypeError, ValueError) as exc:
        raise ProjectError(
            f"{pyproject}: invalid [tool.cayu.evals.default_judge] configuration: {exc}"
        ) from exc
    if judge.max_estimated_cost is not None and eval_price_book is None:
        raise ProjectError(
            f"{pyproject}: a default judge cost ceiling requires "
            '[tool.cayu.evals].price_book = "bundled-public".'
        )
    return judge, eval_price_book


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
    try:
        pool_max_size = configured_database_pool_max()
    except SessionStoreTargetError as exc:
        raise ProjectError(str(exc)) from exc
    # The Evals store keeps its own pool, bounded like the application's shared pool.
    return PostgresEvalStore(target.postgres_dsn, max_size=pool_max_size)
