"""Give each eval trial its own workspace, prepared from its case's workspace fixture.

A trial of a case with a fixture runs in the case's own environment, re-sourced:
the environment's factory builds the same configuration over a fresh checkout of
the fixture's revision, with the fixture's files written in, and that factory
serves the environment name only inside the trial's context. The application's
registration, manifest and execution profile stay as they are, and concurrent
trials never share a source workspace.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from collections import Counter
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from cayu.environments.factory import EnvironmentFactory
from cayu.evals.workspace_fixtures import WorkspaceFixtureSpec
from cayu.runtime import _runtime_records as runtime_records
from cayu.sessions.requests import RunRequest
from cayu.workspaces.local import LocalWorkspace

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.evals.runner import EvalCase

logger = logging.getLogger(__name__)

_TRIAL_DIRECTORY_PREFIX = "cayu-eval-workspace-"
_TRIAL_WORKSPACE_DIRECTORY = "workspace"
_TRIAL_WORKSPACE_ID_PREFIX = "eval-workspace:"


def _registered_environment(
    app: CayuApp, request: RunRequest
) -> runtime_records.RegisteredEnvironment | None:
    """The environment a new run of ``request`` gets, resolved as the runtime resolves it."""

    return app._session_preparation.resolve_new_run_environment(
        request.environment_name,
        registered_agent=app._get_registered_agent(request.agent_name),
    )


def _new_trial_directory() -> Path:
    return Path(tempfile.mkdtemp(prefix=_TRIAL_DIRECTORY_PREFIX))


def _trial_source_workspace(template: LocalWorkspace, root: Path) -> LocalWorkspace:
    return LocalWorkspace(
        root,
        workspace_id=f"{_TRIAL_WORKSPACE_ID_PREFIX}{uuid4().hex}",
        excluded_directory_names=template.excluded_directory_names,
        excluded_path_patterns=template.excluded_path_patterns,
    )


def _resourceable_factory(
    case_id: str, registered: runtime_records.RegisteredEnvironment | None
) -> tuple[runtime_records.RegisteredEnvironment, EnvironmentFactory, LocalWorkspace]:
    if registered is None:
        raise ValueError(
            f"Eval case {case_id!r} has a workspace fixture, but its run has no environment."
        )
    factory = registered.factory
    template = None if factory is None else factory.source_workspace
    if factory is None or template is None:
        raise ValueError(
            f"Eval case {case_id!r} has a workspace fixture, but environment "
            f"{registered.spec.name!r} has no factory whose sessions copy in from a source "
            "workspace, so no trial could get its own copy."
        )
    return registered, factory, template


def require_workspace_fixtures_ready(
    app: CayuApp,
    cases: Sequence[EvalCase],
    *,
    pending_trials: Mapping[str, int],
    max_concurrency: int,
) -> None:
    """Refuse a suite whose trials could not each get their own workspace.

    A case with a workspace fixture needs a registered source and an environment
    whose factory can take another source workspace; each environment is checked
    once, by re-sourcing it over an empty directory. Without a fixture, trials
    that run at once in one environment whose sessions bind its single source
    workspace would refuse each other, so concurrency over such an environment
    is refused before any trial starts.
    """

    checked: set[str] = set()
    shared: Counter[str] = Counter()
    for case in cases:
        pending = pending_trials.get(case.id, 0)
        if not pending:
            continue
        fixture = case.workspace
        if fixture is None:
            if max_concurrency == 1:
                continue
            try:
                registered = _registered_environment(app, case.request)
            except (KeyError, ValueError, RuntimeError):
                continue  # The trial reports its own setup failure.
            if (
                registered is not None
                and registered.factory is not None
                and registered.factory.source_workspace is not None
            ):
                shared[registered.spec.name] += pending
            continue
        try:
            app.get_workspace_source(fixture.repository)
        except KeyError:
            raise ValueError(
                f"Eval case {case.id!r} names workspace source {fixture.repository!r}, which "
                "the application hasn't registered with register_workspace_source."
            ) from None
        try:
            resolved = _registered_environment(app, case.request)
        except (KeyError, ValueError, RuntimeError) as exc:
            raise ValueError(
                f"Eval case {case.id!r} has a workspace fixture, but its environment can't be "
                f"resolved: {exc}"
            ) from None
        registered, factory, template = _resourceable_factory(case.id, resolved)
        if registered.spec.name in checked:
            continue
        with tempfile.TemporaryDirectory(prefix=_TRIAL_DIRECTORY_PREFIX) as probe:
            resourced = factory.with_source_workspace(
                _trial_source_workspace(template, Path(probe))
            )
        if resourced is None:
            raise ValueError(
                f"Eval case {case.id!r} has a workspace fixture, but the factory of environment "
                f"{registered.spec.name!r} can't take another source workspace."
            )
        checked.add(registered.spec.name)
    crowded = sorted(name for name, count in shared.items() if count > 1)
    if crowded:
        raise ValueError(
            "Concurrent eval trials would share the source workspace of environment "
            + ", ".join(repr(name) for name in crowded)
            + ", which one session binds at a time. Give the cases a workspace fixture so "
            "each trial gets its own copy, or run them with max_concurrency=1."
        )


@asynccontextmanager
async def prepared_workspace_fixture(
    app: CayuApp,
    case_id: str,
    request: RunRequest,
    fixture: WorkspaceFixtureSpec,
) -> AsyncIterator[LocalWorkspace]:
    """Run the body with ``request``'s environment re-sourced at a fresh checkout of ``fixture``.

    Yields the checkout, the trial's source workspace. It lives in its own
    temporary directory, which is removed when the body ends, after the trial's
    evidence and assertions have read it.
    """

    registered, factory, template = _resourceable_factory(
        case_id, _registered_environment(app, request)
    )
    source = app.get_workspace_source(fixture.repository)
    trial_directory = await asyncio.to_thread(_new_trial_directory)
    workspace: LocalWorkspace | None = None
    try:
        root = trial_directory / _TRIAL_WORKSPACE_DIRECTORY
        root.mkdir()
        await source.checkout(fixture.revision, root)
        workspace = _trial_source_workspace(template, root)
        for path, content in fixture.files.items():
            await workspace.write_bytes(path, content.encode("utf-8"))
        trial_factory = factory.with_source_workspace(workspace)
        if trial_factory is None:
            raise RuntimeError(
                f"The factory of environment {registered.spec.name!r} no longer takes another "
                "source workspace."
            )
        try:
            with app._environment_registry.scoped_factory_registration(
                registered.spec,
                trial_factory,
                artifact_store=registered.environment.artifact_store,
                registration_site=(
                    registered.registration_source,
                    registered.registration_symbol,
                ),
            ):
                yield workspace
        finally:
            try:
                await trial_factory.close_idle_resources()
            except Exception:
                logger.warning(
                    "Closing the idle resources of eval case %r's workspace fixture failed.",
                    case_id,
                    exc_info=True,
                )
    finally:
        if workspace is not None and app._environment_lifecycle.binds_source_workspace(
            workspace.id
        ):
            # A session's environment cleanup is retained for a finalize retry that
            # copies back into this checkout, so it must outlive the trial.
            logger.warning(
                "Keeping eval case %r's workspace %s: its session's environment cleanup is "
                "retained for a retry that copies back into it. Remove it once "
                "drain_environment_cleanups has settled the session.",
                case_id,
                trial_directory,
            )
        else:
            try:
                await asyncio.to_thread(shutil.rmtree, trial_directory)
            except OSError:
                logger.warning(
                    "Removing eval case %r's workspace %s failed.",
                    case_id,
                    trial_directory,
                    exc_info=True,
                )


__all__ = ["prepared_workspace_fixture", "require_workspace_fixtures_ready"]
