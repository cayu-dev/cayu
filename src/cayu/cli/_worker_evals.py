"""Run a project's eval runs on its eval worker.

``[tool.cayu.evals] worker = "NAME"`` makes ``cayu worker NAME`` run the eval runs
the Agent's server admits for the targets in ``[tool.cayu.evals].targets``; the
server then leaves them to it. On Cayu Cloud that puts them where its deploy drain
and its until-idle wakes apply: a draining worker finishes the run it holds before
it exits, and a worker that runs until idle returns only once no run is left.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from cayu.applications import CayuApp
from cayu.cli._project_evals import load_project_eval_targets
from cayu.cli.project_control_plane import (
    build_project_control_plane_context,
    close_project_control_plane_context,
    project_eval_worker,
)
from cayu.project_control_plane import (
    ProjectControlPlaneContext,
    resolve_project_control_plane_context,
)
from cayu.server.evals_registry import (
    EvalTargetRegistry,
    declared_eval_target_registrations,
    resolved_evals_runtime,
)
from cayu.server.evals_worker import EvalRunCoordinator


@dataclass(frozen=True, slots=True)
class WorkerEvals:
    """The coordinator that runs this worker's eval runs, and the context that owns its store."""

    coordinator: EvalRunCoordinator
    context: ProjectControlPlaneContext
    # The declared targets' own applications, which the worker closes when it stops.
    apps: tuple[CayuApp, ...]

    def close(self) -> None:
        close_project_control_plane_context(self.context)


def worker_evals(root: Path, name: str, app: CayuApp) -> WorkerEvals | None:
    """The eval runs worker ``name`` runs, or None when it runs none.

    Call it inside ``project_context(root)``, outside a running event loop. A
    declared target that can't be loaded is reported and left out, as ``cayu serve``
    does; without a durable eval store the worker runs no eval runs and says so.
    """

    if project_eval_worker(root) != name:
        return None

    def report(reference: str, reason: str) -> None:
        print(
            f"warning: worker {name!r} runs no eval runs for {reference}: {reason}",
            file=sys.stderr,
            flush=True,
        )

    targets = load_project_eval_targets(root, on_error=report)
    context = build_project_control_plane_context(root, mode="production", eval_targets=targets)
    try:
        resolved = resolve_project_control_plane_context(context, app)
        assert resolved is not None
        if not targets or resolved.eval_store is None:
            print(
                f"warning: worker {name!r} runs no eval runs: "
                + (
                    "none of [tool.cayu.evals].targets loaded."
                    if not targets
                    else "the project has no durable eval store (CAYU_DATABASE_URL or "
                    "[tool.cayu.session_store])."
                ),
                file=sys.stderr,
                flush=True,
            )
            close_project_control_plane_context(context)
            return None
        runtime = resolved_evals_runtime(
            explicit=None,
            registry=EvalTargetRegistry(
                declared_eval_target_registrations(
                    targets, project_root=resolved.app_manifest_project_root
                )
            ),
            automatic_store=resolved.eval_store,
        )
        assert runtime is not None
    except BaseException:
        close_project_control_plane_context(context)
        raise
    return WorkerEvals(
        coordinator=EvalRunCoordinator(runtime),
        context=context,
        apps=context.declared_eval_apps(exclude=app),
    )
