"""Load the eval targets a project declares in ``[tool.cayu.evals].targets``.

Each entry names a ``module:function`` that returns a corpus-runnable eval plan: an
``EvalPlan`` with a ``corpus_target`` or a ``workflow_target`` and no direct suite
(or a bare ``WorkflowEvalTarget``). The plan's ``execution_profile_policy``, when it
declares one, sets how many trials and how much concurrency the target allows.

A declaration that fails to load never stops the Agent from serving: the caller is
told which one and why, and the others load as usual.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from pathlib import Path

from cayu.cli._targets import load_target
from cayu.cli.evals import _coerce_plan
from cayu.cli.project_control_plane import project_eval_target_references
from cayu.evals.execution_profiles import EvalExecutionProfilePolicyV1
from cayu.project_control_plane import ProjectEvalTarget

EvalTargetLoadError = Callable[[str, str], None]


def load_project_eval_targets(
    root: Path,
    *,
    on_error: EvalTargetLoadError,
) -> tuple[ProjectEvalTarget, ...]:
    """Load every declared eval target, reporting each one that can't be used.

    Call it with the project's import path in place (inside ``project_context``)
    and outside a running event loop. ``on_error`` receives the declaration and a
    one-line reason.
    """

    loaded: list[ProjectEvalTarget] = []
    keys: set[str] = set()
    for reference in project_eval_target_references(root):
        try:
            declared = _load(reference, project_root=root.resolve())
        # A declaration that exits, say for a missing eval-only credential, is skipped too.
        except (Exception, SystemExit) as exc:
            on_error(reference, f"{type(exc).__name__}: {exc}")
            continue
        if declared.target.key in keys:
            on_error(
                reference, f"another declared eval target already has key {declared.target.key!r}"
            )
            continue
        keys.add(declared.target.key)
        loaded.append(declared)
    return tuple(loaded)


def _load(reference: str, *, project_root: Path) -> ProjectEvalTarget:
    build = load_target(reference, label="Eval target", normalize_errors=True)
    if not callable(build):
        raise TypeError("an eval target must be a function that returns an eval plan")
    value = build()
    if inspect.isawaitable(value):
        value = asyncio.run(_awaited(value))
    plan = _coerce_plan(value)
    if plan.suite is not None:
        raise ValueError(
            "the plan has a direct suite; declare a corpus_target or a workflow_target "
            "without a suite so corpora can run against it"
        )
    target = plan.corpus_target if plan.corpus_target is not None else plan.workflow_target
    if target is None:
        raise ValueError("the plan has no corpus_target or workflow_target")
    policy = plan.execution_profile_policy
    declared = ProjectEvalTarget(
        reference=reference,
        target=target,
        execution_profile_policy=(
            EvalExecutionProfilePolicyV1.safe_default() if policy is None else policy
        ),
    )
    # Registering it checks the policy against the target's limits and its public catalog
    # entry, so a declaration the server couldn't register is reported here.
    from cayu.server.evals_registry import declared_eval_target_registrations

    declared_eval_target_registrations((declared,), project_root=project_root)
    return declared


async def _awaited(value):
    return await value
