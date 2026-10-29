"""Loading the eval targets a project declares in ``[tool.cayu.evals].targets``."""

from __future__ import annotations

import asyncio
from pathlib import Path

from cayu.cli._project_evals import load_project_eval_targets
from cayu.cli.project import project_context
from cayu.cli.serve import (
    _resolve_serve_project,
    parse_serve_arguments,
    serve_project_control_plane,
)
from cayu.evals.execution import CorpusTarget
from cayu.evals.execution_profiles import EvalExecutionProfilePolicyV1

_MODULE = """
from cayu import Message, RunRequest
from cayu.evals import EvalCase, EvalPlan, EvalSuite, FinalOutputContains
from cayu.evals.execution_profiles import EvalExecutionProfilePolicyV1
from cayu.evals.execution import CorpusExecutionLimits
from tests.evals.test_corpus_execution import _provider, _target

POLICY = EvalExecutionProfilePolicyV1(
    reset_strategy="application_managed",
    isolation_revision="sha256:" + "1" * 64,
    max_trials=2,
    max_concurrency=2,
)


def corpus_plan():
    return EvalPlan(corpus_target=_target(_provider(), key="refund-agent"), execution_profile_policy=POLICY)


async def awaited_plan():
    return EvalPlan(corpus_target=_target(_provider(), key="awaited-agent"))


def same_key_plan():
    return EvalPlan(corpus_target=_target(_provider(), key="refund-agent"))


def direct_suite_plan():
    app = _target(_provider()).app
    case = EvalCase(
        id="refund",
        request=RunRequest(agent_name="agent", messages=[Message.text("user", "Refund?")]),
        assertions=[FinalOutputContains("Approved")],
    )
    return EvalPlan(app=app, suite=EvalSuite(id="refunds", cases=[case]))


def over_its_limits_plan():
    return EvalPlan(
        corpus_target=_target(_provider(), key="limited-agent", limits=CorpusExecutionLimits(max_trials=1)),
        execution_profile_policy=POLICY,
    )


def exiting_plan():
    raise SystemExit("missing eval-only credential")


def broken_plan():
    raise RuntimeError("no provider key in this shell")


NOT_CALLABLE = 3
"""


def _project(root: Path, *targets: str) -> Path:
    listed = ", ".join(f'"{target}"' for target in targets)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "declared-evals"\n\n'
        '[tool.cayu]\nfactory = "app:build_app"\n\n'
        f"[tool.cayu.evals]\ntargets = [{listed}]\n",
        encoding="utf-8",
    )
    (root / "declared_eval_targets.py").write_text(_MODULE, encoding="utf-8")
    return root


def _load(root: Path) -> tuple[tuple, list[tuple[str, str]]]:
    errors: list[tuple[str, str]] = []
    with project_context(root):
        loaded = load_project_eval_targets(
            root, on_error=lambda reference, reason: errors.append((reference, reason))
        )
    return loaded, errors


def test_declared_targets_load_with_the_policy_their_plan_declares(tmp_path: Path) -> None:
    root = _project(
        tmp_path,
        "declared_eval_targets:corpus_plan",
        "declared_eval_targets:awaited_plan",
    )

    loaded, errors = _load(root)

    assert errors == []
    assert [item.reference for item in loaded] == [
        "declared_eval_targets:corpus_plan",
        "declared_eval_targets:awaited_plan",
    ]
    assert [item.target.key for item in loaded] == ["refund-agent", "awaited-agent"]
    assert all(type(item.target) is CorpusTarget for item in loaded)
    assert loaded[0].execution_profile_policy.max_trials == 2
    assert loaded[1].execution_profile_policy == EvalExecutionProfilePolicyV1.safe_default()


def test_a_declaration_that_cannot_be_used_is_reported_and_the_others_load(
    tmp_path: Path,
) -> None:
    root = _project(
        tmp_path,
        "declared_eval_targets:broken_plan",
        "declared_eval_targets:corpus_plan",
        "declared_eval_targets:direct_suite_plan",
        "declared_eval_targets:NOT_CALLABLE",
        "declared_eval_targets:same_key_plan",
        "declared_eval_targets:over_its_limits_plan",
        "declared_eval_targets:exiting_plan",
        "missing_module:plan",
    )

    loaded, errors = _load(root)

    assert [item.target.key for item in loaded] == ["refund-agent"]
    reasons = dict(errors)
    assert list(reasons) == [
        "declared_eval_targets:broken_plan",
        "declared_eval_targets:direct_suite_plan",
        "declared_eval_targets:NOT_CALLABLE",
        "declared_eval_targets:same_key_plan",
        "declared_eval_targets:over_its_limits_plan",
        "declared_eval_targets:exiting_plan",
        "missing_module:plan",
    ]
    assert "no provider key in this shell" in reasons["declared_eval_targets:broken_plan"]
    assert "direct suite" in reasons["declared_eval_targets:direct_suite_plan"]
    assert "must be a function" in reasons["declared_eval_targets:NOT_CALLABLE"]
    assert "already has key 'refund-agent'" in reasons["declared_eval_targets:same_key_plan"]
    assert "runtime target authority" in reasons["declared_eval_targets:over_its_limits_plan"]
    assert "missing eval-only credential" in reasons["declared_eval_targets:exiting_plan"]
    assert "missing_module" in reasons["missing_module:plan"]


def test_serve_serves_the_declared_targets_and_warns_about_the_rest(tmp_path: Path, capsys) -> None:
    root = _project(
        tmp_path,
        "declared_eval_targets:corpus_plan",
        "declared_eval_targets:broken_plan",
    )
    args = parse_serve_arguments(["--dev"])
    project, _settings = _resolve_serve_project(args, start=root)

    with project_context(project.root):
        context = serve_project_control_plane(args, project)
    try:
        assert [item.target.key for item in context.eval_targets] == ["refund-agent"]
    finally:
        asyncio.run(context.close())
    assert (
        "warning: eval target declared_eval_targets:broken_plan is not served: "
        "RuntimeError: no provider key in this shell"
    ) in capsys.readouterr().err
