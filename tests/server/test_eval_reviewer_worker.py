"""The eval worker behind `cayu cloud eval run` reviews a declared target's trials with its reviewer."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from tests.evals.test_eval_reviewer import (
    ANSWER,
    SubmitAnswerTool,
    answer_corpus,
    answer_target,
    reviewed_steps,
    reviewer,
)
from tests.server.test_worker_eval_runs import _PROJECT_ROOT, _runtime, _status

from cayu.evals.execution_profiles import EvalExecutionProfilePolicyV1
from cayu.evals.store import (
    EvalRunInvocation,
    EvalRunRequest,
    EvalRunStatus,
    EvalScenarioRunInvocation,
)
from cayu.project_control_plane import ProjectEvalTarget
from cayu.server.evals_registry import (
    declared_eval_target_registrations,
    target_for_eval_invocation,
)
from cayu.server.evals_worker import EvalRunCoordinator
from cayu.storage.evals_sqlite import SQLiteEvalStore


def _declared() -> tuple[ProjectEvalTarget, SubmitAnswerTool]:
    target, tool = answer_target(reviewed_steps(ANSWER), eval_reviewer=reviewer())
    declared = ProjectEvalTarget(
        reference="evals:reviewed_analyst",
        target=target,
        execution_profile_policy=EvalExecutionProfilePolicyV1.safe_default(),
    )
    return declared, tool


async def _admit(runtime, store, declared: ProjectEvalTarget, run_id: str) -> str:
    target = declared.target
    corpus = answer_corpus(trials=1, target_key=target.key)
    registration = runtime.registry.registration(target.key)
    invocation = EvalRunInvocation()
    prepared = await runtime.registry.prepare_execution_profile(
        target.key,
        effective_target=target_for_eval_invocation(registration.execution_target(), invocation),
    )
    await store.save_corpus(corpus, redact_json=target.app.redact_json)
    await store.admit_run(
        EvalRunRequest(
            run_id=run_id,
            idempotency_key="sha256:" + hashlib.sha256(run_id.encode()).hexdigest(),
            corpus_revision=corpus.revision,
            target_key=target.key,
            suite_id=corpus.suites[0].id,
            suite_revision=corpus.suites[0].revision,
            max_concurrency=1,
            invocation=invocation.model_copy(
                update={
                    "execution_profile": prepared.binding,
                    "execution_profile_snapshot": prepared.snapshot,
                },
                deep=True,
            ),
        ),
        redact_json=target.app.redact_json,
    )
    return run_id


def test_the_eval_worker_reviews_a_declared_targets_trials(tmp_path: Path) -> None:
    async def run() -> None:
        store = SQLiteEvalStore(tmp_path / "evals.db")
        declared, tool = _declared()
        runtime = _runtime(store, declared)
        try:
            run_id = await _admit(runtime, store, declared, "reviewed-run")

            assert await EvalRunCoordinator(runtime).run_until_idle() == (1, True)

            assert await _status(store, run_id) is EvalRunStatus.COMPLETED
            result = await store.load_result(run_id)
            assert result is not None
            assert result.target.eval_reviewer == reviewer().identity()
            (trial,) = result.run.cases[0].trials
            assert trial.status == "passed"
            assert trial.execution_failure_category is None
            assert tool.answers == [ANSWER]
        finally:
            await store.close()

    asyncio.run(run())


def test_a_declared_targets_reviewer_reaches_its_runs_but_never_a_scenario() -> None:
    declared, _ = _declared()
    (registration,) = declared_eval_target_registrations((declared,), project_root=_PROJECT_ROOT)

    execution = target_for_eval_invocation(registration.execution_target(), EvalRunInvocation())

    assert execution.eval_reviewer == reviewer()
    scenario = EvalRunInvocation(
        scenario=EvalScenarioRunInvocation(
            scenario_revision="sha256:" + "1" * 64,
            binding_revision="sha256:" + "2" * 64,
            trials=1,
            timeout_seconds=30,
        )
    )
    with pytest.raises(ValueError, match="scenario decides each approval"):
        target_for_eval_invocation(registration.execution_target(), scenario)
