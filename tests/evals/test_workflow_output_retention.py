"""Private report retention survives capture failure and process replacement."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.evals.test_workflow_capture_recovery import _setup
from tests.evals.test_workflow_eval_target import _register_app, _suite, _target, _TwoChildWorkflow

from cayu import (
    FinalOutputContains,
    ModelStreamEvent,
    SessionTrajectoryBounds,
    SQLiteSessionStore,
    capture_workflow_eval_attempt,
    run_workflow_eval_suite,
    score_workflow_eval_capture,
)
from cayu.evals.corpus import FinalOutputEqualsAssertionSpec
from cayu.evals.incremental_recovery import (
    capture_incremental_workflow_eval_attempt,
    score_incremental_workflow_eval_capture,
)
from cayu.evals.models import EvalRun
from cayu.evals.trajectory import SessionTrajectoryError
from cayu.evals.workflow_target import RetainedWorkflowEvalOutput, WorkflowEvalResult
from cayu.runtime.evidence_spool import IncrementalEvidenceAdmission, IncrementalEvidenceLimits


def _forbidden(*args, **kwargs):
    raise AssertionError("Recovery dispatched a callback")


def _project(evidence):
    return WorkflowEvalResult(
        final_output="transformed:" + evidence.completion_event.payload["answer"],
        structured_output={
            "attempt": evidence.attempt_id,
            "completion": evidence.completion_event.id,
            "private": ["秘密", {"answer": "transformed:x"}],
        },
    )


async def _fresh_recover(directory):
    directory = Path(directory)
    store = SQLiteSessionStore(directory / "source.sqlite3")
    try:
        app = _register_app(session_store=store)
        target = _target(app, _TwoChildWorkflow, factory=_forbidden, projector=_forbidden)
        target = target.model_copy(update={"capture_bounds": SessionTrajectoryBounds(max_events=1)})
        app.run = _forbidden
        run = EvalRun.model_validate_json((directory / "report.json").read_text())
        original = run.model_dump_json()
        trial = run.cases[0].trials[0]
        retained = trial.retained_workflow_output
        assert retained.output.structured_output["attempt"] == trial.workflow_attempt.attempt_id
        messages = tuple(_suite().cases[0].request.messages)
        capture = await capture_workflow_eval_attempt(
            target, trial, messages=messages, bounds=SessionTrajectoryBounds()
        )
        assert capture.trajectory.final_output == "transformed:x"
        assert (
            capture.trajectory.workflow_output.structured_output
            == retained.output.structured_output
        )
        specs = (FinalOutputEqualsAssertionSpec(id="answer", expected="transformed:x"),)
        score = await score_workflow_eval_capture(target, capture, specs)
        assert score.score == 1.0 and score.model_calls == 0
        admission = IncrementalEvidenceAdmission()
        incremental = await capture_incremental_workflow_eval_attempt(
            target,
            trial,
            messages=messages,
            limits=IncrementalEvidenceLimits(batch_records=1),
            admission=admission,
        )
        score = await score_incremental_workflow_eval_capture(
            target,
            trial,
            incremental,
            specs,
            messages=messages,
            admission=admission,
        )
        assert score.score == 1.0 and score.model_calls == 0
        assert admission.active_captures == 0
        assert run.model_dump_json() == original
        assert trial.score is None and not trial.evidence_complete
    finally:
        await store.close()


@pytest.mark.parametrize("failure", ["limit", "timeout"])
def test_retained_projection_recovers_in_fresh_process(tmp_path, monkeypatch, failure):
    import cayu.evals.runner as runner

    entered = False

    async def blocked_capture(*args, **kwargs):
        nonlocal entered
        entered = True
        await asyncio.Event().wait()

    if failure == "timeout":
        monkeypatch.setattr(runner, "_build_child_trajectories", blocked_capture)

    async def create():
        store = SQLiteSessionStore(tmp_path / "source.sqlite3")
        try:
            app = _register_app(
                [
                    [
                        ModelStreamEvent.text_delta("x"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ]
                    for _ in range(2)
                ],
                session_store=store,
            )
            target = _target(app, _TwoChildWorkflow, projector=_project).model_copy(
                update={"capture_bounds": SessionTrajectoryBounds(max_events=1)}
            )
            run = await run_workflow_eval_suite(
                target,
                _suite(FinalOutputContains("x")),
                case_timeout_seconds=10 if failure == "timeout" else None,
            )
            trial = run.cases[0].trials[0]
            if failure == "timeout":
                assert entered and trial.capture_diagnostic.code == "deadline_exceeded"
            else:
                assert trial.status == "unavailable"
            assert trial.final_output == "" and trial.structured_output is None
            assert trial.workflow_output_retention == "retained"
            (tmp_path / "report.json").write_text(run.model_dump_json())
        finally:
            await store.close()

    asyncio.run(create())
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(Path.cwd() / "src"), str(Path.cwd()))))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import asyncio,sys; from tests.evals.test_workflow_output_retention import _fresh_recover; asyncio.run(_fresh_recover(sys.argv[1]))",
            str(tmp_path),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # No independent output artifact exists: removal of the private report removes retention.
    (tmp_path / "report.json").unlink()
    assert not (tmp_path / "report.json").exists()


@pytest.mark.parametrize(
    "change",
    [
        "output",
        "structured",
        "target_revision",
        "projector_revision",
        "session_id",
        "attempt_id",
        "completion_event_id",
        "root_sha256",
    ],
)
def test_retained_identity_changes_fail_closed(change):
    async def exercise():
        _, target, suite = _setup(SessionTrajectoryBounds(max_events=1))
        run = await run_workflow_eval_suite(target, suite)
        source = run.cases[0].trials[0]
        retained = source.retained_workflow_output
        if change in ("output", "structured"):
            output = retained.output.model_copy(
                update={
                    "final_output" if change == "output" else "structured_output": "wrong"
                    if change == "output"
                    else {"answer": "wrong"}
                }
            )
            retained = retained.model_copy(update={"output": output})
        else:
            retained = retained.model_copy(
                update={"anchor": retained.anchor.model_copy(update={change: "wrong"})}
            )
        source = source.model_copy(update={"retained_workflow_output": retained})
        with pytest.raises(SessionTrajectoryError):
            await capture_workflow_eval_attempt(
                target,
                source,
                messages=tuple(suite.cases[0].request.messages),
                bounds=SessionTrajectoryBounds(),
            )

    asyncio.run(exercise())


@pytest.mark.parametrize("mode", ["disabled", "missing"])
def test_missing_retention_is_actionable(mode):
    async def exercise():
        _, target, suite = _setup(SessionTrajectoryBounds(max_events=1))
        run = await run_workflow_eval_suite(target, suite, retain_final_output=mode != "disabled")
        source = run.cases[0].trials[0]
        if mode == "missing":
            source = source.model_copy(
                update={"retained_workflow_output": None, "workflow_output_retention": None}
            )
        assert source.retained_workflow_output is None
        with pytest.raises(ValueError, match="trusted original projected"):
            await capture_workflow_eval_attempt(
                target,
                source,
                messages=tuple(suite.cases[0].request.messages),
                bounds=SessionTrajectoryBounds(),
            )

    asyncio.run(exercise())


def test_retained_record_byte_limit():
    async def exercise():
        _, target, suite = _setup(SessionTrajectoryBounds(max_events=1))
        run = await run_workflow_eval_suite(target, suite)
        retained = run.cases[0].trials[0].retained_workflow_output
        with pytest.raises(ValueError, match="1 MiB"):
            RetainedWorkflowEvalOutput(
                anchor=retained.anchor.model_copy(update={"case_id": "x" * (1 << 20)}),
                output=retained.output,
            )

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["storage", "cancel"])
def test_post_projection_failure_owns_no_separate_storage(failure, monkeypatch):
    import cayu.evals.runner as runner

    async def exercise():
        _, target, suite = _setup()
        entered = False

        async def fail_capture(*args, **kwargs):
            nonlocal entered
            entered = True
            if failure == "cancel":
                raise asyncio.CancelledError
            raise OSError("private storage error")

        monkeypatch.setattr(runner, "_build_child_trajectories", fail_capture)
        if failure == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await run_workflow_eval_suite(target, suite)
            assert entered
            return
        run = await run_workflow_eval_suite(target, suite)
        trial = run.cases[0].trials[0]
        assert entered and trial.workflow_output_retention == "retained"
        assert trial.score is None and not trial.evidence_complete
        original = run.model_dump_json()
        monkeypatch.undo()
        capture = await capture_workflow_eval_attempt(
            target,
            trial,
            messages=tuple(suite.cases[0].request.messages),
            bounds=SessionTrajectoryBounds(),
        )
        assert capture.trajectory.final_output == "x"
        assert run.model_dump_json() == original

    asyncio.run(exercise())
