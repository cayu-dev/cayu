from __future__ import annotations

import asyncio

import pytest
from tests.evals.test_workflow_eval_target import (
    _NoChildWorkflow,
    _register_app,
    _suite,
    _target,
    _TwoChildWorkflow,
)

from cayu import (
    FinalOutputContains,
    ModelStreamEvent,
    SessionTrajectoryBounds,
    capture_workflow_eval_attempt,
    run_workflow_eval_suite,
    score_workflow_eval_capture,
)
from cayu.evals.corpus import FinalOutputEqualsAssertionSpec
from cayu.evals.execution import _copy_corpus_target
from cayu.evals.models import EvalStatus
from cayu.evals.trajectory import SessionTrajectoryError
from cayu.evals.workflow_target import WorkflowEvalResult


def _setup(bounds=None, *, chunks=1):
    bounds = bounds or SessionTrajectoryBounds()
    app = _register_app(
        [
            [
                *[ModelStreamEvent.text_delta("x") for _ in range(chunks)],
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
            for _ in range(2)
        ]
    )
    target = _target(app, _TwoChildWorkflow).model_copy(update={"capture_bounds": bounds})
    return app, target, _suite(FinalOutputContains("x"))


def test_capture_policy_changes_identity_and_survives_target_copy():
    _, target, _ = _setup()
    larger = target.model_copy(
        update={"capture_bounds": SessionTrajectoryBounds(max_events=50_000)}
    )
    assert target.identity().revision != larger.identity().revision
    assert _copy_corpus_target(larger).identity() == larger.identity()
    assert larger.identity().capture_bounds.max_events == 50_000
    for invalid in (0, -1, True, 100_001):
        with pytest.raises(ValueError):
            SessionTrajectoryBounds(max_events=invalid)


def test_aggregate_boundary_and_payload_free_diagnostic():
    async def exercise():
        _, target, suite = _setup()
        result = await run_workflow_eval_suite(target, suite, retain_trajectory=True)
        trial = result.cases[0].trials[0]
        count = sum(len(child.events) for child in trial.trajectory.children)
        assert all(len(child.events) < count - 1 for child in trial.trajectory.children)
        for bound, expected in (
            (count, EvalStatus.PASSED),
            (count - 1, EvalStatus.UNAVAILABLE),
            (count + 1, EvalStatus.PASSED),
        ):
            _, configured, suite = _setup(SessionTrajectoryBounds(max_events=bound))
            run = await run_workflow_eval_suite(configured, suite)
            trial = run.cases[0].trials[0]
            assert trial.status is expected, trial.error
            assert trial.execution_status == "completed"
            assert trial.capture_bounds.max_events == bound
            if expected is EvalStatus.UNAVAILABLE:
                assert trial.score is None
                assert not trial.evidence_complete
                diagnostic = trial.capture_diagnostic
                assert diagnostic.terminal_code == "event_limit_exceeded"
                assert diagnostic.observed_lower_bound == diagnostic.limit + 1
                assert diagnostic.consumed_events + diagnostic.limit == bound
                assert trial.workflow_attempt is not None
                assert "text_delta" not in diagnostic.model_dump_json()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "limit",
    [
        {"max_total_bytes": 100},
        {"max_record_bytes": 100},
        {"max_transcript_records": 0},
        {"max_depth": 1},
        {"max_sessions": 1},
    ],
)
def test_raising_events_does_not_disable_other_limits(limit):
    async def exercise():
        _, target, suite = _setup(SessionTrajectoryBounds(max_events=50_000, **limit))
        run = await run_workflow_eval_suite(target, suite)
        trial = run.cases[0].trials[0]
        assert trial.execution_status == "completed"
        assert trial.capture_diagnostic is not None
        assert trial.score is None

    asyncio.run(exercise())


def test_recapture_and_score_do_not_dispatch_and_preserve_source(monkeypatch):
    async def exercise():
        app, target, suite = _setup(SessionTrajectoryBounds(max_events=1))
        run = await run_workflow_eval_suite(target, suite)
        source = run.cases[0].trials[0]
        original = run.model_dump_json()

        def forbidden(*args, **kwargs):
            pytest.fail("Recovery invoked application execution")

        target = target.model_copy(
            update={"workflow_factory": forbidden, "result_projector": forbidden}
        )
        monkeypatch.setattr(app, "run", forbidden)
        capture = await capture_workflow_eval_attempt(
            target,
            source,
            messages=tuple(suite.cases[0].request.messages),
            output=WorkflowEvalResult(final_output="x", structured_output={"answer": "x"}),
            bounds=SessionTrajectoryBounds(max_events=50_000),
        )
        score = await score_workflow_eval_capture(
            target, capture, (FinalOutputEqualsAssertionSpec(id="answer", expected="x"),)
        )
        assert score.score == 1.0
        assert score.model_calls == 0
        assert score.assertion_revisions
        assert score.source_capture_id == capture.capture_id
        assert run.model_dump_json() == original
        assert source.status is EvalStatus.UNAVAILABLE
        # Root identity and child-byte mutation must fail against the sealed capture.
        child = capture.trajectory.children[0].session
        await app.session_store.update_metadata(child.id, {"tampered": True})
        with pytest.raises(SessionTrajectoryError):
            await score_workflow_eval_capture(
                target, capture, (FinalOutputEqualsAssertionSpec(id="answer", expected="x"),)
            )

    asyncio.run(exercise())


@pytest.mark.parametrize("mismatch", ["input", "output", "target", "attempt"])
def test_recovery_identity_mismatch_fails_closed(mismatch):
    async def exercise():
        app = _register_app()
        target = _target(app, _NoChildWorkflow)
        suite = _suite()
        run = await run_workflow_eval_suite(target, suite)
        source = run.cases[0].trials[0]
        messages = tuple(suite.cases[0].request.messages)
        output = WorkflowEvalResult(final_output="done", structured_output={"answer": "done"})
        if mismatch == "input":
            messages = ()
        elif mismatch == "output":
            output = WorkflowEvalResult(final_output="wrong")
        elif mismatch == "target":
            target = target.model_copy(update={"implementation_revision": "sha256:" + "2" * 64})
        else:
            source = source.model_copy(
                update={
                    "workflow_attempt": source.workflow_attempt.model_copy(
                        update={"attempt_id": "wrong"}
                    )
                }
            )
        with pytest.raises(SessionTrajectoryError):
            await capture_workflow_eval_attempt(
                target, source, messages=messages, output=output, bounds=SessionTrajectoryBounds()
            )

    asyncio.run(exercise())


def test_aggregate_default_rejection_recovers_same_paid_attempt():
    from tests.evals.test_session_trajectory import _create_running_session, _finish_session

    from cayu import Event, WorkflowBase, WorkflowSpec

    async def exercise():
        app = _register_app()

        class SyntheticWorkflow(WorkflowBase):
            spec = WorkflowSpec(name="synthetic-large-capture")

            async def run(self, session_id):
                ctx = self.context(session_id)
                yield await ctx.start()
                for child_id in ("synthetic-child-a", "synthetic-child-b"):
                    interaction = await _create_running_session(
                        app.session_store, child_id, parent_session_id=session_id
                    )
                    for index in range(5000):
                        await app.session_store.append_event(
                            child_id,
                            Event(
                                type="custom.synthetic",
                                session_id=child_id,
                                payload={"index": index},
                            ),
                        )
                    await _finish_session(app.session_store, child_id, interaction)
                yield await ctx.completed({"answer": "done"})

        target = _target(app, SyntheticWorkflow)
        suite = _suite(FinalOutputContains("done"))
        run = await run_workflow_eval_suite(target, suite)
        source = run.cases[0].trials[0]
        assert source.execution_status == "completed"
        assert source.capture_diagnostic.terminal_code == "event_limit_exceeded"
        capture = await capture_workflow_eval_attempt(
            target,
            source,
            messages=tuple(suite.cases[0].request.messages),
            output=WorkflowEvalResult(final_output="done", structured_output={"answer": "done"}),
            bounds=SessionTrajectoryBounds(max_events=50_000),
        )
        counts = [len(child.events) for child in capture.trajectory.children]
        assert all(count < 10_000 for count in counts)
        assert sum(counts) > 10_000
        result = await score_workflow_eval_capture(
            target, capture, (FinalOutputEqualsAssertionSpec(id="answer", expected="done"),)
        )
        assert result.score == 1.0

    asyncio.run(exercise())


def test_legacy_import_is_explicit_and_does_not_replace_existing_anchor():
    from cayu import import_workflow_eval_attempt
    from cayu.evals.models import EvalRun

    async def exercise():
        app = _register_app()
        target = _target(app, _NoChildWorkflow)
        suite = _suite()
        run = await run_workflow_eval_suite(target, suite)
        anchor = run.cases[0].trials[0].workflow_attempt
        kwargs = dict(
            case_id=anchor.case_id,
            trial_number=1,
            attempt_id=anchor.attempt_id,
            completion_event_id=anchor.completion_event_id,
            messages=tuple(suite.cases[0].request.messages),
            output=WorkflowEvalResult(final_output="done", structured_output={"answer": "done"}),
        )
        with pytest.raises(ValueError, match="existing attempt anchor"):
            await import_workflow_eval_attempt(target, run, **kwargs)
        document = run.model_dump(mode="json")
        trial = document["cases"][0]["trials"][0]
        for key in ("workflow_attempt", "execution_status", "capture_bounds"):
            trial.pop(key)
        legacy = EvalRun.model_validate(document)
        original = legacy.model_dump_json()
        imported = await import_workflow_eval_attempt(target, legacy, **kwargs)
        assert imported.workflow_attempt.origin == "saved_store_import"
        assert imported.workflow_attempt.source_report_sha256
        assert legacy.model_dump_json() == original
        capture = await capture_workflow_eval_attempt(
            target,
            imported,
            messages=kwargs["messages"],
            output=kwargs["output"],
            bounds=SessionTrajectoryBounds(),
        )
        result = await score_workflow_eval_capture(
            target, capture, (FinalOutputEqualsAssertionSpec(id="answer", expected="done"),)
        )
        assert result.score == 1.0

    asyncio.run(exercise())


def test_corpus_report_keeps_completed_runtime_and_unavailable_scoring():
    from tests.evals.test_workflow_eval_target import _corpus

    from cayu import corpus_execution_result_from_json, corpus_execution_result_to_json
    from cayu.evals.execution import run_corpus_suite

    async def exercise():
        _, target, _ = _setup(SessionTrajectoryBounds(max_events=1))
        result = await run_corpus_suite(
            target,
            _corpus(FinalOutputEqualsAssertionSpec(id="answer", expected="x")),
            "workflow-suite",
        )
        restored = corpus_execution_result_from_json(corpus_execution_result_to_json(result))
        trial = restored.run.cases[0].trials[0]
        assert trial.status == "unavailable"
        assert trial.score is None
        assert trial.code == "workflow_capture_failed"
        assert trial.execution_status == "completed"
        from cayu import render_corpus_execution_html

        assert "event_limit_exceeded" in render_corpus_execution_html(restored)
        assert trial.capture_diagnostic.terminal_code == "event_limit_exceeded"
        assert trial.capture_bounds.max_events == 1
        assert all(assertion.outcome == "unavailable" for assertion in trial.assertions)

    asyncio.run(exercise())


def test_recovery_rejects_deleted_child_and_model_judge(tmp_path, monkeypatch):
    from cayu import SQLiteSessionStore
    from cayu.evals.corpus import ModelJudgeAssertionSpec

    async def exercise():
        store = SQLiteSessionStore(tmp_path / "saved.sqlite3")
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
        try:
            target = _target(app, _TwoChildWorkflow)
            suite = _suite(FinalOutputContains("x"))
            run = await run_workflow_eval_suite(target, suite)
            capture = await capture_workflow_eval_attempt(
                target,
                run.cases[0].trials[0],
                messages=tuple(suite.cases[0].request.messages),
                output=WorkflowEvalResult(final_output="x", structured_output={"answer": "x"}),
                bounds=SessionTrajectoryBounds(),
            )
            with pytest.raises(ValueError, match="model judges"):
                await score_workflow_eval_capture(
                    target,
                    capture,
                    (
                        ModelJudgeAssertionSpec(
                            id="judge", evaluator_key="judge", rubric="correct", rubric_version="v1"
                        ),
                    ),
                )
            await store.delete_session(capture.trajectory.children[0].session.id)
            with pytest.raises(SessionTrajectoryError):
                await score_workflow_eval_capture(
                    target, capture, (FinalOutputEqualsAssertionSpec(id="answer", expected="x"),)
                )
        finally:
            await store.close()

    asyncio.run(exercise())
