from __future__ import annotations

import asyncio

import pytest
from tests.evals.test_workflow_eval_target import _register_app, _suite, _target

from cayu import (
    EvalStatus,
    FinalOutputContains,
    ModelStreamEvent,
    SQLiteSessionStore,
    WorkflowBase,
    WorkflowSpec,
    run_workflow_eval_suite,
    step,
)


class _AfterWorkFailure(WorkflowBase):
    spec = WorkflowSpec(name="after-work-failure")

    async def run(self, session_id):
        ctx = self.context(session_id)
        yield await ctx.start()
        await step(ctx, agent="first", step_id="work", prompt="scripted check")
        raise RuntimeError("private failure text must not be projected")


def _batches():
    return [
        [
            ModelStreamEvent.tool_call(id="call_echo", name="echo", arguments={"text": "fixture"}),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "tool_calls",
                    "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                }
            ),
        ],
        [
            ModelStreamEvent.text_delta("fixture done"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 4, "output_tokens": 1, "total_tokens": 5},
                }
            ),
        ],
    ]


def test_failed_workflow_preserves_known_activity(tmp_path):
    async def scenario():
        store = SQLiteSessionStore(tmp_path / "failure.sqlite")
        try:
            app = _register_app(_batches(), session_store=store)
            result = await run_workflow_eval_suite(
                _target(app, _AfterWorkFailure),
                _suite(FinalOutputContains("fixture")),
            )
            trial = result.cases[0].trials[0]
            assert trial.status is EvalStatus.ERROR
            assert trial.events_count > 0, trial.model_dump_json()
            assert trial.evidence_complete is False
            assert trial.score is None
            assert trial.final_output == ""
            assert trial.usage_summary is not None
        finally:
            await store.close()

    asyncio.run(scenario())


class _BeforeWorkFailure(WorkflowBase):
    spec = WorkflowSpec(name="before-work-failure")

    async def run(self, session_id):
        ctx = self.context(session_id)
        yield await ctx.start()
        raise RuntimeError("private before-work failure")


class _BeforeStartFailure(WorkflowBase):
    spec = WorkflowSpec(name="before-start-failure")

    async def run(self, session_id):
        if False:
            yield
        raise RuntimeError("private before-start failure")


def test_failed_workflow_report_reopens_without_dispatch(tmp_path):
    import hashlib

    from cayu import EventQuery, load_eval_run, render_html_report, write_eval_run_json
    from cayu._validation import canonical_durable_json_bytes

    async def scenario():
        path = tmp_path / "reopen.sqlite"
        store = SQLiteSessionStore(path)
        app = _register_app(_batches(), session_store=store)
        provider = app.get_provider("scripted")
        try:
            result = await run_workflow_eval_suite(
                _target(app, _AfterWorkFailure),
                _suite(FinalOutputContains("fixture")),
            )
            trial = result.cases[0].trials[0]
            assert len(provider.requests) == 2
            assert trial.execution_status == "failed"
            assert trial.failure_evidence.classification == "failure"
            assert trial.failure_capture.model_calls == 2
            assert trial.failure_capture.tool_calls == 1
            assert trial.failure_capture.model_calls_with_usage == 2
            assert trial.usage_summary["usage"]["total_tokens"] == 10
            assert all(assertion.outcome.value == "unavailable" for assertion in trial.assertions)
            report = tmp_path / "report.json"
            write_eval_run_json(result, report)
            original = report.read_bytes()
        finally:
            await store.close()
        reopened = SQLiteSessionStore(path)
        try:
            loaded = load_eval_run(report)
            reloaded = loaded.cases[0].trials[0]
            assert reloaded.failure_capture == trial.failure_capture
            for reference in reloaded.failure_capture.records:
                records = await reopened.query_events(EventQuery(session_id=reference.session_id))
                retained = [
                    record
                    for record in records
                    if reference.first_sequence <= record.sequence <= reference.last_sequence
                ]
                assert len(retained) == reference.record_count
                assert (
                    hashlib.sha256(
                        canonical_durable_json_bytes(
                            [record.model_dump(mode="json") for record in retained],
                            "failure records",
                        )
                    ).hexdigest()
                    == reference.records_sha256
                )
            html = render_html_report(loaded)
            assert "Failed-workflow observations" in html
            assert "Execution failure evidence" in html
            assert "RuntimeError" in html
            assert "Workflow execution: failed" in html
            assert "observed_usage_records" in html
            assert "private failure text" not in html + original.decode()
            assert len(provider.requests) == 2
            assert report.read_bytes() == original
        finally:
            await reopened.close()

    asyncio.run(scenario())


def test_before_work_failure_keeps_usage_unknown():
    async def scenario():
        app = _register_app()
        result = await run_workflow_eval_suite(
            _target(app, _BeforeWorkFailure),
            _suite(FinalOutputContains("unused")),
        )
        trial = result.cases[0].trials[0]
        assert trial.execution_status == "failed"
        assert trial.events_count == 2
        assert trial.failure_capture.state == "partial"
        assert trial.failure_capture.usage_basis == "unavailable"
        assert trial.usage_summary is None
        assert not app.get_provider("scripted").requests

    asyncio.run(scenario())


def test_before_start_failure_has_no_invented_record_identity():
    async def scenario():
        app = _register_app()
        result = await run_workflow_eval_suite(
            _target(app, _BeforeStartFailure),
            _suite(FinalOutputContains("unused")),
        )
        trial = result.cases[0].trials[0]
        assert trial.execution_status == "failed"
        assert trial.events_count == 0
        assert trial.failure_capture.state == "unavailable"
        assert trial.failure_capture.attempt_id is None
        assert trial.usage_summary is None
        assert not app.get_provider("scripted").requests

    asyncio.run(scenario())


def test_capture_event_limit_preserves_root_without_inventing_child_usage(tmp_path):
    from cayu.evals.capture_policy import SessionTrajectoryBounds
    from cayu.runtime.sessions import TerminalSessionEvidenceErrorCode

    async def scenario():
        store = SQLiteSessionStore(tmp_path / "limited.sqlite")
        try:
            app = _register_app(_batches(), session_store=store)
            target = _target(app, _AfterWorkFailure).model_copy(
                update={"capture_bounds": SessionTrajectoryBounds(max_events=5)}
            )
            result = await run_workflow_eval_suite(target, _suite(FinalOutputContains("unused")))
            trial = result.cases[0].trials[0]
            assert trial.status is EvalStatus.ERROR
            assert trial.events_count == 4
            assert trial.failure_capture.state == "partial"
            assert trial.usage_summary is None
            assert (
                trial.failure_capture.diagnostics[0].terminal_code
                == TerminalSessionEvidenceErrorCode.EVENT_LIMIT_EXCEEDED
            )
            assert len(app.get_provider("scripted").requests) == 2
        finally:
            await store.close()

    asyncio.run(scenario())


def test_missing_child_capture_preserves_root_and_execution_failure(tmp_path, monkeypatch):
    import cayu.evals.runner as runner
    from cayu.evals.capture_policy import SessionTrajectoryErrorCode

    async def scenario():
        store = SQLiteSessionStore(tmp_path / "missing.sqlite")
        app = _register_app(_batches(), session_store=store)
        capture = runner.capture_failed_workflow
        inspect = store.inspect_identity

        async def observe(app, **kwargs):
            async def missing_child(session_id, **options):
                if session_id != kwargs["session_id"]:
                    raise KeyError("missing child")
                return await inspect(session_id, **options)

            monkeypatch.setattr(store, "inspect_identity", missing_child)
            return await capture(app, **kwargs)

        monkeypatch.setattr(runner, "capture_failed_workflow", observe)
        try:
            result = await run_workflow_eval_suite(
                _target(app, _AfterWorkFailure), _suite(FinalOutputContains("unused"))
            )
            trial = result.cases[0].trials[0]
            assert trial.status is EvalStatus.ERROR
            assert trial.events_count == 4
            assert (
                trial.failure_capture.diagnostics[0].code
                == SessionTrajectoryErrorCode.EVIDENCE_READ_FAILED
            )
            assert trial.failure_evidence.classification == "failure"
            assert trial.usage_summary is None
        finally:
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "fault", ["wrong_parent", "wrong_event_session", "changed_attempt", "missing_root"]
)
def test_failure_projection_rejects_unrelated_evidence(tmp_path, monkeypatch, fault):
    import cayu.evals.runner as runner

    async def scenario():
        store = SQLiteSessionStore(tmp_path / "unrelated.sqlite")
        app = _register_app(_batches(), session_store=store)
        capture = runner.capture_failed_workflow
        query = store.query_events_bounded
        inspect = store.inspect_identity
        lineage = store.query_session_lineage

        async def observe(app, **kwargs):
            root_id = kwargs["session_id"]
            if fault == "changed_attempt":
                started = kwargs["started"]
                kwargs["started"] = started.model_copy(
                    update={"payload": {**started.payload, "attempt_id": "wrong-attempt"}}
                )
            elif fault == "missing_root":

                async def missing(session_id, **options):
                    if session_id == root_id:
                        raise KeyError("missing root")
                    return await inspect(session_id, **options)

                monkeypatch.setattr(store, "inspect_identity", missing)
            elif fault == "wrong_parent":

                async def wrong(query):
                    result = await lineage(query)
                    if query.parent_session_id == root_id and result.children:
                        return result.model_copy(
                            update={
                                "children": (
                                    result.children[0].model_copy(
                                        update={"parent_session_id": "another-root"}
                                    ),
                                )
                            }
                        )
                    return result

                monkeypatch.setattr(store, "query_session_lineage", wrong)
            else:

                async def wrong_event(request, **options):
                    records = await query(request, **options)
                    if request.session_id != root_id and records:
                        return [
                            records[0].model_copy(
                                update={
                                    "event": records[0].event.model_copy(
                                        update={"session_id": "another-child"}
                                    )
                                }
                            )
                        ]
                    return records

                monkeypatch.setattr(store, "query_events_bounded", wrong_event)
            return await capture(app, **kwargs)

        monkeypatch.setattr(runner, "capture_failed_workflow", observe)
        try:
            result = await run_workflow_eval_suite(
                _target(app, _AfterWorkFailure), _suite(FinalOutputContains("unused"))
            )
            trial = result.cases[0].trials[0]
            assert trial.status is EvalStatus.ERROR
            assert trial.failure_capture.diagnostics
            assert trial.usage_summary is None
            assert trial.score is None
            assert trial.events_count == (0 if fault in {"changed_attempt", "missing_root"} else 4)
            assert len(app.get_provider("scripted").requests) == 2
        finally:
            await store.close()

    asyncio.run(scenario())


def test_failed_workflow_portable_report_preserves_failure_and_usage(tmp_path):
    from tests.evals.test_workflow_eval_target import _corpus

    from cayu import (
        corpus_execution_result_from_json,
        corpus_execution_result_to_json,
        render_corpus_execution_html,
    )
    from cayu.evals.corpus import FinalOutputEqualsAssertionSpec
    from cayu.evals.execution import run_corpus_suite
    from cayu.evals.result_contract import EvalTrialDiagnosticCode

    async def scenario():
        store = SQLiteSessionStore(tmp_path / "portable.sqlite")
        try:
            app = _register_app(_batches(), session_store=store)
            result = await run_corpus_suite(
                _target(app, _AfterWorkFailure),
                _corpus(FinalOutputEqualsAssertionSpec(id="output", expected="unused")),
                "workflow-suite",
                max_concurrency=1,
            )
            trial = result.run.cases[0].trials[0]
            assert trial.code is EvalTrialDiagnosticCode.WORKFLOW_EXECUTION_FAILED
            assert trial.execution_status == "failed"
            assert trial.failure_capture.model_calls == 2
            assert trial.usage.total_tokens == 10
            saved = corpus_execution_result_to_json(result)
            loaded = corpus_execution_result_from_json(saved)
            assert loaded == result
            html = render_corpus_execution_html(loaded)
            assert "Failed-workflow observations" in html
            assert "observed_usage_records" in html
            assert "private failure text" not in saved + html
            assert len(app.get_provider("scripted").requests) == 2
        finally:
            await store.close()

    asyncio.run(scenario())


class _AfterTwoChildrenFailure(WorkflowBase):
    spec = WorkflowSpec(name="two-children-failure")

    async def run(self, session_id):
        ctx = self.context(session_id)
        yield await ctx.start()
        await step(ctx, agent="first", step_id="first", prompt="first scripted check")
        await step(ctx, agent="second", step_id="second", prompt="second scripted check")
        raise RuntimeError("private two-child failure")


def test_capture_limit_retains_known_sibling_usage(tmp_path):
    from cayu.evals.capture_policy import SessionTrajectoryBounds

    async def scenario():
        store = SQLiteSessionStore(tmp_path / "siblings.sqlite")
        try:
            app = _register_app(_batches() + _batches(), session_store=store)
            target = _target(app, _AfterTwoChildrenFailure).model_copy(
                update={"capture_bounds": SessionTrajectoryBounds(max_sessions=2)}
            )
            result = await run_workflow_eval_suite(target, _suite(FinalOutputContains("unused")))
            trial = result.cases[0].trials[0]
            assert len(trial.failure_capture.records) == 2
            assert trial.failure_capture.model_calls == 2
            assert trial.usage_summary["usage"]["total_tokens"] == 10
            assert trial.failure_capture.diagnostics
            assert len(app.get_provider("scripted").requests) == 4
            assert trial.score is None
        finally:
            await store.close()

    asyncio.run(scenario())


def test_work_without_usage_does_not_become_zero_usage():
    async def scenario():
        app = _register_app(
            [
                [
                    ModelStreamEvent.text_delta("fixture"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            ]
        )
        result = await run_workflow_eval_suite(
            _target(app, _AfterWorkFailure), _suite(FinalOutputContains("unused"))
        )
        trial = result.cases[0].trials[0]
        assert trial.failure_capture.model_calls == 1
        assert trial.failure_capture.model_calls_with_usage == 0
        assert trial.events_count > 2
        assert trial.usage_summary is None
        assert trial.failure_capture.usage_basis == "unavailable"

    asyncio.run(scenario())


def build_failure_cli_plan():
    from cayu import EvalPlan

    app = _register_app(_batches())
    return EvalPlan(
        workflow_target=_target(app, _AfterWorkFailure), suite=_suite(FinalOutputContains("unused"))
    )


def test_cli_failed_workflow_writes_inspectable_report(tmp_path, monkeypatch):
    from cayu import load_eval_run
    from cayu.cli import main

    (tmp_path / "pyproject.toml").write_text(
        "[tool.cayu]\n"
        'factory = "tests.evals.test_workflow_eval_target:build_cli_app"\n'
        f'eval_target = "{__name__}:build_failure_cli_plan"\n'
    )
    monkeypatch.chdir(tmp_path)
    assert main(["eval", "run", "--output", "failure.json"]) == 2
    trial = load_eval_run(tmp_path / "failure.json").cases[0].trials[0]
    assert trial.execution_status == "failed"
    assert trial.failure_capture.model_calls == 2
    assert trial.usage_summary["usage"]["total_tokens"] == 10
