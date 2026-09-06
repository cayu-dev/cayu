from __future__ import annotations

import asyncio

import pytest
from tests.evals.test_workflow_eval_target import (
    _register_app,
    _suite,
    _target,
    _TwoChildWorkflow,
)

from cayu import (
    FinalOutputContains,
    ModelStreamEvent,
    SessionTrajectoryBounds,
    SQLiteSessionStore,
    capture_workflow_eval_attempt,
    run_workflow_eval_suite,
    score_workflow_eval_capture,
)
from cayu.evals.corpus import (
    ChildStatusAssertionSpec,
    FinalOutputEqualsAssertionSpec,
    ProcessEventAssertionSpec,
    RootStatusAssertionSpec,
    ToolCalledAssertionSpec,
)
from cayu.evals.incremental_recovery import (
    IncrementalWorkflowCaptureError,
    capture_incremental_workflow_eval_attempt,
    score_incremental_workflow_eval_capture,
)
from cayu.evals.workflow_target import WorkflowEvalResult
from cayu.runtime.evidence_spool import (
    IncrementalEvidenceAdmission,
    IncrementalEvidenceError,
    IncrementalEvidenceLimits,
)


async def _setup(store):
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
    target = _target(app, _TwoChildWorkflow).model_copy(
        update={
            "capture_bounds": SessionTrajectoryBounds(max_events=1),
        }
    )
    suite = _suite(FinalOutputContains("x"))
    run = await run_workflow_eval_suite(target, suite)
    source = run.cases[0].trials[0]
    assert source.execution_status == "completed", source.error
    assert not source.evidence_complete
    return (
        app,
        target,
        run,
        source,
        tuple(suite.cases[0].request.messages),
        WorkflowEvalResult(
            final_output="x",
            structured_output={"answer": "x"},
        ),
    )


def test_incremental_saved_capture_and_scoring_match_eager(tmp_path, monkeypatch):
    async def exercise():
        store = SQLiteSessionStore(tmp_path / "source.sqlite3")
        try:
            app, target, run, source, messages, output = await _setup(store)
            original = run.model_dump_json()

            def forbidden(*args, **kwargs):
                pytest.fail("Saved recovery dispatched application execution")

            target = target.model_copy(
                update={"workflow_factory": forbidden, "result_projector": forbidden}
            )
            monkeypatch.setattr(app, "run", forbidden)
            specs = (
                FinalOutputEqualsAssertionSpec(id="answer", expected="x"),
                RootStatusAssertionSpec(id="root", expected="completed"),
                ChildStatusAssertionSpec(
                    id="children", expected="completed", min_count=2, max_count=2
                ),
                ProcessEventAssertionSpec(
                    id="events", event="session_completed", min_count=2, max_count=2
                ),
            )
            admission = IncrementalEvidenceAdmission()
            capture = await capture_incremental_workflow_eval_attempt(
                target,
                source,
                messages=messages,
                output=output,
                limits=IncrementalEvidenceLimits(batch_records=1),
                admission=admission,
            )
            assert admission.active_captures == 0
            assert capture.retained_evidence == "summary_only"
            assert len(capture.sessions) == 2
            assert capture.progress.processed_events == 2 * sum(
                seal.event_count for seal in capture.sessions
            )
            score = await score_incremental_workflow_eval_capture(
                target,
                source,
                capture,
                specs,
                messages=messages,
                output=output,
                admission=admission,
            )
            eager = await capture_workflow_eval_attempt(
                target,
                source,
                messages=messages,
                output=output,
                bounds=SessionTrajectoryBounds(),
            )
            eager_score = await score_workflow_eval_capture(target, eager, specs)
            assert score.score == eager_score.score == 1.0
            assert [result.model_dump() for result in score.assertions] == [
                result.model_dump() for result in eager_score.assertions
            ]
            assert score.model_calls == 0
            assert run.model_dump_json() == original
            await store.update_metadata(capture.sessions[0].session_id, {"changed": True})
            with pytest.raises(IncrementalWorkflowCaptureError, match="evidence_changed"):
                await score_incremental_workflow_eval_capture(
                    target,
                    source,
                    capture,
                    specs,
                    messages=messages,
                    output=output,
                    admission=admission,
                )
            assert admission.active_captures == 0
        finally:
            await store.close()

    asyncio.run(exercise())


def test_unsupported_assertion_rejected_before_capture(tmp_path, monkeypatch):
    async def exercise():
        store = SQLiteSessionStore(tmp_path / "source.sqlite3")
        try:
            _, target, _, source, messages, output = await _setup(store)
            admission = IncrementalEvidenceAdmission()
            capture = await capture_incremental_workflow_eval_attempt(
                target,
                source,
                messages=messages,
                output=output,
                limits=IncrementalEvidenceLimits(),
                admission=admission,
            )

            async def forbidden(*args, **kwargs):
                pytest.fail("Unsupported assertion caused store I/O")

            monkeypatch.setattr(store, "load_bounded", forbidden)
            with pytest.raises(
                IncrementalEvidenceError, match="assertion_requires_unsupported_evidence"
            ):
                await score_incremental_workflow_eval_capture(
                    target,
                    source,
                    capture,
                    (ToolCalledAssertionSpec(id="tool", tool_name="echo"),),
                    messages=messages,
                    output=output,
                    admission=admission,
                )
        finally:
            await store.close()

    asyncio.run(exercise())


def test_one_hundred_capture_arrivals_share_aggregate_admission(tmp_path):
    async def exercise():
        store = SQLiteSessionStore(tmp_path / "source.sqlite3")
        try:
            _, target, _, source, messages, output = await _setup(store)
            admission = IncrementalEvidenceAdmission(max_captures=4)

            async def capture():
                try:
                    return await capture_incremental_workflow_eval_attempt(
                        target,
                        source,
                        messages=messages,
                        output=output,
                        limits=IncrementalEvidenceLimits(),
                        admission=admission,
                    )
                except IncrementalWorkflowCaptureError as exc:
                    assert exc.code == "aggregate_admission_rejected"
                    return None

            results = await asyncio.gather(*(capture() for _ in range(100)))
            assert sum(result is not None for result in results) == 4
            assert admission.active_captures == 0
            assert admission.reserved_buffer_bytes == 0
            assert admission.reserved_spill_bytes == 0
        finally:
            await store.close()

    asyncio.run(exercise())


def test_incremental_timeout_uses_retained_progress_and_releases_admission(tmp_path, monkeypatch):
    async def exercise():
        store = SQLiteSessionStore(tmp_path / "source.sqlite3")
        try:
            _, target, _, source, messages, output = await _setup(store)
            admission = IncrementalEvidenceAdmission()
            paths = []
            expired = False
            original_read = store.load_bounded

            async def guarded_read(*args, **kwargs):
                assert not expired, "Source read after deadline"
                return await original_read(*args, **kwargs)

            async def blocked_export(session_id, *, spool):
                nonlocal expired
                paths.append(spool.path)
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    expired = True
                    raise

            monkeypatch.setattr(store, "load_bounded", guarded_read)
            monkeypatch.setattr(store, "export_terminal_session_evidence", blocked_export)
            with pytest.raises(IncrementalWorkflowCaptureError) as caught:
                await capture_incremental_workflow_eval_attempt(
                    target,
                    source,
                    messages=messages,
                    output=output,
                    limits=IncrementalEvidenceLimits(max_seconds=1),
                    admission=admission,
                )
            assert caught.value.code == "deadline_exceeded"
            assert caught.value.progress.stage == "snapshot_copy"
            assert paths and all(not path.exists() for path in paths)
            assert admission.active_captures == admission.reserved_spill_bytes == 0
        finally:
            await store.close()

    asyncio.run(exercise())


def test_exact_aggregate_and_root_bounds(tmp_path):
    from cayu._validation import compact_json_utf8_size
    from cayu.evals.workflow_recovery import _read_workflow_attempt_root

    async def exercise():
        store = SQLiteSessionStore(tmp_path / "source.sqlite3")
        try:
            _, target, _, source, messages, output = await _setup(store)
            admission = IncrementalEvidenceAdmission()
            initial = await capture_incremental_workflow_eval_attempt(
                target,
                source,
                messages=messages,
                output=output,
                limits=IncrementalEvidenceLimits(),
                admission=admission,
            )
            root, records = await _read_workflow_attempt_root(target, source.workflow_attempt)
            root_bytes = compact_json_utf8_size(root.model_dump(mode="json")) + sum(
                compact_json_utf8_size(record.model_dump(mode="json")) for record in records
            )
            exact = IncrementalEvidenceLimits(
                max_events=sum(seal.event_count for seal in initial.sessions),
                max_transcript_records=sum(seal.transcript_count for seal in initial.sessions),
                max_total_bytes=sum(seal.evidence_bytes for seal in initial.sessions),
                max_root_bytes=root_bytes,
            )
            recaptured = await capture_incremental_workflow_eval_attempt(
                target,
                source,
                messages=messages,
                output=output,
                limits=exact,
                admission=admission,
                expected_evidence_sha256=initial.evidence_sha256,
            )
            assert recaptured.evidence_sha256 == initial.evidence_sha256
            assert recaptured.capture_policy_revision != initial.capture_policy_revision
            for field in (
                "max_events",
                "max_transcript_records",
                "max_total_bytes",
                "max_root_bytes",
            ):
                with pytest.raises(IncrementalWorkflowCaptureError):
                    await capture_incremental_workflow_eval_attempt(
                        target,
                        source,
                        messages=messages,
                        output=output,
                        limits=exact.model_copy(update={field: getattr(exact, field) - 1}),
                        admission=admission,
                    )
        finally:
            await store.close()

    asyncio.run(exercise())


def test_root_read_cancellation_retains_admission_until_worker_settles(tmp_path, monkeypatch):
    import threading

    import cayu.storage.sqlite as sqlite_adapter

    async def exercise():
        store = SQLiteSessionStore(tmp_path / "source.sqlite3")
        released = threading.Event()
        entered = threading.Event()
        try:
            _, target, _, source, messages, output = await _setup(store)
            admission = IncrementalEvidenceAdmission()
            original_load = sqlite_adapter._load_session

            def blocked(connection, session_id):
                entered.set()
                assert released.wait(5)
                return original_load(connection, session_id)

            monkeypatch.setattr(sqlite_adapter, "_load_session", blocked)
            task = asyncio.create_task(
                capture_incremental_workflow_eval_attempt(
                    target,
                    source,
                    messages=messages,
                    output=output,
                    limits=IncrementalEvidenceLimits(),
                    admission=admission,
                )
            )
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0.02)
            assert not task.done()
            assert admission.active_captures == 1
            released.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert admission.active_captures == admission.reserved_buffer_bytes == 0
            assert not store._detached_read_tasks
        finally:
            released.set()
            await store.close()

    asyncio.run(exercise())


@pytest.mark.stress
def test_large_workflow_saved_scoring_without_provider_dispatch(tmp_path, monkeypatch):
    from benchmarks.incremental_evidence import insert_sqlite_records
    from tests.evals.test_session_trajectory import _create_running_session, _finish_session

    from cayu import WorkflowBase, WorkflowSpec

    async def exercise():
        store = SQLiteSessionStore(tmp_path / "source.sqlite3")

        class SyntheticWorkflow(WorkflowBase):
            spec = WorkflowSpec(name="large-saved-evidence")

            async def run(self, session_id):
                ctx = self.context(session_id)
                yield await ctx.start()
                interaction = await _create_running_session(
                    store, "benchmark", parent_session_id=session_id
                )
                await insert_sqlite_records(store, 125_000, 8)
                await _finish_session(store, "benchmark", interaction)
                yield await ctx.completed({"answer": "done"})

        try:
            app = _register_app(session_store=store)
            target = _target(app, SyntheticWorkflow).model_copy(
                update={
                    "capture_bounds": SessionTrajectoryBounds(max_events=100_000),
                }
            )
            suite = _suite(FinalOutputContains("done"))
            run = await run_workflow_eval_suite(target, suite)
            source = run.cases[0].trials[0]
            assert source.execution_status == "completed", source.error
            assert source.capture_diagnostic.terminal_code == "event_limit_exceeded"

            def forbidden(*args, **kwargs):
                pytest.fail("Saved capture repeated execution")

            target = target.model_copy(
                update={"workflow_factory": forbidden, "result_projector": forbidden}
            )
            monkeypatch.setattr(app, "run", forbidden)
            admission = IncrementalEvidenceAdmission()
            messages = tuple(suite.cases[0].request.messages)
            output = WorkflowEvalResult(final_output="done", structured_output={"answer": "done"})
            capture = await capture_incremental_workflow_eval_attempt(
                target,
                source,
                messages=messages,
                output=output,
                limits=IncrementalEvidenceLimits(
                    max_seconds=600, batch_records=64, max_record_bytes=65_536
                ),
                admission=admission,
            )
            assert capture.sessions[0].event_count > 125_000
            score = await score_incremental_workflow_eval_capture(
                target,
                source,
                capture,
                (
                    FinalOutputEqualsAssertionSpec(id="answer", expected="done"),
                    ProcessEventAssertionSpec(
                        id="terminal", event="session_completed", min_count=1, max_count=1
                    ),
                ),
                messages=messages,
                output=output,
                admission=admission,
            )
            assert score.score == 1.0
            assert score.model_calls == 0
            assert admission.active_captures == 0
        finally:
            await store.close()

    asyncio.run(exercise())
