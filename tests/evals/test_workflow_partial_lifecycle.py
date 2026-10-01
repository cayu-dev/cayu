from __future__ import annotations

import asyncio

import pytest
from tests.evals.test_session_trajectory import _create_running_session, _finish_session
from tests.evals.test_workflow_eval_target import _register_app, _suite, _target

from cayu import (
    Event,
    FinalOutputContains,
    InMemorySessionStore,
    SessionStatus,
    SessionTrajectoryBounds,
    SQLiteSessionStore,
    WorkflowBase,
    WorkflowSpec,
    run_workflow_eval_suite,
)
from cayu.evals.workflow_target import WorkflowEvalExecution


def _lifecycle_target(store, *, bounds, unfinished=False, nested=False, close=None):
    app = _register_app(session_store=store)
    children = []

    class Workflow(WorkflowBase):
        spec = WorkflowSpec(name="partial-capture-lifecycle")

        async def run(self, session_id):
            ctx = self.context(session_id)
            yield await ctx.start()
            first = f"{session_id}-large"
            second = f"{session_id}-second"
            children.extend((first, second))
            interaction = await _create_running_session(store, first, parent_session_id=session_id)
            await store.append_event(
                first,
                Event(type="custom.page", session_id=first, payload={"text": "x" * 8192}),
            )
            second_interaction = await _create_running_session(
                store, second, parent_session_id=first if nested else session_id
            )
            if not unfinished:
                await _finish_session(store, second, second_interaction)
            await _finish_session(store, first, interaction)
            yield await ctx.completed({"answer": "done"})

    async def close_execution():
        if close is not None:
            await close(children)

    target = _target(
        app,
        Workflow,
        factory=lambda invocation: WorkflowEvalExecution(
            app=app, workflow=Workflow(app), close=close_execution
        ),
    ).model_copy(update={"capture_bounds": bounds})
    return target, children


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "bounds",
    [
        {"max_record_bytes": 4096},
        {"max_events": 1},
        {"max_sessions": 1},
        {"max_depth": 1},
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_partial_capture_cannot_hide_running_descendants(tmp_path, backend, bounds, nested):
    async def exercise():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "lifecycle.sqlite")
        )
        try:
            target, children = _lifecycle_target(
                store, bounds=SessionTrajectoryBounds(**bounds), unfinished=True, nested=nested
            )
            run = await run_workflow_eval_suite(target, _suite(FinalOutputContains("done")))
            trial = run.cases[0].trials[0]
            child = await store.load_state(children[1])
            assert child is not None and child.status is SessionStatus.RUNNING
            assert trial.status == "unavailable"
            assert trial.score is None
            assert trial.capture_diagnostic is not None
            assert trial.capture_diagnostic.terminal_code == "session_not_terminal"
            assert trial.capture_diagnostic.session_id == children[1]
            assert all(assertion.outcome == "unavailable" for assertion in trial.assertions)
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("phase", ["assertion", "close"])
@pytest.mark.parametrize("change", ["resume", "metadata", "new_descendant"])
def test_partial_capture_revalidates_child_lifecycle_before_publication(phase, change):
    async def exercise():
        store = InMemorySessionStore()

        async def mutate(children):
            if change == "resume":
                await store.update_status(children[1], SessionStatus.RUNNING)
            elif change == "metadata":
                await store.update_metadata(children[1], {"changed": True})
            else:
                grandchild = f"{children[0]}-new"
                interaction = await _create_running_session(
                    store, grandchild, parent_session_id=children[0]
                )
                await _finish_session(store, grandchild, interaction)

        target, children = _lifecycle_target(
            store,
            bounds=SessionTrajectoryBounds(max_record_bytes=4096),
            close=mutate if phase == "close" else None,
        )

        class ConcurrentChange(FinalOutputContains):
            async def evaluate(self, context):
                if phase == "assertion":
                    await mutate(children)
                return await super().evaluate(context)

        run = await run_workflow_eval_suite(target, _suite(ConcurrentChange("done")))
        trial = run.cases[0].trials[0]
        assert trial.status == ("unavailable" if phase == "assertion" else "error")
        assert trial.score is None
        assert not trial.evidence_complete
        assert trial.trajectory is None

    asyncio.run(exercise())


@pytest.mark.parametrize("unfinished", [False, True])
def test_partial_capture_checks_children_with_unknown_origin(monkeypatch, unfinished):
    async def exercise():
        store = InMemorySessionStore()
        target, children = _lifecycle_target(
            store, bounds=SessionTrajectoryBounds(), unfinished=unfinished
        )
        query = store.query_session_lineage

        async def unknown_origin(request):
            result = await query(request)
            return result.model_copy(
                update={
                    "children": tuple(
                        node.model_copy(update={"origin_events": ()})
                        if node.id == children[0]
                        else node
                        for node in result.children
                    )
                }
            )

        monkeypatch.setattr(store, "query_session_lineage", unknown_origin)
        run = await run_workflow_eval_suite(target, _suite(FinalOutputContains("done")))
        trial = run.cases[0].trials[0]
        assert trial.capture_diagnostic is not None
        if unfinished:
            assert trial.status == "unavailable"
            assert trial.capture_diagnostic.terminal_code == "session_not_terminal"
        else:
            assert trial.status == "passed" and trial.score == 1.0
            assert trial.capture_diagnostic.code == "origin_evidence_rejected"

    asyncio.run(exercise())


def test_partial_capture_without_assertions_returns_an_unavailable_trial():
    async def exercise():
        target, _ = _lifecycle_target(
            InMemorySessionStore(), bounds=SessionTrajectoryBounds(max_events=1)
        )
        run = await run_workflow_eval_suite(target, _suite())
        trial = run.cases[0].trials[0]
        assert trial.status == "unavailable"
        assert trial.score is None
        assert trial.assertions == ()
        assert not trial.evidence_complete
        assert trial.retained_workflow_output is not None
        assert trial.final_output == trial.retained_workflow_output.output.final_output == "done"

    asyncio.run(exercise())


def test_partial_capture_lifecycle_reads_fail_closed(monkeypatch):
    async def exercise():
        store = InMemorySessionStore()
        target, _ = _lifecycle_target(store, bounds=SessionTrajectoryBounds(max_record_bytes=4096))

        async def unreadable(session_id):
            raise RuntimeError("private lifecycle read failure")

        monkeypatch.setattr(store, "inspect_identity", unreadable)
        run = await run_workflow_eval_suite(target, _suite(FinalOutputContains("done")))
        trial = run.cases[0].trials[0]
        assert trial.status == "unavailable"
        assert trial.capture_diagnostic is not None
        assert trial.capture_diagnostic.code == "evidence_read_failed"
        assert trial.unavailable_reason is not None
        assert "private lifecycle" not in trial.unavailable_reason

    asyncio.run(exercise())


def test_partial_capture_cancellation_stops_subsequent_lifecycle_reads(monkeypatch):
    async def exercise():
        store = InMemorySessionStore()
        target, _ = _lifecycle_target(store, bounds=SessionTrajectoryBounds(max_record_bytes=4096))
        entered = asyncio.Event()
        release = asyncio.Event()
        finished = asyncio.Event()
        reads = []
        inspect_identity = store.inspect_identity

        async def held_read(session_id):
            reads.append(session_id)
            entered.set()
            await release.wait()
            result = await inspect_identity(session_id)
            finished.set()
            return result

        monkeypatch.setattr(store, "inspect_identity", held_read)
        task = asyncio.create_task(
            run_workflow_eval_suite(target, _suite(FinalOutputContains("done")))
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
            release.set()
            await asyncio.wait_for(finished.wait(), timeout=2)
            # Allow the retained read to reach the next dispatch seam.
            for _ in range(5):
                await asyncio.sleep(0)
            assert len(reads) == 1
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(exercise())


def test_partial_capture_cannot_waive_hard_lifecycle_depth_limit():
    async def exercise():
        store = InMemorySessionStore()
        app = _register_app(session_store=store)

        class DeepWorkflow(WorkflowBase):
            spec = WorkflowSpec(name="partial-hard-lifecycle-depth")

            async def run(self, session_id):
                ctx = self.context(session_id)
                yield await ctx.start()
                parent = session_id
                for index in range(32):
                    child = f"{session_id}-depth-{index}"
                    interaction = await _create_running_session(
                        store, child, parent_session_id=parent
                    )
                    await _finish_session(store, child, interaction)
                    parent = child
                yield await ctx.completed({"answer": "done"})

        target = _target(app, DeepWorkflow).model_copy(
            update={"capture_bounds": SessionTrajectoryBounds(max_events=1)}
        )
        run = await run_workflow_eval_suite(target, _suite(FinalOutputContains("done")))
        trial = run.cases[0].trials[0]
        assert trial.status == "unavailable" and trial.score is None
        assert trial.capture_diagnostic is not None
        assert trial.capture_diagnostic.stage == "capture_revalidation"
        assert trial.capture_diagnostic.code == "depth_limit_exceeded"
        assert trial.capture_diagnostic.limit == 32

    asyncio.run(exercise())
