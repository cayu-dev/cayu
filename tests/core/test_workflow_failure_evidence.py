from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

import pytest
from pydantic import ValidationError

from cayu import (
    AgentSpec,
    CayuApp,
    ExecutionDeadline,
    ExecutionDeadlineExceeded,
    FailureEvidence,
    ModelStreamEvent,
    ParallelStepError,
    ScriptedModelProvider,
    SQLiteSessionStore,
    StepError,
    StepResult,
    WorkflowBase,
    WorkflowSpec,
    execution_deadline_scope,
    parallel,
    step,
)
from cayu.failure_evidence import event_failure_evidence
from cayu.workflows import StepRunOptions


async def deadline_branch(*, cleanup_failure=False):
    async with execution_deadline_scope(ExecutionDeadline.after(0.01, scope="verification")):
        try:
            await asyncio.Event().wait()
        finally:
            if cleanup_failure:
                raise ExceptionGroup("secret payload", [ValueError("secret"), OSError("secret")])


@pytest.mark.parametrize(
    "kind", ["deadline", "timeout", "failure", "admission", "cleanup", "interruption"]
)
def test_direct_classification_and_fail_closed(kind):
    async def fail():
        if kind in {"deadline", "cleanup"}:
            await deadline_branch(cleanup_failure=kind == "cleanup")
        elif kind == "admission":
            ExecutionDeadline.after(0, scope="verification").require_admission("model")
        elif kind == "interruption":
            raise asyncio.CancelledError()
        elif kind == "timeout":
            raise TimeoutError("secret")
        else:
            raise ValueError("secret")

    async def success():
        return StepResult("negative", "s", output=False)

    async def run():
        result = await parallel([success(), fail()])
        evidence = result.failures[0].evidence
        assert evidence.classification == ("deadline" if kind in {"admission", "cleanup"} else kind)
        assert evidence.settlement == "unknown"
        assert result.successes[0].output is False
        assert "secret" not in evidence.model_dump_json()
        assert (
            FailureEvidence.model_validate_json(evidence.model_dump_json()).model_dump()
            == evidence.model_dump()
        )
        if kind in {"deadline", "cleanup", "admission"}:
            assert evidence.deadline.scope == "verification"
            assert evidence.deadline_phase == ("admission" if kind == "admission" else "in_flight")
        if kind == "cleanup":
            assert evidence.secondary_failures
            assert {"ValueError", "OSError"} <= set(evidence.exception_types)
        with pytest.raises(ParallelStepError) as raised:
            result.raise_for_failures()
        assert raised.value.failures[0].evidence == evidence
        with pytest.raises(ParallelStepError):
            _ = result.outputs
        assert asdict(result.failures[0])["evidence"] == evidence

    asyncio.run(run())


@pytest.mark.parametrize("cancel", [True, False])
def test_parent_stop_propagates(cancel):
    async def run():
        task = asyncio.current_task()

        async def child():
            if cancel:
                task.cancel()
            await asyncio.Event().wait()

        with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
            async with execution_deadline_scope(ExecutionDeadline.after(0.02)):
                await parallel([child(), child()])

    asyncio.run(run())


class Verifiers(WorkflowBase):
    spec = WorkflowSpec(name="verifiers")

    async def run(self, session_id):
        yield await self.context(session_id).start()


class VerifierProvider(ScriptedModelProvider):
    def __init__(self, cleanup_failure=False):
        super().__init__([])
        self.cleanup_failure = cleanup_failure
        self.calls = 0
        self.settled = 0

    async def stream(self, request):
        self.calls += 1
        if request.model == "slow":
            try:
                await asyncio.Event().wait()
            finally:
                self.settled += 1
                if self.cleanup_failure:
                    raise RuntimeError("private cleanup error")
        yield ModelStreamEvent.text_delta("negative verification")
        yield ModelStreamEvent.completed({})


@pytest.mark.parametrize(
    "both_expire,cleanup_failure", [(False, False), (True, False), (False, True)]
)
def test_native_deadline_and_sqlite_replay(tmp_path, both_expire, cleanup_failure):
    async def run():
        path = tmp_path / "sessions.db"
        store = SQLiteSessionStore(path)
        provider = VerifierProvider(cleanup_failure)
        app = CayuApp(enable_logging=False, session_store=store)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="slow", model="slow"))
        app.register_agent(AgentSpec(name="fast", model="fast"))
        ctx = Verifiers(app).context("workflow")
        await ctx.start()
        async with execution_deadline_scope(ExecutionDeadline.after(30, scope="parent")):
            result = await parallel(
                [
                    step(
                        ctx,
                        agent="slow" if both_expire else "fast",
                        step_id="a",
                        prompt="verify",
                        run_options=StepRunOptions(
                            execution_deadline=ExecutionDeadline.after(3, scope="verification")
                        ),
                    ),
                    step(
                        ctx,
                        agent="slow",
                        step_id="b",
                        prompt="verify",
                        run_options=StepRunOptions(
                            execution_deadline=ExecutionDeadline.after(3, scope="verification")
                        ),
                    ),
                ]
            )
        assert len(result.failures) == (2 if both_expire else 1)
        assert len(result.successes) == (0 if both_expire else 1)
        assert provider.settled == len(result.failures)
        for failure in result.failures:
            assert failure.evidence.classification == "deadline"
            assert failure.evidence.deadline.scope == "verification"
            assert failure.evidence.deadline_phase == "in_flight"
            assert failure.evidence.session_id == failure.session_id
            assert failure.evidence.run_epoch is not None
            assert failure.evidence.terminal_event_id is not None
            assert failure.workflow_attempt_id == ctx.attempt_id
            assert failure.evidence.settlement == "unknown"
            if cleanup_failure:
                assert failure.evidence.secondary_failures
        calls = provider.calls
        await store.close()
        store = SQLiteSessionStore(path)
        replay_app = CayuApp(enable_logging=False, session_store=store)
        replay_app.register_provider(provider, default=True)
        replay_app.register_agent(AgentSpec(name="slow", model="slow"))
        replay_ctx = Verifiers(replay_app).context("workflow")
        await replay_ctx.start()
        for failure in result.failures:
            with pytest.raises(StepError) as raised:
                await step(replay_ctx, agent="slow", step_id=failure.step_id, prompt="verify")
            replay = raised.value.evidence
            assert replay.classification == "deadline"
            assert replay.deadline.model_dump() == failure.evidence.deadline.model_dump()
            assert replay.run_epoch == failure.evidence.run_epoch
            assert replay.terminal_event_id == failure.evidence.terminal_event_id
            assert replay.settlement == "unknown"
            if cleanup_failure:
                assert replay.secondary_failures
        assert provider.calls == calls
        await store.close()

    asyncio.run(run())


def test_legacy_event_expired_clock_is_not_causation():
    evidence = event_failure_evidence(
        {"execution_deadline": ExecutionDeadline.after(0).inspection()},
        session_id="s",
        event_id="e",
        interrupted=True,
    )
    assert evidence.classification == "interruption"
    assert evidence.deadline is None
    assert evidence.run_epoch is None
    assert evidence.settlement == "unknown"


def test_diagnostics_bounds():
    from cayu.failure_evidence import exception_evidence

    evidence = exception_evidence(ExceptionGroup("secret", [ValueError("secret")] * 100))
    assert evidence.truncated
    assert len(evidence.exception_types) <= 16
    assert len(evidence.model_dump_json()) < 4096
    assert "secret" not in json.dumps(evidence.model_dump(mode="json"))
    with pytest.raises(ValidationError):
        FailureEvidence(exception_types=("x" * 129,))


@pytest.mark.parametrize("kind", ["timeout", "failure", "admission"])
def test_native_failure_and_explicit_child_replay(kind):
    class FailingProvider(ScriptedModelProvider):
        calls = 0

        async def stream(self, request):
            self.calls += 1
            if kind == "timeout":
                raise TimeoutError("private error")
            raise ValueError("private error")
            yield  # pragma: no cover

    async def run():
        provider = FailingProvider([])
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker", model="test"))
        ctx = Verifiers(app).context("workflow")
        await ctx.start()
        result = await parallel(
            [
                step(
                    ctx,
                    agent="worker",
                    step_id="s",
                    session_id="child",
                    prompt="verify",
                    run_options=StepRunOptions(execution_deadline=ExecutionDeadline.after(0))
                    if kind == "admission"
                    else None,
                )
            ]
        )
        failure = result.failures[0]
        assert failure.evidence.classification == ("deadline" if kind == "admission" else kind)
        assert failure.step_id == "s"
        assert failure.workflow_attempt_id == ctx.attempt_id
        if kind == "admission":
            assert provider.calls == 0
            assert failure.evidence.deadline_phase == "admission"
            assert failure.evidence.run_epoch is None
            return
        assert failure.evidence.run_epoch is not None
        calls = provider.calls
        replay = Verifiers(app).context("workflow")
        await replay.start()
        with pytest.raises(StepError) as raised:
            await step(replay, agent="worker", step_id="s", session_id="child", prompt="verify")
        assert raised.value.evidence.model_dump() == failure.evidence.model_dump()
        assert provider.calls == calls

    asyncio.run(run())


@pytest.mark.parametrize("signal", ["duplicate", "superseded", "cancel"])
def test_grouped_stop_signals_propagate(signal):
    from cayu.workflows.workflow import DuplicateStepIdError, WorkflowSupersededError

    async def fail():
        error = {
            "duplicate": DuplicateStepIdError,
            "superseded": WorkflowSupersededError,
            "cancel": asyncio.CancelledError,
        }[signal]()
        raise BaseExceptionGroup("stop and cleanup", [error, ValueError("cleanup")])

    async def run():
        with pytest.raises(BaseExceptionGroup):
            await parallel([fail()])

    asyncio.run(run())


def test_no_parent_time_without_active_timer():
    from cayu.deadlines import bind_execution_deadline

    async def success():
        return StepResult("s", "child")

    async def run():
        with (
            bind_execution_deadline(ExecutionDeadline.after(0, scope="parent")),
            pytest.raises(ExecutionDeadlineExceeded),
        ):
            await parallel([success()])

    asyncio.run(run())


def test_retained_work_is_not_reported_settled():
    async def run():
        retained = asyncio.create_task(asyncio.Event().wait())
        try:
            result = await parallel([deadline_branch()])
            assert result.failures[0].evidence.classification == "deadline"
            assert result.failures[0].evidence.settlement == "unknown"
            assert not retained.done()
        finally:
            retained.cancel()
            await asyncio.gather(retained, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize(
    "payload",
    [
        {"classification": "deadline"},
        {"classification": "deadline", "deadline_phase": "in_flight", "deadline": {}},
        {"classification": "deadline", "settlement": "safe"},
    ],
)
def test_malformed_durable_evidence_stays_unknown(payload):
    evidence = event_failure_evidence(
        {"failure_evidence": {**payload, "session_id": "s"}},
        session_id="s",
        event_id="e",
        interrupted=False,
    )
    assert evidence.classification == "unknown"
    assert evidence.deadline is None
    assert evidence.settlement == "unknown"


@pytest.mark.parametrize("bounded_parent", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_child_deadline_cancellation_cleanup_group_keeps_sibling(bounded_parent, nested):
    async def run():
        sibling_finished = False

        async def child():
            async with execution_deadline_scope(ExecutionDeadline.after(0.01, scope="child")):
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError as exc:
                    group = BaseExceptionGroup(
                        "deadline and cleanup", [exc, RuntimeError("cleanup")]
                    )
                    if nested:
                        group = BaseExceptionGroup("nested", [group, ValueError("cleanup")])
                    raise group from exc

        async def sibling():
            nonlocal sibling_finished
            await asyncio.sleep(0.03)
            sibling_finished = True
            return StepResult("healthy", "healthy", output="ok")

        async with execution_deadline_scope(
            ExecutionDeadline.after(10 if bounded_parent else None, scope="parent")
        ):
            result = await parallel([child(), sibling()])
        assert sibling_finished
        assert len(result.failures) == len(result.successes) == 1
        assert result.successes[0].output == "ok"
        evidence = result.failures[0].evidence
        assert evidence.classification == "deadline"
        assert evidence.deadline.scope == "child"
        assert evidence.deadline_phase == "in_flight"
        assert evidence.secondary_failures
        assert evidence.settlement == "unknown"
        assert {"CancelledError", "RuntimeError"} <= set(evidence.exception_types)
        assert "_cayu_deadline_expiry_task" not in evidence.model_dump_json()
        with pytest.raises(ParallelStepError):
            result.raise_for_failures()

    asyncio.run(run())


@pytest.mark.parametrize("stop", ["parent_cancel", "parent_deadline", "child_cancel"])
def test_deadline_cleanup_group_preserves_requested_stops(stop):
    async def run():
        parent = asyncio.current_task()
        entered = asyncio.Event()
        sibling_finished = False

        async def child():
            async with execution_deadline_scope(ExecutionDeadline.after(10, scope="child")):
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError as exc:
                    raise BaseExceptionGroup(
                        "stop and cleanup", [exc, RuntimeError("cleanup")]
                    ) from exc

        async def sibling():
            nonlocal sibling_finished
            await asyncio.Event().wait()
            sibling_finished = True
            return StepResult("healthy", "healthy")

        async with execution_deadline_scope(
            ExecutionDeadline.after(0.03 if stop == "parent_deadline" else None, scope="parent")
        ):
            child_task = asyncio.create_task(child())

            async def cancel():
                await entered.wait()
                if stop == "parent_cancel":
                    parent.cancel()
                elif stop == "child_cancel":
                    child_task.cancel()

            canceller = asyncio.create_task(cancel())
            try:
                await parallel([child_task, sibling()])
            finally:
                await asyncio.gather(canceller, return_exceptions=True)
                assert not sibling_finished

    with pytest.raises((asyncio.CancelledError, TimeoutError)):
        asyncio.run(run())


@pytest.mark.parametrize("stop", ["duplicate", "superseded", "fatal"])
def test_child_deadline_does_not_downgrade_other_grouped_stops(stop):
    from cayu.workflows.workflow import DuplicateStepIdError, WorkflowSupersededError

    class FatalStop(BaseException):
        pass

    async def child():
        async with execution_deadline_scope(ExecutionDeadline.after(0.01, scope="child")):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as exc:
                error = {
                    "duplicate": DuplicateStepIdError,
                    "superseded": WorkflowSupersededError,
                    "fatal": FatalStop,
                }[stop]()
                raise BaseExceptionGroup("deadline and stop", [exc, error]) from exc

    async def run():
        with pytest.raises(BaseExceptionGroup):
            await parallel([child()])

    asyncio.run(run())


def test_deadline_metadata_does_not_authorize_group_collection():
    async def child():
        group = BaseExceptionGroup("cancel and cleanup", [asyncio.CancelledError(), RuntimeError()])
        group.execution_deadline = ExecutionDeadline.after(0, scope="child").inspection()
        raise group

    async def run():
        with pytest.raises(BaseExceptionGroup):
            await parallel([child()])

    asyncio.run(run())


def test_parent_expiry_without_timer_stops_child_cleanup_group():
    from cayu.deadlines import bind_execution_deadline

    async def child():
        async with execution_deadline_scope(ExecutionDeadline.after(10, scope="child")):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as exc:
                raise BaseExceptionGroup("deadline and cleanup", [exc, RuntimeError()]) from exc

    async def sibling():
        await asyncio.Event().wait()
        return StepResult("healthy", "healthy")

    async def run():
        with (
            bind_execution_deadline(ExecutionDeadline.after(0.01, scope="parent")),
            pytest.raises(ExecutionDeadlineExceeded),
        ):
            await asyncio.wait_for(parallel([child(), sibling()]), 1)

    asyncio.run(run())


def test_deadline_group_from_another_task_is_not_collection_authority():
    async def child():
        async with execution_deadline_scope(ExecutionDeadline.after(0.01, scope="child")):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as exc:
                raise BaseExceptionGroup("deadline and cleanup", [exc, RuntimeError()]) from exc

    async def run():
        with pytest.raises(BaseExceptionGroup) as raised:
            await asyncio.create_task(child())

        async def rethrow():
            raise raised.value

        with pytest.raises(BaseExceptionGroup):
            await parallel([rethrow()])

    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
def test_parallel_accepts_future_outcomes(outcome):
    async def run():
        future = asyncio.get_running_loop().create_future()
        if outcome == "success":
            future.set_result(StepResult("healthy", "healthy", output="ok"))
            result = await parallel([future])
            assert result.outputs == ("ok",)
        elif outcome == "failure":
            future.set_exception(ValueError("failed"))
            result = await parallel([future])
            assert result.failures[0].evidence.classification == "failure"
        else:
            future.cancel()
            with pytest.raises(asyncio.CancelledError):
                await parallel([future])

    asyncio.run(run())
