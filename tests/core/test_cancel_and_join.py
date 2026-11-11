"""Stopping a child must not swallow a cancellation of the task that stops it."""

from __future__ import annotations

import asyncio
import types
from typing import Any, cast

import pytest
import tests.core.test_dispatch as dispatch_tests
import tests.core.test_run_limits as run_limit_tests

from cayu._task_wait import (
    cancel_and_join,
    consume_pending_task_cancellation,
    restore_task_cancellation_requests,
)
from cayu.events import Event
from cayu.messages import Message
from cayu.runtime import _inbox_continuation
from cayu.runtime import _run_limits as run_limits_module
from cayu.runtime._event_loop_lag_sessions import SessionEventLoopLagRecorder
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.requests import RunRequest
from cayu.tasks.dispatch import TaskStoreDispatcher
from cayu.tasks.records import TaskClaimLost


async def _slow_to_stop(stopping: asyncio.Event, release: asyncio.Event) -> None:
    """A child that takes a while to finish once it is cancelled, like a heartbeat
    whose last renewal is still in flight."""

    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        stopping.set()
        await release.wait()
        raise


async def _cancel_while_joining(join) -> tuple[bool, asyncio.Task[None]]:
    stopping = asyncio.Event()
    release = asyncio.Event()
    child = asyncio.create_task(_slow_to_stop(stopping, release))
    await asyncio.sleep(0)
    caller = asyncio.create_task(join(child))
    await stopping.wait()
    caller.cancel()
    await asyncio.sleep(0)
    release.set()
    try:
        await caller
    except asyncio.CancelledError:
        return True, child
    return False, child


def test_cancel_and_join_keeps_the_callers_cancellation() -> None:
    async def scenario() -> None:
        cancelled, child = await _cancel_while_joining(cancel_and_join)
        assert cancelled
        assert child.done()

    asyncio.run(scenario())


def test_cancel_and_join_returns_the_childs_own_outcome() -> None:
    async def scenario() -> None:
        stopped = asyncio.create_task(asyncio.Event().wait())
        await asyncio.sleep(0)
        assert await cancel_and_join(stopped) is None

        failure = RuntimeError("child failed while stopping")

        async def fails_on_cancel() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise failure from None

        failing = asyncio.create_task(fails_on_cancel())
        await asyncio.sleep(0)
        assert await cancel_and_join(failing) is failure

        async def returns() -> str:
            return "done"

        finished = asyncio.create_task(returns())
        await finished
        assert await cancel_and_join(finished) is None
        current = asyncio.current_task()
        assert current is not None and current.cancelling() == 0

    asyncio.run(scenario())


def test_cancel_and_join_leaves_an_owned_cancellation_to_its_caller() -> None:
    """In a finally on the way out, the cancellation already propagating stays
    the one that propagates: its repeated delivery is absorbed, its count kept."""

    async def scenario() -> None:
        stopping = asyncio.Event()
        release = asyncio.Event()
        child = asyncio.create_task(_slow_to_stop(stopping, release))
        leaving = asyncio.CancelledError("shutdown")
        await asyncio.sleep(0)

        async def finally_block() -> BaseException | None:
            return await cancel_and_join(child, leaving=leaving)

        caller = asyncio.create_task(finally_block())
        await stopping.wait()
        caller.cancel("shutdown")
        await asyncio.sleep(0)
        release.set()
        assert await caller is None
        assert child.done()
        assert caller.cancelling() == 1

    asyncio.run(scenario())


def test_dispatcher_heartbeat_stop_keeps_a_shutdown_cancellation() -> None:
    """The heartbeat stops in the finally around terminalization; a worker
    shutdown arriving then must still end ``process_next``."""

    async def stop_heartbeat(heartbeat: asyncio.Task[None]) -> None:
        await TaskStoreDispatcher._stop_heartbeat(asyncio.Event(), heartbeat)

    async def scenario() -> None:
        cancelled, heartbeat = await _cancel_while_joining(stop_heartbeat)
        assert cancelled
        assert heartbeat.done()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [TaskClaimLost("lease lost"), None])
def test_dispatcher_heartbeat_stop_tolerates_a_lost_claim(failure: Exception | None) -> None:
    async def scenario() -> None:
        async def heartbeat() -> None:
            if failure is not None:
                raise failure
            await asyncio.Event().wait()

        task = asyncio.create_task(heartbeat())
        await asyncio.sleep(0)
        await TaskStoreDispatcher._stop_heartbeat(asyncio.Event(), task)
        assert task.done()

    asyncio.run(scenario())


def test_dispatcher_heartbeat_stop_raises_other_heartbeat_failures() -> None:
    async def scenario() -> None:
        async def heartbeat() -> None:
            raise RuntimeError("renewal failed")

        task = asyncio.create_task(heartbeat())
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="renewal failed"):
            await TaskStoreDispatcher._stop_heartbeat(asyncio.Event(), task)

    asyncio.run(scenario())


def test_budgeted_operation_keeps_a_cancellation_while_its_heartbeat_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operation returned; cancelling the run while the reservation heartbeat
    stops must still cancel it, not return the result as if nothing happened."""

    store = InMemorySessionStore()
    controller = run_limit_tests._controller(store)
    stopping = asyncio.Event()
    release = asyncio.Event()

    async def heartbeat_reservations(*args, **kwargs) -> None:
        await _slow_to_stop(stopping, release)

    async def scenario() -> None:
        session_id = "sess_operation_heartbeat_stop_cancel"
        await run_limit_tests._running_session(store, session_id)
        setup = await controller.reserve_operation_budgets(
            budget_limits=(run_limit_tests._reserved_limit("2"),),
            session_id=session_id,
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=run_limit_tests._model_attempt_identity(),
            rejection_release_reason="reservation rejected",
            accepted_record_error="accepted reservation missing record",
        )
        monkeypatch.setattr(controller, "heartbeat_reservations", heartbeat_reservations)

        async def operation() -> str:
            return "done"

        caller = asyncio.create_task(
            controller.run_operation_with_reservation_heartbeat(
                operation,
                reservations=list(setup.reservations),
                authoritative_failure_types=(),
                lease_lost_before_dispatch_message="lease lost before operation",
                authoritative_failure_note="lease lost as operation failed",
                concurrent_failure_note="operation failed while lease loss was handled",
            )
        )
        await stopping.wait()
        caller.cancel("caller cancelled")
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError, match="caller cancelled"):
            await caller

    asyncio.run(scenario())


def test_lag_recorder_close_keeps_a_cancellation_while_a_write_stops() -> None:
    async def emit(event: Event) -> None:
        return None

    async def stop_writes(write: asyncio.Task[None]) -> None:
        recorder = SessionEventLoopLagRecorder(active_session_ids=lambda: (), emit=emit)
        recorder._write = write
        await recorder.aclose(timeout_s=0.001)

    async def scenario() -> None:
        cancelled, write = await _cancel_while_joining(stop_writes)
        assert cancelled
        assert write.done()

    asyncio.run(scenario())


def test_cancel_and_join_forwards_a_further_cancellation_to_the_child() -> None:
    """A child that finishes in-flight work on its first cancellation is
    cancelled again by the caller's next one, as ``await child`` did."""

    async def scenario() -> None:
        first = asyncio.Event()

        async def stubborn() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                first.set()
                await asyncio.Event().wait()

        child = asyncio.create_task(stubborn())
        await asyncio.sleep(0)
        caller = asyncio.create_task(cancel_and_join(child))
        await first.wait()
        caller.cancel("first")
        with pytest.raises(asyncio.CancelledError, match="first"):
            await caller
        assert child.done()

    asyncio.run(scenario())


def test_cancel_and_join_names_the_childs_failure_on_the_callers_cancellation() -> None:
    async def scenario() -> None:
        stopping = asyncio.Event()
        release = asyncio.Event()

        async def fails_while_stopping() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                stopping.set()
                # Its last renewal finishes even when the stop is repeated.
                while not release.is_set():
                    try:
                        await release.wait()
                    except asyncio.CancelledError:
                        continue
                raise RuntimeError("lease lost") from None

        child = asyncio.create_task(fails_while_stopping())
        await asyncio.sleep(0)

        async def join() -> list[str]:
            try:
                await cancel_and_join(child)
            except asyncio.CancelledError as cancellation:
                return cancellation.__notes__
            return []

        caller = asyncio.create_task(join())
        await stopping.wait()
        caller.cancel()
        await asyncio.sleep(0)
        release.set()
        assert await caller == ["The stopped task also failed: RuntimeError."]

    asyncio.run(scenario())


def test_dispatcher_heartbeat_stop_lets_a_process_control_exit_through() -> None:
    """A lease owner that restores a consumed cancellation and then raises a
    process-control failure leaves a cancellation delivery pending; stopping the
    heartbeat must not raise that delivery over the process-control exit."""

    async def claim(heartbeat: asyncio.Task[Any]) -> None:
        leaving: BaseException | None = None
        try:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as owner:
                consume_pending_task_cancellation(owner)
                restore_task_cancellation_requests(1, cancellation=owner)
                raise BaseExceptionGroup("drain", [GeneratorExit("process control")]) from None
        except BaseException as exiting:
            if not isinstance(exiting, Exception):
                leaving = exiting
            raise
        finally:
            await TaskStoreDispatcher._stop_heartbeat(asyncio.Event(), heartbeat, leaving=leaving)

    async def scenario() -> None:
        heartbeat = asyncio.create_task(asyncio.Event().wait())
        task = asyncio.create_task(claim(heartbeat))
        await asyncio.sleep(0)
        task.cancel("shutdown")
        with pytest.raises(BaseExceptionGroup) as raised:
            await task
        assert isinstance(raised.value.exceptions[0], GeneratorExit)
        assert heartbeat.done()

    asyncio.run(scenario())


def test_streamed_model_step_keeps_a_cancellation_while_its_heartbeat_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemorySessionStore()
    controller = run_limit_tests._controller(store)
    stopping = asyncio.Event()
    release = asyncio.Event()

    async def heartbeat_reservations(*args, **kwargs) -> None:
        await _slow_to_stop(stopping, release)

    async def scenario() -> None:
        session_id = "sess_model_step_heartbeat_stop_cancel"
        await run_limit_tests._running_session(store, session_id)
        setup = await controller.reserve_operation_budgets(
            budget_limits=(run_limit_tests._reserved_limit("2"),),
            session_id=session_id,
            agent_name="assistant",
            provider_name="fake",
            model="fake-model",
            model_attempt_identity=run_limit_tests._model_attempt_identity(),
            rejection_release_reason="reservation rejected",
            accepted_record_error="accepted reservation missing record",
        )
        monkeypatch.setattr(controller, "heartbeat_reservations", heartbeat_reservations)

        async def model_events():
            yield None, None

        async def consume() -> None:
            async for _ in controller.model_step_events_with_heartbeat(
                model_events(),
                reservations=list(setup.reservations),
                lifecycle=run_limits_module.BudgetModelStepLifecycle(),
            ):
                pass

        caller = asyncio.create_task(consume())
        await stopping.wait()
        caller.cancel("caller cancelled")
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError, match="caller cancelled"):
            await caller

    asyncio.run(scenario())


def test_inbox_continuation_keeps_a_cancellation_while_its_receive_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the continuation's worker is done, the pending queue receive is
    stopped; a cancellation of the stream owner then must end the stream."""

    stopping = asyncio.Event()
    release = asyncio.Event()
    worker_done = asyncio.Event()

    class SlowToStopQueue(asyncio.Queue):
        async def get(self):
            await _slow_to_stop(stopping, release)

    monkeypatch.setattr(
        _inbox_continuation,
        "asyncio",
        types.SimpleNamespace(**{**vars(asyncio), "Queue": SlowToStopQueue}),
    )

    class App:
        task_store = object()

        async def _run_worker_step(self, run) -> None:
            await worker_done.wait()

    async def scenario() -> None:
        async def consume() -> None:
            async for _ in _inbox_continuation._owned_inbox_continuation(
                cast("Any", App()), cast("Any", None), cast("Any", None), cast("Any", None)
            ):
                pass

        caller = asyncio.create_task(consume())
        await asyncio.sleep(0)
        worker_done.set()
        await stopping.wait()
        caller.cancel("stream owner cancelled")
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError, match="stream owner cancelled"):
            await caller

    asyncio.run(scenario())


def test_dispatcher_lets_a_process_control_exit_of_a_claim_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lease owner restores the caller's consumed cancellation and raises a
    process-control failure; the claim must end with that failure, not with the
    restored cancellation delivered while the heartbeat stops."""

    async def scenario() -> None:
        h = dispatch_tests._build([dispatch_tests._batch("initial")])
        async for _ in h.app.run(
            RunRequest(
                agent_name="assistant",
                session_id="process-control",
                messages=[Message.text("user", "initial")],
            )
        ):
            pass
        await h.app.dispatch(dispatch_tests._dispatch_request("process-control", "queued"))
        entered = asyncio.Event()

        async def lease_owner(awaitable, *args, **kwargs):
            awaitable.close()
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as owner:
                consume_pending_task_cancellation(owner)
                restore_task_cancellation_requests(1, cancellation=owner)
                raise BaseExceptionGroup(
                    "Dispatch process-control and caller-cancellation evidence",
                    [GeneratorExit("process control")],
                ) from None

        monkeypatch.setattr(h.dispatcher, "_await_with_task_lease", lease_owner)
        processing = asyncio.create_task(h.dispatcher.process_next(h.app, worker_id="owner"))
        await asyncio.wait_for(entered.wait(), 5)
        processing.cancel("shutdown")
        [outcome] = await asyncio.gather(processing, return_exceptions=True)
        assert isinstance(outcome, BaseExceptionGroup), repr(outcome)
        assert isinstance(outcome.exceptions[0], GeneratorExit)

    asyncio.run(scenario())
