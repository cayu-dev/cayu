"""A grouped cancellation settles like the plain cancellation it carries.

A before-stop policy waits until the run is cancelled. Its cleanup then either
re-raises the cancellation or fails too and raises it grouped with an
``OSError``. The shape of that cleanup failure must not change the settlement:
without an interrupt request the run is abandoned (and an expired deadline fails
its task), and with one it ends as the requested interruption.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from datetime import datetime, timedelta

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest, SessionStatus
from cayu._exception_groups import iter_exception_tree
from cayu.deadlines import ExecutionDeadline
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.runtime import BeforeStopContext, BeforeStopDecision, LoopPolicy
from cayu.runtime._run_exit import grouped_caller_cancellation
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions import InMemorySessionStore, InterruptSessionRequest
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskStatus

SESSION_ID = "grouped-cancellation"


class _WaitingStopPolicy(LoopPolicy):
    """Waits in ``before_stop`` until cancelled; its cleanup may fail too."""

    def __init__(self, *, grouped: bool, on_enter=None, on_cancel=None) -> None:
        self.grouped = grouped
        self.on_enter = on_enter
        self.on_cancel = on_cancel
        self.entered = asyncio.Event()
        self.raised: BaseException | None = None

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:grouped-cancellation-stop-policy",
            behavior_version="1",
            implementation_version="1",
        )

    async def before_stop(self, context: BeforeStopContext) -> BeforeStopDecision:
        if self.on_enter is not None:
            self.on_enter()
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as cancellation:
            if self.on_cancel is not None:
                self.on_cancel()
            self.raised = cancellation
            if self.grouped:
                self.raised = BaseExceptionGroup(
                    "policy cancelled and cleanup failed",
                    [cancellation, OSError("policy connection close failed")],
                )
                raise self.raised from None
            raise
        raise AssertionError("the policy was not cancelled")


def _keeps_store_capabilities(store_type: type) -> type:
    """Redeclare capability flags beside the overrides, as the Runtime requires."""

    for name in dir(InMemorySessionStore):
        attribute = inspect.getattr_static(InMemorySessionStore, name)
        if not name.startswith("__") and type(attribute) in (bool, int):
            setattr(store_type, name, attribute)
    return store_type


@_keeps_store_capabilities
class _InterruptReadStore(InMemorySessionStore):
    """Its interrupt-state reads fail or suspend once armed."""

    def __init__(self, *, fail: bool) -> None:
        super().__init__()
        self.fail = fail
        self.armed = False
        self.reading = asyncio.Event()
        self.read_failure = OSError("session state read failed")

    async def load_state(self, session_id: str):
        if self.armed:
            self.armed = False
            self.reading.set()
            if self.fail:
                raise self.read_failure
            await asyncio.sleep(0.05)
        return await super().load_state(session_id)


@_keeps_store_capabilities
class _BlockingSetupStore(InMemorySessionStore):
    """Run setup's initial transcript commits, then its cleanup fails on cancellation."""

    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.raised: BaseExceptionGroup | None = None

    async def replace_initial_transcript_messages(self, *args, **kwargs):
        result = await super().replace_initial_transcript_messages(*args, **kwargs)
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as cancellation:
            self.raised = BaseExceptionGroup(
                "setup cancelled and cleanup failed",
                [cancellation, OSError("setup cleanup failed")],
            )
            raise self.raised from None
        return result


def _app(store: InMemorySessionStore, tasks: InMemoryTaskStore | None = None) -> CayuApp:
    app = CayuApp(session_store=store, task_store=tasks, enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        ),
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="scripted"))
    return app


def _request(policy: LoopPolicy, **options) -> RunRequest:
    return RunRequest(
        agent_name="assistant",
        session_id=SESSION_ID,
        messages=[Message.text("user", "answer")],
        loop_policies=(policy,),
        **options,
    )


def test_only_delivered_cancellation_with_cleanup_failures_is_the_callers() -> None:
    async def scenario() -> None:
        cancellation = asyncio.CancelledError()
        cleanup = OSError("cleanup failed")

        def group(*leaves: BaseException) -> BaseExceptionGroup:
            return BaseExceptionGroup("cleanup", list(leaves))

        task = asyncio.current_task()
        assert task is not None
        # Not delivered: a child's own cancellation inside a failure.
        assert grouped_caller_cancellation(group(cancellation, cleanup)) is None
        task.cancel()
        try:
            assert grouped_caller_cancellation(cancellation) is None
            assert grouped_caller_cancellation(group(cancellation, cleanup)) is cancellation
            nested = group(group(cleanup), cancellation)
            assert grouped_caller_cancellation(nested) is cancellation
            assert (
                grouped_caller_cancellation(group(cancellation, asyncio.CancelledError())) is None
            )
            assert grouped_caller_cancellation(group(cancellation, GeneratorExit())) is None
            assert grouped_caller_cancellation(group(cancellation, KeyboardInterrupt())) is None
            assert (
                grouped_caller_cancellation(group(cancellation, cleanup), cancellation_baseline=1)
                is None
            )
        finally:
            task.uncancel()

    asyncio.run(scenario())


async def _terminal(store: InMemorySessionStore):
    events = await store.load_events(SESSION_ID)
    return next(event for event in reversed(events) if event.type is EventType.SESSION_INTERRUPTED)


@pytest.mark.parametrize("grouped", [False, True], ids=["plain", "grouped"])
def test_cancellation_without_interrupt_request_abandons_the_run(grouped: bool) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        app = _app(store)
        policy = _WaitingStopPolicy(grouped=grouped)

        async def consume() -> None:
            async for _ in app.run(_request(policy)):
                pass

        run = asyncio.create_task(consume())
        try:
            await policy.entered.wait()
            run.cancel()
            with pytest.raises(BaseException) as raised:
                await run
            # The cancellation, or exactly the group the cleanup raised.
            assert raised.value is policy.raised
            assert await app.drain_recovery_cleanups(timeout_s=5)
            assert (await store.load(SESSION_ID)).status is SessionStatus.INTERRUPTED
            terminal = await _terminal(store)
            assert terminal.payload["abandoned"] is True
            assert terminal.payload["reason"] == "event_stream_closed"
            assert terminal.payload["failure_evidence"]["classification"] == "interruption"
            assert "interruption_request_id" not in terminal.payload
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("grouped", [False, True], ids=["plain", "grouped"])
def test_cancellation_with_interrupt_request_ends_as_that_interruption(grouped: bool) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        app = _app(store)
        policy = _WaitingStopPolicy(grouped=grouped)
        stream = app.run(_request(policy))

        async def consume() -> list:
            return [event async for event in stream]

        run = asyncio.create_task(consume())
        try:
            await policy.entered.wait()
            async for _ in app.interrupt_session(
                InterruptSessionRequest(session_id=SESSION_ID, reason="requested pause")
            ):
                pass
            await asyncio.wait_for(asyncio.gather(run, return_exceptions=True), timeout=10)
            assert await app.drain_background_interruptions(timeout_s=5)
            assert (await store.load(SESSION_ID)).status is SessionStatus.INTERRUPTED
            terminal = await _terminal(store)
            assert terminal.payload["reason"] == "requested pause"
            assert terminal.payload.get("abandoned") is not True
            assert terminal.payload["interruption_request_id"]
        finally:
            await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("grouped", [False, True], ids=["plain", "grouped"])
def test_deadline_during_stop_policy_fails_the_attached_task(monkeypatch, grouped: bool) -> None:
    import cayu.deadlines as deadline_module

    async def scenario() -> None:
        advance = timedelta()

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return datetime.now(tz) + advance

        monkeypatch.setattr(deadline_module, "datetime", Clock)
        boundary = ExecutionDeadline.after(600)

        def expire() -> None:
            # Expire the real owned timer only once the policy is reached, so
            # no setup latency selects another failure boundary.
            nonlocal advance
            advance = timedelta(seconds=601)
            timers = [
                timer
                for timer, selected in deadline_module._TIMERS.get()
                if selected.expires_at == boundary.expires_at and timer.when() is not None
            ]
            assert timers
            timers[-1].reschedule(asyncio.get_running_loop().time())

        store = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        task = await tasks.create_task(TaskCreate(type="answer"))
        app = _app(store, tasks)
        policy = _WaitingStopPolicy(grouped=grouped, on_enter=expire)
        try:
            run = app.run(_request(policy, task_id=task.id, execution_deadline=boundary))
            # The deadline's cancellation, with any cleanup failure, may escape.
            with contextlib.suppress(BaseException):
                async for _ in run:
                    pass
            assert policy.entered.is_set()
            assert await app.drain_recovery_cleanups(timeout_s=5)
            assert await app.drain_environment_cleanups(timeout_s=5)
            assert (await store.load(SESSION_ID)).status is SessionStatus.INTERRUPTED
            terminal = await _terminal(store)
            assert terminal.payload["failure_evidence"]["classification"] == "deadline"
            final = await tasks.load_task(task.id)
            assert final.status is TaskStatus.FAILED
            assert final.error["error"] == "ExecutionDeadlineExceeded"
        finally:
            await app.aclose()

    asyncio.run(scenario())


def _assert_abandoned(terminal) -> None:
    assert terminal.payload["abandoned"] is True
    assert terminal.payload["reason"] == "event_stream_closed"
    assert "interruption_request_id" not in terminal.payload


def test_grouped_cancellation_in_run_setup_abandons_the_run() -> None:
    async def scenario() -> None:
        store = _BlockingSetupStore()
        app = _app(store)

        async def consume() -> None:
            async for _ in app.run(_request(_WaitingStopPolicy(grouped=False))):
                pass

        run = asyncio.create_task(consume())
        try:
            await store.entered.wait()
            run.cancel()
            with pytest.raises(BaseExceptionGroup) as raised:
                await run
            assert raised.value is store.raised
            assert await app.drain_recovery_cleanups(timeout_s=5)
            assert (await store.load(SESSION_ID)).status is SessionStatus.INTERRUPTED
            _assert_abandoned(await _terminal(store))
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_failed_interrupt_read_keeps_the_grouped_cancellation() -> None:
    async def scenario() -> None:
        store = _InterruptReadStore(fail=True)
        app = _app(store)
        policy = _WaitingStopPolicy(grouped=True, on_cancel=lambda: setattr(store, "armed", True))

        async def consume() -> None:
            async for _ in app.run(_request(policy)):
                pass

        run = asyncio.create_task(consume())
        try:
            await policy.entered.wait()
            run.cancel()
            with pytest.raises(BaseExceptionGroup) as raised:
                await run
            assert store.reading.is_set()
            assert raised.value is policy.raised
            cause = raised.value.__cause__
            assert cause is not None
            assert any(leaf is store.read_failure for leaf in iter_exception_tree(cause))
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_cancellation_redelivered_during_interrupt_read_still_abandons() -> None:
    async def scenario() -> None:
        store = _InterruptReadStore(fail=False)
        app = _app(store)
        policy = _WaitingStopPolicy(grouped=True, on_cancel=lambda: setattr(store, "armed", True))

        async def consume() -> None:
            async for _ in app.run(_request(policy)):
                pass

        run = asyncio.create_task(consume())
        try:
            await policy.entered.wait()
            run.cancel()
            await store.reading.wait()
            run.cancel()
            with pytest.raises(BaseExceptionGroup) as raised:
                await run
            assert raised.value is policy.raised
            assert await app.drain_recovery_cleanups(timeout_s=5)
            assert (await store.load(SESSION_ID)).status is SessionStatus.INTERRUPTED
            _assert_abandoned(await _terminal(store))
        finally:
            await app.aclose()

    asyncio.run(scenario())
