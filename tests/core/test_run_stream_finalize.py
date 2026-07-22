"""An abandoned run/resume event stream must finalize — not strand — the session.

When a consumer closes the event stream mid-run (client disconnect, abandoned async
generator), GeneratorExit reaches the run generator. The runtime must transition the
still-RUNNING session to INTERRUPTED and persist a terminal event, so the session
stays observable and resumable instead of being stranded in RUNNING forever.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from datetime import timedelta
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    import pytest

from cayu.core import AgentSpec, Message
from cayu.core.events import Event, EventType
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    CayuApp,
    InMemorySessionStore,
    ResumeRequest,
    RunRequest,
    SessionIdentity,
    SessionRunFenced,
    SessionStatus,
)


class FakeProvider(ModelProvider):
    name = "fake"

    def __init__(self, batches: list[list[ModelStreamEvent]]) -> None:
        self.event_batches = batches
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        batch_index = len(self.requests)
        self.requests.append(request)
        if batch_index >= len(self.event_batches):
            raise AssertionError(f"No fake provider event batch for request {batch_index}")
        for event in self.event_batches[batch_index]:
            yield event


class RecordingReleaseStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.release_calls: dict[str, int] = {}

    async def release_run_fence(self, session_id: str) -> None:
        self.release_calls[session_id] = self.release_calls.get(session_id, 0) + 1
        await super().release_run_fence(session_id)


class BlockingReleaseStore(RecordingReleaseStore):
    def __init__(self, *, fail_release: bool = False) -> None:
        super().__init__()
        self.fail_release = fail_release
        self.release_started = asyncio.Event()
        self.allow_release = asyncio.Event()
        self.release_finished = asyncio.Event()

    async def release_run_fence(self, session_id: str) -> None:
        self.release_started.set()
        await self.allow_release.wait()
        try:
            if self.fail_release:
                try:
                    raise OSError("run fence database connection lost")
                except OSError as cause:
                    raise RuntimeError("run fence release failed") from cause
            await super().release_run_fence(session_id)
        finally:
            self.release_finished.set()


class Harness(NamedTuple):
    app: CayuApp
    store: RecordingReleaseStore
    provider: FakeProvider


def _batch(text: str) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.text_delta(text),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]


def _build(batches: list[list[ModelStreamEvent]]) -> Harness:
    store = RecordingReleaseStore()
    provider = FakeProvider(batches)
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    return Harness(app, store, provider)


def _abandoned_terminal_event(events: list[Event]) -> Event:
    interrupted = [event for event in events if event.type == EventType.SESSION_INTERRUPTED]
    assert len(interrupted) == 1
    return interrupted[0]


def _assert_turn_completed_before_abandoned_terminal(events: list[Event]) -> None:
    event_types = [event.type for event in events]
    assert event_types.count(EventType.TURN_COMPLETED) == 1
    assert event_types.index(EventType.TURN_COMPLETED) < event_types.index(
        EventType.SESSION_INTERRUPTED
    )
    turn = next(event for event in events if event.type == EventType.TURN_COMPLETED)
    assert turn.payload["status"] == "interrupted"


def test_abandoned_run_stream_finalizes_running_session() -> None:
    # The first provider batch is consumed by the post-abandon resume, proving the
    # abandoned run never reached the model and the session stayed resumable.
    h = _build([_batch("answer after abandonment")])

    async def scenario() -> tuple[Event, list[Event]]:
        stream = h.app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_abandoned_run",
                messages=[Message.text("user", "hello")],
            )
        )
        first_event = await anext(stream)
        assert first_event.type == EventType.SESSION_STARTED
        # The consumer walks away mid-run (e.g. SSE client disconnect).
        await stream.aclose()
        events = await h.store.load_events("sess_abandoned_run")
        return first_event, events

    _, events = asyncio.run(scenario())

    session = asyncio.run(h.store.load("sess_abandoned_run"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    _assert_turn_completed_before_abandoned_terminal(events)
    terminal = _abandoned_terminal_event(events)
    assert terminal.payload == {
        "interruption_type": "runtime_interrupted",
        "reason": "event_stream_closed",
        "abandoned": True,
    }
    assert h.store.release_calls["sess_abandoned_run"] == 1

    # The finalized session is resumable — it was not stranded in RUNNING.
    async def resume() -> None:
        async for _ in h.app.resume(
            ResumeRequest(
                session_id="sess_abandoned_run",
                messages=[Message.text("user", "continue")],
            )
        ):
            pass

    asyncio.run(resume())
    resumed = asyncio.run(h.store.load("sess_abandoned_run"))
    assert resumed is not None
    assert resumed.status == SessionStatus.COMPLETED
    assert len(h.provider.requests) == 1


def test_injected_run_stream_exception_finalizes_before_return() -> None:
    h = _build([_batch("unused")])

    async def scenario() -> None:
        session_id = "sess_injected_run_stream_exception"
        stream = h.app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            )
        )
        first_event = await anext(stream)
        assert first_event.type == EventType.SESSION_STARTED

        injected = RuntimeError("consumer rejected run event")
        try:
            await stream.athrow(injected)
        except RuntimeError as exc:
            assert exc is injected
        else:
            raise AssertionError("Injected stream exception did not propagate.")

        session = await h.store.load(session_id)
        assert session is not None and session.status == SessionStatus.INTERRUPTED
        events = await h.store.load_events(session_id)
        _assert_turn_completed_before_abandoned_terminal(events)
        terminal = _abandoned_terminal_event(events)
        assert terminal.payload["reason"] == "event_stream_closed"
        assert terminal.payload["abandoned"] is True
        assert h.store.release_calls[session_id] == 1

    asyncio.run(scenario())


def test_cleanup_cancellation_remains_authoritative_and_waits_for_release() -> None:
    store = BlockingReleaseStore()
    provider = FakeProvider([_batch("unused")])
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> None:
        session_id = "sess_cleanup_cancelled_after_stream_failure"
        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            )
        )
        first_event = await anext(stream)
        assert first_event.type == EventType.SESSION_STARTED

        injected = RuntimeError("consumer rejected run event")
        throw_task = asyncio.create_task(stream.athrow(injected))
        await asyncio.wait_for(store.release_started.wait(), timeout=5)
        assert throw_task.cancelling() == 0

        throw_task.cancel()
        await asyncio.sleep(0)
        assert throw_task.cancelling() == 1
        assert throw_task.done() is False
        assert store.release_finished.is_set() is False

        store.allow_release.set()
        try:
            await asyncio.wait_for(throw_task, timeout=5)
        except asyncio.CancelledError as cancellation:
            assert cancellation.__cause__ is injected
        else:
            raise AssertionError("Cleanup cancellation did not remain authoritative.")

        assert throw_task.cancelled() is True
        assert throw_task.cancelling() == 1
        assert store.release_finished.is_set() is True
        session = await store.load(session_id)
        assert session is not None and session.status == SessionStatus.INTERRUPTED
        assert store.release_calls[session_id] == 1

    asyncio.run(scenario())


def test_cancellation_requested_before_cleanup_starts_remains_authoritative() -> None:
    store = RecordingReleaseStore()
    provider = FakeProvider([_batch("unused")])
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> None:
        session_id = "sess_cancelled_before_stream_cleanup"
        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            )
        )
        first_event = await anext(stream)
        assert first_event.type == EventType.SESSION_STARTED

        async def cancel_then_close() -> None:
            current_task = asyncio.current_task()
            assert current_task is not None
            current_task.cancel()
            assert current_task.cancelling() == 1
            await stream.aclose()

        close_task = asyncio.create_task(cancel_then_close())
        try:
            await close_task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("Cancellation requested before cleanup was swallowed.")

        assert close_task.cancelled() is True
        assert close_task.cancelling() == 1
        session = await store.load(session_id)
        assert session is not None and session.status == SessionStatus.INTERRUPTED
        assert store.release_calls[session_id] == 1
        assert app._session_control.has_active_tasks(session_id) is False

    asyncio.run(scenario())


def test_previously_delivered_cancellation_is_not_rediscovered_during_cleanup() -> None:
    h = _build([_batch("unused")])

    async def scenario() -> None:
        session_id = "sess_handled_cancellation_before_stream_cleanup"
        stream = h.app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            )
        )
        first_event = await anext(stream)
        assert first_event.type == EventType.SESSION_STARTED

        async def handle_cancellation_then_close() -> None:
            current_task = asyncio.current_task()
            assert current_task is not None
            current_task.cancel()
            with suppress(asyncio.CancelledError):
                await asyncio.sleep(0)
            assert current_task.cancelling() == 1
            await stream.aclose()

        close_task = asyncio.create_task(handle_cancellation_then_close())
        await close_task

        assert close_task.cancelled() is False
        assert close_task.cancelling() == 1
        session = await h.store.load(session_id)
        assert session is not None and session.status == SessionStatus.INTERRUPTED
        assert h.store.release_calls[session_id] == 1
        assert h.app._session_control.has_active_tasks(session_id) is False

    asyncio.run(scenario())


def test_cleanup_cancellation_preserves_release_failure_without_loop_report() -> None:
    session_id = "sess_cleanup_cancelled_with_release_failure"
    store = BlockingReleaseStore(fail_release=True)
    provider = FakeProvider([_batch("unused")])
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        reported_contexts: list[dict[str, object]] = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: reported_contexts.append(context))
        try:
            stream = app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                )
            )
            first_event = await anext(stream)
            assert first_event.type == EventType.SESSION_STARTED

            injected = RuntimeError("consumer rejected run event")
            throw_task = asyncio.create_task(stream.athrow(injected))
            await asyncio.wait_for(store.release_started.wait(), timeout=5)
            throw_task.cancel()
            await asyncio.sleep(0)
            assert throw_task.cancelling() == 1
            assert throw_task.done() is False

            store.allow_release.set()
            try:
                await asyncio.wait_for(throw_task, timeout=5)
            except asyncio.CancelledError as cancellation:
                assert isinstance(cancellation.__cause__, BaseExceptionGroup)
                earlier_failure, cleanup_failure = cancellation.__cause__.exceptions
                assert earlier_failure is injected
                assert isinstance(cleanup_failure, RuntimeError)
                assert str(cleanup_failure) == "run fence release failed"
                assert isinstance(cleanup_failure.__cause__, OSError)
                assert str(cleanup_failure.__cause__) == ("run fence database connection lost")
            else:
                raise AssertionError("Cleanup cancellation did not remain authoritative.")

            await asyncio.sleep(0)
            assert throw_task.cancelled() is True
            assert throw_task.cancelling() == 1
            assert store.release_finished.is_set() is True
            session = await store.load(session_id)
            assert session is not None and session.status == SessionStatus.INTERRUPTED
            assert reported_contexts == []
        finally:
            loop.set_exception_handler(previous_handler)

    asyncio.run(scenario())


def test_wait_for_stream_resumption_retains_stale_run_fence() -> None:
    h = _build([_batch("stale answer must not persist")])

    async def scenario() -> None:
        session_id = "sess_wait_for_stale_run_fence"
        stream = h.app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            )
        )
        first_event = await asyncio.wait_for(anext(stream), timeout=5)
        assert first_event.type == EventType.SESSION_STARTED
        stale_owner = await h.store.load(session_id)
        assert stale_owner is not None

        async def take_over() -> int:
            replacement = await h.store.fence_stalled_run(
                session_id,
                statuses={SessionStatus.RUNNING},
                inactive_before=stale_owner.last_activity_at + timedelta(seconds=1),
            )
            assert replacement is not None
            return replacement.run_epoch

        replacement_epoch = await asyncio.create_task(take_over())
        try:
            try:
                await asyncio.wait_for(anext(stream), timeout=5)
            except SessionRunFenced:
                pass
            else:
                raise AssertionError("Stale stream resumed after run-fence takeover.")
        finally:
            await stream.aclose()

        after = await h.store.load(session_id)
        assert after is not None
        assert after.status == SessionStatus.RUNNING
        assert after.run_epoch == replacement_epoch
        assert not [
            event
            for event in await h.store.load_events(session_id)
            if event.type
            in {
                EventType.MODEL_STARTED,
                EventType.MODEL_COMPLETED,
                EventType.SESSION_COMPLETED,
            }
        ]

    asyncio.run(scenario())


def test_abandoned_resume_stream_finalizes_running_session() -> None:
    h = _build([_batch("first answer")])

    async def scenario() -> None:
        async for _ in h.app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_abandoned_resume",
                messages=[Message.text("user", "hello")],
            )
        ):
            pass
        stream = h.app.resume(
            ResumeRequest(
                session_id="sess_abandoned_resume",
                messages=[Message.text("user", "continue")],
            )
        )
        first_event = await anext(stream)
        assert first_event.type == EventType.SESSION_RESUMED
        await stream.aclose()

    asyncio.run(scenario())

    session = asyncio.run(h.store.load("sess_abandoned_resume"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    events = asyncio.run(h.store.load_events("sess_abandoned_resume"))
    event_types = [event.type for event in events]
    assert event_types.count(EventType.TURN_COMPLETED) == 2
    assert event_types[-2:] == [EventType.TURN_COMPLETED, EventType.SESSION_INTERRUPTED]
    assert events[-2].payload["status"] == "interrupted"
    terminal = _abandoned_terminal_event(events)
    assert terminal.payload["reason"] == "event_stream_closed"
    assert terminal.payload["abandoned"] is True


def test_finalize_abandoned_session_by_id_finalizes_and_is_idempotent() -> None:
    # The shared finalizer that the run-factory-window and tool-approval GeneratorExit
    # guards call to close a session stranded RUNNING before _run_session's own finalizer.
    h = _build([_batch("x")])

    async def scenario() -> None:
        await h.store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_strand",
                messages=[Message.text("user", "hi")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await h.store.transition_status(
            "sess_strand",
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.RUNNING,
        )

        try:
            await h.app._recovery_coordinator.finalize_abandoned_session_by_id("sess_strand")
            first = await h.store.load("sess_strand")
            assert first is not None and first.status == SessionStatus.INTERRUPTED

            # Idempotent: a second call (e.g. also reached by _run_session's finalizer) no-ops.
            await h.app._recovery_coordinator.finalize_abandoned_session_by_id("sess_strand")
            second = await h.store.load("sess_strand")
            assert second is not None and second.status == SessionStatus.INTERRUPTED

            # Unknown session id is a safe no-op.
            await h.app._recovery_coordinator.finalize_abandoned_session_by_id("does-not-exist")

            events = await h.store.load_events("sess_strand")
            interrupted = [e for e in events if e.type == EventType.SESSION_INTERRUPTED]
            assert len(interrupted) == 1
            assert interrupted[0].payload["abandoned"] is True
        finally:
            await h.store.release_run_fence("sess_strand")

    asyncio.run(scenario())


def test_abandoned_session_contains_cancellation_group_from_terminal_cleanup() -> None:
    h = _build([_batch("x")])

    async def scenario() -> None:
        await h.store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_grouped_abandonment",
                messages=[Message.text("user", "hi")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )

        async def fail_terminal(_request):  # type: ignore[no-untyped-def]
            raise BaseExceptionGroup(
                "terminal cleanup cancelled and failed",
                [asyncio.CancelledError(), RuntimeError("cleanup failed")],
            )
            yield  # pragma: no cover

        h.app._recovery_coordinator._emit_terminal_event_with_hooks = fail_terminal
        await h.app._recovery_coordinator.finalize_abandoned_session_by_id(
            "sess_grouped_abandonment"
        )
        session = await h.store.load("sess_grouped_abandonment")
        assert session is not None
        assert session.status is SessionStatus.INTERRUPTED

    asyncio.run(scenario())


def test_finalize_abandoned_session_does_not_require_registered_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h = _build([_batch("x")])

    async def scenario() -> None:
        await h.store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_missing_environment",
                environment_name="retired-environment",
                messages=[Message.text("user", "hi")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await h.store.transition_status(
            "sess_missing_environment",
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.RUNNING,
        )
        original_emit = h.app._recovery_coordinator._event_writer.emit

        async def emit_then_cancel(event: Event) -> Event:
            await original_emit(event)
            raise BaseExceptionGroup(
                "terminal cleanup cancelled and failed",
                [asyncio.CancelledError(), RuntimeError("cleanup failed")],
            )

        monkeypatch.setattr(
            h.app._recovery_coordinator._event_writer,
            "emit",
            emit_then_cancel,
        )

        try:
            await h.app._recovery_coordinator.finalize_abandoned_session_by_id(
                "sess_missing_environment"
            )
            session = await h.store.load("sess_missing_environment")
            assert session is not None
            assert session.status == SessionStatus.INTERRUPTED
            terminal = _abandoned_terminal_event(
                await h.store.load_events("sess_missing_environment")
            )
            assert terminal.payload["reason"] == "event_stream_closed"
            assert terminal.environment_name == "retired-environment"
        finally:
            await h.store.release_run_fence("sess_missing_environment")

    asyncio.run(scenario())


def test_completed_run_stream_close_is_a_no_op() -> None:
    # Closing an already-finished stream must not rewrite the terminal status.
    h = _build([_batch("first answer")])

    async def scenario() -> None:
        stream = h.app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_completed_close",
                messages=[Message.text("user", "hello")],
            )
        )
        async for _ in stream:
            pass
        await stream.aclose()

    asyncio.run(scenario())

    session = asyncio.run(h.store.load("sess_completed_close"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED
    events = asyncio.run(h.store.load_events("sess_completed_close"))
    assert not [event for event in events if event.type == EventType.SESSION_INTERRUPTED]
