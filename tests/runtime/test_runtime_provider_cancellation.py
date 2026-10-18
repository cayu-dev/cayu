"""Provider cancellation contracts."""

from __future__ import annotations

import asyncio
import json
import threading
import traceback
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from tests.provider_cleanup_assertions import without_redacted_cleanup_context
from tests.provider_traceback_assertions import is_cayu_source_filename
from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_session_support import collect_events, collect_interrupt_events

import cayu.providers._credential_boundary as credential_boundary_module
from cayu._exception_groups import iter_exception_tree
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets.billing import BillingIdentity
from cayu.deadlines import ExecutionDeadline, ExecutionDeadlineExceeded
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelRequest, ModelStreamEvent
from cayu.providers.chat_completions import ChatCompletionsProvider
from cayu.providers.deadlines import ProviderStreamDeadlines
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.recovery import IncompleteSessionRecoveryRequest
from cayu.sessions.requests import InterruptSessionRequest, RunRequest


def test_cayu_app_preserves_caller_cancellation_during_provider_stream_cleanup(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    canary = "provider-stream-cleanup-cancel-canary"

    class FailingCloseIterator:
        def __init__(self) -> None:
            self.reads = 0
            self.close_started = asyncio.Event()
            self.cleanup_release = threading.Event()
            self.cleanup_finished = threading.Event()

        def __aiter__(self) -> FailingCloseIterator:
            return self

        async def __anext__(self) -> ModelStreamEvent:
            self.reads += 1
            if self.reads == 1:
                return ModelStreamEvent.text_delta("partial")
            raise RuntimeError(f"provider stream failed near {canary}")

        async def aclose(self) -> None:
            self.close_started.set()

            def physical_cleanup() -> None:
                self.cleanup_release.wait(timeout=5)
                self.cleanup_finished.set()
                raise RuntimeError(f"cleanup failed near {canary}")

            await asyncio.to_thread(physical_cleanup)

    class StreamCollisionProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.iterator = FailingCloseIterator()
            self.requests = 0

        def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            self.requests += 1
            return self.iterator

    async def run() -> None:
        provider = StreamCollisionProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        live_events: list[Event] = []

        async def consume() -> None:
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_stream_cleanup_cancel",
                    messages=[Message.text("user", "hello")],
                )
            ):
                live_events.append(event)

        task = asyncio.create_task(consume())
        await asyncio.wait_for(provider.iterator.close_started.wait(), timeout=10)
        task.cancel("caller cancelled provider cleanup")
        assert task.cancelling() == 1
        caught = False
        caught_cancellation: asyncio.CancelledError | None = None
        try:
            await task
        except asyncio.CancelledError as exc:
            caught = True
            caught_cancellation = exc
            assert BaseException.__dict__["args"].__get__(exc, BaseException) == (
                "Provider operation cancelled",
            )
        assert caught is True
        assert caught_cancellation is not None
        assert caught_cancellation.__cause__ is None
        assert caught_cancellation.__context__ is None
        retained_cancellation = (
            str(caught_cancellation) + repr(caught_cancellation) + repr(vars(caught_cancellation))
        )
        assert canary not in retained_cancellation
        captured = traceback.TracebackException.from_exception(
            caught_cancellation,
            capture_locals=True,
        )
        cayu_frames = [frame for frame in captured.stack if is_cayu_source_filename(frame.filename)]
        assert canary not in repr([(frame.name, frame.locals) for frame in cayu_frames])
        assert task.cancelling() == 1
        assert task.cancelled() is True
        assert provider.iterator.cleanup_finished.is_set() is False
        provider.iterator.cleanup_release.set()
        assert provider.requests == 1
        assert EventType.MODEL_ERROR not in {event.type for event in live_events}
        assert EventType.SESSION_FAILED not in {event.type for event in live_events}

        session_id = "sess_stream_cleanup_cancel"
        session = await app.session_store.load(session_id)
        assert session is not None
        assert session.status == SessionStatus.INTERRUPTED
        records = await app.session_store.query_events(EventQuery(session_id=session_id, limit=100))
        events = [record.event for record in records]
        event_types = {event.type for event in events}
        assert EventType.MODEL_ERROR not in event_types
        assert EventType.SESSION_FAILED not in event_types
        interrupted_events = [
            event for event in events if event.type == EventType.SESSION_INTERRUPTED
        ]
        assert len(interrupted_events) == 1
        interrupted = interrupted_events[0]
        assert [
            without_redacted_cleanup_context(failure)
            for failure in interrupted.payload["provider_cancellation_failures"]
        ] == [
            {
                "phase": "model_stream",
                "error": "Model provider stream failed before cancellation.",
                "error_type": "ModelProviderStreamError",
            },
            {
                "phase": "provider_stream_cleanup",
                "error": "Provider stream cleanup did not complete normally.",
                "error_type": "ProviderStreamCleanupError",
                "cleanup_diagnostic_version": 1,
                "cleanup_action": "stream_close",
                "cleanup_reason": "cleanup_cancelled",
                "cleanup_exception_type": "CancelledError",
                "cleanup_failure_phase": "close",
                "cleanup_close_exception_type": "CancelledError",
                "cleanup_close_exception_message": "redacted",
                "cancellation_requested": True,
                "stream_close_state": "not_confirmed",
                "remote_cancellation_state": "unknown",
                "remote_settlement_state": "unknown",
                **{
                    key: next(
                        record.event.payload[key]
                        for record in records
                        if record.event.type is EventType.MODEL_STARTED
                    )
                    for key in ("model_step_id", "model_attempt_id")
                },
            },
        ]
        retained = json.dumps(
            [event.model_dump(mode="json") for event in events],
            sort_keys=True,
        )
        assert canary not in retained
        captured_output = capsys.readouterr()
        assert canary not in caplog.text
        assert canary not in captured_output.out
        assert canary not in captured_output.err
        assert all(canary not in str(warning.message) for warning in recwarn)

        provider.iterator.cleanup_release.set()
        await asyncio.wait_for(
            asyncio.to_thread(provider.iterator.cleanup_finished.wait),
            timeout=2,
        )

    asyncio.run(run())


def test_public_stream_abandonment_closes_provider_stream() -> None:
    class CloseTrackedIterator:
        def __init__(self) -> None:
            self.reads = 0
            self.closed = asyncio.Event()

        def __aiter__(self) -> CloseTrackedIterator:
            return self

        async def __anext__(self) -> ModelStreamEvent:
            self.reads += 1
            if self.reads == 1:
                return ModelStreamEvent.text_delta("partial")
            await asyncio.Event().wait()
            raise AssertionError("provider stream unexpectedly resumed")

        async def aclose(self) -> None:
            self.closed.set()

    class CloseTrackedProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.iterator = CloseTrackedIterator()

        def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            return self.iterator

    async def run() -> tuple[SessionStatus, bool]:
        provider = CloseTrackedProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_public_abandonment_provider_cleanup",
                messages=[Message.text("user", "hello")],
            )
        )

        while True:
            event = await anext(stream)
            if event.type is EventType.MODEL_TEXT_DELTA:
                break
        await stream.aclose()
        await asyncio.wait_for(provider.iterator.closed.wait(), timeout=1)

        session = await app.session_store.load("sess_public_abandonment_provider_cleanup")
        assert session is not None
        return session.status, provider.iterator.closed.is_set()

    status, closed = asyncio.run(run())

    assert status is SessionStatus.INTERRUPTED
    assert closed is True


@pytest.mark.parametrize("close_fails", [False, True])
def test_successful_provider_cleanup_race_is_not_published_as_cleanup_failure(
    close_fails: bool,
) -> None:
    class SuccessfulRacingCloseIterator:
        def __init__(self) -> None:
            self.reads = 0
            self.owner: asyncio.Task[list[Event]] | None = None
            self.close_finished = asyncio.Event()

        def __aiter__(self) -> SuccessfulRacingCloseIterator:
            return self

        async def __anext__(self) -> ModelStreamEvent:
            self.reads += 1
            if self.reads == 1:
                return ModelStreamEvent.text_delta("partial")
            raise RuntimeError("authoritative provider failure")

        async def aclose(self) -> None:
            assert self.owner is not None
            self.owner.cancel("caller cancelled as provider cleanup completed")
            self.close_finished.set()
            if close_fails:
                raise RuntimeError("settled close failure")

    class SuccessfulRacingCloseProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.iterator = SuccessfulRacingCloseIterator()

        def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.requests.append(request)
            return self.iterator

    async def run() -> tuple[tuple[object, ...], int, bool, list[Event], bool]:
        provider = SuccessfulRacingCloseProvider()
        store = InMemorySessionStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_successful_provider_cleanup_race",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
        provider.iterator.owner = run_task

        with pytest.raises(asyncio.CancelledError) as exc_info:
            await run_task
        stored_events = await store.load_events("sess_successful_provider_cleanup_race")
        return (
            exc_info.value.args,
            run_task.cancelling(),
            run_task.cancelled(),
            stored_events,
            provider.iterator.close_finished.is_set(),
        )

    cancellation_args, cancelling, cancelled, stored_events, close_finished = asyncio.run(run())

    assert cancellation_args == ("Provider operation cancelled",)
    assert cancelling == 1
    assert cancelled is True
    assert close_finished is True
    assert EventType.MODEL_ERROR not in {event.type for event in stored_events}
    assert EventType.SESSION_FAILED not in {event.type for event in stored_events}
    interrupted = [event for event in stored_events if event.type is EventType.SESSION_INTERRUPTED]
    assert len(interrupted) == 1
    failures = interrupted[0].payload["provider_cancellation_failures"]
    assert failures[:1] == [
        {
            "phase": "model_stream",
            "error": "Model provider stream failed before cancellation.",
            "error_type": "ModelProviderStreamError",
        }
    ]

    if close_fails:
        assert len(failures) == 2
        assert failures[1]["phase"] == "provider_stream_cleanup"
        assert failures[1]["error_type"] == "ProviderStreamCleanupError"
    else:
        assert len(failures) == 1


def test_provider_cancellation_marker_precedes_operator_terminal_transition() -> None:
    class MarkerOrderingStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.marker_observed_before_terminal_transition = False

        async def publish_interaction_transition(self, session_id: str, **kwargs: Any):
            if kwargs.get("to_status") is SessionStatus.INTERRUPTED:
                checkpoint = await self.load_checkpoint(session_id)
                marker = None if checkpoint is None else checkpoint.get("pending_session_interrupt")
                if type(marker) is dict and marker.get("provider_cancellation_failures"):
                    self.marker_observed_before_terminal_transition = True
            return await super().publish_interaction_transition(session_id, **kwargs)

    class FailingCloseIterator:
        def __init__(self) -> None:
            self.reads = 0
            self.close_started = asyncio.Event()
            self.cleanup_release = asyncio.Event()
            self.cleanup_finished = asyncio.Event()
            self.cleanup_cancelled = asyncio.Event()

        def __aiter__(self) -> FailingCloseIterator:
            return self

        async def __anext__(self) -> ModelStreamEvent:
            self.reads += 1
            if self.reads == 1:
                return ModelStreamEvent.text_delta("partial")
            raise RuntimeError("private provider failure")

        async def aclose(self) -> None:
            self.close_started.set()
            try:
                await self.cleanup_release.wait()
            except asyncio.CancelledError:
                self.cleanup_cancelled.set()
                raise
            self.cleanup_finished.set()
            raise RuntimeError("private provider cleanup failure") from None

    class FailingCloseProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.iterator = FailingCloseIterator()

        def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            return self.iterator

    async def run() -> None:
        session_id = "sess_operator_provider_marker_ordering"
        store = MarkerOrderingStore()
        provider = FailingCloseProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        run_task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
        await asyncio.wait_for(provider.iterator.close_started.wait(), timeout=10)
        interrupt_events = await collect_interrupt_events(
            app,
            InterruptSessionRequest(session_id=session_id, reason="operator stop"),
        )
        run_events = await asyncio.wait_for(run_task, timeout=2)

        assert store.marker_observed_before_terminal_transition is True
        assert len(interrupt_events) == 1
        assert interrupt_events[0].type is EventType.SESSION_INTERRUPTED
        assert run_events[-1].id == interrupt_events[0].id
        assert [
            without_redacted_cleanup_context(failure)
            for failure in interrupt_events[0].payload["provider_cancellation_failures"]
        ] == [
            {
                "phase": "model_stream",
                "error": "Model provider stream failed before cancellation.",
                "error_type": "ModelProviderStreamError",
            },
            {
                "phase": "provider_stream_cleanup",
                "error": "Provider stream cleanup did not complete normally.",
                "error_type": "ProviderStreamCleanupError",
                "cleanup_diagnostic_version": 1,
                "cleanup_action": "stream_close",
                "cleanup_reason": "cleanup_cancelled",
                "cleanup_exception_type": "CancelledError",
                "cleanup_failure_phase": "close",
                "cleanup_close_exception_type": "CancelledError",
                "cleanup_close_exception_message": "redacted",
                "cancellation_requested": True,
                "stream_close_state": "not_confirmed",
                "remote_cancellation_state": "unknown",
                "remote_settlement_state": "unknown",
            },
        ]
        assert provider.iterator.cleanup_finished.is_set() is False
        provider.iterator.cleanup_release.set()
        assert provider.iterator.cleanup_cancelled.is_set()

    asyncio.run(run())


def test_cayu_app_does_not_report_caller_stream_read_cancellation_as_provider_failure() -> None:
    class BlockingIterator:
        def __init__(self) -> None:
            self.read_started = asyncio.Event()
            self.close_calls = 0

        def __aiter__(self) -> BlockingIterator:
            return self

        async def __anext__(self) -> ModelStreamEvent:
            self.read_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def aclose(self) -> None:
            self.close_calls += 1

    class BlockingProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.iterator = BlockingIterator()
            self.requests = 0

        def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            self.requests += 1
            return self.iterator

    async def run() -> None:
        provider = BlockingProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        async def consume() -> None:
            async for _event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_stream_read_cancel",
                    messages=[Message.text("user", "hello")],
                )
            ):
                pass

        task = asyncio.create_task(consume())
        await asyncio.wait_for(provider.iterator.read_started.wait(), timeout=10)
        task.cancel("caller cancelled provider read")
        assert task.cancelling() == 1
        caught_cancellation: asyncio.CancelledError | None = None
        try:
            await task
        except asyncio.CancelledError as exc:
            caught_cancellation = exc
        assert provider.requests == 1
        assert provider.iterator.close_calls == 1

        records = await app.session_store.query_events(
            EventQuery(session_id="sess_stream_read_cancel", limit=100)
        )
        events = [record.event for record in records]
        interrupted = [event for event in events if event.type == EventType.SESSION_INTERRUPTED]
        assert caught_cancellation is not None, [event.model_dump(mode="json") for event in events]
        assert str(caught_cancellation) == "Provider operation cancelled"
        assert task.cancelling() == 1
        assert task.cancelled() is True
        assert len(interrupted) == 1
        failures = interrupted[0].payload.get("provider_cancellation_failures", [])
        assert failures == []

    asyncio.run(run())


def test_cayu_app_rejects_provider_dispatch_when_live_cleanup_capacity_is_exhausted() -> None:
    async def run() -> None:
        held = credential_boundary_module.reserve_provider_stream_cleanup(1)

        class CapacityProvider(FakeProvider):
            @property
            def stream_deadlines(self) -> ProviderStreamDeadlines:
                return ProviderStreamDeadlines(max_concurrent_streams=1)

        provider = CapacityProvider([ModelStreamEvent.completed({})])
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        try:
            events = await collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_provider_cleanup_capacity",
                    messages=[Message.text("user", "hello")],
                ),
            )
        finally:
            credential_boundary_module.release_provider_stream_cleanup(held)

        assert provider.requests == []
        assert events[-1].type is EventType.SESSION_FAILED

    asyncio.run(run())


def test_cayu_app_does_not_publish_provider_replacement_cancellation_text(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    canary = "provider-replaced-caller-cancellation-canary"

    class ReplacingIterator:
        def __init__(self) -> None:
            self.read_started = asyncio.Event()

        def __aiter__(self) -> ReplacingIterator:
            return self

        async def __anext__(self) -> ModelStreamEvent:
            self.read_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise asyncio.CancelledError(canary) from None
            raise AssertionError("unreachable")

        async def aclose(self) -> None:
            return None

    class ReplacingProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.iterator = ReplacingIterator()

        def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            return self.iterator

    async def run() -> None:
        provider = ReplacingProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        async def consume() -> None:
            async for _ in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_provider_replaced_cancellation",
                    messages=[Message.text("user", "hello")],
                )
            ):
                pass

        task = asyncio.create_task(consume())
        await asyncio.wait_for(provider.iterator.read_started.wait(), timeout=10)
        task.cancel("caller cancellation")
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        assert raised.value.args == ("Provider operation cancelled",)
        assert task.cancelling() == 1
        assert task.cancelled()
        retained = str(raised.value) + repr(raised.value) + repr(vars(raised.value))
        assert canary not in retained
        assert raised.value.__cause__ is None
        assert raised.value.__context__ is None

    asyncio.run(run())
    captured = capsys.readouterr()
    assert canary not in caplog.text
    assert canary not in captured.out
    assert canary not in captured.err
    assert all(canary not in str(warning.message) for warning in recwarn)


@pytest.mark.parametrize("replacement_kind", ["error", "group", "value"])
def test_cayu_app_preserves_caller_cancellation_when_provider_replaces_its_type(
    replacement_kind: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    canary = f"provider-replaced-cancellation-type-{replacement_kind}-canary"

    class ReplacingIterator:
        def __init__(self) -> None:
            self.read_started = asyncio.Event()
            self.close_calls = 0

        def __aiter__(self) -> ReplacingIterator:
            return self

        async def __anext__(self) -> ModelStreamEvent:
            self.read_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if replacement_kind == "value":
                    return ModelStreamEvent.text_delta(canary)
                failure = RuntimeError(canary)
                if replacement_kind == "group":
                    raise BaseExceptionGroup(canary, [failure]) from None
                raise failure from None
            raise AssertionError("unreachable")

        async def aclose(self) -> None:
            self.close_calls += 1

    class ReplacingProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.iterator = ReplacingIterator()
            self.requests = 0

        def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            self.requests += 1
            return self.iterator

    async def run() -> None:
        session_id = f"sess_provider_replaced_cancellation_type_{replacement_kind}"
        provider = ReplacingProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        live_events: list[Event] = []

        async def consume() -> None:
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                )
            ):
                live_events.append(event)

        task = asyncio.create_task(consume())
        await asyncio.wait_for(provider.iterator.read_started.wait(), timeout=10)
        task.cancel("caller cancellation")
        with pytest.raises(asyncio.CancelledError) as raised:
            await task

        assert raised.value.args == ("Provider operation cancelled",)
        assert raised.value.__cause__ is None
        assert raised.value.__context__ is None
        assert task.cancelling() == 1
        assert task.cancelled()
        assert provider.requests == 1
        assert provider.iterator.close_calls == 1
        assert EventType.MODEL_ERROR not in {event.type for event in live_events}
        assert EventType.SESSION_FAILED not in {event.type for event in live_events}

        session = await app.session_store.load(session_id)
        assert session is not None
        assert session.status is SessionStatus.INTERRUPTED
        records = await app.session_store.query_events(EventQuery(session_id=session_id, limit=100))
        events = [record.event for record in records]
        interrupted = [event for event in events if event.type is EventType.SESSION_INTERRUPTED]
        assert len(interrupted) == 1
        started = next(event for event in events if event.type is EventType.MODEL_STARTED)
        expected_failures = [
            {
                "phase": "provider_stream_cleanup",
                "error": "Provider stream cleanup did not complete normally.",
                "error_type": "ProviderStreamCleanupError",
                "cleanup_diagnostic_version": 1,
                "cleanup_action": "stream_close",
                "cleanup_reason": "read_exception",
                "cleanup_exception_type": "unknown"
                if replacement_kind == "group"
                else "RuntimeError",
                "cleanup_failure_phase": "read",
                "cleanup_read_exception_type": "unknown"
                if replacement_kind == "group"
                else "RuntimeError",
                "cleanup_read_exception_message": "redacted",
                "cancellation_requested": True,
                "stream_close_state": "confirmed",
                "remote_cancellation_state": "unknown",
                "remote_settlement_state": "unknown",
                "model_step_id": started.payload["model_step_id"],
                "model_attempt_id": started.payload["model_attempt_id"],
            }
        ]
        assert [
            without_redacted_cleanup_context(failure)
            for failure in interrupted[0].payload.get("provider_cancellation_failures", [])
        ] == (expected_failures)
        assert EventType.MODEL_ERROR not in {event.type for event in events}
        assert EventType.SESSION_FAILED not in {event.type for event in events}
        assert canary not in json.dumps(
            [event.model_dump(mode="json") for event in events],
            sort_keys=True,
        )

    asyncio.run(run())
    captured = capsys.readouterr()
    assert canary not in caplog.text
    assert canary not in captured.out
    assert canary not in captured.err
    assert all(canary not in str(warning.message) for warning in recwarn)


def test_cayu_app_preserves_caller_cancellation_across_late_explicit_close_failure(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    canary = "provider-late-explicit-close-failure-canary"

    class LateFailingCloseIterator:
        def __init__(self) -> None:
            self.read_started = asyncio.Event()
            self.close_calls = 0

        def __aiter__(self) -> LateFailingCloseIterator:
            return self

        async def __anext__(self) -> ModelStreamEvent:
            self.read_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise asyncio.CancelledError(canary) from None
            raise AssertionError("unreachable")

        async def aclose(self) -> None:
            self.close_calls += 1
            raise RuntimeError(canary)

    class LateFailingCloseProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.iterator = LateFailingCloseIterator()

        def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            return self.iterator

    async def run() -> None:
        session_id = "sess_provider_late_explicit_close_failure"
        provider = LateFailingCloseProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
        await asyncio.wait_for(provider.iterator.read_started.wait(), timeout=10)
        task.cancel("caller cancellation")
        with pytest.raises(asyncio.CancelledError) as raised:
            await task

        assert raised.value.args == ("Provider operation cancelled",)
        assert raised.value.__cause__ is None
        assert raised.value.__context__ is None
        assert task.cancelling() == 1
        assert task.cancelled()
        assert provider.iterator.close_calls == 1
        records = await app.session_store.query_events(EventQuery(session_id=session_id, limit=100))
        events = [record.event for record in records]
        interrupted = [event for event in events if event.type is EventType.SESSION_INTERRUPTED]
        assert len(interrupted) == 1
        assert [
            without_redacted_cleanup_context(failure)
            for failure in interrupted[0].payload["provider_cancellation_failures"]
        ] == [
            {
                "phase": "provider_stream_cleanup",
                "error": "Provider stream cleanup did not complete normally.",
                "error_type": "ProviderStreamCleanupError",
                "cleanup_diagnostic_version": 1,
                "cleanup_action": "stream_close",
                "cleanup_reason": "close_exception",
                "cleanup_exception_type": "RuntimeError",
                "cleanup_failure_phase": "close",
                "cleanup_close_exception_type": "RuntimeError",
                "cleanup_close_exception_message": "redacted",
                "cancellation_requested": True,
                "stream_close_state": "not_confirmed",
                "remote_cancellation_state": "unknown",
                "remote_settlement_state": "unknown",
                **{
                    key: next(
                        record.event.payload[key]
                        for record in records
                        if record.event.type is EventType.MODEL_STARTED
                    )
                    for key in ("model_step_id", "model_attempt_id")
                },
            }
        ]
        assert EventType.MODEL_ERROR not in {event.type for event in events}
        assert EventType.SESSION_FAILED not in {event.type for event in events}
        assert canary not in json.dumps(
            [event.model_dump(mode="json") for event in events],
            sort_keys=True,
        )

    asyncio.run(run())
    captured = capsys.readouterr()
    assert canary not in caplog.text
    assert canary not in captured.out
    assert canary not in captured.err
    assert all(canary not in str(warning.message) for warning in recwarn)


@pytest.mark.parametrize("boundary", ["stream", "billing"])
def test_cayu_app_does_not_authenticate_provider_group_cancellation_without_task_request(
    boundary: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    canary = f"provider-fatal-group-cancellation-{boundary}-canary"

    class FatalGroupProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.cancellation_count: int | None = None

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity | None:
            del request
            if boundary != "billing":
                return None
            task = asyncio.current_task()
            self.cancellation_count = None if task is None else task.cancelling()
            raise BaseExceptionGroup(
                canary,
                [SystemExit(17), asyncio.CancelledError(canary)],
            )

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            if boundary != "stream":
                raise AssertionError("billing failure should precede stream dispatch")
            task = asyncio.current_task()
            self.cancellation_count = None if task is None else task.cancelling()
            raise BaseExceptionGroup(
                canary,
                [SystemExit(17), asyncio.CancelledError(canary)],
            )
            yield  # pragma: no cover

    session_id = f"sess_provider_fatal_group_cancellation_{boundary}"
    provider = FatalGroupProvider()
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    with pytest.raises(BaseExceptionGroup) as raised:
        asyncio.run(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                ),
            )
        )

    leaves = [
        candidate
        for candidate in iter_exception_tree(raised.value)
        if not isinstance(candidate, BaseExceptionGroup)
    ]
    system_exits = [candidate for candidate in leaves if isinstance(candidate, SystemExit)]
    assert len(system_exits) == 1
    assert system_exits[0].code == 17
    assert all(not isinstance(candidate, asyncio.CancelledError) for candidate in leaves)
    assert provider.cancellation_count == 0
    assert canary not in repr(raised.value)

    session = asyncio.run(app.session_store.load(session_id))
    assert session is not None
    assert session.status is not SessionStatus.INTERRUPTED
    records = asyncio.run(
        app.session_store.query_events(EventQuery(session_id=session_id, limit=100))
    )
    retained = json.dumps(
        [record.event.model_dump(mode="json") for record in records],
        sort_keys=True,
    )
    assert EventType.SESSION_INTERRUPTED not in {record.event.type for record in records}
    assert canary not in retained
    captured = capsys.readouterr()
    assert canary not in caplog.text
    assert canary not in captured.out
    assert canary not in captured.err
    assert all(canary not in str(warning.message) for warning in recwarn)


def test_cayu_app_classifies_nonfatal_provider_group_cancellation_as_failure(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    canary = "provider-nonfatal-group-cancellation-canary"

    class GroupCancellingProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.cancellation_count: int | None = None
            self.requests = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            self.requests += 1
            task = asyncio.current_task()
            self.cancellation_count = None if task is None else task.cancelling()
            raise BaseExceptionGroup(
                canary,
                [
                    asyncio.CancelledError(canary),
                    RuntimeError(canary),
                ],
            )
            yield  # pragma: no cover

    session_id = "sess_provider_nonfatal_group_cancellation"
    provider = GroupCancellingProvider()
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert provider.requests == 1
    assert provider.cancellation_count == 0
    assert events[-1].type is EventType.SESSION_FAILED
    assert EventType.SESSION_INTERRUPTED not in {event.type for event in events}
    retained = json.dumps(
        [event.model_dump(mode="json") for event in events],
        sort_keys=True,
    )
    assert canary not in retained
    captured = capsys.readouterr()
    assert canary not in caplog.text
    assert canary not in captured.out
    assert canary not in captured.err
    assert all(canary not in str(warning.message) for warning in recwarn)


@pytest.mark.parametrize("stage", ["construction", "iterator", "iteration"])
def test_cayu_app_classifies_provider_created_stream_cancellation_as_failure(
    stage: str,
) -> None:
    canary = "provider-created-stream-cancellation-canary"

    class CancellingIterator:
        def __aiter__(self) -> CancellingIterator:
            raise asyncio.CancelledError(canary)

        async def __anext__(self) -> ModelStreamEvent:
            raise AssertionError("iterator cancellation must occur before iteration")

        async def aclose(self) -> None:
            return None

    class CancellingProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.requests = 0

        def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            self.requests += 1
            if stage == "construction":
                raise asyncio.CancelledError(canary)
            if stage == "iterator":
                return CancellingIterator()

            async def events() -> AsyncIterator[ModelStreamEvent]:
                raise asyncio.CancelledError(canary)
                yield  # pragma: no cover

            return events()

    provider = CancellingProvider()
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    events = asyncio.run(
        collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_provider_created_stream_cancellation",
                messages=[Message.text("user", "hello")],
            ),
        )
    )

    assert provider.requests == 1
    assert events[-1].type is EventType.SESSION_FAILED
    assert EventType.SESSION_INTERRUPTED not in {event.type for event in events}
    assert events[-1].payload["error"] == "Model provider stream cancelled itself."
    assert canary not in json.dumps(
        [event.model_dump(mode="json") for event in events],
        sort_keys=True,
    )

    error = next(e for e in events if e.type is EventType.MODEL_ERROR)
    assert error.payload["retry_suppression"] == "cancellation"
    assert error.payload["retry"] is False


def test_chat_completions_adapter_preserves_nested_cancellation_diagnostics() -> None:
    class BlockingRawEvents:
        def __init__(self) -> None:
            self.reads = 0
            self.close_started = asyncio.Event()
            self.cleanup_release = asyncio.Event()
            self.cleanup_finished = asyncio.Event()

        def __aiter__(self) -> BlockingRawEvents:
            return self

        async def __anext__(self) -> dict[str, Any]:
            self.reads += 1
            if self.reads == 1:
                return {
                    "id": "chatcmpl-cancellation-diagnostics",
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "partial"},
                            "finish_reason": None,
                        }
                    ],
                }
            raise RuntimeError("private transport failure")

        async def aclose(self) -> None:
            self.close_started.set()
            await self.cleanup_release.wait()
            self.cleanup_finished.set()

    class BlockingTransport:
        def __init__(self) -> None:
            self.events = BlockingRawEvents()
            self.requests = 0

        def stream_chat_completions(self, **_kwargs: Any) -> BlockingRawEvents:
            self.requests += 1
            return self.events

    async def run() -> None:
        transport = BlockingTransport()
        provider = ChatCompletionsProvider(api_key="test-key", transport=transport)
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="test-model"))

        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_chat_nested_cancel_diagnostics",
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
        await asyncio.wait_for(transport.events.close_started.wait(), timeout=10)
        task.cancel("caller cancelled chat cleanup")
        with pytest.raises(asyncio.CancelledError):
            await task

        records = await app.session_store.query_events(
            EventQuery(session_id="sess_chat_nested_cancel_diagnostics", limit=100)
        )
        interrupted = [
            record.event for record in records if record.event.type == EventType.SESSION_INTERRUPTED
        ]
        started = next(
            record.event for record in records if record.event.type is EventType.MODEL_STARTED
        )
        assert transport.requests == 1
        assert len(interrupted) == 1
        assert [
            without_redacted_cleanup_context(failure)
            for failure in interrupted[0].payload["provider_cancellation_failures"]
        ] == [
            {
                "phase": "model_stream",
                "error": "Model provider stream failed before cancellation.",
                "error_type": "ModelProviderStreamError",
            },
            {
                "phase": "provider_stream_cleanup",
                "error": "Provider stream cleanup did not complete normally.",
                "error_type": "ProviderStreamCleanupError",
                "cleanup_diagnostic_version": 1,
                "cleanup_action": "stream_close",
                "cleanup_reason": "cleanup_cancelled",
                "cleanup_exception_type": "CancelledError",
                "cleanup_failure_phase": "close",
                "cleanup_close_exception_type": "CancelledError",
                "cleanup_close_exception_message": "redacted",
                "cancellation_requested": True,
                "stream_close_state": "not_confirmed",
                "remote_cancellation_state": "unknown",
                "remote_settlement_state": "unknown",
                "model_step_id": started.payload["model_step_id"],
                "model_attempt_id": started.payload["model_attempt_id"],
            },
        ]
        assert transport.events.cleanup_finished.is_set() is False
        transport.events.cleanup_release.set()

    asyncio.run(run())


def test_cayu_app_preserves_caller_cancellation_across_billing_task_group_failure(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    canary = "provider-billing-task-group-cancel-canary"

    class BillingCollisionProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.hook_started = asyncio.Event()
            self.release_cleanup = asyncio.Event()
            self.cleanup_failed = asyncio.Event()
            self.requests = 0

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity | None:
            del request

            async def child() -> None:
                await self.release_cleanup.wait()
                self.cleanup_failed.set()
                raise RuntimeError(f"billing cleanup failed near {canary}") from None

            async with asyncio.TaskGroup() as group:
                group.create_task(child())
                self.hook_started.set()
                await asyncio.Event().wait()
            return None

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            self.requests += 1
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    async def run() -> None:
        provider = BillingCollisionProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        live_events: list[Event] = []

        async def consume() -> None:
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_billing_task_group_cancel",
                    messages=[Message.text("user", "hello")],
                )
            ):
                live_events.append(event)

        task = asyncio.create_task(consume())
        await asyncio.wait_for(provider.hook_started.wait(), timeout=10)
        task.cancel("caller cancelled billing")
        provider.release_cleanup.set()
        assert task.cancelling() == 1
        caught = False
        caught_cancellation: asyncio.CancelledError | None = None
        try:
            await task
        except asyncio.CancelledError as exc:
            caught = True
            caught_cancellation = exc
            assert BaseException.__dict__["args"].__get__(exc, BaseException) == (
                "caller cancelled billing",
            )
        assert caught is True
        assert caught_cancellation is not None
        assert caught_cancellation.__cause__ is None
        assert caught_cancellation.__context__ is None
        retained_cancellation = (
            str(caught_cancellation) + repr(caught_cancellation) + repr(vars(caught_cancellation))
        )
        assert canary not in retained_cancellation
        captured = traceback.TracebackException.from_exception(
            caught_cancellation,
            capture_locals=True,
        )
        cayu_frames = [frame for frame in captured.stack if is_cayu_source_filename(frame.filename)]
        assert canary not in repr([(frame.name, frame.locals) for frame in cayu_frames])
        assert task.cancelling() == 1
        assert task.cancelled() is True
        assert provider.cleanup_failed.is_set()
        assert provider.requests == 0
        assert EventType.MODEL_ERROR not in {event.type for event in live_events}
        assert EventType.SESSION_FAILED not in {event.type for event in live_events}

        session = await app.session_store.load("sess_billing_task_group_cancel")
        assert session is not None
        assert session.status == SessionStatus.INTERRUPTED
        records = await app.session_store.query_events(EventQuery(session_id=session.id, limit=100))
        events = [record.event for record in records]
        event_types = {event.type for event in events}
        assert EventType.MODEL_ERROR not in event_types
        assert EventType.SESSION_FAILED not in event_types
        interrupted_events = [
            event for event in events if event.type == EventType.SESSION_INTERRUPTED
        ]
        assert len(interrupted_events) == 1
        interrupted = interrupted_events[0]
        assert interrupted.payload["provider_cancellation_failures"] == [
            {
                "phase": "billing_identity_for_request",
                "error": ("Provider billing identity cleanup failed during caller cancellation."),
                "error_type": "BillingIdentityCleanupError",
            }
        ]
        retained = json.dumps(
            [event.model_dump(mode="json") for event in events],
            sort_keys=True,
        )
        assert canary not in retained
        captured_output = capsys.readouterr()
        assert canary not in caplog.text
        assert canary not in captured_output.out
        assert canary not in captured_output.err
        assert all(canary not in str(warning.message) for warning in recwarn)

    asyncio.run(run())


@pytest.mark.parametrize("expired", [False, True])
def test_provider_cancellation_publication_failure_remains_repairable(
    monkeypatch: pytest.MonkeyPatch,
    expired: bool,
) -> None:
    from cayu.deadlines import _TIMERS

    class BillingCleanupFailureProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.hook_started = asyncio.Event()
            self.release_cleanup = asyncio.Event()

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity | None:
            del request
            self.deadline_timers = tuple(timer for timer, _ in _TIMERS.get())

            async def child() -> None:
                await self.release_cleanup.wait()
                raise RuntimeError("private billing cleanup failure") from None

            async with asyncio.TaskGroup() as group:
                group.create_task(child())
                self.hook_started.set()
                await asyncio.Event().wait()
            return None

    async def run() -> None:
        session_id = "sess_provider_cancel_terminal_repair"
        provider = BillingCleanupFailureProvider()
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        original_terminal_stream = app._terminal_event_publication.publish_recovered

        async def fail_interrupted_publication(request):  # type: ignore[no-untyped-def]
            if request.event.type == EventType.SESSION_INTERRUPTED:
                raise RuntimeError("private interrupted publication failure")
            async for event in original_terminal_stream(request):
                yield event

        monkeypatch.setattr(
            app._terminal_event_publication,
            "publish_recovered",
            fail_interrupted_publication,
        )
        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                    execution_deadline=ExecutionDeadline.after(3600)
                    if expired
                    else ExecutionDeadline(),
                ),
            )
        )
        await asyncio.wait_for(provider.hook_started.wait(), timeout=10)
        if expired:
            expired_wall = datetime.now(UTC) + timedelta(seconds=3601)

            class ExpiredClock(datetime):
                @classmethod
                def now(cls, tz=None):
                    return expired_wall

            monkeypatch.setattr("cayu.deadlines.datetime", ExpiredClock)
            for timer in provider.deadline_timers:
                if timer.when() is not None:
                    timer.reschedule(asyncio.get_running_loop().time())
            await asyncio.sleep(0)
        else:
            task.cancel("caller cancelled billing")
        provider.release_cleanup.set()
        with pytest.raises(
            ExecutionDeadlineExceeded if expired else asyncio.CancelledError
        ) as raised:
            await task

        if not expired:
            assert str(raised.value) == "caller cancelled billing"
            assert type(raised.value.__cause__) is RuntimeError
            assert str(raised.value.__cause__) == (
                "Session interruption finalization failed after provider cancellation."
            )
        session = await app.session_store.load(session_id)
        assert session is not None
        assert session.status is SessionStatus.INTERRUPTED
        checkpoint = await app.session_store.load_checkpoint(session_id)
        assert checkpoint is not None
        pending = checkpoint["pending_session_interrupt"]
        assert pending["interruption_type"] == (
            "limit_reached" if expired else "runtime_interrupted"
        )
        assert pending["provider_cancellation_failures"] == [
            {
                "phase": "billing_identity_for_request",
                "error": "Provider billing identity cleanup failed during caller cancellation.",
                "error_type": "BillingIdentityCleanupError",
            }
        ]
        assert type(pending["interruption_request_id"]) is str
        assert not any(
            event.type == EventType.SESSION_INTERRUPTED
            for event in await app.session_store.load_events(session_id)
        )

        monkeypatch.setattr(
            app._terminal_event_publication,
            "publish_recovered",
            original_terminal_stream,
        )
        recovered = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        interrupted = next(
            event for event in recovered.events if event.type is EventType.SESSION_INTERRUPTED
        )
        assert interrupted.payload["interruption_type"] == pending["interruption_type"]
        if expired:
            assert interrupted.payload["limit"] == "execution_deadline"
            assert interrupted.payload["failure_evidence"]["classification"] == "deadline"
        repaired_checkpoint = await app.session_store.load_checkpoint(session_id)
        assert repaired_checkpoint is not None
        assert "pending_session_interrupt" not in repaired_checkpoint

    asyncio.run(run())


def test_provider_cancellation_live_readback_rejects_conflicting_marker_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BillingCleanupFailureProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__([])
            self.hook_started = asyncio.Event()
            self.release_cleanup = asyncio.Event()

        async def billing_identity_for_request(
            self,
            request: ModelRequest,
        ) -> BillingIdentity | None:
            del request

            async def child() -> None:
                await self.release_cleanup.wait()
                raise RuntimeError("private billing cleanup failure") from None

            async with asyncio.TaskGroup() as group:
                group.create_task(child())
                self.hook_started.set()
                await asyncio.Event().wait()
            return None

    async def run() -> None:
        session_id = "sess_provider_cancel_conflicting_live_readback"
        store = InMemorySessionStore()
        provider = BillingCleanupFailureProvider()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        original_terminal_stream = app._terminal_event_publication.publish_recovered

        async def conflicting_readback(request):  # type: ignore[no-untyped-def]
            async for event in original_terminal_stream(request):
                if (
                    event.type is EventType.SESSION_INTERRUPTED
                    and "provider_cancellation_failures" in event.payload
                ):
                    yield event.model_copy(
                        update={
                            "payload": {
                                **event.payload,
                                "interruption_type": "operator_requested",
                            }
                        },
                        deep=True,
                    )
                else:
                    yield event

        monkeypatch.setattr(
            app._terminal_event_publication,
            "publish_recovered",
            conflicting_readback,
        )
        task = asyncio.create_task(
            collect_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                ),
            )
        )
        await asyncio.wait_for(provider.hook_started.wait(), timeout=10)
        task.cancel("caller cancelled billing")
        provider.release_cleanup.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await task

        assert str(raised.value) == "caller cancelled billing"
        assert task.cancelling() == 1
        assert task.cancelled() is True
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        marker = checkpoint["pending_session_interrupt"]
        assert marker["interruption_type"] == "runtime_interrupted"
        assert marker["provider_cancellation_failures"] == [
            {
                "phase": "billing_identity_for_request",
                "error": "Provider billing identity cleanup failed during caller cancellation.",
                "error_type": "BillingIdentityCleanupError",
            }
        ]
        assert type(raised.value.__cause__) is RuntimeError
        assert str(raised.value.__cause__) == (
            "Session interruption finalization failed after provider cancellation."
        )

    asyncio.run(run())
