from __future__ import annotations

import asyncio

import pytest

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.context.thinking import ThinkingConfig
from cayu.messages import TextPart, ThinkingPart
from cayu.observability.events import EventSink
from cayu.providers import ModelProvider, ModelStreamEvent
from cayu.providers._credential_boundary import (
    credential_safe_provider_cancellation,
    provider_cancellation_failures,
)
from cayu.providers.chat_completions import ChatCompletionsProvider
from cayu.providers.deadlines import ProviderStreamDeadlineController
from cayu.providers.operations import ProviderOperationRecoveryMetadata
from cayu.runtime import _stream_delta_buffer as buffering
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._provider_stream import _owned_model_provider_events
from cayu.runtime._session_control import SessionInterruptedByRequest, StreamInterruptPoll
from cayu.sessions import InMemorySessionStore, InterruptSessionRequest
from cayu.storage import SQLiteSessionStore


class ReasoningTransport:
    async def stream_chat_completions(self, **kwargs):
        for _ in range(6000):
            yield {"choices": [{"delta": {"reasoning": "x"}}]}
        for _ in range(2000):
            yield {"choices": [{"delta": {"content": "y"}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("include_thinking", [True, False])
def test_chat_completions_reasoning_has_bounded_durable_events(
    backend, include_thinking, tmp_path, sqlite_resources
):
    async def scenario():
        async with sqlite_resources as resources:
            store = (
                resources.own(SQLiteSessionStore(tmp_path / "stream.sqlite"))
                if backend == "sqlite"
                else InMemorySessionStore()
            )
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(
                ChatCompletionsProvider(api_key="test", transport=ReasoningTransport()),
                default=True,
            )
            app.register_agent(
                AgentSpec(
                    name="test",
                    model="reasoner",
                    thinking=ThinkingConfig(include_in_transcript=include_thinking),
                )
            )
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="test",
                        session_id="stream",
                        messages=[Message.text("user", "run")],
                    )
                )
            ]
            assert events[-1].type is EventType.SESSION_COMPLETED
            for kind, expected in (
                (EventType.MODEL_THINKING_DELTA, "x" * 6000),
                (EventType.MODEL_TEXT_DELTA, "y" * 2000),
            ):
                deltas = [event for event in events if event.type is kind]
                assert len(deltas) <= 6
                assert "".join(event.payload["delta"] for event in deltas) == expected
                persisted = [
                    event for event in await store.load_events("stream") if event.type is kind
                ]
                assert [event.payload["delta"] for event in persisted] == [
                    event.payload["delta"] for event in deltas
                ]
            transcript = await store.load_transcript("stream")
            assert (
                "".join(part.text for part in transcript[-1].content if isinstance(part, TextPart))
                == "y" * 2000
            )
            thinking = [part for part in transcript[-1].content if isinstance(part, ThinkingPart)]
            assert "".join(part.text for part in thinking) == (
                "x" * 6000 if include_thinking else ""
            )

    asyncio.run(scenario())


async def collect(events):
    async def source():
        for event in events:
            yield event

    result = []
    async for event in buffering.coalesce_model_stream_deltas(source()):
        result.extend(
            [event.delta, event.completion]
            if isinstance(event, buffering.BufferedModelCompletion)
            else [event]
        )
    return result


def test_coalescing_preserves_semantic_boundaries_and_recovery_cursors():
    signed = ModelStreamEvent.thinking("signed", provider_state={"signature": "test"})
    recovered = ModelStreamEvent.thinking("recovered")
    recovered.recovery_metadata = ProviderOperationRecoveryMetadata(cursor=1)
    events = [
        ModelStreamEvent.thinking("a"),
        ModelStreamEvent.thinking("b"),
        ModelStreamEvent.thinking("c"),
        signed,
        ModelStreamEvent.text_delta("d"),
        ModelStreamEvent.text_delta("e"),
        recovered,
        ModelStreamEvent.completed(),
    ]
    result = asyncio.run(collect(events))
    assert [event.delta for event in result] == ["a", "bc", "signed", "d", "e", "recovered", ""]
    assert result[2].payload == signed.payload
    assert result[-2].recovery_metadata == recovered.recovery_metadata


def test_provider_cannot_supply_an_internal_completion_batch():
    batch = buffering.BufferedModelCompletion(
        ModelStreamEvent.text_delta("untrusted"), ModelStreamEvent.completed()
    )
    with pytest.raises(TypeError, match="must yield ModelStreamEvent instances"):
        asyncio.run(collect([batch]))


def test_coalescing_flushes_on_elapsed_arrival(monkeypatch):
    now = 0.0
    monkeypatch.setattr(buffering, "monotonic", lambda: now)

    async def scenario():
        async def source():
            nonlocal now
            yield ModelStreamEvent.thinking("first")
            yield ModelStreamEvent.thinking("second")
            now = 2.0
            yield ModelStreamEvent.thinking("third")
            raise RuntimeError("after flush")

        stream = buffering.coalesce_model_stream_deltas(source())
        assert (await anext(stream)).delta == "first"
        assert (await anext(stream)).delta == "secondthird"
        with pytest.raises(RuntimeError, match="after flush"):
            await anext(stream)

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [RuntimeError("failed"), asyncio.CancelledError()])
def test_coalescing_flushes_accepted_text_before_source_failure(failure):
    async def scenario():
        closed = False

        async def source():
            nonlocal closed
            try:
                yield ModelStreamEvent.thinking("first")
                yield ModelStreamEvent.thinking("tail")
                raise failure
            finally:
                closed = True

        stream = buffering.coalesce_model_stream_deltas(source())
        assert (await anext(stream)).delta == "first"
        assert (await anext(stream)).delta == "tail"
        with pytest.raises(type(failure)):
            await anext(stream)
        assert closed

    asyncio.run(scenario())


def test_closing_coalescer_closes_source_without_yielding():
    async def scenario():
        closed = False

        async def source():
            nonlocal closed
            try:
                yield ModelStreamEvent.text_delta("first")
                yield ModelStreamEvent.text_delta("tail")
            finally:
                closed = True

        stream = buffering.coalesce_model_stream_deltas(source())
        await anext(stream)
        await stream.aclose()
        assert closed

    asyncio.run(scenario())


@pytest.mark.parametrize("session_interrupt", [False, True])
def test_runtime_cancellation_preserves_already_accepted_buffered_reasoning(session_interrupt):
    async def scenario():
        ready = asyncio.Event()
        closed = asyncio.Event()

        class Provider(ModelProvider):
            name = "cancel-buffer"

            async def stream(self, request):
                try:
                    yield ModelStreamEvent.thinking("first")
                    yield ModelStreamEvent.thinking("tail")
                    ready.set()
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    raise credential_safe_provider_cancellation(
                        "Provider operation cancelled",
                        preserve_empty_artifacts=False,
                        provider_cancellation_failures=(
                            {
                                "phase": "provider_stream_cleanup",
                                "error": "Provider stream cleanup did not complete normally.",
                                "error_type": "ProviderStreamCleanupError",
                            },
                        ),
                    ) from None
                finally:
                    closed.set()

        app = CayuApp(enable_logging=False)
        app.register_provider(Provider(), default=True)
        app.register_agent(AgentSpec(name="test", model="test"))

        async def run():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="cancel",
                        agent_name="test",
                        messages=[Message.text("user", "run")],
                    )
                )
            ]

        task = asyncio.create_task(run())
        try:
            await asyncio.wait_for(ready.wait(), 10)
            if session_interrupt:
                receipt = [
                    event
                    async for event in app.interrupt_session(
                        InterruptSessionRequest(session_id="cancel", reason="stop")
                    )
                ]
                await asyncio.wait_for(task, 10)
                assert receipt[-1].type is EventType.SESSION_INTERRUPTED
            else:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 10)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        events = await app.session_store.load_events("cancel")
        assert (
            "".join(
                event.payload["delta"]
                for event in events
                if event.type is EventType.MODEL_THINKING_DELTA
            )
            == "firsttail"
        )
        assert closed.is_set()
        interrupted = next(event for event in events if event.type is EventType.SESSION_INTERRUPTED)
        (failure,) = interrupted.payload["provider_cancellation_failures"]
        assert failure["phase"] == "provider_stream_cleanup"

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["text", "thinking"])
@pytest.mark.parametrize("during_delivery", [False, True])
@pytest.mark.parametrize("session_interrupt", [False, True])
def test_boundary_flush_cancellation_preserves_completed_usage(
    kind, during_delivery, session_interrupt, monkeypatch
):
    async def scenario():
        flushing = asyncio.Event()
        completion_read = False
        provider_closed = False
        delta = ModelStreamEvent.text_delta if kind == "text" else ModelStreamEvent.thinking
        event_type = (
            EventType.MODEL_TEXT_DELTA if kind == "text" else EventType.MODEL_THINKING_DELTA
        )

        class Provider(ModelProvider):
            name = "completed-before-tail-flush"

            async def stream(self, request):
                nonlocal completion_read, provider_closed
                try:
                    yield delta("first")
                    yield delta("tail")
                    completion_read = True
                    yield ModelStreamEvent.completed(
                        {"finish_reason": "stop", "usage": {"input_tokens": 2, "output_tokens": 3}}
                    )
                finally:
                    provider_closed = True

        async def block_tail(event):
            if event.type is event_type and event.payload["delta"] == "tail":
                assert completion_read
                flushing.set()
                await asyncio.Event().wait()

        class Sink(EventSink):
            async def emit(self, event):
                await block_tail(event)

        original_emit = RuntimeEventWriter.emit

        async def emit(self, event):
            await block_tail(event)
            return await original_emit(self, event)

        if not during_delivery:
            monkeypatch.setattr(RuntimeEventWriter, "emit", emit)
        app = CayuApp(event_sinks=[Sink()] if during_delivery else [], enable_logging=False)
        app.register_provider(Provider(), default=True)
        app.register_agent(AgentSpec(name="test", model="test"))

        async def run():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="boundary-cancel",
                        agent_name="test",
                        messages=[Message.text("user", "run")],
                    )
                )
            ]

        task = asyncio.create_task(run())
        try:
            await asyncio.wait_for(flushing.wait(), 10)
            if session_interrupt:
                async for _ in app.interrupt_session(
                    InterruptSessionRequest(session_id="boundary-cancel", reason="stop")
                ):
                    pass
                await asyncio.wait_for(task, 10)
            else:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 10)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        events = await app.session_store.load_events("boundary-cancel")
        completions = [event for event in events if event.type is EventType.MODEL_COMPLETED]
        assert len(completions) == 1
        assert completions[0].payload["usage"]["input_tokens"] == 2
        assert completions[0].payload["usage"]["output_tokens"] == 3
        assert completions[0].payload["step_classification"]["type"] == "failed"
        assert not any(event.type is EventType.SESSION_COMPLETED for event in events)
        assert provider_closed
        if during_delivery:
            assert (
                "".join(event.payload["delta"] for event in events if event.type is event_type)
                == "firsttail"
            )

    asyncio.run(scenario())


def test_interrupt_poll_preserves_primary_failure_when_buffer_source_close_fails(monkeypatch):
    polls = 0

    async def interrupt(self):
        nonlocal polls
        polls += 1
        if polls == 3:
            raise SessionInterruptedByRequest("poll-cleanup")

    monkeypatch.setattr(StreamInterruptPoll, "raise_if_interrupted", interrupt)

    class Provider(ModelProvider):
        name = "poll-cleanup-failure"

        async def stream(self, request):
            try:
                yield ModelStreamEvent.text_delta("first")
                yield ModelStreamEvent.text_delta("tail")
                yield ModelStreamEvent.text_delta("unaccepted")
            finally:
                raise RuntimeError("private provider cleanup failure")

    async def scenario():
        app = CayuApp(enable_logging=False)
        app.register_provider(Provider(), default=True)
        app.register_agent(AgentSpec(name="test", model="test"))
        events = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="poll-cleanup",
                    agent_name="test",
                    messages=[Message.text("user", "run")],
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_FAILED
        assert events[-1].payload["error_type"] == "ExceptionGroup"
        assert "SessionInterruptedByRequest" in events[-1].payload["error"]
        assert "ProviderStreamCleanupError" in events[-1].payload["error"]
        assert "private provider" not in events[-1].payload["error"]
        assert (
            "".join(
                event.payload["delta"]
                for event in events
                if event.type is EventType.MODEL_TEXT_DELTA
            )
            == "firsttail"
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("blocked_delta", ["first", "tail"])
def test_cancelled_delta_delivery_does_not_wait_for_stalled_provider_close(blocked_delta):
    async def scenario():
        delivering = asyncio.Event()
        closing = asyncio.Event()
        release_close = asyncio.Event()
        closed = asyncio.Event()

        class Provider(ModelProvider):
            name = "stalled-completion-close"

            async def stream(self, request):
                try:
                    yield ModelStreamEvent.text_delta("first")
                    yield ModelStreamEvent.text_delta("tail")
                    yield ModelStreamEvent.completed(
                        {"finish_reason": "stop", "usage": {"output_tokens": 3}}
                    )
                finally:
                    closing.set()
                    await release_close.wait()
                    closed.set()

        class Sink(EventSink):
            async def emit(self, event):
                if (
                    event.type is EventType.MODEL_TEXT_DELTA
                    and event.payload["delta"] == blocked_delta
                ):
                    delivering.set()
                    await asyncio.Event().wait()

        app = CayuApp(event_sinks=[Sink()], enable_logging=False)
        app.register_provider(Provider(), default=True)
        app.register_agent(AgentSpec(name="test", model="test"))

        async def run():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="stalled-close",
                        agent_name="test",
                        messages=[Message.text("user", "run")],
                    )
                )
            ]

        task = asyncio.create_task(run())
        try:
            await asyncio.wait_for(delivering.wait(), 10)
            task.cancel()
            # A wait_for timeout would issue a second cancellation and conceal
            # the regression: the first request must settle without that help.
            done, _ = await asyncio.wait({task}, timeout=10)
            assert task in done, "Caller cancellation waited for physical provider close."
            with pytest.raises(asyncio.CancelledError):
                await task
            assert closing.is_set()
            assert not closed.is_set()
            events = await app.session_store.load_events("stalled-close")
            completions = [event for event in events if event.type is EventType.MODEL_COMPLETED]
            assert len(completions) == (1 if blocked_delta == "tail" else 0)
            if completions:
                assert completions[0].payload["usage"]["output_tokens"] == 3
            interrupted = next(
                event for event in events if event.type is EventType.SESSION_INTERRUPTED
            )
            (failure,) = interrupted.payload["provider_cancellation_failures"]
            assert failure["phase"] == "provider_stream_cleanup"
        finally:
            release_close.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.wait_for(closed.wait(), 5)

    asyncio.run(scenario())


@pytest.mark.parametrize("completed", [False, True])
def test_buffering_preserves_owned_cancellation_evidence_and_terminal_handoff(completed):
    async def scenario():
        ready = asyncio.Event()
        accepted = []
        diagnostics = (
            {
                "phase": "provider_stream_cleanup",
                "error": "Provider stream cleanup did not complete normally.",
                "error_type": "ProviderStreamCleanupError",
            },
        )

        async def source():
            yield ModelStreamEvent.text_delta("first")
            yield ModelStreamEvent.text_delta("tail")
            ready.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if completed:
                    yield ModelStreamEvent.completed(
                        {"finish_reason": "stop", "usage": {"output_tokens": 2}}
                    )
                raise credential_safe_provider_cancellation(
                    "Provider operation cancelled",
                    preserve_empty_artifacts=False,
                    provider_cancellation_failures=diagnostics,
                ) from None

        async def consume():
            stream = buffering.coalesce_model_stream_deltas(
                _owned_model_provider_events(
                    source,
                    cancellation_baseline=0,
                    max_concurrent_streams=64,
                )
            )
            try:
                async for event in stream:
                    accepted.extend(
                        [event.delta, event.completion]
                        if isinstance(event, buffering.BufferedModelCompletion)
                        else [event]
                    )
            except asyncio.CancelledError as failure:
                return provider_cancellation_failures(failure)
            raise AssertionError("Caller cancellation was lost.")

        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(ready.wait(), 10)
            task.cancel()
            retained = await asyncio.wait_for(task, 10)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert retained == diagnostics
        assert [event.delta for event in accepted] == (
            ["first", "tail", ""] if completed else ["first", "tail"]
        )
        if completed:
            assert accepted[-1].payload["usage"] == {"output_tokens": 2}

    asyncio.run(scenario())


def test_semantic_progress_observes_every_original_reasoning_chunk(monkeypatch):
    observed = []
    observe = ProviderStreamDeadlineController.observe_semantic

    def record(self, kind):
        observed.append(kind)
        return observe(self, kind)

    monkeypatch.setattr(ProviderStreamDeadlineController, "observe_semantic", record)

    class Provider(ModelProvider):
        name = "reasoner"

        async def stream(self, request):
            for _ in range(2000):
                yield ModelStreamEvent.thinking("x")
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed()

    async def scenario():
        app = CayuApp(enable_logging=False)
        app.register_provider(Provider(), default=True)
        app.register_agent(AgentSpec(name="test", model="reasoner"))
        events = [
            event
            async for event in app.run(
                RunRequest(agent_name="test", messages=[Message.text("user", "run")])
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED
        assert len([e for e in events if e.type is EventType.MODEL_THINKING_DELTA]) <= 3
        assert observed.count("reasoning") == 2000

    asyncio.run(scenario())


@pytest.mark.parametrize("text", ["bad\x00text", "bad\ud800text"])
def test_invalid_provider_delta_is_not_hidden_by_coalescing(text):
    async def scenario():
        async def source():
            yield ModelStreamEvent.thinking("first")
            yield ModelStreamEvent.thinking("accepted")
            yield ModelStreamEvent.thinking(text)

        stream = buffering.coalesce_model_stream_deltas(source())
        assert (await anext(stream)).delta == "first"
        assert (await anext(stream)).delta == "accepted"
        with pytest.raises(ValueError):
            await anext(stream)

    asyncio.run(scenario())
