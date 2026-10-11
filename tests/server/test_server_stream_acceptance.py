"""HTTP server stream acceptance contracts."""

from __future__ import annotations

# ruff: noqa: E402
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
httpx = pytest.importorskip("httpx")

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI  # noqa: TC002 - preserve runtime helper annotation resolution
from fastapi.testclient import TestClient
from tests.server.server_test_support import _LOCAL_SERVER_CONFIG, OneShotProvider, _sse_frames

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.environments.factory import (
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
)
from cayu.events import Event, EventType, event_with_durable_sequence
from cayu.exceptions import TerminalEventPublicationUncertain
from cayu.messages import Message
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime._event_projection import (
    PRIVATE_EVENT_AUTHORITY,
    public_event_id,
    public_event_sequence,
)
from cayu.server import create_server
from cayu.server.routes import (
    _accepted_event_stream_response,
    _detached_event_stream_response,
    _start_detached_event_stream_response,
)
from cayu.server.sse import (
    SSE_ERROR_TEXT_MAX_BYTES,
    SSE_EVENT_DATA_MAX_BYTES,
    SSE_OBSERVER_MAX_BYTES,
    SSE_OBSERVER_MAX_FRAMES,
    SSE_SEND_TIMEOUT_SECONDS,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import EventRecord, SessionIdentity, SessionStatus
from cayu.sessions.requests import (
    InterruptSessionRequest,
    RunRequest,
    run_request_with_runtime_generated_authority,
)
from cayu.tasks.creation import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore
from cayu.tasks.records import Task, TaskStatus
from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor


async def _post_and_disconnect_before_first_body(
    server: FastAPI,
    path: str,
    body: dict[str, Any],
) -> list[dict[str, Any]]:
    """Send one ASGI POST and disconnect immediately after HTTP acceptance."""
    request_sent = False
    response_started = asyncio.Event()
    disconnect_delivered = asyncio.Event()
    messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {
                "type": "http.request",
                "body": json.dumps(body).encode(),
                "more_body": False,
            }
        await response_started.wait()
        disconnect_delivered.set()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body" and disconnect_delivered.is_set():
            # The server may race one write already queued at its ASGI boundary;
            # model the socket loss by dropping it before the client receives it.
            return
        messages.append(message)
        if message["type"] == "http.response.start":
            response_started.set()
            # Do not let the response task emit its first body frame before the
            # disconnect listener has observed the injected network loss.
            await disconnect_delivered.wait()

    await server(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 50000),
            "server": ("testserver", 80),
        },
        receive,
        send,
    )
    return messages


def test_run_accepts_and_detaches_before_environment_factory_finishes() -> None:
    class BlockingEnvironmentFactory(EnvironmentFactory):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def create(
            self,
            _request: EnvironmentFactoryRequest,
        ) -> EnvironmentFactoryResult:
            self.started.set()
            await self.release.wait()
            return EnvironmentFactoryResult(
                environment=Environment(EnvironmentSpec(name="dynamic"))
            )

    factory = BlockingEnvironmentFactory()
    app = CayuApp(enable_logging=False)
    app.register_provider(OneShotProvider(), default=True)
    app.register_environment_factory(
        EnvironmentSpec(name="dynamic"),
        factory,
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    server = create_server(app, config=_LOCAL_SERVER_CONFIG)
    session_id = "session_factory_acceptance_boundary"

    async def exercise() -> tuple[bool, list[dict[str, Any]]]:
        request_task = asyncio.create_task(
            _post_and_disconnect_before_first_body(
                server,
                "/api/run",
                {"prompt": "hello", "session_id": session_id},
            )
        )
        await asyncio.wait_for(factory.started.wait(), timeout=10)
        done, _ = await asyncio.wait({request_task}, timeout=0.2)
        accepted_before_release = request_task in done
        state = await app.session_store.load_state(session_id)
        assert state is not None
        assert state.status is SessionStatus.RUNNING
        active_runs = app._session_control.active_runs(session_id)
        assert len(active_runs) == 1
        assert active_runs[0].runtime_task is not request_task
        assert not active_runs[0].runtime_task.done()

        factory.release.set()
        messages = await asyncio.wait_for(request_task, timeout=5)
        deadline = asyncio.get_running_loop().time() + 5
        while True:
            state = await app.session_store.load_state(session_id)
            if state is not None and state.status is SessionStatus.COMPLETED:
                return accepted_before_release, messages
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("detached run did not complete after factory release")
            await asyncio.sleep(0.01)

    accepted_before_release, messages = asyncio.run(exercise())

    assert accepted_before_release
    starts = [message for message in messages if message["type"] == "http.response.start"]
    assert [message["status"] for message in starts] == [200]


def test_interrupt_after_run_acceptance_cancels_detached_provider() -> None:
    class BlockingProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.started: asyncio.Event | None = None
            self.cancelled: asyncio.Event | None = None
            self.never_complete: asyncio.Event | None = None

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            if self.started is None or self.cancelled is None or self.never_complete is None:
                raise AssertionError("BlockingProvider test events were not initialized.")
            self.started.set()
            try:
                await self.never_complete.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    provider = BlockingProvider()
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    server = create_server(app, config=_LOCAL_SERVER_CONFIG)
    session_id = "session_detached_interrupt_ownership"

    async def exercise() -> tuple[list[dict[str, Any]], list[Event]]:
        provider.started = asyncio.Event()
        provider.cancelled = asyncio.Event()
        provider.never_complete = asyncio.Event()
        request_task = asyncio.create_task(
            _post_and_disconnect_before_first_body(
                server,
                "/api/run",
                {"prompt": "hello", "session_id": session_id},
            )
        )
        await asyncio.wait_for(provider.started.wait(), timeout=10)
        messages = await asyncio.wait_for(request_task, timeout=1)

        active_runs = app._session_control.active_runs(session_id)
        assert len(active_runs) == 1
        assert active_runs[0].runtime_task is not request_task
        assert not active_runs[0].runtime_task.done()

        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(session_id=session_id, reason="operator stop")
            )
        ]
        await asyncio.wait_for(provider.cancelled.wait(), timeout=1)
        assert await app.drain_background_interruptions(timeout_s=1) is True
        return messages, interrupt_events

    messages, interrupt_events = asyncio.run(exercise())

    starts = [message for message in messages if message["type"] == "http.response.start"]
    assert [message["status"] for message in starts] == [200]
    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]


def test_interrupt_after_acceptance_before_observer_start_reaches_runtime() -> None:
    class RecordingProvider(OneShotProvider):
        dispatches = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.dispatches += 1
            async for event in super().stream(request):
                yield event

    provider = RecordingProvider()
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    session_id = "session_interrupt_before_observer_start"

    async def exercise() -> tuple[list[Event], list[dict[str, str]], SessionStatus]:
        response = await _accepted_event_stream_response(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                )
            ),
            cayu_app=app,
            session_id=session_id,
        )
        active_runs = app._session_control.active_runs(session_id)
        assert len(active_runs) == 1
        assert not active_runs[0].runtime_task.done()

        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(session_id=session_id, reason="operator stop")
            )
        ]
        observed = [message async for message in response.body_iterator]
        runtime_task = active_runs[0].runtime_task
        await asyncio.wait_for(asyncio.shield(runtime_task), timeout=5)
        assert not runtime_task.cancelled() and runtime_task.cancelling() == 0
        assert provider.dispatches == 0
        state = await app.session_store.load_state(session_id)
        assert state is not None
        assert await app.drain_background_interruptions(timeout_s=1) is True
        return interrupt_events, observed, state.status

    interrupt_events, observed, status = asyncio.run(exercise())

    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    observed_types = [json.loads(message["data"])["type"] for message in observed]
    assert observed_types[:2] == [
        EventType.INTERACTION_STARTED,
        EventType.SESSION_STARTED,
    ]
    assert observed_types[-1] == EventType.SESSION_INTERRUPTED
    assert status is SessionStatus.INTERRUPTED


def test_interrupt_before_observer_start_cancels_environment_factory() -> None:
    class BlockingEnvironmentFactory(EnvironmentFactory):
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.never_complete = asyncio.Event()

        async def create(
            self,
            _request: EnvironmentFactoryRequest,
        ) -> EnvironmentFactoryResult:
            self.started.set()
            try:
                await self.never_complete.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            raise AssertionError("unreachable")

    factory = BlockingEnvironmentFactory()
    app = CayuApp(enable_logging=False)
    app.register_provider(OneShotProvider(), default=True)
    app.register_environment_factory(
        EnvironmentSpec(name="dynamic"),
        factory,
        default=True,
    )
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    session_id = "session_interrupt_factory_before_observer_start"

    async def exercise() -> tuple[list[Event], list[dict[str, str]], SessionStatus]:
        response = await _accepted_event_stream_response(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                )
            ),
            cayu_app=app,
            session_id=session_id,
        )
        assert not factory.started.is_set()

        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(session_id=session_id, reason="operator stop")
            )
        ]
        observed = [message async for message in response.body_iterator]
        await asyncio.wait_for(factory.started.wait(), timeout=10)
        await asyncio.wait_for(factory.cancelled.wait(), timeout=1)
        state = await app.session_store.load_state(session_id)
        assert state is not None
        assert await app.drain_background_interruptions(timeout_s=1) is True
        return interrupt_events, observed, state.status

    interrupt_events, observed, status = asyncio.run(exercise())

    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    observed_types = [json.loads(message["data"])["type"] for message in observed]
    assert observed_types[:2] == [
        EventType.INTERACTION_STARTED,
        EventType.ENVIRONMENT_FACTORY_STARTED,
    ]
    assert observed_types[-1] == EventType.SESSION_INTERRUPTED
    assert status is SessionStatus.INTERRUPTED


def test_interrupt_during_run_acceptance_finishes_task_bookkeeping() -> None:
    class RecordingProvider(OneShotProvider):
        def __init__(self) -> None:
            self.dispatches = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.dispatches += 1
            async for event in super().stream(request):
                yield event

    class BlockingTaskStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.create_started = asyncio.Event()
            self.release_create = asyncio.Event()

        async def create_running_task(self, request, *, session_invocation):
            self.create_started.set()
            await self.release_create.wait()
            return await super().create_running_task(
                request,
                session_invocation=session_invocation,
            )

    task_store = BlockingTaskStore()
    app = CayuApp(task_store=task_store, enable_logging=False)
    provider = RecordingProvider()
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    session_id = "session_interrupt_during_acceptance_bookkeeping"

    async def exercise() -> tuple[list[Event], list[dict[str, str]], SessionStatus]:
        async def create_task_after_accept(_event: Event) -> None:
            snapshot = await app.session_store.load_invocation_snapshot(session_id)
            if snapshot is None:
                raise AssertionError("Accepted run session was not persisted.")
            await task_store.create_running_task(
                TaskCreate(
                    task_id="task_interrupt_during_acceptance_bookkeeping",
                    type="run",
                    session_id=session_id,
                ),
                session_invocation=snapshot,
            )

        async def interrupt() -> list[Event]:
            return [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(session_id=session_id, reason="operator stop")
                )
            ]

        run_request = run_request_with_runtime_generated_authority(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                task_id="task_interrupt_during_acceptance_bookkeeping",
                messages=[Message.text("user", "hello")],
            ),
            "task_id",
        )
        response_task = asyncio.create_task(
            _accepted_event_stream_response(
                app.run(run_request),
                cayu_app=app,
                session_id=session_id,
                after_accept=create_task_after_accept,
            )
        )
        await asyncio.wait_for(task_store.create_started.wait(), timeout=10)
        interrupt_task = asyncio.create_task(interrupt())
        await asyncio.sleep(0)
        assert not response_task.done()

        task_store.release_create.set()
        response, interrupt_events = await asyncio.gather(response_task, interrupt_task)
        observed = [message async for message in response.body_iterator]
        state = await app.session_store.load_state(session_id)
        assert state is not None
        assert await app.drain_background_interruptions(timeout_s=1) is True
        tasks = await task_store.list_tasks()
        assert [task.id for task in tasks] == ["task_interrupt_during_acceptance_bookkeeping"]
        assert tasks[0].session_id == session_id
        assert tasks[0].status is TaskStatus.RUNNING
        assert provider.dispatches == 0
        records = await app.session_store.query_events(EventQuery(session_id=session_id, limit=100))
        assert [record.event.type for record in records] == [
            json.loads(message["data"])["type"] for message in observed
        ]
        return interrupt_events, observed, state.status

    interrupt_events, observed, status = asyncio.run(exercise())

    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    observed_types = [json.loads(message["data"])["type"] for message in observed]
    # Acceptance is interaction.started, not session.started. Deferred
    # interruption can arrive at the group-membership read before startup or
    # at the next await after startup; neither ordering may lose bookkeeping.
    assert observed_types in (
        [
            EventType.INTERACTION_STARTED,
            EventType.INTERACTION_INTERRUPTED,
            EventType.SESSION_INTERRUPTED,
        ],
        [
            EventType.INTERACTION_STARTED,
            EventType.SESSION_STARTED,
            EventType.INTERACTION_INTERRUPTED,
            EventType.SESSION_INTERRUPTED,
        ],
    )
    assert status is SessionStatus.INTERRUPTED


def test_run_route_interrupt_during_acceptance_state_read_keeps_task_linked() -> None:
    class BlockingAcceptanceStateStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        terminal_interaction_publication_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.block_next_state_read = True
            self.state_read_started = asyncio.Event()
            self.release_state_read = asyncio.Event()

        async def load_invocation_snapshot(self, session_id: str):
            if self.block_next_state_read:
                self.block_next_state_read = False
                self.state_read_started.set()
                await self.release_state_read.wait()
            return await super().load_invocation_snapshot(session_id)

    session_store = BlockingAcceptanceStateStore()
    task_store = InMemoryTaskStore()
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
    )
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    server = create_server(app, config=_LOCAL_SERVER_CONFIG)
    session_id = "session_interrupt_during_acceptance_state_read"

    async def exercise() -> tuple[httpx.Response, list[Event], SessionStatus, list[Task]]:
        async def interrupt() -> list[Event]:
            return [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(session_id=session_id, reason="operator stop")
                )
            ]

        transport = httpx.ASGITransport(app=server)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            request_task = asyncio.create_task(
                client.post(
                    "/api/run",
                    json={"prompt": "hello", "session_id": session_id},
                )
            )
            await asyncio.wait_for(session_store.state_read_started.wait(), timeout=10)
            interrupt_task = asyncio.create_task(interrupt())
            await asyncio.sleep(0)
            session_store.release_state_read.set()
            response, interrupt_events = await asyncio.wait_for(
                asyncio.gather(request_task, interrupt_task),
                timeout=2,
            )

        state = await session_store.load_state(session_id)
        assert state is not None
        return response, interrupt_events, state.status, await task_store.list_tasks()

    response, interrupt_events, status, tasks = asyncio.run(exercise())

    assert response.status_code == 200
    frames = [frame for frame in _sse_frames(response) if "data" in frame]
    assert frames[-1]["data"]["type"] == EventType.SESSION_INTERRUPTED
    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    assert status is SessionStatus.INTERRUPTED
    assert len(tasks) == 1
    assert tasks[0].session_id == session_id
    assert tasks[0].status is TaskStatus.RUNNING


def test_request_cancellation_during_acceptance_does_not_cancel_detached_run() -> None:
    class BlockingProvider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.started.set()
            await self.release.wait()
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    provider = BlockingProvider()
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    session_id = "session_cancelled_acceptance_owner"

    async def exercise() -> tuple[SessionStatus, list[EventType | str]]:
        callback_started = asyncio.Event()
        release_callback = asyncio.Event()

        async def after_accept(_event: Event) -> None:
            callback_started.set()
            await release_callback.wait()

        response_task = asyncio.create_task(
            _accepted_event_stream_response(
                app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "hello")],
                    )
                ),
                cayu_app=app,
                session_id=session_id,
                after_accept=after_accept,
            )
        )
        await asyncio.wait_for(callback_started.wait(), timeout=10)
        response_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await response_task

        release_callback.set()
        await asyncio.wait_for(provider.started.wait(), timeout=10)
        active_runs = app._session_control.active_runs(session_id)
        assert len(active_runs) == 1
        assert not active_runs[0].runtime_task.done()

        provider.release.set()
        deadline = asyncio.get_running_loop().time() + 1
        while True:
            state = await app.session_store.load_state(session_id)
            if state is not None and state.status is SessionStatus.COMPLETED:
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("detached run did not complete after request cancellation")
            await asyncio.sleep(0.01)

        records = await app.session_store.query_events(EventQuery(session_id=session_id, limit=100))
        # Durable completion precedes trailing cleanup and active-task removal.
        await asyncio.wait_for(asyncio.shield(active_runs[0].runtime_task), timeout=5)
        assert app._session_control.active_runs(session_id) == ()
        return state.status, [record.event.type for record in records]

    status, event_types = asyncio.run(exercise())

    assert status is SessionStatus.COMPLETED
    assert event_types[-1] == EventType.SESSION_COMPLETED


def test_event_source_cancellation_before_acceptance_does_not_hang_request() -> None:
    async def cancelled_stream() -> AsyncIterator[Event]:
        raise asyncio.CancelledError
        yield  # pragma: no cover - makes this an async generator

    async def exercise() -> None:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(
                _accepted_event_stream_response(
                    cancelled_stream(),
                    cayu_app=CayuApp(enable_logging=False),
                    session_id="session_cancelled_before_acceptance",
                ),
                timeout=1,
            )

    asyncio.run(exercise())


def test_runtime_failure_survives_cancelled_stream_cleanup() -> None:
    class CancelledCloseStream:
        def __init__(self) -> None:
            self.calls = 0

        def __aiter__(self) -> CancelledCloseStream:
            return self

        async def __anext__(self) -> Event:
            self.calls += 1
            if self.calls == 1:
                return event_with_durable_sequence(
                    Event(
                        id="event_before_runtime_failure",
                        type="custom.before_runtime_failure",
                        session_id="session_cancelled_stream_cleanup",
                    ),
                    1,
                )
            raise RuntimeError("runtime failed")

        async def aclose(self) -> None:
            raise asyncio.CancelledError

    async def exercise() -> list[dict[str, str]]:
        response = await _accepted_event_stream_response(
            CancelledCloseStream(),
            cayu_app=CayuApp(enable_logging=False),
            session_id="session_cancelled_stream_cleanup",
        )

        async def collect() -> list[dict[str, str]]:
            return [message async for message in response.body_iterator]

        return await asyncio.wait_for(collect(), timeout=1)

    messages = asyncio.run(exercise())

    assert messages[0]["id"] == (f"session_cancelled_stream_cleanup:{public_event_id(1)}")
    assert messages[-1]["event"] == "error"
    error = json.loads(messages[-1]["data"])
    assert error["kind"] == "runtime"
    assert error["code"] == "runtime_failed"


def test_terminal_publication_uncertainty_requests_durable_reconciliation() -> None:
    publication_failure = ConnectionError("terminal append acknowledgement lost")
    reconciliation_failure = TimeoutError("terminal reconciliation unavailable")

    async def uncertain_stream() -> AsyncIterator[Event]:
        yield event_with_durable_sequence(
            Event(
                id="event_before_terminal_uncertainty",
                type="custom.before_terminal_uncertainty",
                session_id="session_terminal_uncertainty",
            ),
            1,
        )
        raise TerminalEventPublicationUncertain(
            event=Event(
                id="event_terminal",
                type=EventType.SESSION_COMPLETED,
                session_id="session_terminal_uncertainty",
            ),
            publication_failure=publication_failure,
            reconciliation_failure=reconciliation_failure,
        )

    async def exercise() -> list[dict[str, str]]:
        response = await _accepted_event_stream_response(
            uncertain_stream(),
            cayu_app=CayuApp(enable_logging=False),
            session_id="session_terminal_uncertainty",
        )
        return [message async for message in response.body_iterator]

    messages = asyncio.run(exercise())

    assert messages[0]["id"] == (f"session_terminal_uncertainty:{public_event_id(1)}")
    assert messages[-1]["event"] == "error"
    error = json.loads(messages[-1]["data"])
    assert error["kind"] == "runtime"
    assert error["code"] == "terminal_event_publication_uncertain"
    assert error["retryable"] is True
    assert error["session_id"] == "session_terminal_uncertainty"
    assert error["error_type"] == "TerminalEventPublicationUncertain"


def test_preaccept_terminal_publication_uncertainty_is_not_reported_as_conflict() -> None:
    async def uncertain_stream() -> AsyncIterator[Event]:
        raise TerminalEventPublicationUncertain(
            event=Event(
                id="event_preaccept_terminal_uncertainty",
                type=EventType.SESSION_INTERRUPTED,
                session_id="session_preaccept_terminal_uncertainty",
            ),
            publication_failure=ConnectionError("terminal append acknowledgement lost"),
            reconciliation_failure=TimeoutError("terminal reconciliation unavailable"),
        )
        yield  # pragma: no cover

    async def exercise() -> None:
        with pytest.raises(fastapi.HTTPException) as raised:
            await _accepted_event_stream_response(
                uncertain_stream(),
                cayu_app=CayuApp(enable_logging=False),
                session_id="session_preaccept_terminal_uncertainty",
                conflict_error_types=(RuntimeError,),
            )
        assert raised.value.status_code == 500
        assert raised.value.detail == (
            "Terminal event publication outcome is uncertain; inspect durable session "
            "state before retrying the mutation."
        )

    asyncio.run(exercise())


@pytest.mark.parametrize("terminal_append_committed", [False, True])
def test_interrupt_accepts_before_first_frame_terminal_publication_uncertainty(
    terminal_append_committed: bool,
) -> None:
    class AmbiguousInterruptTerminalStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.armed = False
            self.attempted_terminal_event_id: str | None = None
            self.unreadable_event_id: str | None = None

        async def append_event(self, session_id: str, event: Event) -> None:
            if self.armed and event.type == EventType.SESSION_INTERRUPTED:
                self.armed = False
                self.attempted_terminal_event_id = event.id
                self.unreadable_event_id = event.id
                if terminal_append_committed:
                    await super().append_event(session_id, event)
                raise ConnectionError("terminal append acknowledgement lost")
            await super().append_event(session_id, event)

        async def query_events(self, query: EventQuery):
            if self.unreadable_event_id is not None and query.event_id == self.unreadable_event_id:
                self.unreadable_event_id = None
                raise TimeoutError("terminal reconciliation unavailable")
            return await super().query_events(query)

    store = AmbiguousInterruptTerminalStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    session_id = "session-interrupt-terminal-publication-uncertain"
    mutation_id = "mutation-interrupt-terminal-publication-uncertain"

    async def prepare() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "create pending session")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )

    asyncio.run(prepare())
    store.armed = True
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream(
        "POST",
        f"/api/sessions/{session_id}/interrupt",
        json={"reason": "operator stop"},
        headers={"Cayu-Mutation-ID": mutation_id},
    ) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert len(frames) == 1
    assert frames[0]["event"] == "error"
    assert frames[0]["data"]["code"] == "terminal_event_publication_uncertain"
    assert frames[0]["data"]["retryable"] is True

    events = client.get(
        f"/api/sessions/{session_id}/events",
        params={"order_by": "sequence_asc", "limit": 100},
    ).json()["events"]
    interrupted = [event for event in events if event["type"] == EventType.SESSION_INTERRUPTED]
    accepted = [event for event in events if event["type"] == EventType.SERVER_MUTATION_ACCEPTED]

    assert len(interrupted) == int(terminal_append_committed)
    assert len(accepted) == 1
    expected_accepted_payload = {
        "mutation_id": mutation_id,
        "mutation_kind": "interrupt",
        "accepted_event_id": (
            interrupted[0]["id"] if terminal_append_committed else PRIVATE_EVENT_AUTHORITY
        ),
        "accepted_event_type": EventType.SESSION_INTERRUPTED,
        "accepted_event_publication_uncertain": True,
    }
    if terminal_append_committed:
        expected_accepted_payload["accepted_event_sequence"] = interrupted[0]["sequence"]
    assert accepted[0]["payload"] == expected_accepted_payload

    durable_markers = asyncio.run(
        store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.SERVER_MUTATION_ACCEPTED,
                limit=1,
            )
        )
    )
    assert len(durable_markers) == 1
    if terminal_append_committed:
        assert (
            durable_markers[0].event.payload["accepted_event_sequence"]
            == interrupted[0]["sequence"]
        )
        assert "accepted_event_id" not in durable_markers[0].event.payload
    else:
        assert (
            durable_markers[0].event.payload["accepted_event_id"]
            == store.attempted_terminal_event_id
        )
    if terminal_append_committed:
        assert events.index(interrupted[0]) < events.index(accepted[0])
    state = client.get(f"/api/sessions/{session_id}/state").json()
    assert state["status"] == SessionStatus.INTERRUPTED


def test_request_cancellation_does_not_cancel_uncertain_acceptance_marker() -> None:
    class BlockingUncertainAcceptanceStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.armed = False
            self.unreadable_event_id: str | None = None
            self.marker_started = asyncio.Event()
            self.release_marker = asyncio.Event()

        async def append_event(self, session_id: str, event: Event) -> None:
            if self.armed and event.type == EventType.SESSION_INTERRUPTED:
                self.armed = False
                self.unreadable_event_id = event.id
                raise ConnectionError("terminal append acknowledgement lost")
            if event.type == EventType.SERVER_MUTATION_ACCEPTED:
                self.marker_started.set()
                await self.release_marker.wait()
            await super().append_event(session_id, event)

        async def query_events(self, query: EventQuery):
            if self.unreadable_event_id is not None and query.event_id == self.unreadable_event_id:
                self.unreadable_event_id = None
                raise TimeoutError("terminal reconciliation unavailable")
            return await super().query_events(query)

    async def exercise() -> tuple[SessionStatus, list[Event]]:
        store = BlockingUncertainAcceptanceStore()
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "session-cancelled-uncertain-acceptance"
        mutation_id = "mutation-cancelled-uncertain-acceptance"
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "create pending session")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        store.armed = True
        server = create_server(app, config=_LOCAL_SERVER_CONFIG)
        transport = httpx.ASGITransport(app=server)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            request_task = asyncio.create_task(
                client.post(
                    f"/api/sessions/{session_id}/interrupt",
                    json={"reason": "operator stop"},
                    headers={"Cayu-Mutation-ID": mutation_id},
                )
            )
            await asyncio.wait_for(store.marker_started.wait(), timeout=10)
            request_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request_task
            store.release_marker.set()

            deadline = asyncio.get_running_loop().time() + 1
            while True:
                records = await store.query_events(
                    EventQuery(
                        session_id=session_id,
                        event_type=EventType.SERVER_MUTATION_ACCEPTED,
                        limit=10,
                    )
                )
                if records:
                    break
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError(
                        "Detached uncertainty acceptance did not persist its mutation marker."
                    )
                await asyncio.sleep(0.01)

        state = await store.load_state(session_id)
        assert state is not None
        return state.status, [record.event for record in records]

    status, markers = asyncio.run(exercise())

    assert status is SessionStatus.INTERRUPTED
    assert len(markers) == 1
    assert markers[0].payload["mutation_id"] == "mutation-cancelled-uncertain-acceptance"
    assert markers[0].payload["accepted_event_publication_uncertain"] is True


def test_runtime_cancellation_group_becomes_structured_stream_error() -> None:
    async def grouped_failure_stream() -> AsyncIterator[Event]:
        yield event_with_durable_sequence(
            Event(
                id="event_before_grouped_failure",
                type="custom.before_grouped_failure",
                session_id="session_grouped_stream_failure",
            ),
            1,
        )
        raise BaseExceptionGroup(
            "runtime cleanup cancelled and failed",
            [asyncio.CancelledError(), RuntimeError("cleanup failed")],
        )

    async def exercise() -> list[dict[str, str]]:
        response = await _accepted_event_stream_response(
            grouped_failure_stream(),
            cayu_app=CayuApp(enable_logging=False),
            session_id="session_grouped_stream_failure",
        )
        return [message async for message in response.body_iterator]

    messages = asyncio.run(exercise())

    assert messages[0]["id"] == (f"session_grouped_stream_failure:{public_event_id(1)}")
    assert messages[-1]["event"] == "error"
    error = json.loads(messages[-1]["data"])
    assert error["kind"] == "runtime"
    assert error["code"] == "runtime_failed"


def test_runtime_cancellation_only_group_is_not_reported_as_runtime_failure() -> None:
    cancellation_group = BaseExceptionGroup(
        "runtime cleanup cancelled",
        [asyncio.CancelledError()],
    )

    async def grouped_cancellation_stream() -> AsyncIterator[Event]:
        yield event_with_durable_sequence(
            Event(
                id="event_before_grouped_cancellation",
                type="custom.before_grouped_cancellation",
                session_id="session_grouped_stream_cancellation",
            ),
            1,
        )
        raise cancellation_group

    async def exercise() -> tuple[list[dict[str, str]], BaseExceptionGroup]:
        response, pump_task, _ = _start_detached_event_stream_response(
            grouped_cancellation_stream(),
            cayu_app=CayuApp(enable_logging=False),
            session_id="session_grouped_stream_cancellation",
        )
        messages = [message async for message in response.body_iterator]
        with pytest.raises(BaseExceptionGroup) as exc_info:
            await pump_task
        return messages, exc_info.value

    messages, failure = asyncio.run(exercise())

    assert failure is cancellation_group
    assert [message.get("event") for message in messages] == [None]


def test_live_projection_requires_a_unique_durable_event_record() -> None:
    app = CayuApp(enable_logging=False)
    queried_limits: list[int | None] = []

    async def duplicate_query(query: EventQuery | None = None):
        assert query is not None
        queried_limits.append(query.limit)
        return [
            EventRecord(
                sequence=sequence,
                event=Event(
                    id="duplicate-event",
                    type="custom.duplicate",
                    session_id="duplicate-session",
                ),
            )
            for sequence in (1, 2)
        ]

    app.session_store.query_events = duplicate_query

    async def event_stream() -> AsyncIterator[Event]:
        yield Event(
            id="duplicate-event",
            type="custom.duplicate",
            session_id="duplicate-session",
        )

    async def exercise() -> list[dict[str, str]]:
        response = _detached_event_stream_response(
            event_stream(),
            cayu_app=app,
            session_id="duplicate-session",
        )
        return [message async for message in response.body_iterator]

    messages = asyncio.run(exercise())

    assert queried_limits == [2]
    assert messages[-1]["event"] == "error"


def test_accepted_stream_driver_finishes_when_response_start_send_fails() -> None:
    async def exercise() -> bool:
        completed = asyncio.Event()

        async def event_stream() -> AsyncIterator[Event]:
            yield event_with_durable_sequence(
                Event(
                    id="event_response_start_failure",
                    type="custom.accepted",
                    session_id="session_response_start_failure",
                ),
                1,
            )
            completed.set()

        response = await _accepted_event_stream_response(
            event_stream(),
            cayu_app=CayuApp(enable_logging=False),
            session_id="session_response_start_failure",
        )

        async def receive() -> dict[str, Any]:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def send(_message: dict[str, Any]) -> None:
            raise OSError("response send failed")

        with pytest.raises(OSError, match="response send failed"):
            await response(
                {
                    "type": "http",
                    "asgi": {"version": "3.0", "spec_version": "2.3"},
                    "http_version": "1.1",
                    "method": "POST",
                    "scheme": "http",
                    "path": "/api/run",
                    "raw_path": b"/api/run",
                    "query_string": b"",
                    "root_path": "",
                    "headers": [],
                    "client": ("127.0.0.1", 50000),
                    "server": ("testserver", 80),
                },
                receive,
                send,
            )
        await asyncio.wait_for(completed.wait(), timeout=1)
        return completed.is_set()

    assert asyncio.run(exercise()) is True


def _detached_observer_first_message(events: list[Event]) -> tuple[dict[str, str], float, bool]:
    async def scenario() -> tuple[dict[str, str], float, bool]:
        completed = False

        async def event_stream() -> AsyncIterator[Event]:
            nonlocal completed
            try:
                for sequence, event in enumerate(events, start=1):
                    yield event_with_durable_sequence(event, sequence)
            finally:
                completed = True

        response = _detached_event_stream_response(
            event_stream(),
            cayu_app=CayuApp(),
            session_id="session_observer_bound",
        )
        for _ in range(10):
            await asyncio.sleep(0)
            if completed:
                break
        iterator = response.body_iterator.__aiter__()
        message = await anext(iterator)
        await iterator.aclose()
        await asyncio.sleep(0)
        return message, response.send_timeout, completed

    return asyncio.run(scenario())


def test_detached_observer_frame_count_is_bounded_without_stopping_pump() -> None:
    events = [
        Event(
            id=f"event_{index}",
            type="custom.observer",
            session_id="session_observer_bound",
        )
        for index in range(SSE_OBSERVER_MAX_FRAMES + 1)
    ]

    message, send_timeout, completed = _detached_observer_first_message(events)
    data = json.loads(message["data"])

    assert completed is True
    assert send_timeout == SSE_SEND_TIMEOUT_SECONDS
    assert message["event"] == "error"
    assert data["kind"] == "observer"
    assert data["code"] == "observer_lagged"
    assert data["retryable"] is True


def test_detached_observer_does_not_misclassify_a_healthy_synchronous_burst() -> None:
    app = CayuApp()
    completed = False

    async def synchronous_burst(request: RunRequest) -> AsyncIterator[Event]:
        nonlocal completed
        try:
            for index in range(SSE_OBSERVER_MAX_FRAMES + 1):
                yield event_with_durable_sequence(
                    Event(
                        id=f"event_{index}",
                        type="custom.observer",
                        session_id=request.session_id,
                    ),
                    index + 1,
                )
        finally:
            completed = True

    app.run = synchronous_burst  # type: ignore[method-assign]
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream("POST", "/api/run", json={"prompt": "hello"}) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert completed is True
    assert len(frames) == SSE_OBSERVER_MAX_FRAMES + 1
    assert [frame["data"]["id"] for frame in frames] == [
        public_event_id(index + 1) for index in range(SSE_OBSERVER_MAX_FRAMES + 1)
    ]
    assert all(frame.get("event") != "error" for frame in frames)


def test_detached_observer_serialized_bytes_are_bounded() -> None:
    payload_chars = SSE_OBSERVER_MAX_BYTES // 2 + 1024
    events = [
        Event(
            id=f"event_{index}",
            type="custom.observer",
            session_id="session_observer_bound",
            payload={"value": "x" * payload_chars},
        )
        for index in range(2)
    ]

    message, _, completed = _detached_observer_first_message(events)
    data = json.loads(message["data"])

    assert completed is True
    assert data["kind"] == "observer"
    assert data["code"] == "observer_lagged"


def test_detached_oversized_frame_does_not_stop_runtime_pump() -> None:
    event = Event(
        id="event_large",
        type="custom.observer",
        session_id="session_observer_bound",
        payload={"value": "x" * SSE_EVENT_DATA_MAX_BYTES},
    )

    message, _, completed = _detached_observer_first_message([event])
    data = json.loads(message["data"])

    assert completed is True
    assert data["kind"] == "observer"
    assert data["code"] == "event_frame_too_large"
    assert data["retryable"] is False


def test_streaming_mutation_id_creates_an_exact_durable_acceptance_event() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    session_id = "session-mutation-identity"
    mutation_id = "mutation-run-identity"

    with client.stream(
        "POST",
        "/api/run",
        json={"prompt": "hello", "session_id": session_id},
        headers={"Cayu-Mutation-ID": mutation_id},
    ) as response:
        assert response.status_code == 200
        frames = [frame for frame in _sse_frames(response) if "data" in frame]

    markers = [
        frame["data"]
        for frame in frames
        if frame["data"]["type"] == EventType.SERVER_MUTATION_ACCEPTED
    ]
    assert len(markers) == 1
    marker = markers[0]
    assert marker["session_id"] == session_id
    assert marker["interaction_id"] == frames[0]["data"]["interaction_id"]
    assert marker["interaction_id"] is not None
    assert marker["payload"] == {
        "mutation_id": mutation_id,
        "mutation_kind": "run",
        "accepted_event_id": frames[0]["data"]["id"],
        "accepted_event_sequence": public_event_sequence(frames[0]["data"]["id"]),
        "accepted_event_type": frames[0]["data"]["type"],
    }
    assert frames.index(next(frame for frame in frames if frame["data"] == marker)) > 0

    events = client.get(
        f"/api/sessions/{session_id}/events",
        params={"event_type": EventType.SERVER_MUTATION_ACCEPTED},
    ).json()["events"]
    assert [event["id"] for event in events] == [marker["id"]]
    assert events[0]["interaction_id"] == marker["interaction_id"]


def test_streaming_mutation_id_header_rejects_unsafe_values_before_execution() -> None:
    app = CayuApp()
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.post(
        "/api/run",
        json={"prompt": "hello", "session_id": "session-invalid-mutation-id"},
        headers={"Cayu-Mutation-ID": "invalid mutation id"},
    )

    assert response.status_code == 422
    assert asyncio.run(app.session_store.load_state("session-invalid-mutation-id")) is None


def test_secret_bearing_mutation_id_is_rejected_before_execution() -> None:
    session_id = "sess-secret-id"
    app = CayuApp(
        enable_logging=False,
        secret_redactor=SecretRedactor("u"),
    )
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.post(
        "/api/run",
        json={"prompt": "hello", "session_id": session_id},
        headers={"Cayu-Mutation-ID": "mutation-identity"},
    )

    assert response.status_code == 422
    assert "workload secret" in response.json()["detail"]
    assert asyncio.run(app.session_store.load_state(session_id)) is None


def test_run_disconnect_after_http_acceptance_before_first_body_replays_from_start() -> None:
    task_store = InMemoryTaskStore()
    app = CayuApp(task_store=task_store)
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    server = create_server(app, config=_LOCAL_SERVER_CONFIG)
    session_id = "session_pre_first_disconnect"

    async def exercise_disconnect() -> list[str]:
        messages = await _post_and_disconnect_before_first_body(
            server,
            "/api/run",
            {"prompt": "hello", "session_id": session_id},
        )
        starts = [message for message in messages if message["type"] == "http.response.start"]
        assert [message["status"] for message in starts] == [200]
        assert not any(
            message.get("body") for message in messages if message["type"] == "http.response.body"
        )

        deadline = asyncio.get_running_loop().time() + 5
        while True:
            state = await app.session_store.load_state(session_id)
            if state is not None and state.status is SessionStatus.COMPLETED:
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError(
                    "detached run did not complete after the observer disconnected"
                )
            await asyncio.sleep(0.01)

        tasks = await task_store.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].status is TaskStatus.COMPLETED
        records = await app.session_store.query_events(EventQuery(session_id=session_id, limit=100))
        return [record.event.id for record in records]

    durable_event_ids = asyncio.run(exercise_disconnect())
    assert durable_event_ids

    executed = []

    async def unexpected_run(request):
        executed.append(request)
        if False:
            yield None

    app.run = unexpected_run
    with TestClient(server).stream(
        "POST",
        "/api/run",
        json={"prompt": "ignored", "session_id": session_id},
        headers={"Last-Event-ID": f"{session_id}:"},
    ) as response:
        assert response.status_code == 200
        replayed = [frame["data"]["id"] for frame in _sse_frames(response) if "data" in frame]

    assert replayed == [
        public_event_id(sequence) for sequence in range(1, len(durable_event_ids) + 1)
    ]
    assert executed == []


def test_existing_session_reconnect_cannot_race_accepted_mutation_transition() -> None:
    app = CayuApp()
    session_id = "session_resume_pre_first_disconnect"
    baseline_id = "event_before_resume"
    executions = 0

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_event(
            session_id,
            Event(
                id=baseline_id,
                type=EventType.SESSION_INTERRUPTED,
                session_id=session_id,
                agent_name="assistant",
            ),
        )
        await app.session_store.update_status(session_id, SessionStatus.INTERRUPTED)

    asyncio.run(seed())

    async def accepted_resume(request):
        nonlocal executions
        executions += 1
        await app.session_store.transition_status(
            session_id,
            from_statuses={SessionStatus.INTERRUPTED},
            to_status=SessionStatus.RUNNING,
        )
        resumed = Event(
            id="event_resume_accepted",
            type=EventType.SESSION_RESUMED,
            session_id=session_id,
            agent_name="assistant",
        )
        await app.session_store.append_event(session_id, resumed)
        yield resumed
        await app.session_store.update_status(session_id, SessionStatus.COMPLETED)
        completed = Event(
            id="event_resume_completed",
            type=EventType.SESSION_COMPLETED,
            session_id=session_id,
            agent_name="assistant",
        )
        await app.session_store.append_event(session_id, completed)
        yield completed

    app.resume = accepted_resume
    server = create_server(app, config=_LOCAL_SERVER_CONFIG)

    async def exercise_disconnect() -> None:
        messages = await _post_and_disconnect_before_first_body(
            server,
            "/api/resume",
            {"session_id": session_id, "prompt": "continue"},
        )
        starts = [message for message in messages if message["type"] == "http.response.start"]
        assert [message["status"] for message in starts] == [200]

        deadline = asyncio.get_running_loop().time() + 5
        while True:
            state = await app.session_store.load_state(session_id)
            if state is not None and state.status is SessionStatus.COMPLETED:
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("accepted resume did not complete after disconnect")
            await asyncio.sleep(0.01)

    asyncio.run(exercise_disconnect())
    assert executions == 1

    async def unexpected_resume(request):
        raise AssertionError(f"replay re-executed resume for {request.session_id}")
        yield  # pragma: no cover

    app.resume = unexpected_resume
    with TestClient(server).stream(
        "POST",
        "/api/resume",
        json={"session_id": session_id, "prompt": "ignored"},
        headers={"Last-Event-ID": f"{session_id}:{baseline_id}"},
    ) as response:
        assert response.status_code == 200
        replayed = [frame["data"]["id"] for frame in _sse_frames(response) if "data" in frame]

    assert replayed == [public_event_id(2), public_event_id(3)]
    assert executions == 1


def test_concurrent_client_run_identity_creates_one_session_and_one_task() -> None:
    class CoordinatedRunStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        terminal_interaction_publication_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.claims = 0
            self.both_claiming = asyncio.Event()

        async def create(self, request, *, identity, **kwargs):
            if request.session_id == "session_concurrent_claim":
                self.claims += 1
                if self.claims == 2:
                    self.both_claiming.set()
                await self.both_claiming.wait()
            return await super().create(request, identity=identity, **kwargs)

    store = CoordinatedRunStore()
    task_store = InMemoryTaskStore()
    app = CayuApp(session_store=store, task_store=task_store)
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    server = create_server(app, config=_LOCAL_SERVER_CONFIG)

    async def submit_concurrently():
        transport = httpx.ASGITransport(app=server)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await asyncio.gather(
                client.post(
                    "/api/run",
                    json={"prompt": "first", "session_id": "session_concurrent_claim"},
                ),
                client.post(
                    "/api/run",
                    json={"prompt": "second", "session_id": "session_concurrent_claim"},
                ),
            )

    responses = asyncio.run(submit_concurrently())
    assert sorted(response.status_code for response in responses) == [200, 409]
    conflict = next(response for response in responses if response.status_code == 409)
    assert conflict.json()["detail"] == "Session already exists: session_concurrent_claim"

    tasks = asyncio.run(task_store.list_tasks())
    assert len(tasks) == 1
    assert tasks[0].status is TaskStatus.COMPLETED
    state = asyncio.run(store.load_state("session_concurrent_claim"))
    assert state is not None
    assert state.status is SessionStatus.COMPLETED


def test_client_disconnect_does_not_cancel_detached_run() -> None:
    import time

    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    session_id = None
    with client.stream("POST", "/api/run", json={"prompt": "hello"}) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("data:"):
                session_id = json.loads(line[len("data:") :].strip())["session_id"]
                break  # disconnect after the first event

    assert session_id is not None
    # The run is driven by a detached pump, so it still finishes after the disconnect.
    deadline = time.monotonic() + 10
    status = None
    while time.monotonic() < deadline:
        status = client.get(f"/api/sessions/{session_id}").json()["status"]
        if status == "completed":
            break
        time.sleep(0.05)
    assert status == "completed"


def test_run_stream_failure_emits_terminal_structured_error_frame() -> None:
    app = CayuApp(secret_redactor=SecretRedactor("secret-token"))
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    async def broken_run(request):
        yield event_with_durable_sequence(
            Event(
                type=EventType.SESSION_STARTED,
                session_id=request.session_id,
                agent_name=request.agent_name,
            ),
            1,
        )
        raise RuntimeError("run exploded with secret-token " + "x" * SSE_ERROR_TEXT_MAX_BYTES)

    app.run = broken_run

    with client.stream("POST", "/api/run", json={"prompt": "hello"}) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert frames
    error_frame = frames[-1]
    assert error_frame.get("event") == "error"
    data = error_frame["data"]
    assert data["type"] == "stream.error"
    assert data["kind"] == "runtime"
    assert data["code"] == "runtime_failed"
    assert data["error_type"] == "RuntimeError"
    assert data["retryable"] is False
    assert data["session_id"].startswith("session-")
    assert "secret-token" not in data["error"]
    assert REDACTED_SECRET in data["error"]
    assert data["error"].endswith("... [truncated]")
    assert len(data["error"].encode("utf-8")) <= SSE_ERROR_TEXT_MAX_BYTES


def test_interrupt_stream_uses_same_typed_redacted_runtime_error_contract() -> None:
    app = CayuApp(secret_redactor=SecretRedactor("secret-token"))

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session_interrupt_stream_error",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status(
            "session_interrupt_stream_error", SessionStatus.RUNNING
        )

    asyncio.run(seed())

    async def broken_interrupt(request):
        yield event_with_durable_sequence(
            Event(
                id="event_interrupted",
                type=EventType.SESSION_INTERRUPTED,
                session_id=request.session_id,
                agent_name="assistant",
            ),
            1,
        )
        raise RuntimeError("interrupt failed with secret-token")

    app.interrupt_session = broken_interrupt
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/sessions/session_interrupt_stream_error/interrupt",
        json={"reason": "operator request"},
    ) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert frames[0]["data"]["id"] == public_event_id(1)
    error_frame = frames[-1]
    assert error_frame["event"] == "error"
    assert error_frame["data"]["kind"] == "runtime"
    assert error_frame["data"]["code"] == "runtime_failed"
    assert error_frame["data"]["retryable"] is False
    assert error_frame["data"]["session_id"] == "session_interrupt_stream_error"
    assert "secret-token" not in error_frame["data"]["error"]
    assert REDACTED_SECRET in error_frame["data"]["error"]
