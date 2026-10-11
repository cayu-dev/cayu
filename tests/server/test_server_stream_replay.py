"""HTTP server stream replay contracts."""

from __future__ import annotations

# ruff: noqa: E402
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
httpx = pytest.importorskip("httpx")

import asyncio

from fastapi.testclient import TestClient
from tests.server.server_test_support import _LOCAL_SERVER_CONFIG, OneShotProvider, _sse_frames

from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.runtime._event_projection import (
    REDACTED_CUSTOM_EVENT_TYPE,
    public_event_envelope_alias_field,
    public_event_id,
)
from cayu.server import ServerConfig, ServerLifecycleConfig, create_server
from cayu.server.routes import _next_replay_poll_interval
from cayu.server.sse import SSE_REPLAY_PAGE_EVENTS
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import EventRecord, SessionIdentity, SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.tasks.memory import InMemoryTaskStore
from cayu.vaults.redaction import SecretRedactor

_SHORT_REPLAY_SERVER_CONFIG = ServerConfig.local_development(
    lifecycle=ServerLifecycleConfig(replay_idle_timeout_s=0.01)
)


def test_replay_polling_backs_off_and_resets_after_events() -> None:
    interval = 0.05
    observed = []
    for _ in range(7):
        observed.append(interval)
        interval = _next_replay_poll_interval(interval, received_events=False)

    assert observed == [0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.0]
    assert _next_replay_poll_interval(interval, received_events=True) == 0.05


def test_run_stream_carries_resumable_event_ids_and_replays_on_last_event_id() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    session_id = "session-dashboard-run"
    run_body = {"prompt": "hello", "session_id": session_id}
    with client.stream("POST", "/api/run", json=run_body) as response:
        assert response.status_code == 200
        frames = [frame for frame in _sse_frames(response) if "data" in frame]

    assert frames
    assert frames[0]["data"]["session_id"] == session_id
    # Every frame carries a resumable id of the form `<session_id>:<event_id>`.
    for frame in frames:
        assert frame["id"] == f"{session_id}:{frame['data']['id']}"

    async def fail_unbounded_session_load(*_args, **_kwargs):
        raise AssertionError("SSE replay must use the bounded state projection")

    app.session_store.load = fail_unbounded_session_load  # type: ignore[method-assign]

    executed = []

    async def unexpected_execution(request):
        executed.append(request)
        if False:
            yield None

    app.run = unexpected_execution
    app.resume = unexpected_execution

    # A reconnect with Last-Event-ID replays the persisted events the client missed
    # instead of starting a new run.
    queries = []
    original_query_events = app.session_store.query_events

    async def query_events(query=None):
        queries.append(query)
        return await original_query_events(query)

    app.session_store.query_events = query_events
    with client.stream(
        "POST",
        "/api/run",
        json=run_body,
        headers={"Last-Event-ID": frames[0]["id"]},
    ) as response:
        assert response.status_code == 200
        replayed = [frame for frame in _sse_frames(response) if "data" in frame]

    assert [frame["data"]["id"] for frame in replayed] == [
        frame["data"]["id"] for frame in frames[1:]
    ]
    assert [frame["data"]["interaction_id"] for frame in replayed] == [
        frame["data"]["interaction_id"] for frame in frames[1:]
    ]
    assert replayed[-1]["data"]["type"] == "session.completed"
    alias_lookup = next(query for query in queries if query.after_sequence == 0)
    replay_page = next(query for query in queries if query.after_sequence == 1)
    assert alias_lookup.limit == 1
    assert replay_page.limit == SSE_REPLAY_PAGE_EVENTS

    with client.stream(
        "POST",
        "/api/resume",
        json={"session_id": session_id, "prompt": "ignored during replay"},
        headers={"Last-Event-ID": frames[0]["id"]},
    ) as response:
        assert response.status_code == 200
        resume_replayed = [frame for frame in _sse_frames(response) if "data" in frame]
    assert [frame["data"]["id"] for frame in resume_replayed] == [
        frame["data"]["id"] for frame in frames[1:]
    ]

    with client.stream(
        "POST",
        "/api/run",
        json=run_body,
        headers={"Last-Event-ID": f"{session_id}:"},
    ) as response:
        assert response.status_code == 200
        replayed_from_start = [frame for frame in _sse_frames(response) if "data" in frame]
    assert [frame["data"]["id"] for frame in replayed_from_start] == [
        frame["data"]["id"] for frame in frames
    ]
    assert [frame["data"]["interaction_id"] for frame in replayed_from_start] == [
        frame["data"]["interaction_id"] for frame in frames
    ]

    with client.stream(
        "POST",
        "/api/resume",
        json={"session_id": session_id, "prompt": "ignored during replay"},
        headers={"Last-Event-ID": f"{session_id}:"},
    ) as response:
        assert response.status_code == 200
        resume_replayed_from_start = [frame for frame in _sse_frames(response) if "data" in frame]
    assert [frame["data"]["id"] for frame in resume_replayed_from_start] == [
        frame["data"]["id"] for frame in frames
    ]
    assert executed == []
    # No new session was created by the replay request.
    sessions = client.get("/api/sessions").json()["sessions"]
    assert [session["id"] for session in sessions] == [session_id]
    assert len(client.get("/api/tasks").json()) == 1


def test_legacy_custom_event_uses_one_safe_rest_sse_and_replay_projection() -> None:
    secret = "eventscope-canary"
    session_id = f"session-{secret}"
    private_event_id = f"private-{secret}-event"
    app = CayuApp(
        enable_logging=False,
        secret_redactor=SecretRedactor(secret),
    )

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "inspect legacy event")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_events(
            session_id,
            [
                Event(
                    id=private_event_id,
                    type=f"custom.{secret}",
                    session_id=session_id,
                    payload={
                        secret: secret,
                        "tool_call_id": f"tool-{secret}",
                    },
                ),
                Event(
                    id="private-terminal-event",
                    type=EventType.SESSION_COMPLETED,
                    session_id=session_id,
                ),
            ],
        )
        await app.session_store.update_status(session_id, SessionStatus.COMPLETED)

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    history = client.get(f"/api/sessions/{session_id}/events").json()["events"]
    assert [event["id"] for event in history] == [
        public_event_id(1),
        public_event_id(2),
    ]
    assert history[0]["type"] == REDACTED_CUSTOM_EVENT_TYPE
    assert secret not in repr(history[0])

    for event_id in (public_event_id(1), private_event_id):
        selected = client.get(
            f"/api/sessions/{session_id}/events",
            params={"event_id": event_id},
        ).json()["events"]
        assert [event["id"] for event in selected] == [public_event_id(1)]
        assert secret not in repr(selected)

    with client.stream(
        "POST",
        "/api/run",
        json={"prompt": "ignored", "session_id": session_id},
        headers={"Last-Event-ID": f"{session_id}:"},
    ) as response:
        frames = [frame for frame in _sse_frames(response) if "data" in frame]
    assert [frame["data"]["id"] for frame in frames] == [
        public_event_id(1),
        public_event_id(2),
    ]
    assert frames[0]["data"]["type"] == REDACTED_CUSTOM_EVENT_TYPE
    assert secret not in repr(frames)

    with client.stream(
        "POST",
        "/api/run",
        json={"prompt": "ignored", "session_id": session_id},
        headers={"Last-Event-ID": frames[0]["id"]},
    ) as response:
        projected_session_replay = [frame for frame in _sse_frames(response) if "data" in frame]
    assert [frame["data"]["id"] for frame in projected_session_replay] == [public_event_id(2)]
    assert secret not in repr(projected_session_replay)

    for marker_id in (public_event_id(1), private_event_id):
        with client.stream(
            "POST",
            "/api/run",
            json={"prompt": "ignored", "session_id": session_id},
            headers={"Last-Event-ID": f"{session_id}:{marker_id}"},
        ) as response:
            replay = [frame for frame in _sse_frames(response) if "data" in frame]
        assert [frame["data"]["id"] for frame in replay] == [public_event_id(2)]
        assert secret not in repr(replay)


def test_replay_accepts_server_issued_redacted_colon_bearing_session_marker() -> None:
    secret = "colon-session-canary"
    session_id = f"legacy-{secret}:partition"
    app = CayuApp(
        enable_logging=False,
        secret_redactor=SecretRedactor(secret),
    )

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "inspect legacy session")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_events(
            session_id,
            [
                Event(
                    type=EventType.SESSION_STARTED,
                    session_id=session_id,
                ),
                Event(
                    type=EventType.SESSION_COMPLETED,
                    session_id=session_id,
                ),
            ],
        )
        await app.session_store.update_status(session_id, SessionStatus.COMPLETED)

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/resume",
        json={"prompt": "ignored", "session_id": session_id},
        headers={"Last-Event-ID": f"{session_id}:"},
    ) as response:
        initial = [frame for frame in _sse_frames(response) if "data" in frame]

    assert len(initial) == 2
    assert secret not in initial[0]["id"]
    assert public_event_envelope_alias_field(initial[0]["id"].split(":", 1)[0]) == "session_id"

    with client.stream(
        "POST",
        "/api/resume",
        json={"prompt": "ignored", "session_id": session_id},
        headers={"Last-Event-ID": initial[0]["id"]},
    ) as response:
        replay = [frame for frame in _sse_frames(response) if "data" in frame]

    assert [frame["data"]["id"] for frame in replay] == [public_event_id(2)]
    assert secret not in repr(replay)


def test_unknown_public_event_alias_cannot_collide_with_a_private_event_id() -> None:
    private_sentinel = "__cayu_missing_private_event__"
    app = CayuApp(enable_logging=False)

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session_alias_miss",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_event(
            "session_alias_miss",
            Event(
                id=private_sentinel,
                type=EventType.SESSION_STARTED,
                session_id="session_alias_miss",
            ),
        )

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    missing = client.get(
        "/api/sessions/session_alias_miss/events",
        params={"event_id": public_event_id(999)},
    )
    assert missing.status_code == 200
    assert missing.json()["events"] == []

    legacy_exact = client.get(
        "/api/sessions/session_alias_miss/events",
        params={"event_id": private_sentinel},
    )
    assert [event["id"] for event in legacy_exact.json()["events"]] == [public_event_id(1)]


def test_public_event_aliases_disambiguate_legacy_raw_namespace_collisions() -> None:
    app = CayuApp(enable_logging=False)

    async def seed() -> None:
        for session_id, first_event_id in (
            ("session-alias-ambiguous", public_event_id(2)),
            ("session-alias-raw-only", public_event_id(9)),
            ("session-alias-same-record", public_event_id(5)),
            ("session-alias-malformed-raw", "cayu_event_legacy"),
        ):
            await app.session_store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "hello")],
                ),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
            await app.session_store.append_events(
                session_id,
                [
                    Event(
                        id=first_event_id,
                        type=EventType.SESSION_STARTED,
                        session_id=session_id,
                    ),
                    Event(
                        id=f"{session_id}-completed",
                        type=EventType.SESSION_COMPLETED,
                        session_id=session_id,
                    ),
                ],
            )
            await app.session_store.update_status(session_id, SessionStatus.COMPLETED)

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    ambiguous = client.get(
        "/api/sessions/session-alias-ambiguous/events",
        params={"event_id": public_event_id(2)},
    )
    assert ambiguous.status_code == 409
    assert "ambiguous" in ambiguous.json()["detail"]

    replay_ambiguous = client.post(
        "/api/resume",
        json={"prompt": "ignored", "session_id": "session-alias-ambiguous"},
        headers={"Last-Event-ID": (f"session-alias-ambiguous:{public_event_id(2)}")},
    )
    assert replay_ambiguous.status_code == 409

    raw_only = client.get(
        "/api/sessions/session-alias-raw-only/events",
        params={"event_id": public_event_id(9)},
    )
    assert [event["id"] for event in raw_only.json()["events"]] == [public_event_id(3)]

    same_record = client.get(
        "/api/sessions/session-alias-same-record/events",
        params={"event_id": public_event_id(5)},
    )
    assert [event["id"] for event in same_record.json()["events"]] == [public_event_id(5)]

    malformed_raw = client.get(
        "/api/sessions/session-alias-malformed-raw/events",
        params={"event_id": "cayu_event_legacy"},
    )
    assert [event["id"] for event in malformed_raw.json()["events"]] == [public_event_id(7)]


def test_run_replay_rejects_malformed_last_event_id_and_unknown_session() -> None:
    app = CayuApp()
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    malformed = client.post(
        "/api/run",
        json={"prompt": "hello", "session_id": "session_marker_validation"},
        headers={"Last-Event-ID": "not-a-marker"},
    )
    assert malformed.status_code == 422

    for marker in ("missing_session:event_1", "missing_session:"):
        unknown = client.post(
            "/api/run",
            json={"prompt": "hello", "session_id": "missing_session"},
            headers={"Last-Event-ID": marker},
        )
        assert unknown.status_code == 404


def test_run_replay_missing_session_does_not_expose_resolved_private_identity() -> None:
    private_session_id = "private-session-secret"

    class VanishingSessionStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def load_state(self, session_id: str):
            if session_id == private_session_id:
                return None
            return await super().load_state(session_id)

    store = VanishingSessionStore()
    app = CayuApp(
        session_store=store,
        secret_redactor=SecretRedactor(private_session_id),
        enable_logging=False,
    )

    async def seed() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=private_session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fakemodel"),
        )

    asyncio.run(seed())
    public_session_id = app.project_session_id_for_exposure(private_session_id)
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.post(
        "/api/run",
        json={"prompt": "ignored", "session_id": public_session_id},
        headers={"Last-Event-ID": f"{public_session_id}:"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": f"Session not found: {public_session_id}"}
    assert private_session_id not in response.text


def test_run_replay_rejects_unknown_event_and_mismatched_body_identity() -> None:
    app = CayuApp(task_store=InMemoryTaskStore())

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session_marker_validation",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_event(
            "session_marker_validation",
            Event(
                id="event_seen",
                type=EventType.SESSION_STARTED,
                session_id="session_marker_validation",
                agent_name="assistant",
            ),
        )
        await app.session_store.update_status(
            "session_marker_validation",
            SessionStatus.INTERRUPTED,
        )

    asyncio.run(seed())
    executed = []

    async def unexpected_execution(request):
        executed.append(request)
        if False:
            yield None

    app.run = unexpected_execution
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    unknown_event = client.post(
        "/api/run",
        json={"prompt": "ignored", "session_id": "session_marker_validation"},
        headers={"Last-Event-ID": "session_marker_validation:event_missing"},
    )
    mismatched_session = client.post(
        "/api/run",
        json={"prompt": "ignored", "session_id": "session_marker_validation"},
        headers={"Last-Event-ID": "session_other:event_seen"},
    )
    malformed_public_alias = client.post(
        "/api/run",
        json={"prompt": "ignored", "session_id": "session_marker_validation"},
        headers={
            "Last-Event-ID": (
                f"session_marker_validation:cayu_event_{MAX_DURABLE_JSON_INTEGER + 1}"
            )
        },
    )

    assert unknown_event.status_code == 409
    assert "event was not found" in unknown_event.json()["detail"]
    assert mismatched_session.status_code == 422
    assert "does not match" in mismatched_session.json()["detail"]
    assert malformed_public_alias.status_code == 422
    assert "malformed Cayu public event alias" in malformed_public_alias.json()["detail"]
    assert executed == []
    assert client.get("/api/tasks").json() == []


def test_replay_of_active_session_times_out_with_structured_error() -> None:
    app = CayuApp()

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session_stranded",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.update_status("session_stranded", SessionStatus.RUNNING)
        await app.session_store.append_event(
            "session_stranded",
            Event(
                id="event_seen",
                type=EventType.SESSION_STARTED,
                session_id="session_stranded",
                agent_name="assistant",
            ),
        )

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_SHORT_REPLAY_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/run",
        json={"prompt": "ignored during replay"},
        headers={"Last-Event-ID": "session_stranded:event_seen"},
    ) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert frames[-1]["event"] == "error"
    assert frames[-1]["data"]["kind"] == "observer"
    assert frames[-1]["data"]["code"] == "replay_idle_timeout"
    assert frames[-1]["data"]["retryable"] is True
    assert frames[-1]["data"]["session_id"] == "session_stranded"
    assert frames[-1]["data"]["error_type"] == "TimeoutError"
    assert "session_stranded" in frames[-1]["data"]["error"]


def test_replay_waits_for_terminal_event_after_terminal_status() -> None:
    class DelayedTerminalEventStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.injected_terminal_race = False
            self.terminal_append_task: asyncio.Task[None] | None = None

        async def query_events(
            self,
            query: EventQuery | None = None,
        ) -> list[EventRecord]:
            records = await super().query_events(query)
            if (
                query is not None
                and query.session_id is not None
                and query.after_sequence is not None
                and query.event_id is None
                and query.event_type is None
                and not query.event_types
                and not self.injected_terminal_race
            ):
                self.injected_terminal_race = True
                session_id = query.session_id
                await self.update_status(session_id, SessionStatus.INTERRUPTED)

                async def append_terminal_event() -> None:
                    await asyncio.sleep(0.05)
                    await self.append_event(
                        session_id,
                        Event(
                            id="event_delayed_terminal",
                            type=EventType.SESSION_INTERRUPTED,
                            session_id=session_id,
                            agent_name="assistant",
                        ),
                    )

                self.terminal_append_task = asyncio.create_task(append_terminal_event())
            return records

    store = DelayedTerminalEventStore()
    app = CayuApp(session_store=store, enable_logging=False)
    session_id = "session_replay_terminal_status_race"

    async def exercise() -> httpx.Response:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_events(
            session_id,
            [
                Event(
                    id="event_initial_start",
                    type=EventType.SESSION_STARTED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
                Event(
                    id="event_previous_terminal",
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
            ],
        )
        await store.update_status(session_id, SessionStatus.RUNNING)
        await store.append_event(
            session_id,
            Event(
                id="event_resume_baseline",
                type=EventType.SESSION_RESUMED,
                session_id=session_id,
                agent_name="assistant",
            ),
        )

        transport = httpx.ASGITransport(app=create_server(app, config=_LOCAL_SERVER_CONFIG))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/api/resume",
                json={"session_id": session_id, "prompt": "ignored during replay"},
                headers={"Last-Event-ID": f"{session_id}:event_previous_terminal"},
            )
        if store.terminal_append_task is not None:
            await store.terminal_append_task
        return response

    response = asyncio.run(exercise())

    assert response.status_code == 200
    frames = [frame for frame in _sse_frames(response) if "data" in frame]
    assert [frame["data"]["id"] for frame in frames] == [
        public_event_id(3),
        public_event_id(4),
    ]
    assert frames[-1]["data"]["type"] == EventType.SESSION_INTERRUPTED


def test_replay_recognizes_post_terminal_hook_marker() -> None:
    app = CayuApp(enable_logging=False)
    session_id = "session_replay_post_terminal_hook"

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_events(
            session_id,
            [
                Event(
                    id="event_terminal",
                    type=EventType.SESSION_COMPLETED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
                Event(
                    id="event_hook_completed",
                    type=EventType.HOOK_COMPLETED,
                    session_id=session_id,
                    agent_name="assistant",
                    payload={
                        "terminal_event_id": "event_terminal",
                        "terminal_event_type": str(EventType.SESSION_COMPLETED),
                    },
                ),
            ],
        )
        await app.session_store.update_status(session_id, SessionStatus.COMPLETED)

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_SHORT_REPLAY_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/resume",
        json={"session_id": session_id, "prompt": "ignored during replay"},
        headers={"Last-Event-ID": f"{session_id}:event_hook_completed"},
    ) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert frames == []


@pytest.mark.parametrize(
    "hook_event_type",
    [EventType.HOOK_STARTED, EventType.HOOK_COMPLETED, EventType.HOOK_FAILED],
)
def test_replay_does_not_attach_stale_hook_marker_across_operation_start(
    hook_event_type: EventType,
) -> None:
    class DelayedTerminalStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.terminal_append_task: asyncio.Task[None] | None = None

        async def load_state(self, session_id: str):
            state = await super().load_state(session_id)
            if self.terminal_append_task is None:

                async def append_terminal_event() -> None:
                    await asyncio.sleep(0.05)
                    await self.append_event(
                        session_id,
                        Event(
                            id="event_current_terminal",
                            type=EventType.SESSION_INTERRUPTED,
                            session_id=session_id,
                            agent_name="assistant",
                        ),
                    )

                self.terminal_append_task = asyncio.create_task(append_terminal_event())
            return state

    store = DelayedTerminalStore()
    app = CayuApp(session_store=store, enable_logging=False)
    session_id = f"session_replay_stale_{hook_event_type.value}"

    async def exercise() -> httpx.Response:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_events(
            session_id,
            [
                Event(
                    id="event_previous_terminal",
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
                Event(
                    id="event_operation_start",
                    type=EventType.SESSION_RESUMED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
                Event(
                    id="event_stale_hook",
                    type=hook_event_type,
                    session_id=session_id,
                    agent_name="assistant",
                    payload={
                        "terminal_event_id": "event_previous_terminal",
                        "terminal_event_type": str(EventType.SESSION_INTERRUPTED),
                    },
                ),
            ],
        )
        await store.update_status(session_id, SessionStatus.INTERRUPTED)

        transport = httpx.ASGITransport(app=create_server(app, config=_LOCAL_SERVER_CONFIG))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/api/resume",
                json={"session_id": session_id, "prompt": "ignored during replay"},
                headers={"Last-Event-ID": f"{session_id}:event_stale_hook"},
            )
        if store.terminal_append_task is not None:
            await store.terminal_append_task
        return response

    response = asyncio.run(exercise())

    assert response.status_code == 200
    frames = [frame for frame in _sse_frames(response) if "data" in frame]
    assert [frame["data"]["id"] for frame in frames] == [public_event_id(4)]


def test_replay_unverified_hook_does_not_erase_observed_terminal_boundary() -> None:
    app = CayuApp(enable_logging=False)
    session_id = "session_replay_unverified_hook_after_terminal"

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_events(
            session_id,
            [
                Event(
                    id="event_operation_start",
                    type=EventType.SESSION_STARTED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
                Event(
                    id="event_terminal",
                    type=EventType.SESSION_COMPLETED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
                Event(
                    id="event_unverified_hook",
                    type=EventType.HOOK_COMPLETED,
                    session_id=session_id,
                    agent_name="assistant",
                    payload={
                        "terminal_event_id": "event_missing_terminal",
                        "terminal_event_type": str(EventType.SESSION_COMPLETED),
                    },
                ),
            ],
        )
        await app.session_store.update_status(session_id, SessionStatus.COMPLETED)

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_SHORT_REPLAY_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/resume",
        json={"session_id": session_id, "prompt": "ignored during replay"},
        headers={"Last-Event-ID": f"{session_id}:event_operation_start"},
    ) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert [frame["data"]["id"] for frame in frames] == [
        public_event_id(2),
        public_event_id(3),
    ]


def test_replay_does_not_accept_custom_event_as_terminal_lineage() -> None:
    app = CayuApp(enable_logging=False)
    session_id = "session_replay_forged_terminal_lineage"

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_events(
            session_id,
            [
                Event(
                    id="event_terminal",
                    type=EventType.SESSION_COMPLETED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
                Event(
                    id="event_custom",
                    type="custom.forged_terminal_lineage",
                    session_id=session_id,
                    agent_name="assistant",
                    payload={
                        "terminal_event_id": "event_terminal",
                        "terminal_event_type": str(EventType.SESSION_COMPLETED),
                    },
                ),
            ],
        )
        await app.session_store.update_status(session_id, SessionStatus.COMPLETED)

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_SHORT_REPLAY_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/resume",
        json={"session_id": session_id, "prompt": "ignored during replay"},
        headers={"Last-Event-ID": f"{session_id}:event_custom"},
    ) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert frames[-1]["event"] == "error"
    assert frames[-1]["data"]["code"] == "replay_idle_timeout"


@pytest.mark.parametrize(
    "post_terminal_event_type",
    [
        EventType.SERVER_MUTATION_ACCEPTED,
        EventType.SESSION_INTERRUPTION_CASCADE_RETRY_REQUESTED,
        EventType.SESSION_INTERRUPTION_CASCADE_COMPLETED,
        EventType.SESSION_INTERRUPTION_CASCADE_FAILED,
    ],
)
def test_replay_recognizes_runtime_post_terminal_marker(
    post_terminal_event_type: EventType,
) -> None:
    app = CayuApp(enable_logging=False)
    session_id = f"session_replay_{post_terminal_event_type.value}"

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_events(
            session_id,
            [
                Event(
                    id="event_terminal",
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
                Event(
                    id="event_post_terminal",
                    type=post_terminal_event_type,
                    session_id=session_id,
                    agent_name="assistant",
                ),
            ],
        )
        await app.session_store.update_status(session_id, SessionStatus.INTERRUPTED)

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_SHORT_REPLAY_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/resume",
        json={"session_id": session_id, "prompt": "ignored during replay"},
        headers={"Last-Event-ID": f"{session_id}:event_post_terminal"},
    ) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert frames == []


def test_replay_cascade_marker_uses_latest_completed_operation_boundary() -> None:
    app = CayuApp(enable_logging=False)
    session_id = "session_replay_stale_cascade_after_completion"

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_events(
            session_id,
            [
                Event(
                    id="event_previous_interrupt",
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
                Event(
                    id="event_operation_start",
                    type=EventType.SESSION_RESUMED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
                Event(
                    id="event_current_completion",
                    type=EventType.SESSION_COMPLETED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
                Event(
                    id="event_stale_cascade",
                    type=EventType.SESSION_INTERRUPTION_CASCADE_COMPLETED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
            ],
        )
        await app.session_store.update_status(session_id, SessionStatus.COMPLETED)

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_SHORT_REPLAY_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/resume",
        json={"session_id": session_id, "prompt": "ignored during replay"},
        headers={"Last-Event-ID": f"{session_id}:event_stale_cascade"},
    ) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert frames == []


def test_replay_does_not_attach_stale_cascade_marker_across_operation_start() -> None:
    class DelayedTerminalStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.terminal_append_task: asyncio.Task[None] | None = None

        async def load_state(self, session_id: str):
            state = await super().load_state(session_id)
            if self.terminal_append_task is None:

                async def append_terminal_event() -> None:
                    await asyncio.sleep(0.05)
                    await self.append_event(
                        session_id,
                        Event(
                            id="event_current_terminal",
                            type=EventType.SESSION_INTERRUPTED,
                            session_id=session_id,
                            agent_name="assistant",
                        ),
                    )

                self.terminal_append_task = asyncio.create_task(append_terminal_event())
            return state

    store = DelayedTerminalStore()
    app = CayuApp(session_store=store, enable_logging=False)
    session_id = "session_replay_stale_cascade_marker"

    async def exercise() -> httpx.Response:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_events(
            session_id,
            [
                Event(
                    id="event_previous_terminal",
                    type=EventType.SESSION_INTERRUPTED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
                Event(
                    id="event_operation_start",
                    type=EventType.SESSION_RESUMED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
                Event(
                    id="event_stale_cascade",
                    type=EventType.SESSION_INTERRUPTION_CASCADE_COMPLETED,
                    session_id=session_id,
                    agent_name="assistant",
                ),
            ],
        )
        await store.update_status(session_id, SessionStatus.INTERRUPTED)

        transport = httpx.ASGITransport(app=create_server(app, config=_LOCAL_SERVER_CONFIG))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/api/resume",
                json={"session_id": session_id, "prompt": "ignored during replay"},
                headers={"Last-Event-ID": f"{session_id}:event_stale_cascade"},
            )
        if store.terminal_append_task is not None:
            await store.terminal_append_task
        return response

    response = asyncio.run(exercise())

    assert response.status_code == 200
    frames = [frame for frame in _sse_frames(response) if "data" in frame]
    assert [frame["data"]["id"] for frame in frames] == [public_event_id(4)]


def test_replay_streams_complete_history_in_bounded_pages() -> None:
    app = CayuApp()
    event_count = 1_001

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session_replay_bound",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_events(
            "session_replay_bound",
            [
                Event(
                    id="event_seen",
                    type=EventType.SESSION_STARTED,
                    session_id="session_replay_bound",
                    agent_name="assistant",
                ),
                *[
                    Event(
                        id=f"event_{index}",
                        type="custom.replay",
                        session_id="session_replay_bound",
                        agent_name="assistant",
                    )
                    for index in range(event_count)
                ],
                Event(
                    id="event_replay_terminal",
                    type=EventType.SESSION_INTERRUPTED,
                    session_id="session_replay_bound",
                    agent_name="assistant",
                ),
            ],
        )
        await app.session_store.update_status("session_replay_bound", SessionStatus.INTERRUPTED)

    asyncio.run(seed())
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream(
        "POST",
        "/api/run",
        json={"prompt": "ignored during replay"},
        headers={"Last-Event-ID": "session_replay_bound:event_seen"},
    ) as response:
        assert response.status_code == 200
        frames = _sse_frames(response)

    assert all(frame.get("event") != "error" for frame in frames)
    assert len(frames) == event_count + 1
    event_frames = frames
    assert event_frames[0]["data"]["id"] == public_event_id(2)
    assert event_frames[-2]["data"]["id"] == public_event_id(event_count + 1)
    assert event_frames[-1]["data"]["id"] == public_event_id(event_count + 2)


@pytest.mark.parametrize(
    ("path", "body"),
    [
        (
            "/api/tool-approvals/resolve",
            {
                "session_id": "session_approval_replay",
                "approval_id": "approval_1",
                "tool_round_id": "round_1",
                "tool_call_id": "call_1",
                "decision": "approve",
            },
        ),
        (
            "/api/tool-approvals/recover",
            {
                "session_id": "session_approval_replay",
                "approval_id": "approval_1",
                "tool_round_id": "round_1",
                "tool_call_id": "call_1",
                "outcome": "completed",
                "message": "confirmed externally",
            },
        ),
        (
            "/api/tool-rounds/recover",
            {
                "session_id": "session_approval_replay",
                "round_id": "round_1",
                "tool_call_id": "call_1",
                "outcome": "completed",
                "message": "confirmed externally",
            },
        ),
        (
            "/api/user-input/resolve",
            {
                "session_id": "session_approval_replay",
                "input_id": "input_1",
                "answer": "continue",
            },
        ),
        (
            "/api/user-input/recover",
            {
                "session_id": "session_approval_replay",
                "input_id": "input_1",
                "answer": "continue",
                "tool_call_id": "call_1",
                "outcome": "completed",
                "message": "confirmed externally",
            },
        ),
        (
            "/api/sessions/session_approval_replay/interrupt",
            {},
        ),
        (
            "/api/sessions/session_approval_replay/compact",
            {
                "idempotency_key": "compact-replay",
                "expected_run_epoch": 0,
                "expected_transcript_cursor": 1,
            },
        ),
    ],
)
@pytest.mark.parametrize(
    ("last_event_id", "expected_event_ids"),
    [
        ("session_approval_replay:event_seen", [public_event_id(2)]),
        (
            "session_approval_replay:",
            [public_event_id(1), public_event_id(2)],
        ),
    ],
)
def test_mutation_routes_replay_without_reexecuting(
    path: str,
    body: dict,
    last_event_id: str,
    expected_event_ids: list[str],
) -> None:
    app = CayuApp()

    async def seed() -> None:
        await app.session_store.create(
            RunRequest(
                agent_name="assistant",
                session_id="session_approval_replay",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await app.session_store.append_events(
            "session_approval_replay",
            [
                Event(
                    id="event_seen",
                    type=EventType.SESSION_STARTED,
                    session_id="session_approval_replay",
                    agent_name="assistant",
                ),
                Event(
                    id="event_missed",
                    type=EventType.SESSION_INTERRUPTED,
                    session_id="session_approval_replay",
                    agent_name="assistant",
                ),
            ],
        )
        await app.session_store.update_status("session_approval_replay", SessionStatus.INTERRUPTED)

    asyncio.run(seed())
    executed = []

    async def unexpected_execution(request):
        executed.append(request)
        if False:
            yield None

    app.resolve_tool_approval = unexpected_execution
    app.recover_tool_approval = unexpected_execution
    app.recover_tool_round = unexpected_execution
    app.resolve_user_input = unexpected_execution
    app.recover_user_input = unexpected_execution
    app.interrupt_session = unexpected_execution
    app.compact_session = unexpected_execution
    app.enqueue_session_message = unexpected_execution
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    with client.stream(
        "POST",
        path,
        json=body,
        headers={"Last-Event-ID": last_event_id},
    ) as response:
        assert response.status_code == 200
        frames = [frame for frame in _sse_frames(response) if "data" in frame]

    assert [frame["data"]["id"] for frame in frames] == expected_event_ids
    assert executed == []


def test_session_scoped_replay_rejects_marker_for_different_session() -> None:
    app = CayuApp()
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))

    response = client.post(
        "/api/tool-approvals/resolve",
        json={
            "session_id": "session_requested",
            "approval_id": "approval_1",
            "tool_round_id": "round_1",
            "tool_call_id": "call_1",
            "decision": "approve",
        },
        headers={"Last-Event-ID": "session_other:event_seen"},
    )

    assert response.status_code == 422
    assert "does not match" in response.json()["detail"]
