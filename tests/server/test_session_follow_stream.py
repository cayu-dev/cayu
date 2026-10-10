from __future__ import annotations

# ruff: noqa: E402
import asyncio
import itertools
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
httpx = pytest.importorskip("httpx")

from fastapi import HTTPException, Request

from cayu import CayuApp, PostgresSessionStore, SQLiteSessionStore
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.runtime._event_projection import REDACTED_CUSTOM_EVENT_TYPE, public_event_id
from cayu.server import ServerConfig, ServerLifecycleConfig, create_server
from cayu.server.sse import SSE_REPLAY_PAGE_EVENTS
from cayu.sessions.base import InMemorySessionStore
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.records import EventRecord, SessionIdentity, SessionStateSnapshot, SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.storage.migrations import SchemaMode
from cayu.vaults.redaction import SecretRedactor

_STORE_KINDS = ("memory", "sqlite", "postgres")


def _parse_sse(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Split an SSE body into frames and comment lines."""

    frames: list[dict[str, Any]] = []
    comments: list[str] = []
    current: dict[str, Any] = {}
    for line in text.replace("\r\n", "\n").split("\n"):
        if not line:
            if current:
                frames.append(current)
                current = {}
            continue
        if line.startswith(":"):
            comments.append(line[1:].strip())
        elif line.startswith("id:"):
            current["id"] = line[len("id:") :].strip()
        elif line.startswith("event:"):
            current["event"] = line[len("event:") :].strip()
        elif line.startswith("data:"):
            current["data"] = json.loads(line[len("data:") :].strip())
    if current:
        frames.append(current)
    return frames, comments


def _event_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [frame for frame in frames if "event" not in frame]


def _stream_path(session_id: str) -> str:
    return f"/api/sessions/{session_id}/events/stream"


async def _create_session(
    store: Any,
    session_id: str,
    *,
    status: SessionStatus = SessionStatus.RUNNING,
    events: tuple[Event, ...] = (),
) -> None:
    await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[Message.text("user", "hello")],
        ),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )
    if events:
        await store.append_events(session_id, list(events))
    await store.update_status(session_id, status)


def _event(session_id: str, event_id: str, event_type: Any, **fields: Any) -> Event:
    return Event(
        id=event_id,
        type=event_type,
        session_id=session_id,
        agent_name="assistant",
        **fields,
    )


@asynccontextmanager
async def _asgi_client(app: CayuApp, config: ServerConfig) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=create_server(app, config=config))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


async def _drop_postgres_cayu_tables(dsn: str) -> None:
    import psycopg
    from psycopg import sql

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = current_schema() AND tablename LIKE 'cayu\\_%' ESCAPE '\\'"
            )
            for (table_name,) in await cur.fetchall():
                await cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(table_name))
                )
        await conn.commit()


@asynccontextmanager
async def _durable_store_opener(
    kind: str,
    fixture: Any,
) -> AsyncIterator[Callable[[], Any]]:
    """Yield a factory that opens a fresh store over one durable database.

    A second store instance stands in for a restarted server process.
    """

    if kind == "memory":
        store = InMemorySessionStore()
        yield lambda: store
        return
    if kind == "sqlite":
        async with fixture as resources:
            path = resources.path()
            yield lambda: resources.own(SQLiteSessionStore(path))
        return
    await _drop_postgres_cayu_tables(fixture)
    opened: list[PostgresSessionStore] = []

    def open_postgres() -> PostgresSessionStore:
        store = PostgresSessionStore(fixture, schema_mode=SchemaMode.CREATE)
        opened.append(store)
        return store

    try:
        yield open_postgres
    finally:
        for store in opened:
            await store.close()
        await _drop_postgres_cayu_tables(fixture)


@pytest.mark.parametrize("store_kind", _STORE_KINDS)
def test_follow_stream_long_poll_and_state_etag_match_across_stores(
    store_kind: str,
    request: pytest.FixtureRequest,
) -> None:
    fixture = None
    if store_kind == "sqlite":
        fixture = request.getfixturevalue("sqlite_resources")
    elif store_kind == "postgres":
        fixture = request.getfixturevalue("postgres_dsn")
    session_id = f"session-follow-{store_kind}"
    config = ServerConfig.local_development()

    async def scenario() -> None:
        async with _durable_store_opener(store_kind, fixture) as open_store:
            store = open_store()
            app = CayuApp(session_store=store, enable_logging=False)
            await _create_session(
                store,
                session_id,
                events=(
                    _event(session_id, "e1", EventType.SESSION_STARTED, interaction_id="turn-1"),
                    _event(session_id, "e2", "custom.delta", interaction_id="turn-1"),
                    _event(session_id, "e3", "custom.progress", interaction_id="turn-2"),
                ),
            )
            async with _asgi_client(app, config) as client:
                follow = asyncio.create_task(client.get(_stream_path(session_id)))
                long_poll = asyncio.create_task(
                    client.get(
                        f"/api/sessions/{session_id}/events",
                        params={"after_sequence": 3, "wait_seconds": 20},
                    )
                )
                state = await client.get(f"/api/sessions/{session_id}/state")
                assert state.status_code == 200
                etag = state.headers["etag"]
                assert state.headers["cache-control"] == "private, no-cache"
                unchanged = await client.get(
                    f"/api/sessions/{session_id}/state",
                    headers={"If-None-Match": etag},
                )
                assert unchanged.status_code == 304
                assert unchanged.content == b""
                assert unchanged.headers["etag"] == etag

                await store.append_event(
                    session_id,
                    _event(session_id, "e4", "custom.progress", interaction_id="turn-2"),
                )
                polled = await long_poll
                assert polled.status_code == 200
                assert [event["sequence"] for event in polled.json()["events"]] == [4]

                await store.append_events(
                    session_id,
                    [
                        _event(session_id, "e5", "custom.delta", interaction_id="turn-2"),
                        _event(session_id, "e6", EventType.SESSION_COMPLETED),
                    ],
                )
                await store.update_status(session_id, SessionStatus.COMPLETED)
                response = await follow
                changed = await client.get(
                    f"/api/sessions/{session_id}/state",
                    headers={"If-None-Match": etag},
                )
                assert changed.status_code == 200
                assert changed.json()["status"] == "completed"
                assert changed.headers["etag"] != etag

            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            frames, _ = _parse_sse(response.text)
            events = _event_frames(frames)
            assert [frame["data"]["id"] for frame in events] == [
                public_event_id(sequence) for sequence in range(1, 7)
            ]
            assert [frame["id"] for frame in events] == [
                f"{session_id}:{public_event_id(sequence)}" for sequence in range(1, 7)
            ]
            assert frames[-1] == {
                "event": "end",
                "data": {
                    "type": "session.follow.end",
                    "session_id": session_id,
                    "status": "completed",
                },
            }

            # A restarted server over the same durable store resumes exactly
            # after the last received marker, with no gap or duplicate.
            restarted_app = CayuApp(session_store=open_store(), enable_logging=False)
            async with _asgi_client(restarted_app, config) as client:
                resumed = await client.get(
                    _stream_path(session_id),
                    params={"after_sequence": 1},
                    headers={"Last-Event-ID": events[2]["id"]},
                )
                resumed_frames, _ = _parse_sse(resumed.text)
                assert [frame["data"]["id"] for frame in _event_frames(resumed_frames)] == [
                    public_event_id(4),
                    public_event_id(5),
                    public_event_id(6),
                ]
                assert resumed_frames[-1]["event"] == "end"

                from_sequence = await client.get(
                    _stream_path(session_id), params={"after_sequence": 4}
                )
                sequence_frames, _ = _parse_sse(from_sequence.text)
                assert [frame["data"]["id"] for frame in _event_frames(sequence_frames)] == [
                    public_event_id(5),
                    public_event_id(6),
                ]

                listed = await client.get(f"/api/sessions/{session_id}/events")
                public_turn_2 = listed.json()["events"][2]["interaction_id"]
                for params in (
                    {"event_type": "custom.delta"},
                    {"exclude_event_type": "custom.delta"},
                    {"interaction_id": public_turn_2},
                    {"interaction_id": public_turn_2, "exclude_event_type": "custom.delta"},
                ):
                    expected = await client.get(f"/api/sessions/{session_id}/events", params=params)
                    streamed = await client.get(_stream_path(session_id), params=params)
                    streamed_frames, _ = _parse_sse(streamed.text)
                    assert [frame["data"] for frame in _event_frames(streamed_frames)] == [
                        {
                            key: value
                            for key, value in event.items()
                            if key != "sequence"
                            and not (key in {"environment_name", "workflow_name"} and value is None)
                        }
                        for event in expected.json()["events"]
                    ], params
                    assert streamed_frames[-1]["event"] == "end"

                unknown = await client.get(
                    _stream_path(session_id),
                    headers={"Last-Event-ID": f"{session_id}:{public_event_id(99)}"},
                )
                assert unknown.status_code == 409

                drained = await client.get(
                    f"/api/sessions/{session_id}/events",
                    params={"after_sequence": 6, "wait_seconds": 20},
                )
                assert drained.status_code == 200
                assert drained.json()["events"] == []

    asyncio.run(scenario())


def test_follow_stream_is_read_only() -> None:
    app = CayuApp(enable_logging=False)
    session_id = "session-follow-read-only"

    async def scenario() -> None:
        store = app.session_store
        await _create_session(
            store,
            session_id,
            status=SessionStatus.INTERRUPTED,
            events=(
                _event(session_id, "e1", EventType.SESSION_STARTED),
                _event(session_id, "e2", EventType.SESSION_INTERRUPTED),
            ),
        )
        before_session = await store.load(session_id)
        before_events = await store.query_events(EventQuery(session_id=session_id, limit=100))
        async with _asgi_client(app, ServerConfig.local_development()) as client:
            for headers in (
                {},
                {"Cayu-Mutation-ID": "follow-mutation"},
                {"Last-Event-ID": f"{session_id}:", "Cayu-Mutation-ID": "follow-mutation"},
            ):
                response = await client.get(_stream_path(session_id), headers=headers)
                assert response.status_code == 200
                frames, _ = _parse_sse(response.text)
                assert frames[-1]["data"]["status"] == "interrupted"
            rejected = await client.post(_stream_path(session_id), json={"prompt": "resume"})
            assert rejected.status_code == 405
        after_events = await store.query_events(EventQuery(session_id=session_id, limit=100))
        assert after_events == before_events
        assert all(
            record.event.type != EventType.SERVER_MUTATION_ACCEPTED for record in after_events
        )
        assert await store.load(session_id) == before_session

    asyncio.run(scenario())


def test_follow_stream_uses_list_endpoint_exposure_projection() -> None:
    secret = "follow-canary"
    session_id = f"session-{secret}"
    app = CayuApp(enable_logging=False, secret_redactor=SecretRedactor(secret))

    async def scenario() -> None:
        await _create_session(
            app.session_store,
            session_id,
            status=SessionStatus.COMPLETED,
            events=(
                _event(session_id, f"private-{secret}", f"custom.{secret}", payload={secret: 1}),
                _event(session_id, "terminal", EventType.SESSION_COMPLETED),
            ),
        )
        async with _asgi_client(app, ServerConfig.local_development()) as client:
            history = (await client.get(f"/api/sessions/{session_id}/events")).json()["events"]
            response = await client.get(_stream_path(session_id))
            frames, _ = _parse_sse(response.text)
            events = _event_frames(frames)
            assert [frame["data"]["id"] for frame in events] == [e["id"] for e in history]
            assert events[0]["data"]["type"] == REDACTED_CUSTOM_EVENT_TYPE
            assert events[0]["data"]["payload"] == history[0]["payload"]
            assert secret not in response.text
            public_session_id = frames[-1]["data"]["session_id"]
            assert public_session_id == history[0]["session_id"]

            # The server-issued projected marker resumes the same stream.
            resumed = await client.get(
                _stream_path(session_id),
                headers={"Last-Event-ID": events[0]["id"]},
            )
            resumed_frames, _ = _parse_sse(resumed.text)
            assert [frame["data"]["id"] for frame in _event_frames(resumed_frames)] == [
                public_event_id(2)
            ]
            assert secret not in resumed.text

    asyncio.run(scenario())


def test_follow_stream_requires_the_list_endpoint_authentication() -> None:
    app = CayuApp(enable_logging=False)

    def auth(request: Request) -> dict[str, str]:
        if request.headers.get("x-test-user") != "operator":
            raise HTTPException(status_code=401, detail="Unauthorized")
        return {"subject": "operator"}

    async def scenario() -> None:
        await _create_session(
            app.session_store,
            "session-auth",
            status=SessionStatus.COMPLETED,
            events=(_event("session-auth", "e1", EventType.SESSION_COMPLETED),),
        )
        async with _asgi_client(app, ServerConfig.protected(auth)) as client:
            for path in ("/api/sessions/session-auth/events", _stream_path("session-auth")):
                assert (await client.get(path)).status_code == 401
                allowed = await client.get(path, headers={"x-test-user": "operator"})
                assert allowed.status_code == 200

    asyncio.run(scenario())


def test_follow_stream_rejects_invalid_markers_and_filters() -> None:
    app = CayuApp(enable_logging=False)
    session_id = "session-follow-markers"

    async def scenario() -> None:
        await _create_session(
            app.session_store,
            session_id,
            status=SessionStatus.COMPLETED,
            events=(_event(session_id, "e1", EventType.SESSION_COMPLETED),),
        )
        await _create_session(app.session_store, "session-other", status=SessionStatus.COMPLETED)
        async with _asgi_client(app, ServerConfig.local_development()) as client:
            path = _stream_path(session_id)
            for marker, status_code in (
                ("no-separator", 422),
                (f"session-other:{public_event_id(1)}", 422),
                (f"{session_id}:unknown-event", 409),
            ):
                response = await client.get(path, headers={"Last-Event-ID": marker})
                assert response.status_code == status_code, marker
            assert (await client.get(_stream_path("session-missing"))).status_code == 404
            assert (await client.get(path, params={"after_sequence": -1})).status_code == 422
            events_path = f"/api/sessions/{session_id}/events"
            for params in (
                {"wait_seconds": 26},
                {"wait_seconds": 1, "before_sequence": 2},
                {"wait_seconds": 1, "order_by": "sequence_desc"},
            ):
                assert (await client.get(events_path, params=params)).status_code == 422

    asyncio.run(scenario())


def test_follow_stream_closes_terminal_session_after_post_terminal_boundary() -> None:
    app = CayuApp(enable_logging=False)
    session_id = "session-follow-post-terminal"

    async def scenario() -> None:
        await _create_session(
            app.session_store,
            session_id,
            status=SessionStatus.FAILED,
            events=(
                _event(session_id, "e1", EventType.SESSION_STARTED),
                _event(session_id, "e2", EventType.SESSION_FAILED),
                _event(session_id, "e3", "custom.audit"),
            ),
        )
        async with _asgi_client(
            app, ServerConfig.local_development(lifecycle=ServerLifecycleConfig())
        ) as client:
            for params in ({"after_sequence": 3}, {"after_sequence": 50}):
                response = await asyncio.wait_for(
                    client.get(_stream_path(session_id), params=params),
                    timeout=10,
                )
                frames, _ = _parse_sse(response.text)
                assert frames == [
                    {
                        "event": "end",
                        "data": {
                            "type": "session.follow.end",
                            "session_id": session_id,
                            "status": "failed",
                        },
                    }
                ]

    asyncio.run(scenario())


class _ObservedSessionStore(InMemorySessionStore):
    """Record follow-loop store reads and signal when a follower is polling."""

    def __init__(self) -> None:
        super().__init__()
        self.page_reads: list[float] = []
        self.state_reads = 0
        self.follower_polling = asyncio.Event()

    async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
        if (
            query is not None
            and query.limit == SSE_REPLAY_PAGE_EVENTS
            and query.event_type is None
            and not query.event_types
        ):
            self.page_reads.append(asyncio.get_running_loop().time())
            self.follower_polling.set()
        return await super().query_events(query)

    async def load_state(self, session_id: str) -> SessionStateSnapshot | None:
        self.state_reads += 1
        return await super().load_state(session_id)


def test_follow_stream_sends_heartbeats_and_closes_idle_streams() -> None:
    store = _ObservedSessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    session_id = "session-follow-heartbeat"
    config = ServerConfig.local_development(
        lifecycle=ServerLifecycleConfig(
            replay_idle_timeout_s=0.5,
            session_follow_heartbeat_s=0.05,
        )
    )

    async def scenario() -> None:
        await _create_session(
            store,
            session_id,
            events=(_event(session_id, "e1", EventType.SESSION_STARTED),),
        )
        async with _asgi_client(app, config) as client:
            response = await client.get(_stream_path(session_id))
            contract = (await client.get("/api/contract")).json()["sse"]["session_follow"]
        frames, comments = _parse_sse(response.text)
        assert "heartbeat" in comments
        assert [frame["data"]["id"] for frame in _event_frames(frames)] == [public_event_id(1)]
        assert frames[-1]["event"] == "error"
        assert frames[-1]["data"]["code"] == "replay_idle_timeout"
        assert frames[-1]["data"]["retryable"] is True
        assert all(frame.get("event") != "end" for frame in frames)
        assert contract["heartbeat_interval_seconds"] == 0.05
        assert contract["idle_timeout_seconds"] == 0.5

    asyncio.run(scenario())


def test_idle_follower_of_quiet_running_session_has_bounded_store_reads() -> None:
    store = _ObservedSessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    session_id = "session-follow-quiet"
    idle_timeout_s = 3.5
    config = ServerConfig.local_development(
        lifecycle=ServerLifecycleConfig(replay_idle_timeout_s=idle_timeout_s)
    )

    async def scenario() -> None:
        await _create_session(
            store,
            session_id,
            events=(_event(session_id, "e1", EventType.SESSION_STARTED),),
        )
        store.state_reads = 0
        async with _asgi_client(app, config) as client:
            response = await client.get(_stream_path(session_id))
        frames, _ = _parse_sse(response.text)
        assert frames[-1]["data"]["code"] == "replay_idle_timeout"

    asyncio.run(scenario())

    gaps = [later - earlier for earlier, later in itertools.pairwise(store.page_reads)]
    # Polling backs off 50 ms, 100 ms, 200 ms, 400 ms, 800 ms, then holds at
    # the 1 s ceiling; a quiet follower never polls faster than that again.
    assert gaps[0] < 0.5
    assert gaps[-2] >= 0.9
    # Without backoff a 50 ms loop would read the store ~70 times here.
    assert len(store.page_reads) <= 10
    assert store.state_reads <= len(store.page_reads) + 2


def test_follow_stream_caps_concurrent_streams_per_principal_and_session() -> None:
    store = _ObservedSessionStore()
    app = CayuApp(session_store=store, enable_logging=False)

    def auth(request: Request) -> dict[str, str]:
        subject = request.headers.get("x-test-user")
        if not subject:
            raise HTTPException(status_code=401, detail="Unauthorized")
        return {"subject": subject}

    config = ServerConfig.protected(
        auth,
        lifecycle=ServerLifecycleConfig(
            session_follow_max_streams_per_principal=1,
            session_follow_max_streams_per_session=2,
        ),
    )

    async def scenario() -> None:
        await _create_session(store, "session-live")
        await _create_session(
            store,
            "session-done",
            status=SessionStatus.COMPLETED,
            events=(_event("session-done", "e1", EventType.SESSION_COMPLETED),),
        )

        def as_user(user: str) -> dict[str, str]:
            return {"x-test-user": user}

        async with _asgi_client(app, config) as client:
            alice = asyncio.create_task(
                client.get(_stream_path("session-live"), headers=as_user("alice"))
            )
            await asyncio.wait_for(store.follower_polling.wait(), timeout=10)
            store.follower_polling.clear()

            principal_limited = await client.get(
                _stream_path("session-done"), headers=as_user("alice")
            )
            assert principal_limited.status_code == 429
            assert principal_limited.headers["retry-after"] == "5"

            bob = asyncio.create_task(
                client.get(_stream_path("session-live"), headers=as_user("bob"))
            )
            await asyncio.wait_for(store.follower_polling.wait(), timeout=10)
            session_limited = await client.get(
                _stream_path("session-live"), headers=as_user("carol")
            )
            assert session_limited.status_code == 429
            assert session_limited.headers["retry-after"] == "5"
            contract = (await client.get("/api/contract", headers=as_user("carol"))).json()
            assert contract["sse"]["session_follow"]["max_streams_per_principal"] == 1
            assert contract["sse"]["session_follow"]["max_streams_per_session"] == 2

            await store.append_event(
                "session-live",
                _event("session-live", "e1", EventType.SESSION_COMPLETED),
            )
            await store.update_status("session-live", SessionStatus.COMPLETED)
            for follower in (await alice, await bob):
                frames, _ = _parse_sse(follower.text)
                assert frames[-1]["event"] == "end"

            released = await client.get(_stream_path("session-done"), headers=as_user("alice"))
            assert released.status_code == 200

    asyncio.run(scenario())
