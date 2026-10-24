"""Exact trace linkage through real runtime execution and the HTTP read boundary."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from tests.core.test_tool_completion import FinalTool, app_for, call
from tests.core.test_tool_round_publication_failure_matrix import _TwoCallProvider

from cayu import Message, RunRequest
from cayu.providers import ModelStreamEvent
from cayu.server import ServerConfig, create_server
from cayu.sessions import InMemorySessionStore
from cayu.sessions.event_queries import EventQuery
from cayu.storage import SQLiteSessionStore


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_trace_pages_preserve_exact_rounds_and_transcript_links(tmp_path, backend):
    store = (
        InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(tmp_path / "trace.db")
    )
    _assert_trace_pages(tmp_path, backend, store)


def _assert_trace_pages(tmp_path, backend, store):
    provider = _TwoCallProvider(
        [call(), call(), [ModelStreamEvent.text_delta("Done"), ModelStreamEvent.completed()]]
    )
    app = app_for(store, provider, FinalTool())

    async def run():
        return [
            event
            async for event in app.run(
                RunRequest(
                    session_id="trace-session",
                    agent_name="support",
                    messages=[Message.text("user", "Help")],
                )
            )
        ]

    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        events = client.portal.call(run)
        assert events[-1].type == "session.completed"
        response = client.get("/api/sessions/trace-session/trace", params={"limit": 1000})
        assert response.status_code == 200, response.text
        whole = response.json()

        async def durable_events():
            return await store.query_events(EventQuery(session_id="trace-session", limit=1000))

        raw_records = client.portal.call(durable_events)
        public_json = json.dumps(whole)
        for record in raw_records:
            for field in ("model_step_id", "model_attempt_id", "execution_id"):
                private = record.event.payload.get(field)
                if type(private) is str and private:
                    assert private not in public_json
        assert (
            whole["events"]
            == client.get("/api/sessions/trace-session/events", params={"limit": 1000}).json()[
                "events"
            ]
        )
        assert all(
            span["status"] == "completed" and span["ended_at"] is not None
            for span in whole["spans"]
            if span["kind"] in {"session", "interaction", "model", "round", "tool"}
        ), whole["spans"]
        model_starts = {
            event.timestamp.isoformat() for event in events if event.type == "model.started"
        }
        assert {
            span["started_at"] for span in whole["spans"] if span["kind"] == "model"
        } == model_starts
        stale = client.get(
            "/api/sessions/trace-session/trace",
            params={"after_sequence": 1, "session_instance_id": "wrong-incarnation"},
        )
        assert stale.status_code == 409
        tools = [span for span in whole["spans"] if span["kind"] == "tool"]
        assert len(tools) == 2
        assert len({span["id"] for span in tools}) == 2
        assert len({span["parent_id"] for span in tools}) == 2
        transcript = client.get("/api/sessions/trace-session/transcript").json()["messages"]
        expected_links = {
            (message["index"], position, part["tool_call_id"])
            for message in transcript
            for position, part in enumerate(message["content"])
            if part["type"] == "tool_call"
        }
        actual_links = {
            (link["message_index"], link["part_index"], link["tool_call_id"])
            for span in tools
            for link in span["transcript_links"]
        }
        assert actual_links == expected_links
        after = 0
        span_ids = set()
        event_ids = []
        while True:
            page_response = client.get(
                "/api/sessions/trace-session/trace",
                params={
                    "limit": 1,
                    "after_sequence": after,
                    "session_instance_id": whole["session_instance_id"],
                },
            )
            assert page_response.status_code == 200, page_response.text
            page = page_response.json()
            span_ids.update(span["id"] for span in page["spans"])
            event_ids.extend(event["id"] for event in page["events"])
            if not page["has_more"]:
                break
            assert page["next_sequence"] > after
            after = page["next_sequence"]
        assert span_ids == {span["id"] for span in whole["spans"]}
        assert event_ids == [event["id"] for event in whole["events"]]

    if backend == "sqlite":
        reopened = SQLiteSessionStore(tmp_path / "trace.db")
        reopened_app = app_for(reopened, _TwoCallProvider([]), FinalTool())
        with TestClient(
            create_server(reopened_app, config=ServerConfig.local_development())
        ) as client:
            result = client.get("/api/sessions/trace-session/trace", params={"limit": 1000})
            assert result.status_code == 200
            assert [
                (span["id"], span["parent_id"], span["status"]) for span in result.json()["spans"]
            ] == [(span["id"], span["parent_id"], span["status"]) for span in whole["spans"]]


def test_trace_user_input_wait_resolves_with_same_span_identity():
    from tests.server.server_test_support import _sse_events
    from tests.server.test_server_session_actions import _ask_user_client

    with _ask_user_client() as client:
        events = _sse_events(client, "/api/run", {"prompt": "deploy"})
        awaiting = next(event for event in events if event["type"] == "session.awaiting_user_input")
        session_id = awaiting["session_id"]
        paused = client.get(f"/api/sessions/{session_id}/trace", params={"limit": 1000}).json()
        waits = [span for span in paused["spans"] if span["kind"] == "wait"]
        assert len(waits) == 1
        assert (
            next(span for span in paused["spans"] if span["id"] == waits[0]["parent_id"])["kind"]
            == "tool"
        )
        assert waits[0]["status"] == "waiting"
        assert waits[0]["ended_at"] is None
        resolved = _sse_events(
            client,
            "/api/user-input/resolve",
            {
                "session_id": session_id,
                "input_id": awaiting["payload"]["input_id"],
                "answer": "staging",
            },
        )
        assert resolved[-1]["type"] == "session.completed"
        final = client.get(f"/api/sessions/{session_id}/trace", params={"limit": 1000}).json()
        finished = next(span for span in final["spans"] if span["id"] == waits[0]["id"])
        assert finished["status"] == "completed"
        assert finished["ended_at"] is not None
        resumed = next(event for event in final["events"] if event["type"] == "session.resumed")
        narrow = client.get(
            f"/api/sessions/{session_id}/trace",
            params={
                "after_sequence": resumed["sequence"] - 1,
                "limit": 1,
                "through_sequence": final["through_sequence"],
            },
        ).json()
        complete = {span["id"]: span for span in final["spans"]}
        for after in range(final["through_sequence"]):
            page = client.get(
                f"/api/sessions/{session_id}/trace",
                params={
                    "after_sequence": after,
                    "limit": 1,
                    "through_sequence": final["through_sequence"],
                },
            ).json()
            for span in page["spans"]:
                assert span["name"] == complete[span["id"]]["name"]
        for span in narrow["spans"]:
            expected = complete[span["id"]]
            assert (span["parent_id"], span["status"], span["started_at"], span["ended_at"]) == (
                expected["parent_id"],
                expected["status"],
                expected["started_at"],
                expected["ended_at"],
            )


def test_trace_member_read_has_no_mutation_authority():
    from cryptography.hazmat.primitives.asymmetric import ec
    from tests.server.test_proxy_identity import FakeJwks, _as, _auth, _token

    from cayu import CayuApp

    key = ec.generate_private_key(ec.SECP256R1())
    with TestClient(
        create_server(CayuApp(), config=ServerConfig.protected(_auth(FakeJwks(key))))
    ) as client:
        headers = _as(_token(key, role="member"))
        assert client.get("/api/sessions/missing/trace", headers=headers).status_code == 404
        assert client.get("/api/sessions/missing/trace").status_code == 401
        assert client.post("/api/run", json={"prompt": "hello"}, headers=headers).status_code == 403


def test_trace_command_links_to_tool_and_keeps_execution_id_private():
    from tests.core.test_builtin_tools import RecordingRunner

    from cayu.environments import Environment, EnvironmentSpec
    from cayu.runners import ExecCommand
    from cayu.tools import ToolResult

    class CommandTool(FinalTool):
        async def run(self, ctx, args):
            result = await ctx.runner.exec(ExecCommand(argv=["echo", "hello"]))
            return ToolResult(content=result.stdout)

    app = app_for(
        InMemorySessionStore(),
        _TwoCallProvider(
            [call(), [ModelStreamEvent.text_delta("Done"), ModelStreamEvent.completed()]]
        ),
        CommandTool(),
    )
    app.register_environment(
        Environment(EnvironmentSpec(name="runner"), runner=RecordingRunner()), default=True
    )

    async def run():
        return [
            event
            async for event in app.run(
                RunRequest(
                    session_id="command",
                    agent_name="support",
                    messages=[Message.text("user", "run")],
                )
            )
        ]

    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        events = client.portal.call(run)
        assert events[-1].type == "session.completed"
        trace = client.get("/api/sessions/command/trace", params={"limit": 1000}).json()
        commands = [span for span in trace["spans"] if span["kind"] == "exec"]
        assert len(commands) == 1
        assert commands[0]["status"] == "completed"
        assert (
            next(span for span in trace["spans"] if span["id"] == commands[0]["parent_id"])["kind"]
            == "tool"
        )

        async def raw_commands():
            return await app.session_store.query_events(
                EventQuery(session_id="command", event_type="runner.exec.started", limit=1)
            )

        started = client.portal.call(raw_commands)[0].event
        assert started.payload["execution_id"] not in json.dumps(trace)


def test_trace_approval_wait_uses_durable_nested_approval_identity():
    from tests.core.test_require_final_tool import _ApprovalPolicy

    app = app_for(
        InMemorySessionStore(),
        _TwoCallProvider([call()]),
        FinalTool(),
        tool_policy=_ApprovalPolicy(),
    )

    async def run():
        return [
            event
            async for event in app.run(
                RunRequest(
                    session_id="approval",
                    agent_name="support",
                    messages=[Message.text("user", "run")],
                )
            )
        ]

    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        events = client.portal.call(run)
        assert any(event.type == "tool.call.approval_requested" for event in events)
        response = client.get("/api/sessions/approval/trace", params={"limit": 1000})
        assert response.status_code == 200
        trace = response.json()
        waits = [span for span in trace["spans"] if span["kind"] == "wait"]
        assert len(waits) == 1
        assert waits[0]["status"] == "waiting"
        assert (
            next(span for span in trace["spans"] if span["id"] == waits[0]["parent_id"])["kind"]
            == "tool"
        )


def test_trace_reads_beyond_ten_thousand_events_without_history_cap():
    from cayu import CayuApp
    from cayu.events import Event, EventType
    from cayu.sessions.records import SessionIdentity

    app = CayuApp(enable_logging=False)
    total = 10_005

    async def seed():
        await app.session_store.create(
            RunRequest(
                session_id="long", agent_name="support", messages=[Message.text("user", "read")]
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        for index in range(total):
            await app.session_store.append_event(
                "long",
                Event(
                    type=EventType.KNOWLEDGE_LOOKUP,
                    session_id="long",
                    payload={"query": f"lookup-{index}"},
                ),
            )

    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        client.portal.call(seed)
        after = 0
        seen = []
        through = None
        while True:
            params = {"after_sequence": after, "limit": 1000}
            if through is not None:
                params["through_sequence"] = through
            response = client.get("/api/sessions/long/trace", params=params)
            assert response.status_code == 200, response.text
            page = response.json()
            if through is None:
                through = page["through_sequence"]

                async def append_live():
                    await app.session_store.append_event(
                        "long",
                        Event(
                            type=EventType.KNOWLEDGE_LOOKUP,
                            session_id="long",
                            payload={"query": "live"},
                        ),
                    )

                client.portal.call(append_live)
            assert len(page["events"]) <= 1000
            assert len([span for span in page["spans"] if span["kind"] == "knowledge"]) == len(
                page["events"]
            )
            seen.extend(event["sequence"] for event in page["events"])
            if not page["has_more"]:
                break
            assert page["next_sequence"] > after
            after = page["next_sequence"]
        assert seen == list(range(1, total + 1))
        tail = client.get("/api/sessions/long/trace", params={"after_sequence": through}).json()
        assert [event["sequence"] for event in tail["events"]] == [total + 1]
        assert client.get("/api/sessions/long/trace", params={"limit": 1001}).status_code == 422
        empty = client.get("/api/sessions/long/trace", params={"through_sequence": 0}).json()
        assert empty["events"] == []
        assert empty["has_more"] is False


def test_trace_http_cancellation_propagates_without_mutating_session():
    import asyncio

    import httpx

    from cayu import CayuApp
    from cayu.sessions.records import SessionIdentity

    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()

        class WaitingStore(InMemorySessionStore):
            async def query_events(self, query):
                result = await super().query_events(query)
                entered.set()
                await release.wait()
                return result

        store = WaitingStore()
        await store.create(
            RunRequest(
                session_id="cancel-read",
                agent_name="support",
                messages=[Message.text("user", "read")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        before = await store.load("cancel-read")
        server = create_server(
            CayuApp(session_store=store), config=ServerConfig.local_development()
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server), base_url="http://test"
        ) as client:
            task = asyncio.create_task(client.get("/api/sessions/cancel-read/trace"))
            await asyncio.wait_for(entered.wait(), timeout=5)
            assert task.cancel()
            assert task.cancelling() == 1
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled()
            release.set()
            assert await store.load("cancel-read") == before
            response = await client.get("/api/sessions/cancel-read/trace")
            assert response.status_code == 200
            assert response.json()["events"] == []

    asyncio.run(scenario())


def test_trace_postgres_exact_pages(tmp_path, postgres_dsn):
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.postgres import PostgresSessionStore

    _assert_trace_pages(
        tmp_path, "postgres", PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
    )


def test_trace_capability_is_read_only():
    from cayu import CayuApp

    with TestClient(create_server(CayuApp(), config=ServerConfig.local_development())) as client:
        capability = (
            client.get("/api/contract").json()["capabilities"]["surfaces"].get("session_trace")
        )
        assert capability is not None
        assert capability["configured"] is True
        assert capability["read"]["enabled"] is True
        assert capability["mutate"]["enabled"] is False


def test_trace_read_failure_returns_no_partial_projection():
    import asyncio

    import httpx

    async def scenario():
        class FailedStore(InMemorySessionStore):
            invocation_lifecycle_command_version = 1
            fail_reads = False

            async def query_transcript(self, query):
                if self.fail_reads:
                    raise RuntimeError("private-store-diagnostic-canary")
                return await super().query_transcript(query)

        store = FailedStore()
        app = app_for(
            store,
            _TwoCallProvider(
                [call(), [ModelStreamEvent.text_delta("Done"), ModelStreamEvent.completed()]]
            ),
            FinalTool(),
        )
        # Runtime writes use their native transcript owners; fail only the
        # paged read after durable execution has succeeded.
        events = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="read-fail",
                    agent_name="support",
                    messages=[Message.text("user", "read")],
                )
            )
        ]
        assert events[-1].type == "session.completed"
        before = await store.load("read-fail")
        store.fail_reads = True
        server = create_server(app, config=ServerConfig.local_development())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.get("/api/sessions/read-fail/trace")
            assert response.status_code == 500
            assert "private-store-diagnostic-canary" not in response.text
            assert "spans" not in response.text
        assert await store.load("read-fail") == before

    asyncio.run(scenario())


def test_trace_deleted_during_read_returns_conflict_without_partial_result():
    import asyncio

    import httpx

    from cayu import CayuApp
    from cayu.sessions.records import SessionIdentity

    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()

        class BarrierStore(InMemorySessionStore):
            async def query_events(self, query):
                result = await super().query_events(query)
                if not entered.is_set():
                    entered.set()
                    await release.wait()
                return result

        store = BarrierStore()
        await store.create(
            RunRequest(
                session_id="deleted", agent_name="support", messages=[Message.text("user", "read")]
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        server = create_server(
            CayuApp(session_store=store), config=ServerConfig.local_development()
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server), base_url="http://test"
        ) as client:
            reading = asyncio.create_task(client.get("/api/sessions/deleted/trace"))
            await asyncio.wait_for(entered.wait(), timeout=5)
            try:
                await store.delete_session("deleted")
            finally:
                release.set()
            response = await reading
            assert response.status_code == 409
            assert "spans" not in response.text

    asyncio.run(scenario())
