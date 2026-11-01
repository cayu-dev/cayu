"""Closing a session that waits on a person ends it as failed and keeps its history."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from tests.server.test_server_inbox import (
    _app,
    _as,
    _authenticate,
    _items,
    _pause,
    _portal,
    _settle,
)

from cayu import EventType, Message, ResumeRequest, RunRequest, SessionStatus
from cayu.approvals.actors import ResolutionActor
from cayu.server import ServerConfig, create_server
from cayu.sessions.records import SessionStatusConflict


def _client(app) -> TestClient:
    return TestClient(create_server(app, config=ServerConfig.protected(_authenticate)))


def _close(client: TestClient, session_id: str, person: str, **body):
    return client.post(f"/api/sessions/{session_id}/close", json=body, headers=_as(person))


@pytest.mark.parametrize(
    ("agent", "kind", "field"),
    [("asker", "question", "input_id"), ("approver", "approval", "approval_id")],
)
def test_closing_a_waiting_session_fails_it_and_clears_its_inbox_item(agent, kind, field) -> None:
    app = _app()
    with _client(app) as client:
        _pause(client, app, "waiting", agent)
        [item] = _items(client, "priya")
        assert item["kind"] == kind
        transcript_before = _portal(client).call(app.session_store.load_transcript, "waiting")

        closed = _close(client, "waiting", "priya", reason="Stranded by an old release.")
        assert closed.status_code == 200, closed.text
        body = closed.json()
        assert body["session"]["status"] == "failed"
        assert "closed" in body["actions"]

        assert _items(client, "priya") == []
        state = client.get(f"/api/inbox/{item['id']}", headers=_as("tom"))
        assert state.status_code == 200, state.text
        settlement = state.json()["settlement"]
        assert (settlement["decision"], settlement["reason"]) == (
            "dismiss",
            "Stranded by an old release.",
        )
        assert settlement["decided_by"]["subject"] == "priya"

        events = _portal(client).call(app.session_store.load_events, "waiting")
        [failed] = [event for event in events if event.type is EventType.SESSION_FAILED]
        assert failed.payload["closed"] is True
        assert failed.payload["error_type"] == "SessionAbandoned"
        assert failed.payload["metadata"]["reason"] == "Stranded by an old release."
        assert failed.payload["metadata"]["resolved_by"]["subject"] == "priya"
        assert isinstance(failed.payload[field], str)
        # History stays.
        assert _portal(client).call(app.session_store.load_transcript, "waiting") == (
            transcript_before
        )

        # Repeating the close is answered with the closed session; nothing changes.
        again = _close(client, "waiting", "priya", reason="Stranded by an old release.")
        assert again.status_code == 200, again.text
        events_again = _portal(client).call(app.session_store.load_events, "waiting")
        assert [e.id for e in events_again] == [e.id for e in events]

        async def resume() -> None:
            async for _event in app.resume(
                ResumeRequest(session_id="waiting", messages=[Message.text("user", "hi")])
            ):
                pass

        with pytest.raises((SessionStatusConflict, ValueError, RuntimeError)):
            _portal(client).call(resume)


@pytest.mark.parametrize(("agent", "kind"), [("asker", "question"), ("approver", "approval")])
def test_inbox_dismiss_closes_the_session_of_an_approval_or_question(agent, kind) -> None:
    app = _app()
    with _client(app) as client:
        _pause(client, app, "dismissed", agent)
        [item] = _items(client, "priya")
        assert item["kind"] == kind
        assert "dismiss" not in item["decisions"]

        unexplained = _settle(client, item["id"], "priya", decision="dismiss")
        assert unexplained.status_code == 422, unexplained.text
        dismissed = _settle(client, item["id"], "priya", decision="dismiss", reason="Won't resume.")
        assert dismissed.status_code == 200, dismissed.text
        settlement = dismissed.json()["item"]["settlement"]
        assert (settlement["decision"], settlement["reason"]) == ("dismiss", "Won't resume.")
        assert _items(client, "priya") == []
        session = _portal(client).call(app.session_store.load, "dismissed")
        assert session.status is SessionStatus.FAILED

        again = _settle(client, item["id"], "priya", decision="dismiss", reason="Won't resume.")
        assert again.status_code == 200, again.text
        assert again.json()["replayed"] is True
        other = _settle(client, item["id"], "tom", decision="dismiss", reason="Won't resume.")
        assert other.status_code == 409, other.text


def test_close_refuses_a_finished_session_and_an_unknown_one() -> None:
    app = _app()
    with _client(app) as client:
        _pause(client, app, "answered", "asker")
        [item] = _items(client, "priya")
        answer = _settle(client, item["id"], "priya", decision="answer", answer="staging")
        assert answer.status_code == 200, answer.text

        async def finished() -> None:
            import asyncio

            for _attempt in range(500):
                session = await app.session_store.load("answered")
                if session.status is SessionStatus.COMPLETED:
                    return
                await asyncio.sleep(0.01)
            raise AssertionError("answered session did not complete")

        _portal(client).call(finished)
        refused = _close(client, "answered", "priya", reason="Too late.")
        assert refused.status_code == 409, refused.text
        assert "nothing to close" in refused.json()["detail"]
        assert _close(client, "missing", "priya", reason="x").status_code == 404
        assert _close(client, "answered", "priya", reason=" ").status_code == 422


def test_close_session_records_the_given_actor_in_process() -> None:
    app = _app()
    with _client(app) as client:
        _pause(client, app, "local", "asker")
        actor = ResolutionActor(subject="local:ops", source="request")

        async def close():
            return await app.close_session("local", reason="Cleanup.", closed_by=actor)

        result = _portal(client).call(close)
        assert result.status is SessionStatus.FAILED
        events = _portal(client).call(app.session_store.load_events, "local")
        failed = next(event for event in events if event.type is EventType.SESSION_FAILED)
        assert failed.payload["metadata"]["resolved_by"]["subject"] == "local:ops"
        assert _items(client, "priya") == []


def test_a_session_that_resumes_first_is_not_closed(monkeypatch) -> None:
    app = _app()
    with _client(app) as client:
        _pause(client, app, "racing", "asker")
        [item] = _items(client, "priya")
        store = app._runtime_session_store
        original = store.transition_status_and_checkpoint

        async def resumed_first(session_id, *, from_statuses, to_status, **options):
            if to_status is SessionStatus.FAILED and SessionStatus.INTERRUPTED in from_statuses:
                # An answer claims the pause between the close's read and its write.
                await original(
                    session_id,
                    from_statuses={SessionStatus.INTERRUPTED},
                    to_status=SessionStatus.RUNNING,
                    checkpoint_transform=lambda _session, checkpoint: checkpoint,
                )
            return await original(
                session_id, from_statuses=from_statuses, to_status=to_status, **options
            )

        monkeypatch.setattr(store, "transition_status_and_checkpoint", resumed_first)
        refused = _close(client, "racing", "priya", reason="Too late.")
        assert refused.status_code == 409, refused.text
        session = _portal(client).call(app.session_store.load, "racing")
        assert session.status is SessionStatus.RUNNING
        checkpoint = _portal(client).call(app.session_store.load_checkpoint, "racing")
        assert "abandoned_session" not in (checkpoint or {})
        events = _portal(client).call(app.session_store.load_events, "racing")
        assert not any(event.type is EventType.SESSION_FAILED for event in events)
        assert item["kind"] == "question"


def test_closing_a_question_stranded_by_a_changed_execution_profile(tmp_path) -> None:
    import asyncio

    from cayu.approvals.user_input import UserInputResponse
    from cayu.runtime.execution_profiles import ExecutionProfileMismatchError
    from cayu.storage import SQLiteSessionStore

    path = tmp_path / "stranded.db"

    async def scenario() -> None:
        before = _app(session_store=SQLiteSessionStore(path))
        try:
            async for _event in before.run(
                RunRequest(
                    agent_name="asker", session_id="stranded", messages=[Message.text("user", "go")]
                )
            ):
                pass
            [pending] = (await before.session_store.query_pending_actions(_query())).actions
        finally:
            await before.aclose()
        # A new application instance has a different (opaque) provider identity, like a
        # release that changed the execution profile: the answer can't resume.
        after = _app(session_store=SQLiteSessionStore(path))
        try:
            with pytest.raises(ExecutionProfileMismatchError):
                async for _event in after.resolve_user_input(
                    UserInputResponse(
                        session_id="stranded", input_id=pending.input_id, answer="staging"
                    )
                ):
                    pass
            result = await after.close_session("stranded", reason="Stranded by a release.")
            assert result.status is SessionStatus.FAILED
            assert not (await after.session_store.query_pending_actions(_query())).actions
            events = await after.session_store.load_events("stranded")
            assert [event.type for event in events].count(EventType.SESSION_FAILED) == 1
        finally:
            await after.aclose()

    asyncio.run(scenario())


def _query(session_id: str = "stranded"):
    from cayu.sessions.pending_action_contracts import PendingActionQuery

    return PendingActionQuery(session_id=session_id)


def test_close_refuses_a_session_no_longer_waiting_on_the_named_pause() -> None:
    import asyncio

    async def scenario() -> None:
        app = _app()
        try:
            async for _event in app.run(
                RunRequest(
                    agent_name="asker", session_id="moved", messages=[Message.text("user", "go")]
                )
            ):
                pass
            with pytest.raises(SessionStatusConflict, match="no longer waits"):
                await app.close_session("moved", reason="Stale item.", waiting_on="another-input")
            assert (await app.session_store.load("moved")).status is SessionStatus.INTERRUPTED
            [pending] = (await app.session_store.query_pending_actions(_query("moved"))).actions
            closed = await app.close_session(
                "moved", reason="Current item.", waiting_on=pending.input_id
            )
            assert closed.status is SessionStatus.FAILED
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_a_sleeping_session_is_interrupted_before_it_is_closed() -> None:
    import asyncio

    from tests.core.test_sleep_until import _app as _sleeper_app
    from tests.core.test_sleep_until import _Clock, _sleeping

    from cayu.sessions.memory import InMemorySessionStore
    from cayu.sessions.requests import InterruptSessionRequest
    from cayu.tasks.memory import InMemoryTaskStore
    from cayu.tasks.records import TaskStatus

    async def scenario() -> None:
        clock = _Clock()
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore(clock=clock)
        app = _sleeper_app(sessions, tasks, clock)
        try:
            pending = await _sleeping(app, sessions, "asleep")
            with pytest.raises(SessionStatusConflict, match="asleep"):
                await app.close_session("asleep", reason="Nobody needs it.")
            assert "pending_wake" in await sessions.load_checkpoint("asleep")
            async for _event in app.interrupt_session(
                InterruptSessionRequest(session_id="asleep", reason="stop waiting")
            ):
                pass
            assert (await tasks.load_task(pending["wake"]["task_id"])).status is (
                TaskStatus.CANCELLED
            )
            closed = await app.close_session("asleep", reason="Nobody needs it.")
            assert closed.status is SessionStatus.FAILED
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_a_waiting_subagent_is_closed_with_its_parent() -> None:
    import asyncio

    from tests.core.test_foreground_subagent_recovery import _identity, _Provider
    from tests.core.test_tool_round_execution_identities import _RecordingTool

    from cayu import AgentSpec, CayuApp
    from cayu.providers import ModelStreamEvent
    from cayu.sessions.queries import SessionQuery
    from cayu.tools import AlwaysRequireApprovalToolPolicy
    from cayu.tools.subagents import SubagentSpec, SubagentTool

    async def scenario() -> None:
        provider = _Provider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="spawn", name="subagent", arguments={"agent": "child", "task": "work"}
                    ),
                    ModelStreamEvent.completed(),
                ],
                [
                    ModelStreamEvent.tool_call(id="act", name="record", arguments={"value": 7}),
                    ModelStreamEvent.completed(),
                ],
            ]
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="test"),
            tools=[
                SubagentTool(
                    app,
                    agents={"child": SubagentSpec(agent_name="child")},
                    execution_profile_identity=_identity("closed-subagent"),
                )
            ],
        )
        app.register_agent(
            AgentSpec(name="child", model="test"),
            tools=[_RecordingTool()],
            tool_policy=AlwaysRequireApprovalToolPolicy(tools=["record"]),
        )
        try:
            async for _event in app.run(
                RunRequest(
                    session_id="parent", agent_name="parent", messages=[Message.text("user", "go")]
                )
            ):
                pass
            [child] = (
                await app.session_store.list_sessions(SessionQuery(parent_session_id="parent"))
            ).sessions
            with pytest.raises(SessionStatusConflict, match="parent session waits on it"):
                await app.close_session(child.id, reason="Nobody will approve it.")
            # A refused parent close leaves its children as they were.
            with pytest.raises(SessionStatusConflict, match="no longer waits"):
                await app.close_session("parent", reason="Stale item.", waiting_on="not-this-pause")
            assert (await app.session_store.load(child.id)).status is SessionStatus.INTERRUPTED
            assert (await app.session_store.load("parent")).status is SessionStatus.INTERRUPTED
            closed = await app.close_session("parent", reason="Nobody will approve it.")
            assert closed.status is SessionStatus.FAILED
            assert (await app.session_store.load(child.id)).status is SessionStatus.FAILED
            parent_checkpoint = await app.session_store.load_checkpoint("parent")
            assert "foreground_child_wait" not in parent_checkpoint
            for session_id in ("parent", child.id):
                assert not (
                    await app.session_store.query_pending_actions(_query(session_id))
                ).actions
        finally:
            await app.aclose()

    asyncio.run(scenario())


def test_close_accepts_the_public_id_the_pending_action_api_shows() -> None:
    app = _app()
    with _client(app) as client:
        _pause(client, app, "public", "asker")
        actions = client.get("/api/pending-actions", headers=_as("priya"))
        assert actions.status_code == 200, actions.text
        [action] = [row for row in actions.json()["actions"] if row["session"]["id"] == "public"]
        assert action["input_id"].startswith("cayu_event_")
        closed = _close(client, "public", "priya", reason="Stale.", waiting_on=action["input_id"])
        assert closed.status_code == 200, closed.text
        assert closed.json()["session"]["status"] == "failed"
