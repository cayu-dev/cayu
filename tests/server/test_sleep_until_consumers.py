"""A sleeping session waits on its timer: nothing that serves people lists it."""

from __future__ import annotations

from fastapi.testclient import TestClient
from tests.core.test_sleep_until import _app as _sleeping_app
from tests.core.test_sleep_until import _Clock, _run
from tests.server.test_server_inbox import _as, _authenticate, _items

from cayu.cloud_reporting import read_cloud_activity
from cayu.runtime.human_attention import HumanAttentionRequest
from cayu.server import ServerConfig, create_server
from cayu.sessions import InMemorySessionStore
from cayu.sessions.pending_action_contracts import PendingActionQuery
from cayu.sessions.records import PendingActionKind
from cayu.tasks import InMemoryTaskStore


def test_a_sleep_is_a_wake_for_cloud_and_never_a_question_for_a_person() -> None:
    clock = _Clock()
    app = _sleeping_app(InMemorySessionStore(), InMemoryTaskStore(clock=clock), clock)
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        portal = client.portal
        assert portal is not None
        portal.call(lambda: _run(app, "s_consumers"))

        assert _items(client, "priya") == []
        # The control-plane API lists what people settle or follow; a sleep is neither.
        listed = client.get("/api/pending-actions", headers=_as("tom"))
        assert listed.status_code == 200, listed.text
        assert listed.json()["actions"] == []
        assert (
            client.get(
                "/api/pending-actions", params={"kind": "wake"}, headers=_as("tom")
            ).status_code
            == 422
        )

        async def observe() -> None:
            page = await app.session_store.query_pending_actions(
                PendingActionQuery(session_id="s_consumers")
            )
            [record] = page.actions
            assert record.kind is PendingActionKind.WAKE
            assert HumanAttentionRequest.from_pending_action(record) is None
            questions = await app.session_store.query_pending_actions(
                PendingActionQuery(kind=PendingActionKind.USER_INPUT)
            )
            assert questions.actions == []
            activity = await read_cloud_activity(app)
            [run] = [run for run in activity.runs if run.session_id == "s_consumers"]
            [waiting] = run.waiting_for
            assert waiting.kind == "wake"
            assert waiting.attention_id is None
            assert waiting.wake_at is not None and waiting.wake_at.isoformat().startswith(
                "2027-01-15T12:00:30"
            )

        portal.call(observe)


def test_the_pending_action_total_leaves_out_every_wake_not_only_this_pages() -> None:
    from tests.server.test_server_inbox import _app as _inbox_app

    from cayu.agents import AgentSpec
    from cayu.messages import Message
    from cayu.sessions.requests import RunRequest
    from cayu.tools import SleepUntilTool

    clock = _Clock()
    app = _inbox_app(task_store=InMemoryTaskStore(clock=clock), clock=clock)
    from tests.core.test_sleep_until import _sleep_then_answer, _StableScriptedProvider

    app.register_provider(
        _StableScriptedProvider(
            name="sleepy", response_factory=_sleep_then_answer({"duration_seconds": 30})
        )
    )
    app.register_agent(
        AgentSpec(name="napper", model="scripted-model", provider_name="sleepy"),
        tools=[SleepUntilTool()],
    )
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        portal = client.portal
        assert portal is not None

        async def pause() -> None:
            for session_id in ("s_nap_1", "s_nap_2", "s_nap_3"):
                async for _ in app.run(
                    RunRequest(
                        agent_name="napper",
                        session_id=session_id,
                        messages=[Message.text("user", "nap")],
                    )
                ):
                    pass
            async for _ in app.run(
                RunRequest(
                    agent_name="asker",
                    session_id="s_question",
                    messages=[Message.text("user", "ask")],
                )
            ):
                pass

        portal.call(pause)
        page = client.get("/api/pending-actions", params={"limit": 2}, headers=_as("tom")).json()
        # Two of the four candidates fit; the store cannot count them all, so neither can we.
        assert page["total_count"] is None
        assert all(action["kind"] == "user_input" for action in page["actions"])
        page = client.get("/api/pending-actions", params={"limit": 50}, headers=_as("tom")).json()
        assert [action["kind"] for action in page["actions"]] == ["user_input"]
        assert page["total_count"] == 1


def test_the_total_leaves_out_wakes_beyond_this_page_where_the_store_counts_them() -> None:
    from tests.core.test_sleep_until import _sleep_then_answer, _StableScriptedProvider
    from tests.server.test_server_inbox import _app as _inbox_app

    from cayu.agents import AgentSpec
    from cayu.messages import Message
    from cayu.sessions.requests import RunRequest
    from cayu.tools import SleepUntilTool

    clock = _Clock()
    store = InMemorySessionStore()
    app = _inbox_app(session_store=store, task_store=InMemoryTaskStore(clock=clock), clock=clock)
    paged = app._runtime_session_store.query_pending_actions

    async def counting(query=None, **options):
        # A store that counts every match even when a page holds only some.
        result = await paged(query, **options)
        everything = await paged(query.model_copy(update={"limit": 200, "cursor": None}), **options)
        return result.model_copy(update={"total_count": len(everything.actions)})

    app._runtime_session_store.query_pending_actions = counting
    app.register_provider(
        _StableScriptedProvider(
            name="sleepy", response_factory=_sleep_then_answer({"duration_seconds": 30})
        )
    )
    app.register_agent(
        AgentSpec(name="napper", model="scripted-model", provider_name="sleepy"),
        tools=[SleepUntilTool()],
    )
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        portal = client.portal
        assert portal is not None

        async def pause() -> None:
            async for _ in app.run(
                RunRequest(
                    agent_name="asker", session_id="s_q", messages=[Message.text("user", "ask")]
                )
            ):
                pass
            for session_id in ("s_nap_1", "s_nap_2"):
                async for _ in app.run(
                    RunRequest(
                        agent_name="napper",
                        session_id=session_id,
                        messages=[Message.text("user", "nap")],
                    )
                ):
                    pass

        portal.call(pause)
        page = client.get("/api/pending-actions", params={"limit": 1}, headers=_as("tom")).json()
        assert page["total_count"] == 1
