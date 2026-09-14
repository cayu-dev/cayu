"""Public action routing and closure cannot cross a suspended child boundary."""

import asyncio

import pytest
from tests.core.test_foreground_child_resolution_contention import _app, _ContendedTool
from tests.core.test_foreground_subagent_recovery import _Provider

from cayu.approvals.user_input import UserInputResponse
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.base import InMemorySessionStore, RunRequest, SessionQuery, SessionStatus
from cayu.storage.sqlite import SQLiteSessionStore


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_nested_input_routing_and_parent_closure_are_isolated(tmp_path, backend):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "isolation.sqlite")
        )
        opening = [
            [
                ModelStreamEvent.tool_call(
                    id="spawn", name="subagent", arguments={"agent": "child", "task": "work"}
                ),
                ModelStreamEvent.completed(),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="ask", name="ask_user", arguments={"question": "Value?"}
                ),
                ModelStreamEvent.completed(),
            ],
        ]
        provider = _Provider(
            opening
            + opening
            + [
                [ModelStreamEvent.text_delta("child complete"), ModelStreamEvent.completed()],
                [ModelStreamEvent.text_delta("parent complete"), ModelStreamEvent.completed()],
            ]
        )
        app = _app(store, provider, _ContendedTool(), "input")
        try:
            for parent in ("parent", "other-parent"):
                _ = [
                    event
                    async for event in app.run(
                        RunRequest(
                            session_id=parent,
                            agent_name="parent",
                            messages=[Message.text("user", "go")],
                        )
                    )
                ]
            child = (await store.list_sessions(SessionQuery(parent_session_id="parent"))).sessions[
                0
            ]
            sibling = (
                await store.list_sessions(SessionQuery(parent_session_id="other-parent"))
            ).sessions[0]
            pending = next(
                event
                for event in await store.load_events(child.id)
                if event.type == "session.awaiting_user_input"
            )
            input_id = pending.payload["input_id"]

            async def snapshot():
                return [
                    (
                        await store.load(id),
                        await store.load_checkpoint(id),
                        await store.load_transcript(id),
                    )
                    for id in ("parent", child.id, "other-parent", sibling.id)
                ]

            before = await snapshot()
            # The parent cannot resolve its delegated reference; another child
            # with the same provider call ID cannot consume this action either.
            for target, action_id in (
                ("parent", input_id),
                (sibling.id, input_id),
                (child.id, "stale-input"),
            ):
                with pytest.raises((ValueError, RuntimeError)):
                    _ = [
                        event
                        async for event in app.resolve_user_input(
                            UserInputResponse(session_id=target, input_id=action_id, answer="wrong")
                        )
                    ]
                assert await snapshot() == before
                assert len(provider.requests) == 4
            # Current deletion policy refuses a parent with descendants; it
            # must not partially erase the wait or invalidate the child action.
            with pytest.raises(ValueError, match="child-session policy"):
                await app.erase_session_closure("parent")
            assert await snapshot() == before
            assert len(provider.requests) == 4
            response = UserInputResponse(session_id=child.id, input_id=input_id, answer="right")
            _ = [event async for event in app.resolve_user_input(response)]
            assert await app.drain_background_interruptions(timeout_s=10)
            assert (await store.load("parent")).status is SessionStatus.COMPLETED
            assert len(provider.requests) == 6
            completed = await snapshot()
            # Exact replay is idempotent; a different answer cannot borrow it.
            _ = [event async for event in app.resolve_user_input(response)]
            with pytest.raises((ValueError, RuntimeError)):
                _ = [
                    event
                    async for event in app.resolve_user_input(
                        response.model_copy(update={"answer": "changed"})
                    )
                ]
            await app.recover_persisted_event_side_effects()
            assert await snapshot() == completed
            assert completed[2:] == before[2:]
            assert len(provider.requests) == 6
        finally:
            assert await app.drain_background_interruptions(timeout_s=10)
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
