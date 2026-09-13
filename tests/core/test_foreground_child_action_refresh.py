"""Public repeated-child actions keep refresh writes owned after cancellation."""

import asyncio

import pytest
from tests.core.test_foreground_subagent_recovery import _identity, _Provider
from tests.core.test_tool_round_execution_identities import _RecordingTool

from cayu import (
    AgentSpec,
    CayuApp,
    InMemorySessionStore,
    InterruptSessionRequest,
    Message,
    RunRequest,
    SessionQuery,
    SQLiteSessionStore,
    SubagentSpec,
    SubagentTool,
    ToolApprovalDecision,
)
from cayu.providers import ModelStreamEvent
from cayu.runtime import ToolApprovalRequest, UserInputResponse
from cayu.runtime.sessions import PendingActionQuery
from cayu.runtime.tool_policy import AlwaysRequireApprovalToolPolicy
from cayu.tools.user_input import UserInputTool


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("delay_refresh", [False, True])
@pytest.mark.parametrize(
    "actions", [("approval", "approval"), ("approval", "input"), ("input", "input")]
)
def test_repeated_child_actions_continue_original_parent_once(
    tmp_path, monkeypatch, backend, actions, delay_refresh
):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "repeated-actions.sqlite")
        )
        protected = _RecordingTool()
        provider = _Provider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="spawn", name="subagent", arguments={"agent": "child", "task": "work"}
                    ),
                    ModelStreamEvent.completed(),
                ],
                *[
                    [
                        ModelStreamEvent.tool_call(
                            id="same-call",
                            name="record" if action == "approval" else "ask_user",
                            arguments={"value": index}
                            if action == "approval"
                            else {"question": f"Question {index}?"},
                        ),
                        ModelStreamEvent.completed(),
                    ]
                    for index, action in enumerate(actions)
                ],
                [ModelStreamEvent.text_delta("child finished"), ModelStreamEvent.completed()],
                [ModelStreamEvent.text_delta("parent finished"), ModelStreamEvent.completed()],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="test"),
            tools=[
                SubagentTool(
                    app,
                    agents={"child": SubagentSpec(agent_name="child")},
                    execution_profile_identity=_identity("repeated-child-actions"),
                )
            ],
        )
        app.register_agent(
            AgentSpec(name="child", model="test"),
            tools=[protected, UserInputTool()],
            tool_policy=AlwaysRequireApprovalToolPolicy(tools=["record"]),
        )
        try:
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="parent",
                        agent_name="parent",
                        messages=[Message.text("user", "go")],
                    )
                )
            ]
            child = (await store.list_sessions(SessionQuery(parent_session_id="parent"))).sessions[
                0
            ]
            original = (await store.load_checkpoint("parent"))["foreground_child_wait"]
            deliver = app._event_writer._continue_foreground_parent

            async def delayed_discovery(claim):
                if (
                    delay_refresh
                    and claim.session_id == child.id
                    and (claim.event.type == "session.interrupted")
                ):
                    return False
                return await deliver(claim)

            monkeypatch.setattr(app._event_writer, "_continue_foreground_parent", delayed_discovery)
            action_ids = []
            for index, action in enumerate(actions):
                wait = (await store.load_checkpoint("parent"))["foreground_child_wait"]
                assert wait["parent_effect"] == original["parent_effect"]
                assert wait["revision"] == (1 if delay_refresh else index + 1)
                assert wait["child_session_id"] == child.id
                child_checkpoint = await store.load_checkpoint(child.id)
                action_id = (
                    child_checkpoint["pending_tool_approval"]["approval_id"]
                    if action == "approval"
                    else child_checkpoint["pending_user_input"]["input_id"]
                )
                assert action_id not in action_ids
                action_ids.append(action_id)
                assert wait["child_action_id"] == (action_ids[0] if delay_refresh else action_id)
                assert len(provider.requests) == index + 2
                assert not any(
                    event.type == "tool.call.completed"
                    for event in await store.load_events("parent")
                )
                if action == "approval":
                    approval = (await store.load_checkpoint(child.id))["pending_tool_approval"]
                    assert approval["approval_id"] == action_id
                    resolution = app.resolve_tool_approval(
                        ToolApprovalRequest(
                            session_id=child.id,
                            approval_id=approval["approval_id"],
                            tool_round_id=approval["tool_round_id"],
                            tool_call_id=approval["tool_call_id"],
                            decision=ToolApprovalDecision.APPROVE,
                        )
                    )
                else:
                    resolution = app.resolve_user_input(
                        UserInputResponse(
                            session_id=child.id,
                            input_id=action_id,
                            answer=f"Answer {index}",
                        )
                    )
                _ = [event async for event in resolution]
                assert await app.drain_background_interruptions(timeout_s=10)
                await app.recover_persisted_event_side_effects()
            assert (await store.load(child.id)).status == "completed"
            assert (await store.load("parent")).status == "completed"
            assert protected.values == [
                i for i, action in enumerate(actions) if action == "approval"
            ]
            assert len(provider.requests) == 5
            events = await store.load_events("parent")
            for event_type in (
                "session.interrupted",
                "session.delegated_action.updated",
                "tool.call.completed",
                "interaction.started",
                "interaction.completed",
            ):
                expected = (
                    0 if delay_refresh and event_type == "session.delegated_action.updated" else 1
                )
                assert sum(event.type == event_type for event in events) == expected, event_type
            started = next(event for event in events if event.type == "interaction.started")
            completed = next(event for event in events if event.type == "interaction.completed")
            assert completed.interaction_id == started.interaction_id
            assert (
                len(
                    [
                        message
                        for message in await store.load_transcript("parent")
                        if message.role == "tool"
                    ]
                )
                == 1
            )
            await app.recover_persisted_event_side_effects()
            assert await store.load_events("parent") == events
            assert len(provider.requests) == 5
        finally:
            assert await app.drain_background_interruptions(timeout_s=10)
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("stop_parent", [False, True])
def test_cancelled_refresh_retains_dispatched_write_and_finishes_original_parent(
    tmp_path, monkeypatch, backend, stop_parent, caplog
):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "refresh-cancellation.sqlite")
        )
        provider = _Provider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="spawn", name="subagent", arguments={"agent": "child", "task": "work"}
                    ),
                    ModelStreamEvent.completed(),
                ],
                *[
                    [
                        ModelStreamEvent.tool_call(
                            id="same-call", name="ask_user", arguments={"question": question}
                        ),
                        ModelStreamEvent.completed(),
                    ]
                    for question in ("First?", "Second?")
                ],
                [ModelStreamEvent.text_delta("child finished"), ModelStreamEvent.completed()],
                [ModelStreamEvent.text_delta("parent finished"), ModelStreamEvent.completed()],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="test"),
            tools=[
                SubagentTool(
                    app,
                    agents={"child": SubagentSpec(agent_name="child")},
                    execution_profile_identity=_identity("refresh-cancellation"),
                )
            ],
        )
        app.register_agent(AgentSpec(name="child", model="test"), tools=[UserInputTool()])
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = []
        caught = []
        continuation_errors = []
        resume = app._session_engine.resume_foreground_child

        async def record_continuation_error(*args, **kwargs):
            try:
                return await resume(*args, **kwargs)
            except Exception as error:
                continuation_errors.append(error)
                raise

        monkeypatch.setattr(
            app._session_engine, "resume_foreground_child", record_continuation_error
        )
        task = None
        try:
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="parent",
                        agent_name="parent",
                        messages=[Message.text("user", "go")],
                    )
                )
            ]
            children = await store.list_sessions(SessionQuery(parent_session_id="parent"))
            child = children.sessions[0]
            initial = (await store.load_checkpoint("parent"))["foreground_child_wait"]
            actions = await store.query_pending_actions(PendingActionQuery(session_id=child.id))
            first = actions.actions[0].input_id
            publish = app._runtime_session_store.publish_session_operation

            async def blocked_refresh(session_id, **kwargs):
                if kwargs["idempotency_key"].startswith("foreground-action:"):
                    calls.append(kwargs["idempotency_key"])
                    entered.set()
                    await release.wait()
                return await publish(session_id, **kwargs)

            monkeypatch.setattr(
                app._runtime_session_store, "publish_session_operation", blocked_refresh
            )

            async def resolve_first():
                try:
                    return [
                        event
                        async for event in app.resolve_user_input(
                            UserInputResponse(
                                session_id=child.id, input_id=first, answer="first answer"
                            )
                        )
                    ]
                except asyncio.CancelledError:
                    caught.append("normal cancellation")
                    raise

            task = asyncio.create_task(resolve_first())
            await asyncio.wait_for(entered.wait(), 15)
            task.cancel("cancel refresh caller")
            done, _ = await asyncio.wait({task}, timeout=5)
            assert task in done, "Cancellation must not await the blocked store publication"
            with pytest.raises(asyncio.CancelledError):
                await task
            assert caught == ["normal cancellation"]
            assert task.cancelled() and task.cancelling() == 1
            assert app._foreground_child_delivery_owner.active(child.id)
            assert not await app.drain_background_interruptions(timeout_s=0.01)
            assert (await store.load_checkpoint("parent"))["foreground_child_wait"] == initial
            await app.recover_persisted_event_side_effects()
            assert len(calls) == 1
            assert len(provider.requests) == 3
            if stop_parent:
                _ = [
                    event
                    async for event in app.interrupt_session(
                        InterruptSessionRequest(
                            session_id="parent", reason="Stop while child discovery is settling"
                        )
                    )
                ]
                stopped_parent = await store.load("parent")
                stopped_events = await store.load_events("parent")
                assert not (
                    await store.query_pending_actions(
                        PendingActionQuery(session_id="parent", kind="delegated_action")
                    )
                ).actions
            release.set()
            assert await app.drain_background_interruptions(timeout_s=10)
            await app.recover_persisted_event_side_effects()
            current = (await store.load_checkpoint("parent"))["foreground_child_wait"]
            assert current["revision"] == (1 if stop_parent else 2)
            assert current["parent_effect"] == initial["parent_effect"]
            if stop_parent:
                # The stop's own deferred cleanup may finish while the refresh
                # owner drains. Freeze its settled state before child completion.
                settled_parent = await store.load("parent")
                assert settled_parent.model_dump(
                    exclude={"last_activity_at"}
                ) == stopped_parent.model_dump(exclude={"last_activity_at"})
                assert await store.load_events("parent") == stopped_events
                stopped_parent = settled_parent
            actions = await store.query_pending_actions(PendingActionQuery(session_id=child.id))
            second = actions.actions[0].input_id
            assert second != first
            assert current["child_action_id"] == (first if stop_parent else second)
            _ = [
                event
                async for event in app.resolve_user_input(
                    UserInputResponse(session_id=child.id, input_id=second, answer="second answer")
                )
            ]
            assert await app.drain_background_interruptions(timeout_s=10)
            await app.recover_persisted_event_side_effects()
            assert "CancelledError exception in shielded future" not in caplog.text
            if stop_parent:
                final_parent = await store.load("parent")
                assert final_parent == stopped_parent, {
                    key: (value, final_parent.model_dump()[key])
                    for key, value in stopped_parent.model_dump().items()
                    if value != final_parent.model_dump()[key]
                }
                assert await store.load_events("parent") == stopped_events
                assert len(provider.requests) == 4
                assert not any(
                    event.id.startswith("foreground-action:") for event in stopped_events
                )
                return
            assert (await store.load("parent")).status == "completed", continuation_errors
            assert len(provider.requests) == 5
            parent_events = await store.load_events("parent")
            assert (
                len([event for event in parent_events if event.type == "session.interrupted"]) == 1
            )
            assert (
                len([event for event in parent_events if event.type == "tool.call.completed"]) == 1
            )
            assert (
                len([event for event in parent_events if event.type == "interaction.started"]) == 1
            )
            assert (
                len([event for event in parent_events if event.type == "interaction.completed"])
                == 1
            )
            assert (
                len([event for event in parent_events if event.id.startswith("foreground-action:")])
                == 1
            )
        finally:
            release.set()
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            assert await app.drain_background_interruptions(timeout_s=10)
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())
