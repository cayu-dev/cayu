"""Parent interruption must invalidate automatic foreground continuation."""

import asyncio

import pytest
from tests.core.test_foreground_subagent_recovery import _identity, _Provider
from tests.core.test_tool_round_execution_identities import _RecordingTool

from cayu import (
    AgentSpec,
    CayuApp,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    InterruptSessionRequest,
    Message,
    RunRequest,
    SessionQuery,
    SessionStatus,
    SQLiteSessionStore,
    SubagentSpec,
    SubagentTool,
    ToolApprovalDecision,
)
from cayu.providers import ModelStreamEvent
from cayu.runtime import ToolApprovalRequest, UserInputResponse
from cayu.runtime.sessions import PersistedEventSideEffectStatus
from cayu.runtime.tool_policy import AlwaysRequireApprovalToolPolicy
from cayu.tools.user_input import UserInputTool


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("action", ["approve", "deny", "input"])
@pytest.mark.parametrize("cancel_after_claim", [False, True])
def test_parent_interruption_prevents_late_child_action_from_resurrecting_parent(
    tmp_path, backend, action, cancel_after_claim
):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "invalidation.sqlite")
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
                [
                    ModelStreamEvent.tool_call(
                        id="child-action",
                        name="ask_user" if action == "input" else "record",
                        arguments={"question": "Which value?"}
                        if action == "input"
                        else {"value": 7},
                    ),
                    ModelStreamEvent.completed(),
                ],
                [ModelStreamEvent.text_delta("child finished"), ModelStreamEvent.completed()],
                [ModelStreamEvent.text_delta("must not run"), ModelStreamEvent.completed()],
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
                    execution_profile_identity=_identity("interrupted-parent-subagent"),
                )
            ],
        )
        app.register_agent(
            AgentSpec(name="child", model="test"),
            tools=[UserInputTool()] if action == "input" else [protected],
            tool_policy=None
            if action == "input"
            else AlwaysRequireApprovalToolPolicy(tools=["record"]),
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
            assert len(provider.requests) == 2
            children = await store.list_sessions(SessionQuery(parent_session_id="parent"))
            assert len(children.sessions) == 1
            child = children.sessions[0]
            pending = next(
                event
                for event in await store.load_events(child.id)
                if event.type
                == (
                    "session.awaiting_user_input"
                    if action == "input"
                    else "tool.call.approval_requested"
                )
            )

            async def stop_parent():
                return [
                    event
                    async for event in app.interrupt_session(
                        InterruptSessionRequest(session_id="parent", reason="Stop this delegation")
                    )
                ]

            if cancel_after_claim:
                committed = asyncio.Event()
                transition = app._runtime_session_store.transition_status_and_checkpoint

                async def block_after_claim(session_id, **kwargs):
                    result = await transition(session_id, **kwargs)
                    if (
                        session_id == "parent"
                        and kwargs.get("to_status") is SessionStatus.INTERRUPTING
                    ):
                        committed.set()
                        await asyncio.Event().wait()
                    return result

                app._runtime_session_store.transition_status_and_checkpoint = block_after_claim
                stop_task = asyncio.create_task(stop_parent())
                try:
                    await asyncio.wait_for(committed.wait(), timeout=10)
                    stop_task.cancel()
                    assert stop_task.cancelling() == 1
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(stop_task, timeout=10)
                    assert stop_task.cancelled()
                    assert stop_task.cancelling() == 1
                finally:
                    app._runtime_session_store.transition_status_and_checkpoint = transition
                    if not stop_task.done():
                        stop_task.cancel()
                        await asyncio.gather(stop_task, return_exceptions=True)
                await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id="parent", inactive_for_seconds=0)
                )
            interruption_events = await stop_parent()
            parent_transcript = await store.load_transcript("parent")
            parent_events = await store.load_events("parent")
            closed = [event for event in parent_events if event.type == "interaction.interrupted"]
            assert len(closed) == 1
            started = next(event for event in parent_events if event.type == "interaction.started")
            assert closed[0].interaction_id == started.interaction_id
            repeated = [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(session_id="parent", reason="Stop this delegation")
                )
            ]
            assert repeated[-1].payload == interruption_events[-1].payload
            assert await store.load_events("parent") == parent_events
            if action == "input":
                resolution = app.resolve_user_input(
                    UserInputResponse(
                        session_id=child.id,
                        input_id=pending.payload["input_id"],
                        answer="late answer",
                    )
                )
            else:
                resolution = app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=child.id,
                        approval_id=pending.payload["approval"]["approval_id"],
                        tool_round_id=pending.payload["tool_round_id"],
                        tool_call_id=pending.payload["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE
                        if action == "approve"
                        else ToolApprovalDecision.DENY,
                    )
                )
            _ = [event async for event in resolution]
            await app.recover_persisted_event_side_effects()
            terminal_child = next(
                event
                for event in await store.load_events(child.id)
                if event.type == "session.completed"
            )
            delivery = await store.get_persisted_event_side_effect_delivery(
                session_id=child.id, event_id=terminal_child.id
            )
            assert (
                delivery is not None and delivery.status is PersistedEventSideEffectStatus.DELIVERED
            )
            with pytest.raises(RuntimeError, match="no authoritative open interaction"):
                await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id="parent", inactive_for_seconds=0)
                )
            parent = await store.load("parent")
            assert parent is not None and parent.status is SessionStatus.INTERRUPTED
            assert len(provider.requests) == 3
            assert protected.values == ([7] if action == "approve" else [])
            assert await store.load_transcript("parent") == parent_transcript
            assert await store.load_events("parent") == parent_events
            assert interruption_events[-1].payload["interruption_type"] == "operator_requested"
            assert not any(
                event.type == "interaction.completed" for event in await store.load_events("parent")
            )
        finally:
            await app.drain_background_interruptions(timeout_s=10)
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())
