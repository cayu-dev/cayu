"""Child terminal alternatives must release exactly the original parent wait."""

import asyncio

import pytest
from tests.core.test_foreground_subagent_recovery import _identity, _Provider
from tests.core.test_tool_round_execution_identities import _RecordingTool

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.approvals.user_input import UserInputResponse
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions.base import (
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    InterruptSessionRequest,
    RunRequest,
    SessionQuery,
)
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.policy import AlwaysRequireApprovalToolPolicy
from cayu.tools.subagents import SubagentSpec, SubagentTool
from cayu.tools.user_input import UserInputTool


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("action", ["input", "approval"])
@pytest.mark.parametrize(
    ("resolution_wins", "cancel_stop"),
    [
        (False, False),
        (True, False),
        ("claimed", False),
        (False, True),
        (False, "tampered_source_round"),
        (False, "tampered_source_call"),
        (False, "tampered_source_attempt"),
        (False, "tampered_source_epoch"),
        (False, "tampered_source_interaction"),
    ],
)
def test_stopping_paused_child_continues_parent_with_terminal_error(
    tmp_path, backend, action, resolution_wins, cancel_stop, monkeypatch
):
    async def scenario():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "child-terminal.sqlite")
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
                        id="action",
                        name="ask_user" if action == "input" else "record",
                        arguments={"question": "Continue?"} if action == "input" else {"value": 7},
                    ),
                    ModelStreamEvent.completed(),
                ],
                [
                    ModelStreamEvent.text_delta("parent handled child interruption"),
                    ModelStreamEvent.completed(),
                ],
            ]
            + (
                [[ModelStreamEvent.text_delta("parent finished"), ModelStreamEvent.completed()]]
                if resolution_wins
                else []
            )
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="test"),
            tools=[
                SubagentTool(
                    app,
                    agents={"child": SubagentSpec(agent_name="child")},
                    execution_profile_identity=_identity("child-terminal-alternatives"),
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
            children = await store.list_sessions(SessionQuery(parent_session_id="parent"))
            child = children.sessions[0]
            paused_child_checkpoint = await store.load_checkpoint(child.id)
            initial = (await store.load_checkpoint("parent"))["foreground_child_wait"]
            if resolution_wins:
                entered = asyncio.Event()
                release = asyncio.Event()
                transition = app._runtime_session_store.transition_status_and_checkpoint

                async def pause_stop(session_id, **kwargs):
                    if session_id == child.id and kwargs["to_status"] == "interrupting":
                        entered.set()
                        await release.wait()
                    return await transition(session_id, **kwargs)

                monkeypatch.setattr(
                    app._runtime_session_store, "transition_status_and_checkpoint", pause_stop
                )

                async def stop_child():
                    return [
                        event
                        async for event in app.interrupt_session(
                            InterruptSessionRequest(
                                session_id=child.id, reason="Stale stop competing with resolution"
                            )
                        )
                    ]

                stopping = asyncio.create_task(stop_child())
                resolution_task = None
                release_resolution = asyncio.Event()
                try:
                    await asyncio.wait_for(entered.wait(), 10)
                    resolution_claimed = asyncio.Event()
                    if resolution_wins == "claimed":
                        apply_command = (
                            app._runtime_session_store.apply_invocation_lifecycle_command
                        )

                        async def pause_claimed_resolution(command):
                            result = await apply_command(command)
                            if command.session_id == child.id:
                                checkpoint = await store.load_checkpoint(child.id)
                                key = (
                                    "user_input_resolution_intent"
                                    if action == "input"
                                    else "approval_resolution_intent"
                                )
                                if checkpoint is not None and key in checkpoint:
                                    resolution_claimed.set()
                                    await release_resolution.wait()
                            return result

                        monkeypatch.setattr(
                            app._runtime_session_store,
                            "apply_invocation_lifecycle_command",
                            pause_claimed_resolution,
                        )
                    if action == "input":
                        resolution = app.resolve_user_input(
                            UserInputResponse(
                                session_id=child.id,
                                input_id=initial["child_action_id"],
                                answer="accepted",
                            )
                        )
                    else:
                        approval = paused_child_checkpoint["pending_tool_approval"]
                        resolution = app.resolve_tool_approval(
                            ToolApprovalRequest(
                                session_id=child.id,
                                approval_id=approval["approval_id"],
                                tool_round_id=approval["tool_round_id"],
                                tool_call_id=approval["tool_call_id"],
                                decision=ToolApprovalDecision.APPROVE,
                            )
                        )

                    async def resolve_child():
                        return [event async for event in resolution]

                    resolution_task = asyncio.create_task(resolve_child())
                    if resolution_wins == "claimed":
                        await asyncio.wait_for(resolution_claimed.wait(), 10)
                        claimed_child = await store.load(child.id)
                        claimed_checkpoint = await store.load_checkpoint(child.id)
                        claimed_events = await store.load_events(child.id)
                        assert len(provider.requests) == 2 and protected.values == []
                        release.set()
                        with pytest.raises((ValueError, RuntimeError)):
                            await stopping
                        assert await store.load(child.id) == claimed_child
                        assert await store.load_checkpoint(child.id) == claimed_checkpoint
                        assert await store.load_events(child.id) == claimed_events
                        assert len(provider.requests) == 2 and protected.values == []
                        assert not resolution_task.done()
                        release_resolution.set()
                    await resolution_task
                    assert await app.drain_background_interruptions(timeout_s=10)
                    child_after_resolution = await store.load(child.id)
                    child_events_after_resolution = await store.load_events(child.id)
                    assert child_after_resolution.status == "completed"
                    release.set()
                    with pytest.raises((ValueError, RuntimeError)):
                        await stopping
                    after_stop = await store.load(child.id)
                    assert after_stop == child_after_resolution, {
                        key: (value, after_stop.model_dump()[key])
                        for key, value in child_after_resolution.model_dump().items()
                        if value != after_stop.model_dump()[key]
                    }
                    assert await store.load_events(child.id) == child_events_after_resolution
                    await app.recover_persisted_event_side_effects()
                    assert (await store.load("parent")).status == "completed"
                    assert len(provider.requests) == 4
                    assert protected.values == ([7] if action == "approval" else [])
                finally:
                    release.set()
                    release_resolution.set()
                    if not stopping.done():
                        stopping.cancel()
                    tasks = [stopping]
                    if resolution_task is not None:
                        if not resolution_task.done():
                            resolution_task.cancel()
                        tasks.append(resolution_task)
                    await asyncio.gather(*tasks, return_exceptions=True)
                return

            async def stop_paused_child():
                return [
                    event
                    async for event in app.interrupt_session(
                        InterruptSessionRequest(
                            session_id=child.id, reason="Operator stopped the delegated child"
                        )
                    )
                ]

            if cancel_stop:
                committed = asyncio.Event()
                transition = app._runtime_session_store.transition_status_and_checkpoint
                cancellations = []

                async def block_committed_stop(session_id, **kwargs):
                    result = await transition(session_id, **kwargs)
                    if session_id == child.id and kwargs.get("to_status") == "interrupting":
                        committed.set()
                        await asyncio.Event().wait()
                    return result

                monkeypatch.setattr(
                    app._runtime_session_store,
                    "transition_status_and_checkpoint",
                    block_committed_stop,
                )

                async def cancelled_stop():
                    try:
                        return await stop_paused_child()
                    except asyncio.CancelledError:
                        cancellations.append("normal cancellation")
                        raise

                stop_task = asyncio.create_task(cancelled_stop())
                try:
                    await asyncio.wait_for(committed.wait(), 10)
                    assert (await store.load(child.id)).status == "interrupting"
                    assert len(provider.requests) == 2 and protected.values == []
                    stop_task.cancel("cancel child stop after commit")
                    done, _ = await asyncio.wait({stop_task}, timeout=10)
                    assert stop_task in done
                    with pytest.raises(asyncio.CancelledError):
                        await stop_task
                    assert stop_task.cancelled() and stop_task.cancelling() == 1
                    assert cancellations == ["normal cancellation"]
                finally:
                    monkeypatch.setattr(
                        app._runtime_session_store, "transition_status_and_checkpoint", transition
                    )
                    if not stop_task.done():
                        stop_task.cancel()
                    await asyncio.gather(stop_task, return_exceptions=True)
                if isinstance(cancel_stop, str):
                    load_stage = app._runtime_session_store.load_model_completion_stage
                    observed = []

                    async def contradictory_source_round(session_id, stage_id):
                        stage = await load_stage(session_id, stage_id)
                        if session_id != child.id or stage is None or stage.publication is None:
                            return stage
                        stage = stage.model_copy(deep=True)
                        for operation in stage.publication.mutation.operations:
                            if operation.key != "pending_tool_round" or operation.action != "set":
                                continue
                            observed.append(stage.stage_id)
                            if cancel_stop == "tampered_source_round":
                                operation.value["tool_round_id"] = "foreign-round"
                            elif cancel_stop == "tampered_source_attempt":
                                operation.value["model_attempt_id"] = "foreign-attempt"
                            elif cancel_stop == "tampered_source_epoch":
                                operation.value["source_run_epoch"] += 1
                            elif cancel_stop == "tampered_source_interaction":
                                operation.value["interaction_id"] = "foreign-interaction"
                            else:
                                operation.value["tool_calls"][0]["tool_call_id"] = "foreign-call"
                        return stage

                    monkeypatch.setattr(
                        app._runtime_session_store,
                        "load_model_completion_stage",
                        contradictory_source_round,
                    )
                    parent_events_before = await store.load_events("parent")
                    with pytest.raises((ValueError, RuntimeError)):
                        await app.recover_incomplete_session(
                            IncompleteSessionRecoveryRequest(
                                session_id=child.id, inactive_for_seconds=0
                            )
                        )
                    assert observed, "Recovery must inspect the retained model source"
                    assert len(provider.requests) == 2 and protected.values == []
                    assert await store.load_events("parent") == parent_events_before
                    assert "foreground_child_wait" in (await store.load_checkpoint("parent"))
                    return
                else:
                    await app.recover_incomplete_session(
                        IncompleteSessionRecoveryRequest(
                            session_id=child.id, inactive_for_seconds=0
                        )
                    )
            await stop_paused_child()
            assert await app.drain_background_interruptions(timeout_s=10)
            await app.recover_persisted_event_side_effects()
            assert (await store.load(child.id)).status == "interrupted"
            if (await store.load("parent")).status != "completed":
                from cayu.runtime.execution_profiles import (
                    active_invocation_execution_profile_from_checkpoint,
                    active_invocation_execution_profile_is_released,
                )

                current = await store.load(child.id)
                cp = await store.load_checkpoint(child.id)
                profile = active_invocation_execution_profile_from_checkpoint(cp)
                terminal = (await store.summarize_outcome(child.id)).terminal_event
                delivery = await store.get_persisted_event_side_effect_delivery(
                    session_id=child.id, event_id=terminal.event.id
                )
                pytest.fail(
                    str(
                        {
                            "epoch": current.run_epoch,
                            "released": profile is not None
                            and active_invocation_execution_profile_is_released(
                                profile, session_id=child.id, run_epoch=current.run_epoch
                            ),
                            "active": app._session_control.has_active_tasks(child.id),
                            "checkpoint_keys": tuple(cp),
                            "event": (
                                terminal.event.type,
                                terminal.event.payload.get("interruption_type"),
                            ),
                            "delivery": delivery,
                        }
                    )
                )
            assert protected.values == [] and len(provider.requests) == 3
            parent_events = await store.load_events("parent")
            terminal_tools = [
                event
                for event in parent_events
                if event.type in {"tool.call.completed", "tool.call.failed"}
            ]
            assert len(terminal_tools) == 1
            assert terminal_tools[0].type == "tool.call.failed"
            assert (
                terminal_tools[0].payload["tool_call_id"]
                == initial["parent_effect"]["tool_call_id"]
            )
            assert "foreground_child_wait" not in (await store.load_checkpoint("parent"))
            assert (
                len([event for event in parent_events if event.type == "interaction.started"]) == 1
            )
            assert (
                len([event for event in parent_events if event.type == "interaction.completed"])
                == 1
            )
            await app.recover_persisted_event_side_effects()
            assert await store.load_events("parent") == parent_events
            assert len(provider.requests) == 3
            if action == "input":
                late_resolution = app.resolve_user_input(
                    UserInputResponse(
                        session_id=child.id, input_id=initial["child_action_id"], answer="too late"
                    )
                )
            else:
                approval = paused_child_checkpoint["pending_tool_approval"]
                late_resolution = app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=child.id,
                        approval_id=approval["approval_id"],
                        tool_round_id=approval["tool_round_id"],
                        tool_call_id=approval["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                    )
                )
            with pytest.raises((ValueError, RuntimeError)):
                _ = [event async for event in late_resolution]
            assert protected.values == [] and len(provider.requests) == 3
            assert await store.load_events("parent") == parent_events
        finally:
            assert await app.drain_background_interruptions(timeout_s=10)
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())
