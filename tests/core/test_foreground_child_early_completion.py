"""Child completion racing parent wait persistence must not lose its wakeup."""

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
from cayu.runtime import _foreground_child_wait as child_wait
from cayu.runtime._tool_effect_state import ToolEffectStateOwner
from cayu.runtime.authority import SessionRunFenced
from cayu.runtime.execution_profiles import active_invocation_execution_profile_from_checkpoint
from cayu.sessions.base import InMemorySessionStore, RunRequest
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.policy import AlwaysRequireApprovalToolPolicy
from cayu.tools.subagents import SubagentSpec, SubagentTool
from cayu.tools.user_input import UserInputTool


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("action", ["approval", "input"])
@pytest.mark.parametrize(
    "boundary",
    [
        "persist",
        "observe",
        "unreleased",
        "running",
        "running_missing_open",
        "running_missing_settlement",
        "running_conflicting_open",
        "running_corrupt_settlement",
    ],
)
def test_child_terminal_before_parent_wait_remains_deliverable(
    tmp_path, monkeypatch, backend, action, boundary
):
    async def scenario():
        running = boundary.startswith("running")
        invalid_evidence = running and boundary != "running"
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "early-child.sqlite")
        )
        protected = _RecordingTool()
        child_running = asyncio.Event()
        finish_child = asyncio.Event()

        class BarrierProvider(_Provider):
            async def stream(self, request):
                async for event in super().stream(request):
                    if running and len(self.requests) == 3:
                        child_running.set()
                        await finish_child.wait()
                    yield event

        provider = BarrierProvider(
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
                        name="record" if action == "approval" else "ask_user",
                        arguments={"value": 7}
                        if action == "approval"
                        else {"question": "Continue?"},
                    ),
                    ModelStreamEvent.completed(),
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
                    execution_profile_identity=_identity("early-child"),
                )
            ],
        )
        app.register_agent(
            AgentSpec(name="child", model="test"),
            tools=[protected] if action == "approval" else [UserInputTool()],
            tool_policy=AlwaysRequireApprovalToolPolicy(tools=["record"])
            if action == "approval"
            else None,
        )
        observed = asyncio.Event()
        release = asyncio.Event()
        waits = []
        observation_errors = []
        retain = child_wait.retain_foreground_child_wait

        async def pause_before_wait(*args, **kwargs):
            waits.append(kwargs["wait"])
            observed.set()
            await release.wait()
            return await retain(*args, **kwargs)

        if boundary == "persist":
            monkeypatch.setattr(child_wait, "retain_foreground_child_wait", pause_before_wait)
        else:
            observe = child_wait.observe_foreground_child_wait

            async def pause_before_observation(*args, **kwargs):
                initial = await observe(*args, **kwargs)
                if initial is not None:
                    waits.append(initial)
                    observed.set()
                    await release.wait()
                try:
                    return await observe(*args, **kwargs)
                except Exception as exc:
                    observation_errors.append(exc)
                    raise

            monkeypatch.setattr(
                child_wait, "observe_foreground_child_wait", pause_before_observation
            )

        async def run_parent():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="parent",
                        agent_name="parent",
                        messages=[Message.text("user", "go")],
                    )
                )
            ]

        parent_task = asyncio.create_task(run_parent())
        resolution_task = None
        try:
            await asyncio.wait_for(observed.wait(), 15)
            wait = waits[0]
            assert "foreground_child_wait" not in (await store.load_checkpoint("parent"))
            if boundary == "unreleased":
                apply_command = app._runtime_session_store.apply_invocation_lifecycle_command

                async def hold_child_release(command):
                    if command.session_id == wait.child_session_id and command.kind == "release":
                        child = await store.load(wait.child_session_id)
                        if child is not None and child.status == "completed":
                            child_running.set()
                            await finish_child.wait()
                    return await apply_command(command)

                monkeypatch.setattr(
                    app._runtime_session_store,
                    "apply_invocation_lifecycle_command",
                    hold_child_release,
                )
            if action == "approval":
                approval = (await store.load_checkpoint(wait.child_session_id))[
                    "pending_tool_approval"
                ]
                resolution = app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=wait.child_session_id,
                        approval_id=approval["approval_id"],
                        tool_round_id=approval["tool_round_id"],
                        tool_call_id=approval["tool_call_id"],
                        decision=ToolApprovalDecision.APPROVE,
                    )
                )
            else:
                resolution = app.resolve_user_input(
                    UserInputResponse(
                        session_id=wait.child_session_id,
                        input_id=wait.child_action_id,
                        answer="yes",
                    )
                )

            async def resolve_child():
                return [event async for event in resolution]

            resolution_task = asyncio.create_task(resolve_child())
            if boundary == "unreleased":
                await asyncio.wait_for(child_running.wait(), 15)
                assert (await store.load(wait.child_session_id)).status == "completed"
            elif running:
                await asyncio.wait_for(child_running.wait(), 15)
                running_child = await store.load(wait.child_session_id)
                assert running_child.status == "running"
                running_checkpoint = await store.load_checkpoint(wait.child_session_id)
                active = active_invocation_execution_profile_from_checkpoint(running_checkpoint)
                assert active is not None
                child_events = await store.load_events(wait.child_session_id)
                pause_event = [
                    event for event in child_events if event.type == "interaction.paused"
                ][-1]
                historical = await store.load_historical_interaction_settlement(
                    running_child.id,
                    expected_session_instance_id=running_child.instance_id,
                    expected_event=pause_event,
                    expected_profile=active.profile,
                )
                assert historical.session.run_epoch + 1 == wait.child_action_run_epoch
                for incarnation, event in (
                    ("00000000-0000-4000-8000-000000000001", pause_event),
                    (
                        running_child.instance_id,
                        pause_event.model_copy(
                            update={"payload": {**pause_event.payload, "active_duration_ms": -1}}
                        ),
                    ),
                ):
                    with pytest.raises(SessionRunFenced):
                        await store.load_historical_interaction_settlement(
                            running_child.id,
                            expected_session_instance_id=incarnation,
                            expected_event=event,
                            expected_profile=active.profile,
                        )
                with pytest.raises(SessionRunFenced):
                    await store.load_invocation_settlement_transition(
                        running_child.id,
                        expected_session_instance_id=running_child.instance_id,
                        expected_active_invocation_profile=active.model_copy(
                            update={"run_epoch": historical.session.run_epoch}
                        ),
                    )
                assert await store.load(running_child.id) == running_child
                assert await store.load_checkpoint(running_child.id) == running_checkpoint
                assert await store.load_events(running_child.id) == child_events
                if boundary in {"running_missing_open", "running_conflicting_open"}:
                    load_receipt = store.load_runtime_publication_receipt

                    async def missing_open(session_id, publication_id):
                        if session_id == running_child.id and publication_id.startswith(
                            ("approval-open:", "user-input-open:")
                        ):
                            if boundary == "running_missing_open":
                                return None
                            receipt = await load_receipt(session_id, publication_id)
                            assert receipt is not None
                            return receipt.model_copy(
                                update={"interaction_id": "foreign-interaction"}
                            )
                        return await load_receipt(session_id, publication_id)

                    monkeypatch.setattr(store, "load_runtime_publication_receipt", missing_open)
                elif boundary in {"running_missing_settlement", "running_corrupt_settlement"}:
                    load_record = store._load_historical_interaction_settlement_record

                    async def missing_settlement(session_id, event_id):
                        if session_id == running_child.id and event_id == pause_event.id:
                            if boundary == "running_missing_settlement":
                                return None
                            record = await load_record(session_id, event_id)
                            assert record is not None
                            return {**record, "record_digest": "0" * 64}
                        return await load_record(session_id, event_id)

                    monkeypatch.setattr(
                        store, "_load_historical_interaction_settlement_record", missing_settlement
                    )
            else:
                await resolution_task
                assert (await store.load(wait.child_session_id)).status == "completed"
            assert not parent_task.done()
            assert len(provider.requests) == 3
            release.set()
            await asyncio.wait_for(parent_task, 15)
            if invalid_evidence:
                assert observation_errors
                assert len(provider.requests) == 3
                parent_events = await store.load_events("parent")
                assert not any(
                    event.type in {"tool.call.completed", "tool.call.failed"}
                    for event in parent_events
                )
                assert "foreground_child_wait" not in (await store.load_checkpoint("parent"))
                parent = await store.load("parent")
                assert parent is not None
                effect = await ToolEffectStateOwner(store).resolve_call(
                    parent,
                    tool_round_id=wait.parent_effect.tool_round_id,
                    tool_call_id=wait.parent_effect.tool_call_id,
                )
                # Failed observation must retain the dispatched effect, not
                # fabricate settlement. Recovery can classify an executing
                # record as unknown after reclaiming its original run fence.
                assert effect is not None and effect.state in {"executing", "outcome_unknown"}
                assert effect.intent == wait.parent_effect
                assert effect.child_recovery_arguments is not None
                finish_child.set()
                await asyncio.wait_for(resolution_task, 15)
                assert len(provider.requests) == 3
                assert not any(
                    event.type in {"tool.call.completed", "tool.call.failed"}
                    for event in await store.load_events("parent")
                )
                settled_parent = await store.load("parent")
                assert settled_parent is not None
                assert (
                    await ToolEffectStateOwner(store).resolve_call(
                        settled_parent,
                        tool_round_id=wait.parent_effect.tool_round_id,
                        tool_call_id=wait.parent_effect.tool_call_id,
                    )
                    == effect
                )
                return
            if observation_errors:
                raise observation_errors[0]
            if running or boundary == "unreleased":
                retained = (await store.load_checkpoint("parent"))["foreground_child_wait"]
                assert retained == wait.model_dump(mode="json")
                assert (await store.load("parent")).status == "interrupted"
                parent_events = await store.load_events("parent")
                assert not any(
                    event.type in {"tool.call.completed", "tool.call.failed"}
                    for event in parent_events
                )
                assert len(provider.requests) == 3
                finish_child.set()
                await asyncio.wait_for(resolution_task, 15)
            assert await app.drain_background_interruptions(timeout_s=10)
            assert (await store.load("parent")).status == "completed"
            assert len(provider.requests) == 4
            assert protected.values == ([7] if action == "approval" else [])
            events = await store.load_events("parent")
            assert sum(event.type == "tool.call.completed" for event in events) == 1
            assert sum(event.type == "interaction.started" for event in events) == 1
            assert sum(event.type == "interaction.completed" for event in events) == 1
            await app.recover_persisted_event_side_effects()
            assert await store.load_events("parent") == events
            assert len(provider.requests) == 4
        finally:
            release.set()
            finish_child.set()
            if not parent_task.done():
                parent_task.cancel()
            await asyncio.gather(parent_task, return_exceptions=True)
            if resolution_task is not None:
                if not resolution_task.done():
                    resolution_task.cancel()
                await asyncio.gather(resolution_task, return_exceptions=True)
            assert await app.drain_background_interruptions(timeout_s=10)
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())
