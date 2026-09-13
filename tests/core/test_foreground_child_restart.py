"""Fresh-process qualification of a foreground human-action wait."""

from __future__ import annotations

import asyncio
import os
import select
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import SecretStr
from tests.core.test_foreground_subagent_recovery import _identity, _Provider
from tests.core.test_tool_round_execution_identities import _RecordingTool

from cayu import (
    AgentSpec,
    CayuApp,
    ExecutionProfileMismatchError,
    IncompleteSessionRecoveryRequest,
    InterruptSessionRequest,
    Message,
    ModelCompletionManualRecoveryRequired,
    RecoveryPlanRequest,
    RecoveryPlanSelection,
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
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.runtime.tool_policy import AlwaysRequireApprovalToolPolicy
from cayu.tools.user_input import UserInputTool
from cayu.vaults import SecretRedactor

_TEST_DELIVERY_LEASE_SECONDS = 5.0
_TEST_TERMINAL_CLAIM_LEASE_SECONDS = 5.0


class _RestartRecordingTool(_RecordingTool):
    spec = _RecordingTool.spec.model_copy(
        update={"execution_profile_identity": _identity("restart-record")}
    )


async def _worker(path: str, action: str, phase: str) -> None:
    redacted_ids = action.endswith("-redacted")
    action = action.removesuffix("-redacted")
    if phase in {"child-stop-claim", "claim-action-close", "dispatch-action-close"}:
        from cayu.runtime import _session_engine

        # Keep the real store-clock lease and wait for expiry after SIGKILL.
        # The test must not steal a claim merely because its process disappeared.
        _session_engine._INCOMPLETE_RECOVERY_CLAIM_LEASE = timedelta(
            seconds=_TEST_TERMINAL_CLAIM_LEASE_SECONDS
        )
        if phase in {"claim-action-close", "dispatch-action-close"}:
            from cayu.runtime import _recovery_coordinator

            _recovery_coordinator._INCOMPLETE_RECOVERY_CLAIM_LEASE = timedelta(
                seconds=_TEST_TERMINAL_CLAIM_LEASE_SECONDS
            )
    codec = (
        PublicAuthorityAliasCodec(
            PublicAuthorityAliasKeyring(active_key_id="test", keys={"test": SecretStr("A" * 43)})
        )
        if redacted_ids
        else None
    )
    store = SQLiteSessionStore(path, public_authority_alias_codec=codec)
    claim_delivery = store.claim_persisted_event_side_effect

    async def short_delivery_lease(**kwargs):
        # Exercise real expiry after process death, without rewriting timestamps.
        # A one-second lease gives renewal only 333ms to get the SQLite writer
        # during full recovery, testing scheduler/lock latency instead of crash
        # recovery. Renewal expiry and cancellation have dedicated short tests.
        return await claim_delivery(**{**kwargs, "lease_seconds": _TEST_DELIVERY_LEASE_SECONDS})

    store.claim_persisted_event_side_effect = short_delivery_lease
    tool = _RestartRecordingTool()
    initial = [
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
                arguments={"question": "Which value?"} if action == "input" else {"value": 7},
            ),
            ModelStreamEvent.completed(),
        ],
    ]
    final = [
        [ModelStreamEvent.text_delta("child finished"), ModelStreamEvent.completed()],
        [ModelStreamEvent.text_delta("parent finished"), ModelStreamEvent.completed()],
    ]
    batches = {
        "pause": initial,
        "parent-result": initial + final,
        "child-terminal": initial + final,
        "before-wait": initial,
        "reconstruct-pause": final,
        "child-resolution": initial + final,
        "action-close": initial + final,
        "recover-action-close": final,
        "claim-action-close": final,
        "dispatch-action-close": final,
        "inspect-action-close-dispatch": [],
        "replay-resolution": final,
        "recover": final[-1:],
        "resolve": final,
        "parent-stop": initial,
        "parent-stop-claim": initial,
        "resolve-stopped-parent": final,
        "child-stop-claim": initial,
        "recover-stopped-child": final[-1:],
        "resolve-running": final,
        "reconstruct-running": [],
    }

    class RestartProvider(_Provider):
        async def stream(self, request):
            async for event in super().stream(request):
                if phase == "dispatch-action-close" and len(self.requests) == 1:
                    print("PAUSED", flush=True)
                    await asyncio.Event().wait()
                if phase == "resolve-running" and len(self.requests) == 1:
                    print("CHILD_RUNNING", flush=True)
                    while not Path(path + ".release-child").exists():
                        await asyncio.sleep(0.02)
                yield event

    provider = RestartProvider(batches[phase])

    def build_app(store, provider, tool):
        app = CayuApp(
            session_store=store,
            enable_logging=False,
            secret_redactor=SecretRedactor("cayu-child") if redacted_ids else None,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="test"),
            tools=[
                SubagentTool(
                    app,
                    agents={"child": SubagentSpec(agent_name="child")},
                    execution_profile_identity=_identity("restart-subagent"),
                )
            ],
        )
        app.register_agent(
            AgentSpec(name="child", model="test"),
            tools=[UserInputTool()] if action == "input" else [tool],
            tool_policy=None
            if action == "input"
            else AlwaysRequireApprovalToolPolicy(tools=["record"]),
        )
        return app

    app = build_app(store, provider, tool)
    if phase == "before-wait":
        from cayu.runtime import _foreground_child_wait as child_wait_runtime

        observe = child_wait_runtime.observe_foreground_child_wait

        async def stop_before_wait_persistence(*args, **kwargs):
            wait = await observe(*args, **kwargs)
            if wait is not None and wait.parent_effect.session_id == "parent":
                checkpoint = await store.load_checkpoint("parent")
                assert checkpoint is not None and "pending_tool_round" in checkpoint
                assert "foreground_child_wait" not in checkpoint
                print("PAUSED", flush=True)
                await asyncio.Event().wait()
            return wait

        child_wait_runtime.observe_foreground_child_wait = stop_before_wait_persistence
    if phase == "child-resolution":
        apply_command = app._runtime_session_store.apply_invocation_lifecycle_command

        async def stop_after_resolution_claim(command):
            result = await apply_command(command)
            if command.session_id != "parent":
                checkpoint = await store.load_checkpoint(command.session_id)
                key = (
                    "user_input_resolution_intent"
                    if action == "input"
                    else "approval_resolution_intent"
                )
                if checkpoint is not None and key in checkpoint:
                    assert len(provider.requests) == 2
                    assert tool.values == []
                    print("PAUSED", flush=True)
                    await asyncio.Event().wait()
            return result

        app._runtime_session_store.apply_invocation_lifecycle_command = stop_after_resolution_claim
    if phase == "parent-result":
        publish = app._runtime_session_store.publish_runtime_publication

        async def stop_after_parent_result(session_id, **kwargs):
            result = await publish(session_id, **kwargs)
            if session_id == "parent" and kwargs["request"].kind == "tool-round":
                print("PAUSED", flush=True)
                await asyncio.Event().wait()
            return result

        app._runtime_session_store.publish_runtime_publication = stop_after_parent_result
    if phase == "action-close":
        publish = app._runtime_session_store.publish_runtime_publication

        async def stop_after_action_close(session_id, **kwargs):
            request = kwargs["request"]
            if request.kind == "user-input-close":
                before = (
                    await store.load(session_id),
                    await store.load_checkpoint(session_id),
                    await store.load_events(session_id),
                    await store.load_transcript(session_id),
                )
                for field in ("completed_model_step", "continuation_revision", "max_steps"):
                    changed = request.model_copy(deep=True)
                    operation = next(
                        operation
                        for operation in changed.mutation.operations
                        if operation.key == "foreground_child_post_action_continuation"
                    )
                    if field == "max_steps":
                        operation.value["pending_tool_round"][field] += 1
                    else:
                        operation.value[field] += 1
                    with pytest.raises(ValueError, match="continuation conflicts"):
                        await publish(session_id, **{**kwargs, "request": changed})
                    assert before == (
                        await store.load(session_id),
                        await store.load_checkpoint(session_id),
                        await store.load_events(session_id),
                        await store.load_transcript(session_id),
                    )
                    assert (
                        await store.load_runtime_publication_receipt(
                            session_id, request.publication_id
                        )
                        is None
                    )
            result = await publish(session_id, **kwargs)
            if kwargs["request"].kind in {"approval-close", "user-input-close"}:
                checkpoint = await store.load_checkpoint(session_id)
                assert "foreground_child_post_action_continuation" in checkpoint
                assert "pending_tool_round" not in checkpoint
                assert len(provider.requests) == 2
                assert tool.values == ([] if action == "input" else [7])
                print("PAUSED", flush=True)
                await asyncio.Event().wait()
            return result

        app._runtime_session_store.publish_runtime_publication = stop_after_action_close
    if phase == "child-terminal":
        recover_side_effects = app._event_writer.recover_persisted_side_effects

        async def stop_before_parent_wakeup(**kwargs):
            children = await store.list_sessions(SessionQuery(parent_session_id="parent"))
            if children.sessions and children.sessions[0].status is SessionStatus.COMPLETED:
                from cayu.runtime.execution_profiles import (
                    active_invocation_execution_profile_from_checkpoint,
                    active_invocation_execution_profile_is_released,
                )

                child = children.sessions[0]
                profile = active_invocation_execution_profile_from_checkpoint(
                    await store.load_checkpoint(child.id)
                )
                assert profile is not None
                assert active_invocation_execution_profile_is_released(
                    profile, session_id=child.id, run_epoch=child.run_epoch
                )
                parent_checkpoint = await store.load_checkpoint("parent")
                assert parent_checkpoint is not None
                assert "foreground_child_wait" in parent_checkpoint
                assert "foreground_child_terminal" not in parent_checkpoint
                print("PAUSED", flush=True)
                await asyncio.Event().wait()
            return await recover_side_effects(**kwargs)

        app._event_writer.recover_persisted_side_effects = stop_before_parent_wakeup
    if phase in {
        "pause",
        "parent-result",
        "child-terminal",
        "before-wait",
        "child-resolution",
        "action-close",
        "parent-stop",
        "parent-stop-claim",
        "child-stop-claim",
    }:
        events = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="parent",
                    agent_name="parent",
                    messages=[Message.text("user", "delegate")],
                )
            )
        ]
        assert events[-1].type == "session.interrupted", events[-1].payload
        checkpoint = await store.load_checkpoint("parent")
        assert checkpoint is not None and "foreground_child_wait" in checkpoint
        if phase == "pause":
            print("PAUSED", flush=True)
            # The harness kills this real process with the SQLite connection open.
            await asyncio.Event().wait()
        if phase in {"parent-stop", "parent-stop-claim", "child-stop-claim"}:
            stop_target = (
                checkpoint["foreground_child_wait"]["child_session_id"]
                if phase == "child-stop-claim"
                else "parent"
            )
            if phase in {"parent-stop-claim", "child-stop-claim"}:
                transition = app._runtime_session_store.transition_status_and_checkpoint

                async def stop_after_operator_claim(session_id, **kwargs):
                    result = await transition(session_id, **kwargs)
                    if (
                        session_id == stop_target
                        and kwargs.get("to_status") is SessionStatus.INTERRUPTING
                    ):
                        assert len(provider.requests) == 2
                        print("PAUSED", flush=True)
                        await asyncio.Event().wait()
                    return result

                app._runtime_session_store.transition_status_and_checkpoint = (
                    stop_after_operator_claim
                )
            _ = [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(session_id=stop_target, reason="Stop delegation")
                )
            ]
            assert len(provider.requests) == 2
            print("PAUSED", flush=True)
            await asyncio.Event().wait()
    if phase == "recover-stopped-child":
        await asyncio.sleep(_TEST_TERMINAL_CLAIM_LEASE_SECONDS + 0.1)
        children = await store.list_sessions(SessionQuery(parent_session_id="parent"))
        assert len(children.sessions) == 1
        child = children.sessions[0]
        recovery = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=app.project_session_id_for_exposure(child.id), inactive_for_seconds=0
            )
        )
        await app.recover_persisted_event_side_effects()
        assert (await store.load(child.id)).status is SessionStatus.INTERRUPTED, recovery
        assert (await store.load("parent")).status is SessionStatus.COMPLETED
        assert len(provider.requests) == 1 and tool.values == []
        events = await store.load_events("parent")
        for event_type in ("tool.call.failed", "interaction.started", "interaction.completed"):
            assert sum(event.type == event_type for event in events) == 1
        started = next(event for event in events if event.type == "interaction.started")
        completed = next(event for event in events if event.type == "interaction.completed")
        assert started.interaction_id == completed.interaction_id
        assert "foreground_child_wait" not in (await store.load_checkpoint("parent"))
        child_events = await store.load_events(child.id)
        child_started = [event for event in child_events if event.type == "interaction.started"]
        child_closed = [event for event in child_events if event.type == "interaction.interrupted"]
        assert len(child_started) == len(child_closed) == 1
        assert child_started[0].interaction_id == child_closed[0].interaction_id
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
        assert len(provider.requests) == 1 and tool.values == []
        assert await app.drain_background_interruptions(timeout_s=10)
        await store.close()
        return
    if phase == "resolve-stopped-parent":
        parent_events = await store.load_events("parent")
        if any(event.type == "interaction.interrupted" for event in parent_events):
            with pytest.raises(RuntimeError, match="no authoritative open interaction"):
                await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id="parent", inactive_for_seconds=0)
                )
        else:
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id="parent", inactive_for_seconds=0)
            )
        assert provider.requests == []
    if phase in {"reconstruct-pause", "reconstruct-running"}:
        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id="parent", inactive_for_seconds=0)
        )
        reconstructed = await store.load_checkpoint("parent")
        assert reconstructed is not None and "foreground_child_wait" in reconstructed
        assert provider.requests == []
        if phase == "reconstruct-running":
            parent = await store.load("parent")
            assert parent is not None and parent.status is SessionStatus.INTERRUPTED
            child = await store.load(reconstructed["foreground_child_wait"]["child_session_id"])
            assert child is not None and child.status is SessionStatus.RUNNING
            assert not any(
                event.type in {"tool.call.completed", "tool.call.failed"}
                for event in await store.load_events("parent")
            )
            assert await app.drain_background_interruptions(timeout_s=10)
            await store.close()
            return
    if phase == "inspect-action-close-dispatch":
        children = await store.list_sessions(SessionQuery(parent_session_id="parent"))
        assert len(children.sessions) == 1
        child = children.sessions[0]
        stage = await store.load_active_model_completion_stage(child.id)
        assert stage is not None
        checkpoint = await store.load_checkpoint(child.id)
        assert "foreground_child_post_action_continuation" in checkpoint
        with pytest.raises(ModelCompletionManualRecoveryRequired):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=child.id, inactive_for_seconds=0)
            )
        assert provider.requests == [] and tool.values == []
        assert await store.load_active_model_completion_stage(child.id) is not None
        assert (await store.load_checkpoint(child.id))[
            "foreground_child_post_action_continuation"
        ] == checkpoint["foreground_child_post_action_continuation"]
        assert not any(
            event.type == "tool.call.completed" for event in await store.load_events("parent")
        )
        assert await app.drain_background_interruptions(timeout_s=10)
        await store.close()
        return
    if phase in {"recover-action-close", "claim-action-close", "dispatch-action-close"}:
        children = await store.list_sessions(SessionQuery(parent_session_id="parent"))
        assert len(children.sessions) == 1
        child = children.sessions[0]
        checkpoint = await store.load_checkpoint(child.id)
        marker = checkpoint["foreground_child_post_action_continuation"]

        async def snapshot():
            return (
                await store.load(child.id),
                await store.load_checkpoint(child.id),
                await store.load_events(child.id),
                await store.load_transcript(child.id),
                await store.load_runtime_publication_receipt(
                    child.id, marker["close_publication_id"]
                ),
                await store.load("parent"),
                await store.load_checkpoint("parent"),
                await store.load_events("parent"),
            )

        before = await snapshot()
        for _ in range(2):
            await app.plan_recovery(
                RecoveryPlanRequest(
                    selection=RecoveryPlanSelection(
                        session_ids=(app.project_session_id_for_exposure(child.id),),
                        inactive_for_seconds=0,
                    )
                )
            )
            assert await snapshot() == before
        assert provider.requests == [] and tool.values == []
        request = IncompleteSessionRecoveryRequest(
            session_id=app.project_session_id_for_exposure(child.id), inactive_for_seconds=0
        )
        unregistered = CayuApp(session_store=store, enable_logging=False)
        rejected = await unregistered.recover_incomplete_session(request)
        assert [action.value for action in rejected.actions] == ["skipped_unregistered_agent"]
        assert await snapshot() == before

        class IncompatibleProvider(RestartProvider):
            @property
            def execution_profile_identity(self):
                return _identity("incompatible-post-close-provider")

        incompatible_provider = IncompatibleProvider(final)
        incompatible = build_app(store, incompatible_provider, _RestartRecordingTool())
        plan = await incompatible.plan_recovery(
            RecoveryPlanRequest(
                selection=RecoveryPlanSelection(
                    session_ids=(request.session_id,), inactive_for_seconds=0
                )
            )
        )
        assert plan.items[0].registration.status.value == "incompatible"
        assert await snapshot() == before
        with pytest.raises(ExecutionProfileMismatchError):
            await incompatible.recover_incomplete_session(request)
        # A rejection diagnostic is allowed; execution state and the exact
        # committed close must stay unchanged, and no provider may be called.
        rejected_snapshot = await snapshot()
        for index in (1, 3, 4, 5, 6, 7):
            assert rejected_snapshot[index] == before[index]
        assert incompatible_provider.requests == []

        if phase in {"claim-action-close", "dispatch-action-close"}:
            if phase == "claim-action-close":
                apply_command = app._runtime_session_store.apply_invocation_lifecycle_command

                async def stop_after_recovery_fence(command):
                    result = await apply_command(command)
                    if command.session_id == child.id:
                        current = await store.load_checkpoint(child.id)
                        fenced = await store.load(child.id)
                        if fenced.run_epoch > child.run_epoch:
                            assert current["foreground_child_post_action_continuation"] == marker
                            assert "pending_tool_round" not in current
                            assert provider.requests == [] and tool.values == []
                            print("PAUSED", flush=True)
                            await asyncio.Event().wait()
                    return result

                app._runtime_session_store.apply_invocation_lifecycle_command = (
                    stop_after_recovery_fence
                )
            await app.recover_incomplete_session(request)
            pytest.fail("Recovery did not reach its process-loss barrier")

        peer_store = SQLiteSessionStore(path, public_authority_alias_codec=codec)
        peer_provider = RestartProvider(final)
        peer_tool = _RestartRecordingTool()
        peer = build_app(peer_store, peer_provider, peer_tool)
        claimed = asyncio.Event()
        release = asyncio.Event()
        reserve = app._runtime_session_store.reserve_stalled_run_recovery

        async def hold_claim(*args, **kwargs):
            result = await reserve(*args, **kwargs)
            if result is not None:
                claimed.set()
                await release.wait()
            return result

        app._runtime_session_store.reserve_stalled_run_recovery = hold_claim
        recovery_task = asyncio.create_task(app.recover_incomplete_session(request))
        try:
            await asyncio.wait_for(claimed.wait(), timeout=10)
            owned = await snapshot()
            assert owned[1]["foreground_child_post_action_continuation"] == marker
            assert "pending_tool_round" not in owned[1]
            competing = await peer.recover_incomplete_session(request)
            assert [action.value for action in competing.actions] == ["skipped_active"]
            assert await snapshot() == owned
            assert peer_provider.requests == [] and peer_tool.values == []
        finally:
            release.set()
            recovery = await asyncio.wait_for(recovery_task, timeout=30)
            assert await peer.drain_background_interruptions(timeout_s=10)
            await peer_store.close()
        assert await app.drain_background_interruptions(timeout_s=10)
        await app.recover_persisted_event_side_effects()
        assert (await store.load("parent")).status is SessionStatus.COMPLETED, recovery
        assert len(provider.requests) == 2 and tool.values == []
        assert "foreground_child_post_action_continuation" not in (
            await store.load_checkpoint(child.id)
        )
        assert (
            sum(event.type == "tool.call.completed" for event in await store.load_events(child.id))
            == 1
        )
    elif phase == "recover":
        await asyncio.sleep(_TEST_DELIVERY_LEASE_SECONDS + 0.1)
        await app.recover_persisted_event_side_effects()
        parent = await store.load("parent")
        assert parent is not None and parent.status is SessionStatus.COMPLETED, (
            await store.summarize_outcome("parent"),
            [
                (delivery.event_id, delivery.status, delivery.last_error)
                for delivery in await store.list_persisted_event_side_effect_deliveries(limit=100)
                if delivery.last_error is not None
            ],
        )
        assert len(provider.requests) == 1
        assert tool.values == []
    elif phase != "pause":
        children = await store.list_sessions(SessionQuery(parent_session_id="parent"))
        assert len(children.sessions) == 1
        child = children.sessions[0]
        if phase == "replay-resolution":
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=app.project_session_id_for_exposure(child.id), inactive_for_seconds=0
                )
            )
            assert provider.requests == []
            assert tool.values == []
        events = await store.load_events(child.id)
        if action == "input":
            pending = next(event for event in events if event.type == "session.awaiting_user_input")
            stream = app.resolve_user_input(
                UserInputResponse(
                    session_id=app.project_session_id_for_exposure(child.id),
                    input_id=pending.payload["input_id"],
                    answer="7",
                )
            )
        else:
            pending = next(
                event for event in events if event.type == "tool.call.approval_requested"
            )
            stream = app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id=app.project_session_id_for_exposure(child.id),
                    approval_id=pending.payload["approval"]["approval_id"],
                    tool_round_id=pending.payload["tool_round_id"],
                    tool_call_id=pending.payload["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        resolved_events = [event async for event in stream]
        if phase == "action-close":
            pytest.fail(f"Action close did not reach its committed barrier: {resolved_events}")
        parent = await store.load("parent")
        assert parent is not None and parent.status is (
            SessionStatus.INTERRUPTED
            if phase == "resolve-stopped-parent"
            else SessionStatus.COMPLETED
        ), await store.summarize_outcome("parent")
        assert len(provider.requests) == (1 if phase == "resolve-stopped-parent" else 2), [
            (event.type, event.payload) for event in resolved_events
        ]
        assert tool.values == ([] if action == "input" else [7])
    requests_before_redelivery = len(provider.requests)
    await app.recover_persisted_event_side_effects()
    assert len(provider.requests) == requests_before_redelivery
    events = await store.load_events("parent")
    if phase == "resolve-stopped-parent":
        assert not any(
            event.type in {"tool.call.completed", "interaction.completed"} for event in events
        )
        assert [event.payload["step"] for event in events if event.type == "model.started"] == [1]
        interrupted = [event for event in events if event.type == "interaction.interrupted"]
        started = [event for event in events if event.type == "interaction.started"]
        assert len(interrupted) == len(started) == 1
        assert interrupted[0].interaction_id == started[0].interaction_id
        assert await app.drain_background_interruptions(timeout_s=10)
        await store.close()
        return
    assert len([event for event in events if event.type == "tool.call.completed"]) == 1
    assert (
        len(
            [message for message in await store.load_transcript("parent") if message.role == "tool"]
        )
        == 1
    )
    assert [event.payload["step"] for event in events if event.type == "model.started"] == [1, 2]
    from cayu.runtime.execution_profiles import (
        active_invocation_execution_profile_from_checkpoint,
        active_invocation_execution_profile_is_released,
    )

    children = await store.list_sessions(SessionQuery(parent_session_id="parent"))
    assert len(children.sessions) == 1
    for session_id in ("parent", children.sessions[0].id):
        settled = await store.load(session_id)
        assert settled is not None and settled.status is SessionStatus.COMPLETED
        profile = active_invocation_execution_profile_from_checkpoint(
            await store.load_checkpoint(session_id)
        )
        assert profile is not None
        assert active_invocation_execution_profile_is_released(
            profile, session_id=session_id, run_epoch=settled.run_epoch
        )
        interaction_events = await store.load_events(session_id)
        started = [event for event in interaction_events if event.type == "interaction.started"]
        completed = [event for event in interaction_events if event.type == "interaction.completed"]
        assert len(started) == len(completed) == 1
        assert started[0].interaction_id == completed[0].interaction_id
    assert await app.drain_background_interruptions(timeout_s=10)
    await store.close()


@pytest.mark.skipif(sys.platform == "win32", reason="Requires POSIX pipe readiness and SIGKILL")
@pytest.mark.parametrize("action", ["approve", "input"])
def test_parent_reconstruction_while_resolved_child_runs_in_another_process(tmp_path, action):
    path = tmp_path / "running-restart.sqlite"
    command = [
        sys.executable,
        "-m",
        "tests.core.test_foreground_child_restart",
        str(path),
        action,
    ]
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(Path.cwd() / "src"), str(Path.cwd()))),
    }
    processes = []

    def start(phase):
        process = subprocess.Popen(
            [*command, phase],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
        processes.append(process)
        return process

    def await_marker(process, marker):
        assert process.stdout is not None
        output = []
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            ready, _, _ = select.select(
                [process.stdout], [], [], max(0, deadline - time.monotonic())
            )
            assert ready, output
            line = process.stdout.readline()
            output.append(line)
            if line.strip() == marker:
                return
            assert line, "".join(output)
        pytest.fail(f"Worker did not reach {marker}: {output}")

    try:
        parent = start("before-wait")
        await_marker(parent, "PAUSED")
        parent.kill()
        parent.communicate(timeout=10)
        assert parent.returncode == -9
        child = start("resolve-running")
        await_marker(child, "CHILD_RUNNING")
        recovered = subprocess.run(
            [*command, "reconstruct-running"],
            capture_output=True,
            text=True,
            env=environment,
            timeout=60,
        )
        assert recovered.returncode == 0, recovered.stdout + recovered.stderr
        Path(str(path) + ".release-child").touch()
        output, _ = child.communicate(timeout=60)
        assert child.returncode == 0, output
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)


@pytest.mark.skipif(sys.platform == "win32", reason="Requires POSIX pipe readiness and SIGKILL")
@pytest.mark.parametrize("action", ["approve", "input"])
@pytest.mark.parametrize(
    "boundary",
    [
        "pause",
        "parent-result",
        "child-terminal",
        "before-wait",
        "child-resolution",
        "action-close",
        "action-close-claim",
        "action-close-dispatch",
        "parent-stop",
        "parent-stop-claim",
        "child-stop-claim",
    ],
)
def test_foreground_wait_survives_process_kill_and_public_child_resolution(
    tmp_path, action, boundary
):
    command = [
        sys.executable,
        "-m",
        "tests.core.test_foreground_child_restart",
        str(tmp_path / "restart.sqlite"),
        action,
    ]
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(Path.cwd() / "src"), str(Path.cwd()))),
    }
    process = subprocess.Popen(
        [*command, "action-close" if boundary.startswith("action-close") else boundary],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
    )
    output = []
    try:
        assert process.stdout is not None
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            ready, _, _ = select.select(
                [process.stdout], [], [], max(0, deadline - time.monotonic())
            )
            assert ready, output
            line = process.stdout.readline()
            output.append(line)
            if line.strip() == "PAUSED":
                break
            assert line, "".join(output)
        else:
            pytest.fail(f"Worker did not reach durable pause: {output}")
        process.kill()
        process.communicate(timeout=10)
        assert process.returncode == -9
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
    if boundary in {"action-close-claim", "action-close-dispatch"}:
        claimant = subprocess.Popen(
            [
                *command,
                "claim-action-close"
                if boundary == "action-close-claim"
                else "dispatch-action-close",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
        claim_output = []
        try:
            assert claimant.stdout is not None
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                ready, _, _ = select.select(
                    [claimant.stdout], [], [], max(0, deadline - time.monotonic())
                )
                assert ready, claim_output
                line = claimant.stdout.readline()
                claim_output.append(line)
                if line.strip() == "PAUSED":
                    break
                assert line, "".join(claim_output)
            else:
                pytest.fail(f"Claim worker did not reach its durable barrier: {claim_output}")
            claimant.kill()
            claimant.communicate(timeout=10)
            assert claimant.returncode == -9
        finally:
            if claimant.poll() is None:
                claimant.kill()
                claimant.communicate(timeout=10)
        # Respect the real store-clock claim, even though the owner was killed.
        time.sleep(_TEST_TERMINAL_CLAIM_LEASE_SECONDS + 0.1)
    result = subprocess.run(
        [
            *command,
            "recover-stopped-child"
            if boundary == "child-stop-claim"
            else "inspect-action-close-dispatch"
            if boundary == "action-close-dispatch"
            else "recover-action-close"
            if boundary in {"action-close", "action-close-claim"}
            else "resolve-stopped-parent"
            if boundary in {"parent-stop", "parent-stop-claim"}
            else "resolve"
            if boundary == "pause"
            else "reconstruct-pause"
            if boundary == "before-wait"
            else "replay-resolution"
            if boundary == "child-resolution"
            else "recover",
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("action", ["approve", "input"])
@pytest.mark.parametrize("boundary", ["pause", "child-resolution", "action-close"])
def test_redacted_child_identity_survives_process_kill(tmp_path, action, boundary):
    test_foreground_wait_survives_process_kill_and_public_child_resolution(
        tmp_path, action + "-redacted", boundary
    )


if __name__ == "__main__":
    asyncio.run(_worker(*sys.argv[1:]))
