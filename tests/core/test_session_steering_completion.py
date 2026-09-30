"""Stop acceptance and completion serialize at the native settlement boundary."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from uuid import uuid4

import pytest
from tests.core.test_tool_completion import FinalTool, app_for, call, request
from tests.core.test_tool_round_publication_failure_matrix import (
    _SimulatedProcessLoss,
    _TwoCallProvider,
)
from tests.core.test_tool_round_publication_failure_matrix import store_factory as store_factory

from cayu import (
    CayuApp,
    EnqueueSessionMessageRequest,
    EventType,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    Message,
    ModelStreamEvent,
    ResumeRequest,
    SessionStatus,
)
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.environments.bindings import SyncBinding
from cayu.runtime._session_steering import SessionSteeringBoundaryReached
from cayu.runtime.execution_profiles import active_invocation_execution_profile_from_checkpoint
from cayu.runtime.session_steering import (
    SessionSteeringConflict,
    StopAfterCurrentToolRoundRequest,
)
from cayu.sessions.base import ModelCompletionStageRequest, SessionRunFenced
from cayu.workspaces.local import LocalWorkspace


class CompletionBarrier:
    invocation_lifecycle_command_version = 1
    session_steering_version = 1

    def __init__(self, *args, boundary, fault="none", **kwargs):
        super().__init__(*args, **kwargs)
        self.boundary = boundary
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.blocked = False
        self.fault = fault
        self.faulted = False

    async def publish_session_operation(self, *args, **kwargs):
        result = await super().publish_session_operation(*args, **kwargs)
        if (
            self.fault == "steering-ack"
            and kwargs["idempotency_key"].startswith("cayu.session-steering.v1:")
            and not self.faulted
        ):
            self.faulted = True
            raise ConnectionError("lost steering acknowledgement")
        return result

    async def settle_session_invocation(self, command):
        block = (
            command.transition.event.type == EventType.INTERACTION_COMPLETED and not self.blocked
        )
        if block:
            self.blocked = True
            if self.boundary == "before-commit":
                self.entered.set()
                await self.release.wait()
        interrupted = command.transition.event.type == EventType.INTERACTION_INTERRUPTED
        if interrupted and self.fault == "process-loss" and not self.faulted:
            self.faulted = True
            raise _SimulatedProcessLoss("lost after cooperative stop promotion")
        try:
            result = await super().settle_session_invocation(command)
        except SessionSteeringBoundaryReached:
            if self.fault == "rejection-ack" and not self.faulted:
                self.faulted = True
                raise ConnectionError("lost stop rejection acknowledgement") from None
            raise
        if interrupted and self.fault == "interruption-ack" and not self.faulted:
            self.faulted = True
            raise ConnectionError("lost interruption acknowledgement")
        if block and self.boundary == "after-commit":
            self.entered.set()
            await self.release.wait()
        if block and self.fault == "completion-ack" and not self.faulted:
            self.faulted = True
            raise ConnectionError("lost completion acknowledgement")
        return result


def responses(final_tool, *, queued):
    ordinary = [ModelStreamEvent.text_delta("Question ready."), ModelStreamEvent.completed()]
    initial = [call()] if final_tool else [call(), ordinary]
    return initial + ([ordinary, ordinary] if queued else [])


async def stop_request(store, session_id):
    session = await store.load(session_id)
    active = active_invocation_execution_profile_from_checkpoint(
        await store.load_checkpoint(session_id)
    )
    assert active is not None
    return StopAfterCurrentToolRoundRequest(
        session_id=session.id,
        session_instance_id=session.instance_id,
        interaction_id=active.interaction_id,
        expected_run_epoch=session.run_epoch,
        idempotency_key="stop",
    )


async def queue_message(controller, session_id):
    await controller.enqueue_session_message(
        EnqueueSessionMessageRequest(
            session_id=session_id,
            idempotency_key="correction",
            content="Help with another order",
            delivery_mode="on_idle",
        )
    )


async def assert_closed_interaction_blocks_new_model_work(store, session_id):
    session = await store.load(session_id)
    events = await store.load_events(session_id)
    with pytest.raises(SessionRunFenced, match="already completed"):
        await store.prepare_model_completion_stage(
            session_id,
            request=ModelCompletionStageRequest(
                stage_id="late-attempt",
                logical_step_id="late-step",
                dispatch_ordinal=0,
                intent={"logical_step": "late-step"},
            ),
            expected_statuses={SessionStatus.RUNNING},
            expected_run_epoch=session.run_epoch,
            expected_transcript_cursor=len(await store.load_transcript(session_id)),
        )
    assert await store.load_active_model_completion_stage(session_id) is None
    assert await store.load_events(session_id) == events


@pytest.mark.parametrize("final_tool", [False, True])
@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("winner", ["stop", "completion"])
def test_stop_and_completion_have_one_winner(store_factory, final_tool, queued, winner):
    async def scenario():
        session_id = f"stop-completion-{uuid4()}"
        boundary = "before-commit" if winner == "stop" else "after-commit"
        fault = "completion-ack" if winner == "completion" else "none"
        async with store_factory(CompletionBarrier, boundary=boundary, fault=fault) as store:
            provider = _TwoCallProvider(responses(final_tool, queued=queued))
            tool = FinalTool()
            app = app_for(store, provider, tool)
            controller = CayuApp(session_store=store, enable_logging=False)
            options = {"tool_completion": {"tool_names": ["ask_customer"]}} if final_tool else {}

            # Queue before completion makes its transaction close the interaction
            # while retaining RUNNING for the separately admitted successor.
            if queued:
                original = tool.run

                async def run_tool(ctx, args):
                    await queue_message(controller, session_id)
                    return await original(ctx, args)

                tool.run = run_tool

            async def collect():
                return [event async for event in app.run(request(session_id=session_id, **options))]

            running = asyncio.create_task(collect())
            try:
                await asyncio.wait_for(store.entered.wait(), 20)
                stop = await stop_request(store, session_id)
                if winner == "stop":
                    receipt = await controller.stop_after_current_tool_round(stop)
                    assert not running.done() and running.cancelling() == 0
                else:
                    with pytest.raises(SessionSteeringConflict):
                        await controller.stop_after_current_tool_round(stop)
                    if queued:
                        await assert_closed_interaction_blocks_new_model_work(store, session_id)
                store.release.set()
                await asyncio.wait_for(running, 20)
                events = await store.load_events(session_id)
                assert tool.calls == 1
                assert sum(event.type == EventType.TOOL_CALL_COMPLETED for event in events) == 1
                if winner == "stop":
                    assert (await store.load(session_id)).status == SessionStatus.INTERRUPTED
                    for kind in (EventType.SESSION_INTERRUPTED, EventType.INTERACTION_INTERRUPTED):
                        assert sum(event.type == kind for event in events) == 1
                    assert not any(
                        event.type
                        in {
                            EventType.SESSION_COMPLETED,
                            EventType.INTERACTION_COMPLETED,
                            EventType.SESSION_FAILED,
                            EventType.INTERACTION_FAILED,
                            EventType.SESSION_MESSAGE_DELIVERED,
                        }
                        for event in events
                    )
                    assert await controller.stop_after_current_tool_round(stop) == receipt
                    assert len(provider.requests) == (1 if final_tool else 2)
                    if queued:
                        [
                            event
                            async for event in app.resume(
                                ResumeRequest(
                                    session_id=session_id,
                                    messages=[Message.text("user", "continue")],
                                    **options,
                                )
                            )
                        ]
                        continued = await store.load_events(session_id)
                        assert (
                            sum(
                                event.type == EventType.SESSION_MESSAGE_DELIVERED
                                for event in continued
                            )
                            == 1
                        )
                        assert len(provider.requests) == (1 if final_tool else 2) + 2
                        assert (await store.load(session_id)).status == SessionStatus.COMPLETED
                        assert tool.calls == 1
                        assert await controller.stop_after_current_tool_round(stop) == receipt
                else:
                    assert (await store.load(session_id)).status == SessionStatus.COMPLETED
                    assert not any(event.type == EventType.SESSION_INTERRUPTED for event in events)
                    assert sum(
                        event.type == EventType.SESSION_MESSAGE_DELIVERED for event in events
                    ) == int(queued)
                    assert len(provider.requests) == (1 if final_tool else 2) + int(queued)
            finally:
                store.release.set()
                if not running.done():
                    running.cancel()
                await asyncio.gather(running, return_exceptions=True)
                await app.drain_background_interruptions()

    asyncio.run(scenario())


@pytest.mark.parametrize("winner", ["stop", "completion"])
def test_completion_critical_finalization_keeps_the_same_winner(store_factory, tmp_path, winner):
    async def scenario():
        session_id = f"stop-completion-workspace-{uuid4()}"
        boundary = "before-commit" if winner == "stop" else "after-commit"
        async with store_factory(
            CompletionBarrier, boundary=boundary, fault="completion-ack"
        ) as store:
            provider = _TwoCallProvider([call()])
            tool = FinalTool()
            app = app_for(store, provider, tool)
            source, target = tmp_path / "source", tmp_path / "target"
            source.mkdir()
            target.mkdir()
            app.register_environment(
                Environment(
                    EnvironmentSpec(name="sync"),
                    workspace=LocalWorkspace(source, workspace_id="stop-source"),
                    binding=SyncBinding(
                        target_workspace=LocalWorkspace(target, workspace_id="stop-target")
                    ),
                ),
                default=True,
            )
            controller = CayuApp(session_store=store, enable_logging=False)

            async def collect():
                return [
                    event
                    async for event in app.run(
                        request(
                            session_id=session_id, tool_completion={"tool_names": ["ask_customer"]}
                        )
                    )
                ]

            running = asyncio.create_task(collect())
            try:
                await asyncio.wait_for(store.entered.wait(), 20)
                assert (await store.load(session_id)).status == SessionStatus.RUNNING
                stop = await stop_request(store, session_id)
                if winner == "stop":
                    receipt = await controller.stop_after_current_tool_round(stop)
                else:
                    with pytest.raises(SessionSteeringConflict):
                        await controller.stop_after_current_tool_round(stop)
                    await assert_closed_interaction_blocks_new_model_work(store, session_id)
                store.release.set()
                await asyncio.wait_for(running, 20)
                status = SessionStatus.INTERRUPTED if winner == "stop" else SessionStatus.COMPLETED
                assert (await store.load(session_id)).status == status
                events = await store.load_events(session_id)
                terminal = (
                    EventType.INTERACTION_INTERRUPTED
                    if winner == "stop"
                    else EventType.INTERACTION_COMPLETED
                )
                assert sum(event.type == terminal for event in events) == 1
                assert not any(
                    event.type in {EventType.SESSION_FAILED, EventType.INTERACTION_FAILED}
                    for event in events
                )
                if winner == "stop":
                    assert await controller.stop_after_current_tool_round(stop) == receipt
                assert tool.calls == len(provider.requests) == 1
            finally:
                store.release.set()
                if not running.done():
                    running.cancel()
                await asyncio.gather(running, return_exceptions=True)
                await app.drain_background_interruptions()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "fault", ["steering-ack", "rejection-ack", "interruption-ack", "process-loss", "cancel"]
)
def test_stop_winner_survives_settlement_faults(store_factory, fault):
    async def scenario():
        session_id = f"stop-completion-fault-{uuid4()}"
        async with AsyncExitStack() as stores:
            store = await stores.enter_async_context(
                store_factory(CompletionBarrier, boundary="before-commit", fault=fault)
            )
            provider = _TwoCallProvider([call()])
            tool = FinalTool()
            app = app_for(store, provider, tool)
            controller = CayuApp(session_store=store, enable_logging=False)

            async def collect():
                return [
                    event
                    async for event in app.run(
                        request(
                            session_id=session_id, tool_completion={"tool_names": ["ask_customer"]}
                        )
                    )
                ]

            running = asyncio.create_task(collect())
            try:
                await asyncio.wait_for(store.entered.wait(), 20)
                stop = await stop_request(store, session_id)
                if fault == "steering-ack":
                    with pytest.raises(ConnectionError, match="lost steering acknowledgement"):
                        await controller.stop_after_current_tool_round(stop)
                receipt = await controller.stop_after_current_tool_round(stop)
                if fault == "cancel":
                    running.cancel("caller cancellation")
                store.release.set()
                if fault == "cancel":
                    with pytest.raises(asyncio.CancelledError, match="caller cancellation"):
                        await running
                elif fault == "process-loss":
                    with pytest.raises(_SimulatedProcessLoss):
                        await running
                    if not isinstance(store, InMemorySessionStore):
                        await store.close()
                        store = await stores.enter_async_context(
                            store_factory(CompletionBarrier, boundary="none")
                        )
                    reconstructed_provider, reconstructed_tool = _TwoCallProvider([]), FinalTool()
                    recovered = app_for(store, reconstructed_provider, reconstructed_tool)
                    controller = CayuApp(session_store=store, enable_logging=False)
                    recovery = IncompleteSessionRecoveryRequest(
                        session_id=session_id, inactive_for_seconds=0
                    )
                    assert (
                        await recovered.recover_incomplete_session(recovery)
                    ).status == SessionStatus.INTERRUPTED
                    assert reconstructed_provider.requests == [] and reconstructed_tool.calls == 0
                    assert (
                        await recovered.recover_incomplete_session(recovery)
                    ).status == SessionStatus.INTERRUPTED
                else:
                    await asyncio.wait_for(running, 20)
                assert (await store.load(session_id)).status == SessionStatus.INTERRUPTED
                events = await store.load_events(session_id)
                for kind in (
                    EventType.TOOL_CALL_COMPLETED,
                    EventType.SESSION_INTERRUPTED,
                    EventType.INTERACTION_INTERRUPTED,
                ):
                    assert sum(event.type == kind for event in events) == 1
                assert (
                    next(
                        event for event in events if event.type == EventType.SESSION_INTERRUPTED
                    ).payload["interruption_type"]
                    == "operator_requested"
                )
                assert not any(
                    event.type
                    in {
                        EventType.INTERACTION_COMPLETED,
                        EventType.SESSION_COMPLETED,
                        EventType.SESSION_FAILED,
                    }
                    for event in events
                )
                assert await controller.stop_after_current_tool_round(stop) == receipt
                assert tool.calls == len(provider.requests) == 1
            finally:
                store.release.set()
                if not running.done():
                    running.cancel()
                await asyncio.gather(running, return_exceptions=True)
                await app.drain_background_interruptions()

    asyncio.run(scenario())
