"""Accepted cooperative stops precede final-tool completion and result replay."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from tests.core.test_require_final_tool import _ApprovalPolicy
from tests.core.test_tool_completion import CrashStore, FinalTool, app_for, call, request
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
    SessionStatus,
    ToolApprovalDecision,
    ToolApprovalRequest,
)
from cayu.runtime.execution_profiles import active_invocation_execution_profile_from_checkpoint
from cayu.runtime.session_steering import StopAfterCurrentToolRoundRequest


async def accept_stop(store, session_id, *, queued):
    controller = CayuApp(session_store=store, enable_logging=False)
    if queued:
        await controller.enqueue_session_message(
            EnqueueSessionMessageRequest(
                session_id=session_id,
                idempotency_key="correction",
                content="Help with another order",
                delivery_mode="on_idle",
            )
        )
    session = await store.load(session_id)
    active = active_invocation_execution_profile_from_checkpoint(
        await store.load_checkpoint(session_id)
    )
    assert active is not None
    receipt = await controller.stop_after_current_tool_round(
        StopAfterCurrentToolRoundRequest(
            session_id=session.id,
            session_instance_id=session.instance_id,
            interaction_id=active.interaction_id,
            expected_run_epoch=session.run_epoch,
            idempotency_key="stop",
        )
    )
    assert (await store.load(session_id)).status == SessionStatus.RUNNING
    return controller, receipt


async def assert_stopped(store, controller, receipt):
    session_id = receipt.request.session_id
    assert (await store.load(session_id)).status == SessionStatus.INTERRUPTED
    events = await store.load_events(session_id)
    assert not any(
        event.type
        in {
            EventType.SESSION_COMPLETED,
            EventType.SESSION_FAILED,
            EventType.INTERACTION_COMPLETED,
            EventType.INTERACTION_FAILED,
            EventType.SESSION_MESSAGE_DELIVERED,
        }
        for event in events
    )
    tool_events = [event for event in events if event.type == EventType.TOOL_CALL_COMPLETED]
    terminal = [
        event
        for event in events
        if event.type == EventType.SESSION_INTERRUPTED
        and event.payload.get("interruption_type") == "operator_requested"
    ]
    interaction = [event for event in events if event.type == EventType.INTERACTION_INTERRUPTED]
    assert len(tool_events) == len(terminal) == len(interaction) == 1
    assert events.index(tool_events[0]) < events.index(terminal[0])
    assert terminal[0].payload["interruption_type"] == "operator_requested"
    assert "tool_completion" not in interaction[0].payload
    assert [message.role.value for message in await store.load_transcript(session_id)] == [
        "user",
        "assistant",
        "tool",
    ]
    assert await controller.stop_after_current_tool_round(receipt.request) == receipt


@pytest.mark.parametrize("queued", [False, True])
def test_accepted_stop_precedes_final_tool_completion(store_factory, queued):
    async def scenario():
        session_id = f"completion-stop-{uuid4()}"
        entered, release = asyncio.Event(), asyncio.Event()

        class BlockingTool(FinalTool):
            async def run(self, ctx, args):
                entered.set()
                await release.wait()
                return await super().run(ctx, args)

        async with store_factory(CrashStore, boundary="none") as store:
            provider = _TwoCallProvider([call()])
            tool = BlockingTool()
            app = app_for(store, provider, tool)

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
                await asyncio.wait_for(entered.wait(), 10)
                controller, receipt = await accept_stop(store, session_id, queued=queued)
                assert not running.done() and running.cancelling() == 0
                release.set()
                await asyncio.wait_for(running, 20)
                await assert_stopped(store, controller, receipt)
                assert tool.calls == len(provider.requests) == 1
            finally:
                release.set()
                if not running.done():
                    running.cancel()
                await asyncio.gather(running, return_exceptions=True)
                await app.drain_background_interruptions()

    asyncio.run(scenario())


@pytest.mark.parametrize("queued", [False, True])
def test_approval_retains_accepted_stop_before_final_tool_completion(store_factory, queued):
    async def scenario():
        session_id = f"completion-stop-{uuid4()}"
        entered, release = asyncio.Event(), asyncio.Event()

        class BlockingProvider(_TwoCallProvider):
            async def stream(self, request):
                entered.set()
                await release.wait()
                async for event in super().stream(request):
                    yield event

        async with store_factory(CrashStore, boundary="none") as store:
            provider = BlockingProvider([call()])
            tool = FinalTool()
            app = app_for(store, provider, tool, tool_policy=_ApprovalPolicy())

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
                await asyncio.wait_for(entered.wait(), 10)
                controller, receipt = await accept_stop(store, session_id, queued=queued)
                release.set()
                events = await asyncio.wait_for(running, 20)
                pause = next(
                    event
                    for event in events
                    if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
                )
                assert tool.calls == 0
                assert (await store.load(session_id)).status == SessionStatus.INTERRUPTED
                [
                    event
                    async for event in app.resolve_tool_approval(
                        ToolApprovalRequest(
                            session_id=session_id,
                            approval_id=pause.payload["approval"]["approval_id"],
                            tool_call_id=pause.payload["tool_call_id"],
                            tool_round_id=pause.payload["tool_round_id"],
                            decision=ToolApprovalDecision.APPROVE,
                        )
                    )
                ]
                await assert_stopped(store, controller, receipt)
                assert tool.calls == len(provider.requests) == 1
            finally:
                release.set()
                if not running.done():
                    running.cancel()
                await asyncio.gather(running, return_exceptions=True)
                await app.drain_background_interruptions()

    asyncio.run(scenario())


@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("boundary", ["tool-event", "tool-publication"])
def test_recovery_retains_accepted_stop_after_final_tool_success(store_factory, boundary, queued):
    async def scenario():
        session_id = f"completion-stop-{uuid4()}"
        entered, release = asyncio.Event(), asyncio.Event()

        class BlockingTool(FinalTool):
            async def run(self, ctx, args):
                entered.set()
                await release.wait()
                return await super().run(ctx, args)

        async with store_factory(CrashStore, boundary=boundary) as store:
            provider = _TwoCallProvider([call()])
            tool = BlockingTool()
            app = app_for(store, provider, tool)

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
                await asyncio.wait_for(entered.wait(), 10)
                controller, receipt = await accept_stop(store, session_id, queued=queued)
                release.set()
                with pytest.raises(_SimulatedProcessLoss):
                    await asyncio.wait_for(running, 20)
                assert store.crashed and tool.calls == len(provider.requests) == 1
                recovered_provider = _TwoCallProvider([])
                recovered_tool = FinalTool()
                reconstructed = app_for(store, recovered_provider, recovered_tool)
                recovery = IncompleteSessionRecoveryRequest(
                    session_id=session_id, inactive_for_seconds=0
                )
                result = await reconstructed.recover_incomplete_session(recovery)
                assert result.status == SessionStatus.INTERRUPTED
                await assert_stopped(store, controller, receipt)
                again = await reconstructed.recover_incomplete_session(recovery)
                assert again.status == SessionStatus.INTERRUPTED
                await assert_stopped(store, controller, receipt)
                assert recovered_tool.calls == 0 and recovered_provider.requests == []
            finally:
                release.set()
                if not running.done():
                    running.cancel()
                await asyncio.gather(running, return_exceptions=True)

    asyncio.run(scenario())
