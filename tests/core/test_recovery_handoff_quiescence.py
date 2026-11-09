"""Retained recovery finalization must stay protected by its claim and epoch."""

import asyncio

from tests.runtime.runtime_provider_support import FakeProvider
from tests.runtime.runtime_recovery_support import FailingTerminalToolEventStore
from tests.runtime.runtime_tool_support import PureRecordingTool

from cayu import AgentSpec, CayuApp, CayuConfig, EventType, Message, RunRequest
from cayu.approvals.tools import ToolApprovalRecoveryOutcome
from cayu.configuration import OperationsConfig
from cayu.observability.hooks import RuntimeHook
from cayu.providers import ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions.cleanup import RecoveryCleanupPolicy
from cayu.tools.rounds import ToolRoundRecoveryRequest


def test_manual_recovery_keeps_claim_until_retained_terminal_hook_settles(monkeypatch):
    async def scenario():
        hook_started = asyncio.Event()
        hook_cancelled = asyncio.Event()
        allow_hook = asyncio.Event()
        hook_settled = asyncio.Event()

        class BlockingInterruptionHook(RuntimeHook):
            execution_profile_identity = ExecutionProfileBehaviorIdentity(
                name="tests:retained-recovery-hook",
                behavior_version="1",
                implementation_version="1",
            )

            async def after_session_interrupted(self, context):
                hook_started.set()
                while not allow_hook.is_set():
                    try:
                        await allow_hook.wait()
                    except asyncio.CancelledError:
                        hook_cancelled.set()
                hook_settled.set()

        store = FailingTerminalToolEventStore()
        app = CayuApp(
            session_store=store,
            runtime_hooks=[BlockingInterruptionHook()],
            config=CayuConfig(
                operations=OperationsConfig(
                    recovery_cleanup_policy=RecoveryCleanupPolicy(
                        step_timeout_seconds=0.05,
                        overall_timeout_seconds=1,
                    )
                )
            ),
            enable_logging=False,
        )
        app.register_provider(
            FakeProvider(
                [
                    [
                        ModelStreamEvent.tool_call(id="call_1", name="side_effect", arguments={}),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                    [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()],
                ]
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"), tools=[PureRecordingTool()]
        )
        initial = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="retained-handoff",
                    messages=[Message.text("user", "use the tool")],
                )
            )
        ]
        assert initial[-1].type is EventType.SESSION_FAILED
        checkpoint = await store.load_checkpoint("retained-handoff")
        assert checkpoint is not None
        request = ToolRoundRecoveryRequest(
            session_id="retained-handoff",
            round_id=checkpoint["pending_tool_round"]["tool_round_id"],
            tool_call_id="call_1",
            outcome=ToolApprovalRecoveryOutcome.COMPLETED,
            message="verified outside",
        )
        fail_heartbeat = asyncio.Event()

        async def heartbeat(**kwargs):
            await fail_heartbeat.wait()
            raise RuntimeError("claim heartbeat lost")

        monkeypatch.setattr(app._recovery_ownership, "heartbeat", heartbeat)
        stream = app.recover_tool_round(request)
        releases_before = store.release_calls
        releases_while_hook_pending = None
        claim_retained = None
        try:
            assert (await anext(stream)).type is EventType.SESSION_RESUMED
            fail_heartbeat.set()
            await asyncio.wait_for(hook_started.wait(), 5)
            await asyncio.wait_for(hook_cancelled.wait(), 5)
            # The bounded observer returns, but the hook is still a live owner.
            assert not await app.drain_recovery_cleanups(timeout_s=0.3)
            assert not hook_settled.is_set()
            current = await store.load_checkpoint(request.session_id)
            releases_while_hook_pending = store.release_calls - releases_before
            claim_retained = current is not None and "incomplete_session_recovery_claim" in current
        finally:
            allow_hook.set()
            await app.drain_recovery_cleanups(timeout_s=5)
            try:  # noqa: SIM105
                await stream.aclose()
            except BaseException:
                # The original heartbeat/cleanup error is not this test's subject.
                pass

        assert hook_settled.is_set()
        final_checkpoint = await store.load_checkpoint(request.session_id)
        assert final_checkpoint is not None
        assert "incomplete_session_recovery_claim" not in final_checkpoint
        completed_hooks = [
            event
            for event in await store.load_events(request.session_id)
            if event.type is EventType.HOOK_COMPLETED
            and event.payload.get("phase") == "after_session_interrupted"
        ]
        assert releases_while_hook_pending == 0 and claim_retained, {
            "invocation_releases_while_hook_pending": releases_while_hook_pending,
            "claim_retained": claim_retained,
            "durable_hook_completions": len(completed_hooks),
        }
        assert len(completed_hooks) == 1

    asyncio.run(scenario())
