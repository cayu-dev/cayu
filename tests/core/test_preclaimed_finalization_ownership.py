"""Public interruption retains its preclaimed worker after observer cancellation."""

import asyncio
from contextlib import aclosing

import pytest
from tests.core.test_session_store_shared_conformance import (
    _close_store,
    _collect_events,
    _open_store,
    _SimulatedProcessLoss,
    _UserInputRecoveryProvider,
)
from tests.core.test_session_store_shared_conformance import (
    conformance_postgres_dsn as conformance_postgres_dsn,
)
from tests.core.test_session_store_shared_conformance import (
    session_store_case as session_store_case,
)

from cayu import (
    AgentSpec,
    CayuApp,
    CayuConfig,
    EventType,
    InterruptSessionRequest,
    Message,
    RunRequest,
    RuntimeHook,
    SessionStatus,
    UserInputTool,
)
from cayu.configuration import OperationsConfig
from cayu.runtime._session_engine import suppress_interruption_cascade
from cayu.sessions.cleanup import RecoveryCleanupDeadlineExceeded, RecoveryCleanupPolicy


@pytest.mark.parametrize("cancel_count", [0, 1, 2], ids=["close", "cancel", "repeat-cancel"])
def test_public_preclaimed_finalizer_retains_claim_until_worker_settles(
    session_store_case, cancel_count
):
    async def scenario():
        class Hook(RuntimeHook):
            def __init__(self):
                self.armed = False
                self.entered = asyncio.Event()
                self.release = asyncio.Event()
                self.calls = 0
                self.cancelled = False

            async def after_session_interrupted(self, context):
                if not self.armed:
                    return
                self.calls += 1
                self.entered.set()
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise

        store = await _open_store(session_store_case)
        hook = Hook()
        provider = _UserInputRecoveryProvider()
        app = CayuApp(
            session_store=store,
            runtime_hooks=[hook],
            enable_logging=False,
            config=CayuConfig(
                operations=OperationsConfig(
                    recovery_cleanup_policy=RecoveryCleanupPolicy(
                        step_timeout_seconds=0.2, overall_timeout_seconds=1
                    )
                )
            ),
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[UserInputTool()])
        session_id = f"preclaimed-worker-{cancel_count}-{session_store_case[0]}"
        owner = None
        try:
            await _collect_events(
                app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "ask a question")],
                    )
                )
            )
            interrupt = InterruptSessionRequest(session_id=session_id, reason="supersede input")
            original_transition = store.transition_status_and_checkpoint

            async def lose_after_supersession(*args, **kwargs):
                result = await original_transition(*args, **kwargs)
                if kwargs.get("to_status") is SessionStatus.INTERRUPTING:
                    raise _SimulatedProcessLoss("supersession committed")
                return result

            store.transition_status_and_checkpoint = lose_after_supersession
            try:
                with suppress_interruption_cascade(), pytest.raises(_SimulatedProcessLoss):
                    await _collect_events(app.interrupt_session(interrupt))
            finally:
                store.transition_status_and_checkpoint = original_transition

            def abandon_lost_process_claim(session, checkpoint):
                retained = dict(checkpoint)
                retained.pop("incomplete_session_recovery_claim", None)
                return retained

            await store.transform_checkpoint(session_id, abandon_lost_process_claim)
            prior_events = {event.id for event in await store.load_events(session_id)}
            hook.armed = True
            close_requested = asyncio.Event()

            async def observe_finalization():
                async with aclosing(app.interrupt_session(interrupt)) as stream:
                    async for event in stream:
                        if cancel_count == 0 and event.type is EventType.SESSION_INTERRUPTED:
                            await close_requested.wait()
                            break

            with suppress_interruption_cascade():
                owner = asyncio.create_task(observe_finalization())
                entered = asyncio.create_task(hook.entered.wait())
                done, _ = await asyncio.wait(
                    (owner, entered), timeout=15, return_when=asyncio.FIRST_COMPLETED
                )
                if owner in done:
                    await owner
                assert entered in done, "Public interruption did not enter terminal finalization"
                before = await store.load_checkpoint(session_id)
                claim = before["incomplete_session_recovery_claim"]
                if cancel_count:
                    for _ in range(cancel_count):
                        assert owner.cancel("cancel finalization observer")
                        await asyncio.sleep(0)
                    assert owner.cancelling() == cancel_count
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(asyncio.shield(owner), 10)
                    assert owner.cancelled() and owner.cancelling() == cancel_count
                else:
                    close_requested.set()
                    # Explicit close reports unresolved cleanup using its typed
                    # deadline, rather than pretending the blocked work stopped.
                    with pytest.raises(RecoveryCleanupDeadlineExceeded):
                        await asyncio.wait_for(asyncio.shield(owner), 10)
                    assert not owner.cancelled() and owner.cancelling() == 0
                assert not hook.cancelled and hook.calls == 1
                retained = await store.load_checkpoint(session_id)
                assert retained["incomplete_session_recovery_claim"] == claim
                assert not await app.drain_recovery_cleanups(timeout_s=0.01)
                # An exact public retry must not redispatch the blocked hook or
                # remove the original worker's still-live responsibility.
                await _collect_events(app.interrupt_session(interrupt))
                assert hook.calls == 1
                assert "incomplete_session_recovery_claim" in await store.load_checkpoint(
                    session_id
                )
                hook.release.set()
                assert await app.drain_recovery_cleanups(timeout_s=15)
                assert "incomplete_session_recovery_claim" not in await store.load_checkpoint(
                    session_id
                )
                assert not app._recovery_coordinator._recovery_claim_workers
                assert len(provider.requests) == 1
                published = [
                    event
                    for event in await store.load_events(session_id)
                    if event.id not in prior_events
                ]
                assert sum(event.type is EventType.HOOK_STARTED for event in published) == 1
                assert sum(event.type is EventType.HOOK_COMPLETED for event in published) == 1
                assert not any(event.type is EventType.HOOK_FAILED for event in published)
        finally:
            hook.release.set()
            if owner is not None and not owner.done():
                owner.cancel()
                await asyncio.gather(owner, return_exceptions=True)
            await app.drain_recovery_cleanups(timeout_s=15)
            await _close_store(store)

    asyncio.run(scenario())
