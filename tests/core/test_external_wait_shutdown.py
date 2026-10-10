"""Application shutdown accounts for writes that outlive a cancelled observer."""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from tests.external_wait_support import CONTEXT, Policy, registration, reservation, stores

from cayu import CayuApp
from cayu.agents import AgentSpec
from cayu.evals.testing import ScriptedModelProvider
from cayu.external_waits import ExternalEventWaits
from cayu.messages import Message
from cayu.runtime.application_lifecycle import ApplicationAdmissionsSealed
from cayu.session_external_waits import SessionExternalWaitAdapter
from cayu.sessions.requests import RunRequest


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("phase", ["external-wait", "external-create"])
def test_cancelled_external_creation_remains_in_application_drain(
    backend, phase, tmp_path, request, monkeypatch
):
    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            registered = registration(correlation)
            await waits.register(registered, context=CONTEXT)
            app = CayuApp(session_store=store, enable_logging=False)
            provider = ScriptedModelProvider([])
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="root", model="model"))
            adapter = SessionExternalWaitAdapter(app, waits)
            session_id = "external-drain-" + uuid4().hex
            entered, release = asyncio.Event(), asyncio.Event()
            observe = waits._observe_operation

            async def pause(operation, *, key, expectation):
                async def retained():
                    entered.set()
                    await release.wait()
                    return await operation()

                return await observe(
                    retained if key[0] == phase else operation, key=key, expectation=expectation
                )

            monkeypatch.setattr(waits, "_observe_operation", pause)
            task = asyncio.create_task(
                adapter.run_to_wait(
                    RunRequest(
                        agent_name="root",
                        session_id=session_id,
                        messages=[Message.text("user", "submit")],
                    ),
                    registered,
                    context=CONTEXT,
                )
            )
            await asyncio.wait_for(entered.wait(), 10)
            try:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert task.cancelled() and task.cancelling() == 1
                assert app._request_coordinator.owners.outstanding()
                closing = await app.aclose(timeout_s=0.1)
                assert not closing.settled
                assert app._request_coordinator.owners.outstanding()
                with pytest.raises(ApplicationAdmissionsSealed):
                    await adapter.service_wait(registered, context=CONTEXT)
                assert len(provider.requests) == 0
            finally:
                release.set()
            assert await waits.aclose(timeout_s=5)
            closed = await app.aclose(timeout_s=5)
            assert closed.settled
            restored = await reopen()._read_external_wait(
                correlation.request.scope, correlation.request.correlation_key
            )
            assert restored.execution is not None
            assert restored.pending_handoff and not restored.execution_excluded
            assert restored.outcome is None
            assert (await store.load(session_id) is not None) == (phase == "external-create")
            assert len(provider.requests) == 0

            # The application owns only work admitted through its adapter. A
            # separately composed webhook owner remains usable on the shared
            # caller-owned store and must not inherit the closed app's tracker.
            standalone = ExternalEventWaits(store=store, access_policy=Policy())
            independent = await standalone.reserve_correlation(
                reservation().model_copy(update={"correlation_key": "independent-" + uuid4().hex}),
                context=CONTEXT,
            )
            assert independent is not None
            assert not app._request_coordinator.owners.outstanding()
            assert await standalone.aclose()

    asyncio.run(scenario())
