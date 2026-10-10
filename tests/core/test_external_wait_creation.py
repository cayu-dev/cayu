"""Public initial-execution exclusion arbitrates with real native CREATE."""

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
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait
from cayu.session_external_waits import SessionExternalWaitAdapter
from cayu.sessions.external_waits import (
    ExternalEventDelivery,
    ExternalWaitConflict,
    ExternalWaitUnavailable,
)
from cayu.sessions.requests import RunRequest


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("created_first", [False, True])
@pytest.mark.parametrize("cancel_observer", [False, True])
@pytest.mark.parametrize("event_first", [False, True])
def test_prepared_creation_exclusion_fences_delayed_worker(
    backend, created_first, cancel_observer, event_first, tmp_path, request, monkeypatch
):
    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            registered = registration(correlation)
            await waits.register(registered, context=CONTEXT)
            app = CayuApp(session_store=store, enable_logging=False)
            provider = ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.text_delta("submitted"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ]
                ]
            )
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="root", model="model"))
            adapter = SessionExternalWaitAdapter(app, waits)
            session_id = "external-create-" + uuid4().hex
            entered, release = asyncio.Event(), asyncio.Event()
            original = _ExternalExecutionToWait.create_invocation
            workers = []

            async def delayed(boundary, command, execution):
                result = await original(boundary, command, execution) if created_first else None
                entered.set()
                await release.wait()
                return result if created_first else await original(boundary, command, execution)

            async def intercept(boundary, command, execution):
                worker = asyncio.create_task(delayed(boundary, command, execution))
                workers.append(worker)
                return await asyncio.shield(worker)

            monkeypatch.setattr(_ExternalExecutionToWait, "create_invocation", intercept)
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
                if cancel_observer:
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    assert task.cancelled() and task.cancelling() == 1
                restored = reopen()
                restored_waits = ExternalEventWaits(store=restored, access_policy=Policy())
                restored_app = CayuApp(session_store=restored, enable_logging=False)
                receiver = SessionExternalWaitAdapter(restored_app, restored_waits)
                if event_first:
                    await restored_waits.deliver(
                        ExternalEventDelivery(
                            correlation=correlation, delivery_id="result", payload_json="{}"
                        ),
                        context=CONTEXT,
                    )
                if created_first:
                    with pytest.raises(
                        ExternalWaitUnavailable, match="awaiting native writer release"
                    ):
                        await receiver.exclude_prepared_execution(registered, context=CONTEXT)
                    with pytest.raises(ValueError, match="running|pending external-wait handoff"):
                        await restored.delete_session(session_id)
                else:
                    if not event_first:
                        await restored_waits.cancel(
                            correlation, operation_key="cancel-prepared", context=CONTEXT
                        )
                    excluded = await receiver.exclude_prepared_execution(
                        registered, context=CONTEXT
                    )
                    assert excluded.outcome.kind == ("event" if event_first else "cancelled")
                    assert excluded.execution_excluded and not excluded.pending_handoff
                    assert (
                        await receiver.exclude_prepared_execution(registered, context=CONTEXT)
                        == excluded
                    )
                assert len(provider.requests) == 0
            finally:
                release.set()
            if created_first:
                await workers[0]
                if not cancel_observer:
                    await task
                    assert len(provider.requests) == 1
            else:
                with pytest.raises(ExternalWaitConflict):
                    await workers[0]
                if not cancel_observer:
                    with pytest.raises(ExternalWaitConflict):
                        await task
                assert await restored.load(session_id) is None
                assert len(provider.requests) == 0
            await restored_waits.aclose()
            await waits.aclose()

    asyncio.run(scenario())
