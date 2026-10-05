"""Public existing-session whole-turn parking and exact event continuation."""

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
from cayu.session_external_waits import SessionExternalWaitAdapter
from cayu.sessions.base import ResumeRequest, RunRequest
from cayu.sessions.external_waits import ExternalEventDelivery, ExternalWaitConflict


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("cancel_observer", [False, True])
@pytest.mark.parametrize("event_first", [False, True])
def test_exclusion_fences_delayed_resume_admission(
    backend, cancel_observer, event_first, tmp_path, request, monkeypatch
):
    from cayu.sessions.base import SessionRunFenced

    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            provider = ScriptedModelProvider(
                [[ModelStreamEvent.completed({"finish_reason": "stop"})]]
            )
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="root", model="model"))
            session_id = "resume-race-" + uuid4().hex
            async for _ in app.run(
                RunRequest(
                    agent_name="root",
                    session_id=session_id,
                    messages=[Message.text("user", "ready")],
                )
            ):
                pass
            source = await store.load(session_id)
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            registered = registration(correlation)
            await waits.register(registered, context=CONTEXT)
            adapter = SessionExternalWaitAdapter(app, waits)
            entered, release, ended = asyncio.Event(), asyncio.Event(), asyncio.Event()
            observe = waits._observe_operation
            failures = []

            async def pause_admission(operation, *, key, expectation):
                if key[0] != "external-resume":
                    return await observe(operation, key=key, expectation=expectation)

                async def delayed():
                    entered.set()
                    await release.wait()
                    try:
                        return await operation()
                    except BaseException as exc:
                        failures.append(exc)
                        raise
                    finally:
                        ended.set()

                return await observe(delayed, key=key, expectation=expectation)

            monkeypatch.setattr(waits, "_observe_operation", pause_admission)
            running = asyncio.create_task(
                adapter.resume_to_wait(
                    ResumeRequest(session_id=session_id, messages=[Message.text("user", "submit")]),
                    registered,
                    context=CONTEXT,
                )
            )
            await asyncio.wait_for(entered.wait(), 15)
            try:
                if cancel_observer:
                    running.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await running
                    assert running.cancelled() and running.cancelling() == 1
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
                else:
                    await restored_waits.cancel(
                        correlation, operation_key="cancel-resume", context=CONTEXT
                    )
                result = await receiver.exclude_prepared_execution(registered, context=CONTEXT)
                assert result.execution_excluded and not result.pending_handoff
                assert result.outcome.kind == ("event" if event_first else "cancelled")
                assert (
                    await receiver.exclude_prepared_execution(registered, context=CONTEXT) == result
                )
            finally:
                release.set()
            if not cancel_observer:
                with pytest.raises(SessionRunFenced):
                    await running
            await asyncio.wait_for(ended.wait(), 15)
            assert len(failures) == 1 and isinstance(failures[0], SessionRunFenced)
            assert await restored.load(session_id) == source
            assert len(provider.requests) == 1
            assert await restored_waits.inspect(correlation, context=CONTEXT) == result
            await restored_waits.aclose()
            await waits.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("cancel_wait", [False, True])
def test_public_resume_parks_then_consumes_one_event(
    backend, cancel_wait, tmp_path, request, monkeypatch
):
    from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait

    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            provider = ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.text_delta(text),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ]
                    for text in ("Ready.", "Job submitted.", "Result received.")
                ]
            )
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="root", model="model"))
            session_id = "resume-external-" + uuid4().hex
            async for _ in app.run(
                RunRequest(
                    agent_name="root",
                    session_id=session_id,
                    messages=[Message.text("user", "ready")],
                )
            ):
                pass
            source = await store.load(session_id)
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            registered = registration(correlation)
            await waits.register(registered, context=CONTEXT)
            adapter = SessionExternalWaitAdapter(app, waits)
            resume = ResumeRequest(
                session_id=session_id, messages=[Message.text("user", "submit job")]
            )
            original_park = _ExternalExecutionToWait.park

            async def park_with_live_writer(boundary, invocation):
                from cayu.sessions.base import _current_session_run_epoch

                assert _current_session_run_epoch(session_id) == invocation.binding.run_epoch
                assert (await store.inspect_session_execution(session_id)).state == "executing"
                return await original_park(boundary, invocation)

            monkeypatch.setattr(_ExternalExecutionToWait, "park", park_with_live_writer)
            receipt = await adapter.resume_to_wait(resume, registered, context=CONTEXT)
            assert len(provider.requests) == 2
            assert receipt.wait.handoff == "pending"
            assert await adapter.resume_to_wait(resume, registered, context=CONTEXT) == receipt
            with pytest.raises(ExternalWaitConflict):
                await adapter.resume_to_wait(
                    resume.model_copy(update={"messages": [Message.text("user", "changed")]}),
                    registered,
                    context=CONTEXT,
                )
            retained = await store._read_external_wait(
                correlation.request.scope, correlation.request.correlation_key
            )
            assert retained.execution.intent.mode == "resume"
            assert retained.execution.intent.expected_run_epoch == source.run_epoch
            assert retained.execution.session_instance_id == source.instance_id
            reopened = reopen()
            restored_app = CayuApp(session_store=reopened, enable_logging=False)
            restored_app.register_provider(provider, default=True)
            restored_app.register_agent(AgentSpec(name="root", model="model"))
            restored_waits = ExternalEventWaits(store=reopened, access_policy=Policy())
            restored = SessionExternalWaitAdapter(restored_app, restored_waits)
            assert await restored.resume_to_wait(resume, registered, context=CONTEXT) == receipt
            if cancel_wait:
                await restored_waits.cancel(
                    correlation, operation_key="cancel-parked", context=CONTEXT
                )
                result = await restored.service_wait(registered, context=CONTEXT)
                assert result.wait.handoff == "excluded" and not result.wait.pending_handoff
                assert await restored.service_wait(registered, context=CONTEXT) == result
                assert len(provider.requests) == 2
                await reopened.delete_session(session_id)
                await restored_waits.aclose()
                await waits.aclose()
                return
            await restored_waits.deliver(
                ExternalEventDelivery(
                    correlation=correlation, delivery_id="done", payload_json='{"result":42}'
                ),
                context=CONTEXT,
            )
            result = await restored.service_wait(registered, context=CONTEXT)
            assert result.wait.handoff == "settled"
            assert len(provider.requests) == 3
            assert '{"result":42}' in str(provider.requests[-1].messages)
            assert await restored.service_wait(registered, context=CONTEXT) == result
            assert len(provider.requests) == 3
            await restored_waits.aclose()
            await waits.aclose()

    asyncio.run(scenario())
