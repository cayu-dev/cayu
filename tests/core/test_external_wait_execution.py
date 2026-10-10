"""Native preparation owns stable identity, not execution or dispatch permission."""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from tests.external_wait_support import CONTEXT, Policy, registration, reservation, stores

from cayu.external_waits import ExternalEventWaits
from cayu.runtime._external_wait_execution_scope import execution_preparation_scope
from cayu.sessions.external_waits import ExternalWaitConflict, ExternalWaitExecutionIntent


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("cancel_wait", [False, True])
@pytest.mark.parametrize("generated_session_id", [False, True])
def test_real_initial_execution_parks_the_exact_external_binding(
    backend, cancel_wait, generated_session_id, tmp_path, request, monkeypatch
):
    from cayu import CayuApp
    from cayu.agents import AgentSpec
    from cayu.evals.testing import ScriptedModelProvider
    from cayu.external_wait_host import ExternalWaitHost
    from cayu.messages import Message
    from cayu.providers.base import ModelStreamEvent
    from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait
    from cayu.session_external_waits import SessionExternalWaitAdapter
    from cayu.sessions._session_continuation import ContinuationConflict
    from cayu.sessions.external_waits import ExternalEventDelivery
    from cayu.sessions.requests import ResumeRequest, RunRequest

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
                        ModelStreamEvent.text_delta("Job submitted."),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ],
                    [
                        ModelStreamEvent.text_delta("Result received."),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ],
                ]
            )
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="root", model="model"))
            adapter = SessionExternalWaitAdapter(app, waits)
            host = ExternalWaitHost(adapter, context=CONTEXT)
            session_id = None if generated_session_id else "external-runtime-" + uuid4().hex
            run = RunRequest(
                agent_name="root",
                session_id=session_id,
                messages=[Message.text("user", "Submit job")],
            )
            original_park = _ExternalExecutionToWait.park

            async def park_with_live_writer(boundary, invocation):
                from cayu.sessions.base import _current_session_run_epoch

                executing_id = invocation.binding.session_id
                assert _current_session_run_epoch(executing_id) == invocation.binding.run_epoch
                assert (await store.inspect_session_execution(executing_id)).state == "executing"
                return await original_park(boundary, invocation)

            monkeypatch.setattr(_ExternalExecutionToWait, "park", park_with_live_writer)
            receipt = await adapter.run_to_wait(run, registered, context=CONTEXT)
            if session_id is not None:
                assert receipt.session_id == session_id
            session_id = receipt.session_id
            assert await adapter.run_to_wait(run, registered, context=CONTEXT) == receipt
            if generated_session_id:
                # A generated ID must not replace the caller's original request
                # commitment, even when an explicit ID names that same session.
                with pytest.raises(ExternalWaitConflict):
                    await adapter.run_to_wait(
                        run.model_copy(update={"session_id": session_id}),
                        registered,
                        context=CONTEXT,
                    )
            with pytest.raises(ExternalWaitConflict):
                await adapter.resume_to_wait(
                    ResumeRequest(session_id=session_id, messages=run.messages),
                    registered,
                    context=CONTEXT,
                )
            retained = await reopen()._read_external_wait(
                correlation.request.scope, correlation.request.correlation_key
            )
            assert retained.execution is not None
            assert retained.continuation is not None
            native = await store.load_continuation_ticket(
                session_id,
                registration_key=retained.continuation.intent.registration_key,
                session_instance_id=retained.execution.session_instance_id,
            )
            assert native.ticket.state == "WAITING"
            assert len(provider.requests) == 1
            before_session = await store.load(session_id)
            before_events = await store.load_events(session_id)
            competing = CayuApp(session_store=reopen(), enable_logging=False)
            competing.register_provider(provider, default=True)
            competing.register_agent(AgentSpec(name="root", model="model"))
            for caller in (app, competing):
                with pytest.raises(ContinuationConflict, match="exact continuation admission"):
                    async for _ in caller.resume(
                        ResumeRequest(
                            session_id=session_id, messages=[Message.text("user", "bypass wait")]
                        )
                    ):
                        pass
            assert await store.load(session_id) == before_session
            assert await store.load_events(session_id) == before_events
            assert len(provider.requests) == 1
            before = await host.service_once(scope=correlation.request.scope, source="renderer")
            assert before.pending == (correlation.request.correlation_key,)
            assert before.settled == ()
            if cancel_wait:
                from cayu.runtime import _continuation_wait_settlement as settlement

                await waits.cancel(correlation, operation_key="cancel-after-park", context=CONTEXT)
                acknowledge = settlement.acknowledge_retirement

                async def lose_ack(*args, **kwargs):
                    await acknowledge(*args, **kwargs)
                    raise RuntimeError("retirement acknowledgement lost")

                with monkeypatch.context() as patch:
                    patch.setattr(settlement, "acknowledge_retirement", lose_ack)
                    with pytest.raises(RuntimeError, match="retirement acknowledgement lost"):
                        await host.service_once(scope=correlation.request.scope, source="renderer")
                outstanding = await store._read_external_wait(
                    correlation.request.scope, correlation.request.correlation_key
                )
                assert outstanding.handoff == "excluded"
                assert outstanding.pending_handoff
                with pytest.raises(ValueError, match="pending external-wait handoff"):
                    await store.delete_session(session_id)
                restored = reopen()
                restored_app = CayuApp(session_store=restored, enable_logging=False)
                restored_app.register_provider(provider, default=True)
                restored_app.register_agent(AgentSpec(name="root", model="model"))
                restored_waits = ExternalEventWaits(store=restored, access_policy=Policy())
                restored_adapter = SessionExternalWaitAdapter(restored_app, restored_waits)
                host = ExternalWaitHost(restored_adapter, context=CONTEXT)
                page = await host.service_once(scope=correlation.request.scope, source="renderer")
                assert page.settled == (correlation.request.correlation_key,)
                cancelled = await adapter.service_wait(registered, context=CONTEXT)
                assert cancelled.wait.handoff == "excluded"
                assert cancelled.wait.outcome.kind == "cancelled"
                assert await adapter.service_wait(registered, context=CONTEXT) == cancelled
                assert len(provider.requests) == 1
                await store.delete_session(session_id)
                assert await store.load(session_id) is None
                assert await adapter.service_wait(registered, context=CONTEXT) == cancelled
                await restored_waits.aclose()
                await waits.aclose()
                return
            await waits.deliver(
                ExternalEventDelivery(
                    correlation=correlation, delivery_id="result", payload_json='{"result":42}'
                ),
                context=CONTEXT,
            )
            page = await host.service_once(scope=correlation.request.scope, source="renderer")
            assert page.settled == (correlation.request.correlation_key,)
            consumed = await adapter.service_wait(registered, context=CONTEXT)
            assert consumed.wait.handoff == "settled"
            assert len(provider.requests) == 2
            completed = await store.load(session_id)
            assert completed.status.value == "completed"
            assert '{"result":42}' in str(provider.requests[-1].messages)
            assert await adapter.service_wait(registered, context=CONTEXT) == consumed
            assert len(provider.requests) == 2
            reopened_store = reopen()
            reopened_app = CayuApp(session_store=reopened_store, enable_logging=False)
            reopened_app.register_provider(provider, default=True)
            reopened_app.register_agent(AgentSpec(name="root", model="model"))
            reopened_waits = ExternalEventWaits(store=reopened_store, access_policy=Policy())
            reopened_adapter = SessionExternalWaitAdapter(reopened_app, reopened_waits)
            assert await reopened_adapter.service_wait(registered, context=CONTEXT) == consumed
            reopened_host = ExternalWaitHost(reopened_adapter, context=CONTEXT)
            replay = await reopened_host.service_once(
                scope=correlation.request.scope, source="renderer"
            )
            assert replay.settled == replay.pending == ()
            assert len(provider.requests) == 2
            await reopened_waits.aclose()
            await waits.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_runtime_execution_preparation_retains_exact_identity_after_reopen(
    backend, tmp_path, request
):
    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            registered = registration(correlation)
            await waits.register(registered, context=CONTEXT)
            intent = ExternalWaitExecutionIntent(
                mode="run",
                request_sha256="1" * 64,
                profile_sha256="2" * 64,
                session_id="initial",
            )
            command = waits._command(
                "prepare_execution",
                correlation,
                registration=registered,
                execution_intent=intent,
                preparation_owner_id="test-worker",
            )
            with pytest.raises(PermissionError):
                await store._mutate_external_wait(command)
            with execution_preparation_scope(command):
                prepared = await store._mutate_external_wait(command)
            assert prepared.execution is not None
            assert prepared.execution.intent == intent
            assert prepared.continuation is None
            assert await store.load("initial") is None
            restored = reopen()
            with execution_preparation_scope(command):
                assert await restored._mutate_external_wait(command) == prepared
            for field, value in (
                ("request_sha256", "3" * 64),
                ("profile_sha256", "4" * 64),
                ("session_id", "different"),
            ):
                changed = command.model_copy(
                    update={"execution_intent": intent.model_copy(update={field: value})}
                )
                with execution_preparation_scope(changed), pytest.raises(ExternalWaitConflict):
                    await restored._mutate_external_wait(changed)
            assert (
                await restored._read_external_wait(
                    correlation.request.scope, correlation.request.correlation_key
                )
                == prepared
            )
            await waits.aclose()

    asyncio.run(scenario())
