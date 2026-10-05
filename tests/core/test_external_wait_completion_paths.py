"""Every successful whole-turn completion can park an ordinary external wait."""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from tests.core.test_tool_completion import FinalTool
from tests.external_wait_support import CONTEXT, Policy, registration, reservation, stores

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    SessionStatus,
    StructuredOutputSpec,
    ToolCompletionPolicy,
)
from cayu.evals.testing import scripted_structured_output
from cayu.external_waits import ExternalEventWaits
from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait
from cayu.session_external_waits import SessionExternalWaitAdapter
from cayu.sessions.external_waits import ExternalEventDelivery, ExternalWaitUnavailable


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("completion", ["native", "tool", "final_tool"])
@pytest.mark.parametrize("failure", ["none", "service_ack_loss", "cancel_before_park"])
def test_whole_turn_completion_parks_and_continues_after_restart(
    backend, completion, failure, tmp_path, request, monkeypatch
):
    _whole_turn_completion_scenario(backend, completion, failure, tmp_path, request, monkeypatch)


def _whole_turn_completion_scenario(backend, completion, failure, tmp_path, request, monkeypatch):
    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            registered = registration(correlation)
            await waits.register(registered, context=CONTEXT)
            if completion == "tool":
                script = list(scripted_structured_output({"answer": "OK"}, id="answer"))
            elif completion == "final_tool":
                script = [
                    ModelStreamEvent.tool_call(name="ask_customer", arguments={}, id="question"),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            else:
                script = [
                    ModelStreamEvent.text_delta('{"answer":"OK"}'),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            provider = ScriptedModelProvider(
                [script, script], supports_native_structured_output=completion == "native"
            )
            tool = FinalTool()

            def build_app(native_store):
                app = CayuApp(session_store=native_store, enable_logging=False)
                app.register_provider(provider, default=True)
                app.register_agent(
                    AgentSpec(name="root", model="model"),
                    tools=[tool] if completion == "final_tool" else [],
                )
                return app

            app = build_app(store)
            adapter = SessionExternalWaitAdapter(app, waits)
            run = RunRequest(
                agent_name="root",
                session_id="completion-wait-" + uuid4().hex,
                messages=[Message.text("user", "Complete this turn, then wait.")],
                max_steps=3,
                structured_output=(
                    StructuredOutputSpec(
                        json_schema={
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                            "additionalProperties": False,
                        },
                        strategy=completion,
                    )
                    if completion != "final_tool"
                    else None
                ),
                tool_completion=(
                    ToolCompletionPolicy(tool_names=["ask_customer"])
                    if completion == "final_tool"
                    else None
                ),
            )
            if failure in {
                "cancel_before_park",
                "validation_conflict",
                "validation_numeric_conflict",
            }:
                entered = asyncio.Event()

                async def paused_park(self, invocation):
                    entered.set()
                    await asyncio.Event().wait()

                with monkeypatch.context() as patch:
                    patch.setattr(_ExternalExecutionToWait, "park", paused_park)
                    task = asyncio.create_task(
                        adapter.run_to_wait(run, registered, context=CONTEXT)
                    )
                    try:
                        await asyncio.wait_for(entered.wait(), 30)
                    finally:
                        task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    assert task.cancelled() and task.cancelling() == 1
                assert len(provider.requests) == 1
                restored = reopen()
                restored_waits = ExternalEventWaits(store=restored, access_policy=Policy())
                recovering = SessionExternalWaitAdapter(build_app(restored), restored_waits)
                if failure in {"validation_conflict", "validation_numeric_conflict"}:
                    read_events = restored.query_events

                    async def conflicting_validation(query):
                        rows = await read_events(query)
                        if query.event_type == EventType.STRUCTURED_OUTPUT_VALIDATED and rows:
                            rows[0] = rows[0].model_copy(deep=True)
                            if failure == "validation_numeric_conflict":
                                rows[0].event.payload["step"] = True
                            else:
                                rows[0].event.payload["valid"] = False
                        return rows

                    with monkeypatch.context() as patch:
                        patch.setattr(restored, "query_events", conflicting_validation)
                        with pytest.raises(ExternalWaitUnavailable):
                            await recovering.recover_to_wait(
                                registered, context=CONTEXT, inactive_for_seconds=0
                            )
                    assert len(provider.requests) == 1
                receipt = await recovering.recover_to_wait(
                    registered, context=CONTEXT, inactive_for_seconds=0
                )
                assert (
                    await recovering.recover_to_wait(
                        registered, context=CONTEXT, inactive_for_seconds=0
                    )
                    == receipt
                )
                await restored_waits.aclose()
            else:
                receipt = await adapter.run_to_wait(run, registered, context=CONTEXT)
            assert len(provider.requests) == 1
            assert tool.calls == (1 if completion == "final_tool" else 0)
            assert (await store.load(receipt.session_id)).status == SessionStatus.INTERRUPTED
            events = await store.load_events(receipt.session_id)
            assert not any(event.type == EventType.SESSION_COMPLETED for event in events)
            if completion != "final_tool":
                assert (
                    sum(event.type == EventType.STRUCTURED_OUTPUT_VALIDATED for event in events)
                    == 1
                )
            retained = await store._read_external_wait(
                correlation.request.scope, correlation.request.correlation_key
            )
            native = await store.load_continuation_ticket(
                receipt.session_id,
                registration_key=retained.continuation.intent.registration_key,
                session_instance_id=retained.execution.session_instance_id,
            )
            assert native.ticket.state == "WAITING"
            restored = reopen()
            restored_waits = ExternalEventWaits(store=restored, access_policy=Policy())
            restored_adapter = SessionExternalWaitAdapter(build_app(restored), restored_waits)
            assert await restored_adapter.run_to_wait(run, registered, context=CONTEXT) == receipt
            assert len(provider.requests) == 1
            await restored_waits.deliver(
                ExternalEventDelivery(
                    correlation=correlation, delivery_id="ready", payload_json='{"result":42}'
                ),
                context=CONTEXT,
            )
            if failure == "service_ack_loss":
                original_mutation = restored_waits._mutate

                async def lose_preparation_ack(command):
                    result = await original_mutation(command)
                    if command.kind == "prepare_service":
                        raise ConnectionError(
                            "Service preparation committed without acknowledgement"
                        )
                    return result

                with monkeypatch.context() as patch:
                    patch.setattr(restored_waits, "_mutate", lose_preparation_ack)
                    with pytest.raises(ConnectionError, match="without acknowledgement"):
                        await restored_adapter.service_wait(registered, context=CONTEXT)
                assert len(provider.requests) == 1
                prepared = await restored._read_external_wait(
                    correlation.request.scope, correlation.request.correlation_key
                )
                assert prepared.service_stage_id is not None
                from cayu.runtime._external_wait_settlement import settlement_scope
                from cayu.sessions.external_waits import ExternalWaitConflict

                conflicting = restored_waits._command(
                    "prepare_service",
                    correlation,
                    registration=registered,
                    continuation=prepared.continuation,
                    service=prepared.service,
                    service_stage_id="another-stage",
                )
                with settlement_scope(conflicting), pytest.raises(ExternalWaitConflict):
                    await restored_waits._mutate(conflicting)
                assert (
                    await restored._read_external_wait(
                        correlation.request.scope, correlation.request.correlation_key
                    )
                    == prepared
                )
                await restored_waits.aclose()
                restored = reopen()
                restored_waits = ExternalEventWaits(store=restored, access_policy=Policy())
                restored_adapter = SessionExternalWaitAdapter(build_app(restored), restored_waits)
            if failure == "consumption_ack_loss":
                from cayu.runtime._session_continuation_owner import SessionContinuationOwner

                service = SessionContinuationOwner.service

                async def lose_consumption_ack(owner, *args, **kwargs):
                    await service(owner, *args, **kwargs)
                    raise ConnectionError("Completed continuation acknowledgement lost")

                with monkeypatch.context() as patch:
                    patch.setattr(SessionContinuationOwner, "service", lose_consumption_ack)
                    with pytest.raises(ConnectionError, match="acknowledgement lost"):
                        await restored_adapter.service_wait(registered, context=CONTEXT)
                assert len(provider.requests) == 2
                assert (await restored_waits.inspect(correlation, context=CONTEXT)).pending_handoff
                await restored_adapter.app.aclose()
                await restored_waits.aclose()
                restored = reopen()
                restored_waits = ExternalEventWaits(store=restored, access_policy=Policy())
                restored_adapter = SessionExternalWaitAdapter(build_app(restored), restored_waits)
            settled = await restored_adapter.service_wait(registered, context=CONTEXT)
            assert settled.wait.handoff == "settled"
            assert len(provider.requests) == 2
            assert '{"result":42}' in str(provider.requests[-1].messages)
            assert (await restored.load(receipt.session_id)).status == SessionStatus.COMPLETED
            assert await restored_adapter.service_wait(registered, context=CONTEXT) == settled
            assert len(provider.requests) == 2
            assert tool.calls == (2 if completion == "final_tool" else 0)
            await restored.delete_session(receipt.session_id)
            assert await restored_adapter.service_wait(registered, context=CONTEXT) == settled
            assert len(provider.requests) == 2
            await restored_waits.aclose()
            await waits.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_completed_continuation_lost_ack_reconstructs_without_dispatch(
    backend, tmp_path, request, monkeypatch
):
    _whole_turn_completion_scenario(
        backend, "native", "consumption_ack_loss", tmp_path, request, monkeypatch
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("failure", ["validation_conflict", "validation_numeric_conflict"])
def test_external_native_validation_replay_rejects_changed_evidence(
    backend, failure, tmp_path, request, monkeypatch
):
    _whole_turn_completion_scenario(backend, "native", failure, tmp_path, request, monkeypatch)
