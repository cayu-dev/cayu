"""Explicit native retirement preserves accepted outcomes and receiving ownership."""

import asyncio
from datetime import UTC, datetime

import pytest
from tests.external_wait_support import CONTEXT, Policy, registration, reservation, stores

from cayu import CayuApp
from cayu.agents import AgentSpec
from cayu.evals.testing import ScriptedModelProvider
from cayu.external_wait_host import ExternalWaitHost
from cayu.external_waits import ExternalEventWaits
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime import _session_continuation_owner
from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait
from cayu.runtime._session_continuation_owner import SessionContinuationOwner
from cayu.session_external_waits import SessionExternalWaitAdapter
from cayu.sessions._session_continuation import ContinuationConflict, ContinuationUnavailable
from cayu.sessions.base import RunRequest
from cayu.sessions.external_waits import (
    ExternalEventDelivery,
    ExternalWaitConflict,
    ExternalWaitUnavailable,
)


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("before_park", [False, True])
@pytest.mark.parametrize("lost_ack", ["none", "native", "external", "cancel_native"])
def test_explicit_retirement_preserves_event_and_reconstructs_exact_cleanup(
    backend, before_park, lost_ack, tmp_path, request, monkeypatch
):
    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            policy = Policy()
            waits = ExternalEventWaits(store=store, access_policy=policy)
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            registered = registration(correlation)
            await waits.register(registered, context=CONTEXT)
            app = CayuApp(session_store=store, enable_logging=False)
            provider = ScriptedModelProvider(
                [[ModelStreamEvent.completed({"finish_reason": "stop"})]]
            )
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="root", model="model"))
            adapter = SessionExternalWaitAdapter(app, waits)
            entered = asyncio.Event()

            async def block_park(boundary, invocation):
                entered.set()
                await asyncio.Event().wait()

            with monkeypatch.context() as patch:
                if before_park:
                    patch.setattr(_ExternalExecutionToWait, "park", block_park)
                running = asyncio.create_task(
                    adapter.run_to_wait(
                        RunRequest(
                            agent_name="root",
                            session_id="retire-execution",
                            messages=[Message.text("user", "Submit job")],
                        ),
                        registered,
                        context=CONTEXT,
                    )
                )
                if before_park:
                    await asyncio.wait_for(entered.wait(), 20)
                else:
                    await running
                await waits.deliver(
                    ExternalEventDelivery(
                        correlation=correlation, delivery_id="result", payload_json='{"done":true}'
                    ),
                    context=CONTEXT,
                )
                elected = (await waits.inspect(correlation, context=CONTEXT)).outcome
                assert elected.kind == "event"
                if before_park:
                    # The exact control persists but cannot discharge a live writer.
                    with pytest.raises(ContinuationConflict) as refusal:
                        await adapter.retire_execution(
                            registered, operation_key="retire", context=CONTEXT
                        )
                    assert "writer-release" in str(refusal.value.__cause__)
                    pending = await waits.inspect(correlation, context=CONTEXT)
                    assert pending.retirement_requested and pending.pending_handoff
                    assert pending.handoff == "pending" and pending.outcome == elected
                    running.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await running
                    assert running.cancelled() and running.cancelling() == 1

            native_retire = SessionContinuationOwner.retire_released
            mutate = waits._mutate
            native_committed = asyncio.Event()

            async def lose_native_ack(owner, candidate):
                await native_retire(owner, candidate)
                native_committed.set()
                if lost_ack == "cancel_native":
                    await asyncio.Event().wait()
                raise RuntimeError("native retirement acknowledgement lost")

            async def lose_external_ack(command):
                result = await mutate(command)
                if command.kind == "settle":
                    raise RuntimeError("external retirement acknowledgement lost")
                return result

            with monkeypatch.context() as patch:
                if lost_ack in {"native", "cancel_native"}:
                    patch.setattr(SessionContinuationOwner, "retire_released", lose_native_ack)
                elif lost_ack == "external":
                    patch.setattr(waits, "_mutate", lose_external_ack)
                if lost_ack != "none":
                    if lost_ack == "cancel_native":
                        retiring = asyncio.create_task(
                            adapter.retire_execution(
                                registered, operation_key="retire", context=CONTEXT
                            )
                        )
                        await asyncio.wait_for(native_committed.wait(), 20)
                        retiring.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await retiring
                        assert retiring.cancelled() and retiring.cancelling() == 1
                    else:
                        with pytest.raises(RuntimeError, match="acknowledgement lost"):
                            await adapter.retire_execution(
                                registered, operation_key="retire", context=CONTEXT
                            )
                    uncertain = await waits.inspect(correlation, context=CONTEXT)
                    assert uncertain.pending_handoff and uncertain.outcome == elected
                    with pytest.raises(ValueError):
                        await store.delete_session("retire-execution")
                else:
                    await adapter.retire_execution(
                        registered, operation_key="retire", context=CONTEXT
                    )

            restored = reopen()
            restored_waits = ExternalEventWaits(store=restored, access_policy=policy)
            restored_app = CayuApp(session_store=restored, enable_logging=False)
            recovered = SessionExternalWaitAdapter(restored_app, restored_waits)
            before = await restored_waits.inspect(correlation, context=CONTEXT)
            policy.revoked = True
            with pytest.raises(PermissionError):
                await recovered.retire_execution(
                    registered, operation_key="retire", context=CONTEXT
                )
            policy.revoked = False
            assert await restored_waits.inspect(correlation, context=CONTEXT) == before
            # Restart discovers the pending exact control; no provider is needed.
            host = ExternalWaitHost(recovered, context=CONTEXT)
            await host.service_once(scope=correlation.request.scope, source="renderer")
            terminal = await recovered.retire_execution(
                registered, operation_key="retire", context=CONTEXT
            )
            assert terminal.handoff == "excluded" and not terminal.pending_handoff
            assert terminal.retirement_requested and terminal.outcome == elected
            assert len(provider.requests) == 1
            with pytest.raises(ExternalWaitConflict, match="differently"):
                await recovered.retire_execution(
                    registered, operation_key="changed", context=CONTEXT
                )
            assert await restored_waits.inspect(correlation, context=CONTEXT) == terminal
            await restored.delete_session("retire-execution")
            assert (
                await recovered.retire_execution(
                    registered, operation_key="retire", context=CONTEXT
                )
                == terminal
            )
            await restored_app.aclose()
            await app.aclose()
            await restored_waits.aclose()
            await waits.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_explicit_control_reconciles_automatic_retirement_that_lost_acknowledgement(
    backend, tmp_path, request, monkeypatch
):
    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            registered = registration(correlation)
            await waits.register(registered, context=CONTEXT)
            app = CayuApp(session_store=store, enable_logging=False)
            provider = ScriptedModelProvider(
                [[ModelStreamEvent.completed({"finish_reason": "stop"})]]
            )
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="root", model="model"))
            adapter = SessionExternalWaitAdapter(app, waits)
            await adapter.run_to_wait(
                RunRequest(agent_name="root", messages=[Message.text("user", "submit")]),
                registered,
                context=CONTEXT,
            )
            await waits.cancel(correlation, operation_key="cancel", context=CONTEXT)
            native_retire = SessionContinuationOwner.retire_released
            entered = [asyncio.Event(), asyncio.Event()]
            release = [asyncio.Event(), asyncio.Event()]

            async def pause_retirement(owner, candidate):
                index = int(candidate.retirement.control_id.startswith("external-retire-control:"))
                entered[index].set()
                await release[index].wait()
                result = await native_retire(owner, candidate)
                if index == 0:
                    raise RuntimeError("automatic retirement acknowledgement lost")
                return result

            with monkeypatch.context() as patch:
                patch.setattr(SessionContinuationOwner, "retire_released", pause_retirement)
                automatic = asyncio.create_task(adapter.service_wait(registered, context=CONTEXT))
                await asyncio.wait_for(entered[0].wait(), 20)
                explicit = asyncio.create_task(
                    adapter.retire_execution(
                        registered,
                        operation_key="explicit",
                        context=CONTEXT,
                    )
                )
                await asyncio.wait_for(entered[1].wait(), 20)
                release[0].set()
                with pytest.raises(RuntimeError, match="acknowledgement lost"):
                    await automatic
                release[1].set()
                with pytest.raises(ContinuationConflict):
                    await explicit
            restored = reopen()
            recovered_waits = ExternalEventWaits(store=restored, access_policy=Policy())
            recovered_app = CayuApp(session_store=restored, enable_logging=False)
            recovered = SessionExternalWaitAdapter(recovered_app, recovered_waits)
            assert (await recovered_waits.inspect(correlation, context=CONTEXT)).pending_handoff
            terminal = await recovered.retire_execution(
                registered, operation_key="explicit", context=CONTEXT
            )
            assert terminal.handoff == "excluded" and not terminal.pending_handoff
            assert terminal.outcome.kind == "cancelled" and len(provider.requests) == 1
            assert (
                await recovered.retire_execution(
                    registered, operation_key="explicit", context=CONTEXT
                )
                == terminal
            )
            await recovered_app.aclose()
            await app.aclose()
            await recovered_waits.aclose()
            await waits.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("winner", ["retirement", "admission", "prepared"])
def test_retirement_and_inflight_service_use_the_native_decision(
    backend, winner, tmp_path, request, monkeypatch
):
    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            registered = registration(correlation)
            await waits.register(registered, context=CONTEXT)
            provider = ScriptedModelProvider(
                [
                    [ModelStreamEvent.completed({"finish_reason": "stop"})],
                    [ModelStreamEvent.completed({"finish_reason": "stop"})],
                ]
            )
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="root", model="model"))
            adapter = SessionExternalWaitAdapter(app, waits)
            await adapter.run_to_wait(
                RunRequest(agent_name="root", messages=[Message.text("user", "submit")]),
                registered,
                context=CONTEXT,
            )
            await waits.deliver(
                ExternalEventDelivery(
                    correlation=correlation, delivery_id="result", payload_json="{}"
                ),
                context=CONTEXT,
            )
            native_admit = SessionContinuationOwner._admit_owned
            native_retire = SessionContinuationOwner.retire_released
            admission_entered, admit_release = asyncio.Event(), asyncio.Event()
            retirement_entered, retire_release = asyncio.Event(), asyncio.Event()

            async def pause_admission(owner, candidate, command, *, invocation):
                admission_entered.set()
                await admit_release.wait()
                return await native_admit(owner, candidate, command, invocation=invocation)

            async def pause_retirement(owner, candidate):
                retirement_entered.set()
                await retire_release.wait()
                return await native_retire(owner, candidate)

            async def interrupt_prepared_admission(native_store, consumption, command):
                await native_store.consume_continuation(consumption)
                raise RuntimeError("interrupted after native consumption preparation")

            other_store = reopen()
            other_app = CayuApp(session_store=other_store, enable_logging=False)
            other_app.register_provider(provider, default=True)
            other_app.register_agent(AgentSpec(name="root", model="model"))
            other_waits = ExternalEventWaits(store=other_store, access_policy=Policy())
            other = SessionExternalWaitAdapter(other_app, other_waits)
            with monkeypatch.context() as patch:
                patch.setattr(SessionContinuationOwner, "_admit_owned", pause_admission)
                patch.setattr(SessionContinuationOwner, "retire_released", pause_retirement)
                if winner == "prepared":
                    patch.setattr(
                        _session_continuation_owner,
                        "admit_continuation",
                        interrupt_prepared_admission,
                    )
                servicing = asyncio.create_task(adapter.service_wait(registered, context=CONTEXT))
                await asyncio.wait_for(admission_entered.wait(), 20)
                retiring = asyncio.create_task(
                    other.retire_execution(
                        registered,
                        operation_key="retire-race",
                        context=CONTEXT,
                    )
                )
                await asyncio.wait_for(retirement_entered.wait(), 20)
                if winner != "retirement":
                    admit_release.set()
                    if winner == "prepared":
                        with pytest.raises(ExternalWaitUnavailable) as pending:
                            await servicing
                        assert isinstance(pending.value.__cause__, ContinuationUnavailable)
                        assert len(provider.requests) == 1
                        assert (await waits.inspect(correlation, context=CONTEXT)).pending_handoff
                    else:
                        await servicing
                    retire_release.set()
                    with pytest.raises(ContinuationConflict):
                        await retiring
                else:
                    retire_release.set()
                    await retiring
                    admit_release.set()
                    with pytest.raises(ContinuationConflict):
                        await servicing
            settled = await other.retire_execution(
                registered, operation_key="retire-race", context=CONTEXT
            )
            assert settled.handoff == ("excluded" if winner == "retirement" else "settled")
            assert not settled.pending_handoff and settled.outcome.kind == "event"
            assert len(provider.requests) == (1 if winner == "retirement" else 2)
            assert (
                await other.retire_execution(
                    registered, operation_key="retire-race", context=CONTEXT
                )
                == settled
            )
            await other_app.aclose()
            await app.aclose()
            await other_waits.aclose()
            await waits.aclose()

    asyncio.run(scenario())
