"""Recover native preparation through the public host after a failed handoff."""

import asyncio
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from tests.external_wait_support import CONTEXT, Policy, registration, reservation, stores

from cayu import CayuApp
from cayu.agents import AgentSpec
from cayu.evals.testing import ScriptedModelProvider
from cayu.external_wait_host import ExternalWaitHost
from cayu.external_waits import ExternalEventWaits
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait
from cayu.session_external_waits import SessionExternalWaitAdapter
from cayu.sessions.base import ResumeRequest, RunRequest, SessionRunFenced
from cayu.sessions.external_waits import (
    ExternalEventDelivery,
    ExternalWaitConflict,
    ExternalWaitUnavailable,
    external_wait_digest,
)


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("resume_existing", [False, True])
@pytest.mark.parametrize("event_first", [False, True])
def test_cancelled_created_writer_without_ticket_is_excluded_after_release(
    backend, resume_existing, event_first, tmp_path, request, monkeypatch
):
    _cancelled_created_writer_scenario(
        backend, resume_existing, event_first, tmp_path, request, monkeypatch
    )


def _cancelled_created_writer_scenario(
    backend, resume_existing, event_first, tmp_path, request, monkeypatch, rebind_cycles=0
):
    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            registered = registration(correlation)
            await waits.register(registered, context=CONTEXT)
            app = CayuApp(session_store=store, enable_logging=False)
            provider = ScriptedModelProvider(
                [[ModelStreamEvent.completed({"finish_reason": "stop"})]] if resume_existing else []
            )
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="root", model="model"))
            adapter = SessionExternalWaitAdapter(app, waits)
            entered = asyncio.Event()

            async def pause_before_ticket(boundary, invocation):
                entered.set()
                await asyncio.Event().wait()

            monkeypatch.setattr(_ExternalExecutionToWait, "prepare", pause_before_ticket)
            session_id = "unbound-created-" + uuid4().hex
            expected_calls = int(resume_existing)
            if resume_existing:
                async for _ in app.run(
                    RunRequest(
                        agent_name="root",
                        session_id=session_id,
                        messages=[Message.text("user", "ready")],
                    )
                ):
                    pass
                operation = adapter.resume_to_wait(
                    ResumeRequest(session_id=session_id, messages=[Message.text("user", "submit")]),
                    registered,
                    context=CONTEXT,
                )
            else:
                operation = adapter.run_to_wait(
                    RunRequest(
                        agent_name="root",
                        session_id=session_id,
                        messages=[Message.text("user", "submit")],
                    ),
                    registered,
                    context=CONTEXT,
                )
            running = asyncio.create_task(operation)
            await asyncio.wait_for(entered.wait(), 10)
            if event_first:
                await waits.deliver(
                    ExternalEventDelivery(
                        correlation=correlation, delivery_id="result", payload_json="{}"
                    ),
                    context=CONTEXT,
                )
            else:
                await waits.cancel(correlation, operation_key="cancel-start", context=CONTEXT)

            # An elected outcome alone does not release its execution writer.
            with pytest.raises(ExternalWaitUnavailable, match="awaiting native writer release"):
                await adapter.exclude_prepared_execution(registered, context=CONTEXT)
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
            assert running.cancelled() and running.cancelling() == 1
            assert len(provider.requests) == expected_calls
            if rebind_cycles:
                from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
                from cayu.runtime._invocation_lifecycle import prepare_rebind_invocation_command
                from cayu.sessions._execution_profile_checkpoint import (
                    active_invocation_execution_profile_from_checkpoint,
                    checkpoint_with_active_invocation_execution_profile,
                )
                from cayu.sessions._invocation_lifecycle import (
                    INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
                    _invocation_lifecycle_receipt_ledger_from_checkpoint,
                    _InvocationLifecycleCommandReceipt,
                    _InvocationLifecycleReceiptLedger,
                )
                from cayu.sessions.base import IncompleteSessionRecoveryRequest

                runtime_store = runtime_checkpoint_session_store(store)
                original = _invocation_lifecycle_receipt_ledger_from_checkpoint(
                    await runtime_store.load_checkpoint(session_id)
                )
                origin = next(
                    item.external_execution_origin
                    for item in original.receipts
                    if item.external_execution_origin is not None
                )
                for _ in range(rebind_cycles):
                    session = await store.load(session_id)
                    checkpoint = await runtime_store.load_checkpoint(session_id)
                    profile = active_invocation_execution_profile_from_checkpoint(checkpoint)

                    def transfer(current_session, current_checkpoint, profile=profile):
                        return checkpoint_with_active_invocation_execution_profile(
                            current_checkpoint,
                            session_id=session_id,
                            interaction_id=profile.interaction_id,
                            run_epoch=current_session.run_epoch + 1,
                            profile=profile.profile,
                            expected=profile,
                        )

                    command = prepare_rebind_invocation_command(
                        session,
                        checkpoint,
                        expected_statuses={session.status},
                        checkpoint_transform=transfer,
                    )
                    await runtime_store.apply_invocation_lifecycle_command(command)
                compacted = _invocation_lifecycle_receipt_ledger_from_checkpoint(
                    await runtime_store.load_checkpoint(session_id)
                )
                assert not set(item.command_identity for item in original.receipts) & {
                    item.command_identity for item in compacted.receipts
                }
                assert all(item.external_execution_origin == origin for item in compacted.receipts)
                with pytest.raises(ExternalWaitUnavailable, match="native writer release"):
                    await adapter.exclude_prepared_execution(registered, context=CONTEXT)
                await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id=session_id, inactive_for_seconds=0)
                )
                # Independently valid content hashes are not proof of this
                # expected preparation. Exercise corrupt durable readback at
                # the public cleanup entrance, not only the comparison helper.
                saved = store._checkpoints[session_id]
                final_ledger = _invocation_lifecycle_receipt_ledger_from_checkpoint(saved)
                released = max(
                    final_ledger.receipts, key=lambda item: item.result_session.run_epoch
                )
                before = await waits.inspect(correlation, context=CONTEXT)
                for field, value in (
                    ("registration_sha256", "f" * 64),
                    ("execution_sha256", "f" * 64),
                    ("admission_epoch", origin.admission_epoch + 1),
                    ("admission_kind", "create" if resume_existing else "admit"),
                ):
                    material = released.model_dump(mode="json")
                    material["external_execution_origin"][field] = value
                    material["record_sha256"] = ""
                    changed = _InvocationLifecycleCommandReceipt.model_validate(material)
                    altered = _InvocationLifecycleReceiptLedger(
                        receipts=tuple(
                            changed if item == released else item for item in final_ledger.receipts
                        )
                    )
                    store._checkpoints[session_id] = {
                        **saved,
                        INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY: altered.model_dump(
                            mode="json"
                        ),
                    }
                    try:
                        with pytest.raises(ExternalWaitConflict, match="release conflicts"):
                            await adapter.exclude_prepared_execution(registered, context=CONTEXT)
                        assert await waits.inspect(correlation, context=CONTEXT) == before
                    finally:
                        store._checkpoints[session_id] = saved
            restored = reopen()
            restored_waits = ExternalEventWaits(store=restored, access_policy=Policy())
            restored_app = CayuApp(session_store=restored, enable_logging=False)
            restored_app.register_provider(provider, default=True)
            restored_app.register_agent(AgentSpec(name="root", model="model"))
            receiver = SessionExternalWaitAdapter(restored_app, restored_waits)
            before_resume = await restored.load(session_id)
            before_events = await restored.load_events(session_id)
            with pytest.raises(SessionRunFenced, match="External execution preparation"):
                async for _ in restored_app.resume(
                    ResumeRequest(session_id=session_id, messages=[Message.text("user", "bypass")])
                ):
                    pass
            assert await restored.load(session_id) == before_resume
            assert await restored.load_events(session_id) == before_events
            assert len(provider.requests) == expected_calls
            if event_first:
                # A winning event is never automatically abandoned by the host.
                # The authenticated application explicitly excludes this execution.
                mutate = restored_waits._mutate

                async def lose_exclusion_ack(command):
                    result = await mutate(command)
                    if command.kind == "exclude_execution":
                        raise RuntimeError("execution exclusion acknowledgement lost")
                    return result

                with monkeypatch.context() as patch:
                    patch.setattr(restored_waits, "_mutate", lose_exclusion_ack)
                    with pytest.raises(RuntimeError, match="exclusion acknowledgement lost"):
                        await receiver.exclude_prepared_execution(registered, context=CONTEXT)
                # Do not rely on an observer's success or process-local result cache.
                replay_store = reopen()
                restored_waits = ExternalEventWaits(store=replay_store, access_policy=Policy())
                replay_app = CayuApp(session_store=replay_store, enable_logging=False)
                receiver = SessionExternalWaitAdapter(replay_app, restored_waits)
            else:
                host = ExternalWaitHost(receiver, context=CONTEXT)
                page = await host.service_once(scope=correlation.request.scope, source="renderer")
                assert page.settled == (correlation.request.correlation_key,)
            terminal = await restored_waits.inspect(correlation, context=CONTEXT)
            assert terminal.execution_excluded and not terminal.pending_handoff
            assert terminal.outcome.kind == ("event" if event_first else "cancelled")
            assert (
                await receiver.exclude_prepared_execution(registered, context=CONTEXT) == terminal
            )
            await restored.delete_session(session_id)
            assert (
                await receiver.exclude_prepared_execution(registered, context=CONTEXT) == terminal
            )
            assert len(provider.requests) == expected_calls
            await restored_waits.aclose()
            await waits.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("resume_existing", [False, True])
def test_memory_pre_ticket_cleanup_survives_rebind_compaction(
    resume_existing, tmp_path, request, monkeypatch
):
    from cayu.sessions import _invocation_lifecycle

    monkeypatch.setattr(_invocation_lifecycle, "INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_ITEMS", 4)
    _cancelled_created_writer_scenario(
        "memory", resume_existing, True, tmp_path, request, monkeypatch, rebind_cycles=5
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("lost_repair_ack", [False, True])
def test_host_recovers_ticket_committed_before_external_binding(
    backend, lost_repair_ack, tmp_path, request, monkeypatch
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
            mutate = waits._mutate

            async def fail_binding(command):
                if command.kind == "bind":
                    raise RuntimeError("binding publication unavailable")
                return await mutate(command)

            monkeypatch.setattr(waits, "_mutate", fail_binding)
            session_id = "binding-recovery-" + uuid4().hex
            with pytest.raises(ExternalWaitUnavailable):
                await adapter.run_to_wait(
                    RunRequest(
                        agent_name="root",
                        session_id=session_id,
                        messages=[Message.text("user", "submit")],
                    ),
                    registered,
                    context=CONTEXT,
                )
            before = await store._read_external_wait(
                correlation.request.scope, correlation.request.correlation_key
            )
            assert before.execution is not None and before.continuation is None
            assert not provider.requests
            native = await store.load_continuation_ticket(
                session_id,
                registration_key="external-wait:" + external_wait_digest(registered),
                session_instance_id=before.execution.session_instance_id,
            )
            assert native is not None
            raw = waits._command(
                "reconcile_binding",
                correlation,
                registration=registered,
                continuation=native.preparation,
            )
            with pytest.raises(PermissionError):
                await store._mutate_external_wait(raw)
            assert (
                await store._read_external_wait(
                    correlation.request.scope, correlation.request.correlation_key
                )
                == before
            )
            restored = reopen()
            restored_waits = ExternalEventWaits(store=restored, access_policy=Policy())
            restored_app = CayuApp(session_store=restored, enable_logging=False)
            restored_adapter = SessionExternalWaitAdapter(restored_app, restored_waits)
            await restored_waits.cancel(
                correlation, operation_key="cancel-failed-start", context=CONTEXT
            )
            host = ExternalWaitHost(restored_adapter, context=CONTEXT)
            if lost_repair_ack:
                receiving_mutate = restored_waits._mutate

                async def lose_repair_ack(command):
                    result = await receiving_mutate(command)
                    if command.kind == "reconcile_binding":
                        raise RuntimeError("repair acknowledgement lost")
                    return result

                with monkeypatch.context() as patch:
                    patch.setattr(restored_waits, "_mutate", lose_repair_ack)
                    with pytest.raises(RuntimeError, match="repair acknowledgement lost"):
                        await host.service_once(scope=correlation.request.scope, source="renderer")
                repaired = await restored._read_external_wait(
                    correlation.request.scope, correlation.request.correlation_key
                )
                assert repaired.continuation == native.preparation
                assert repaired.pending_handoff
                assert not provider.requests
            page = await host.service_once(scope=correlation.request.scope, source="renderer")
            assert page.settled == (correlation.request.correlation_key,)
            terminal = await restored_waits.inspect(correlation, context=CONTEXT)
            assert terminal.handoff == "excluded" and not terminal.pending_handoff
            assert not provider.requests
            await restored.delete_session(session_id)
            replay = await host.service_once(scope=correlation.request.scope, source="renderer")
            assert replay.pending == replay.settled == ()
            assert await restored_waits.inspect(correlation, context=CONTEXT) == terminal
            await restored_waits.aclose()
            await waits.aclose()

    asyncio.run(scenario())
