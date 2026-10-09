"""Reconstruct the real interrupted wait entrance, not a fabricated invocation."""

import asyncio
import json
import os
import signal
import sys
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from tests.core.test_external_wait_request_policies import RequestStopPolicy
from tests.core.test_tool_completion import FinalTool, call
from tests.external_wait_support import CONTEXT, Policy, registration, reservation, stores

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.evals.testing import ScriptedModelProvider, scripted_structured_output
from cayu.external_waits import ExternalEventWaits, ExternalWaitContext
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.loop_policies import BeforeStopDecision, LoopPolicy
from cayu.session_external_waits import SessionExternalWaitAdapter
from cayu.sessions.external_waits import ExternalEventDelivery, ExternalWaitUnavailable


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize(
    "recovery_case",
    ["normal", "policy_interrupt", "policy_continue", "two_workers", "lost_ack", "cancel_race"],
)
def test_completed_turn_recovers_to_wait_without_another_provider_dispatch(
    backend, recovery_case, tmp_path, request, monkeypatch
):
    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            registered = registration(correlation)
            await waits.register(registered, context=CONTEXT)
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
                    [
                        ModelStreamEvent.text_delta("Result acknowledged."),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ],
                ]
            )

            class StopGate(LoopPolicy):
                interrupted = False
                calls = 0

                @property
                def execution_profile_identity(self):
                    return ExecutionProfileBehaviorIdentity(
                        name="tests:external-wait-stop-gate",
                        behavior_version="1",
                        implementation_version="1",
                    )

                async def before_stop(self, context):
                    self.calls += 1
                    if recovery_case == "policy_continue" and self.calls == 2:
                        return BeforeStopDecision.continue_with(
                            Message.text("user", "Verify the submission before waiting.")
                        )
                    return (
                        BeforeStopDecision.interrupt("Qualification stop gate")
                        if self.interrupted
                        else BeforeStopDecision.complete()
                    )

            policy = StopGate()

            def app_for(native):
                app = CayuApp(session_store=native, enable_logging=False, loop_policies=[policy])
                app.register_provider(provider, default=True)
                app.register_agent(AgentSpec(name="root", model="model"))
                return app

            app = app_for(store)
            session_id = "external-recovered-" + uuid4().hex
            entered = asyncio.Event()

            async def pause_before_park(self, invocation):
                entered.set()
                await asyncio.Event().wait()

            original_park = _ExternalExecutionToWait.park
            monkeypatch.setattr(_ExternalExecutionToWait, "park", pause_before_park)
            task = asyncio.create_task(
                SessionExternalWaitAdapter(app, waits).run_to_wait(
                    RunRequest(
                        agent_name="root",
                        session_id=session_id,
                        messages=[Message.text("user", "Submit job")],
                    ),
                    registered,
                    context=CONTEXT,
                )
            )
            try:
                await asyncio.wait_for(entered.wait(), 30)
            finally:
                task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled() and task.cancelling() == 1
            monkeypatch.setattr(_ExternalExecutionToWait, "park", original_park)
            assert len(provider.requests) == 1
            before = await store._read_external_wait(
                correlation.request.scope, correlation.request.correlation_key
            )
            assert before.continuation is not None
            original_ticket = before.continuation.intent
            interrupted_events = await store.load_events(session_id)
            assert any(
                event.type == EventType.INTERACTION_PAUSED
                and event.interaction_id == original_ticket.interaction_id
                and event.payload.get("pending_action_kind") == "external_wait_recovery"
                for event in interrupted_events
            )
            assert not any(
                event.type == EventType.INTERACTION_INTERRUPTED
                and event.interaction_id == original_ticket.interaction_id
                for event in interrupted_events
            )
            await app.aclose()
            await waits.aclose()

            restored_store = reopen()
            restored_waits = ExternalEventWaits(store=restored_store, access_policy=Policy())
            restored = app_for(restored_store)
            adapter = SessionExternalWaitAdapter(restored, restored_waits)
            prior_session = await restored_store.load(session_id)
            with pytest.raises(PermissionError):
                await adapter.recover_to_wait(
                    registered,
                    context=ExternalWaitContext(principal="unauthorized"),
                    inactive_for_seconds=0,
                )
            assert await restored_store.load(session_id) == prior_session
            assert policy.calls == 1
            if recovery_case == "policy_interrupt":
                policy.interrupted = True
                with pytest.raises(ExternalWaitUnavailable):
                    await adapter.recover_to_wait(
                        registered, context=CONTEXT, inactive_for_seconds=0
                    )
                assert len(provider.requests) == 1
                stopped = await restored_store.load_continuation_ticket(
                    prior_session.id,
                    registration_key=original_ticket.registration_key,
                    session_instance_id=original_ticket.session_instance_id,
                )
                assert stopped.ticket.state == "ARMING"
                policy.interrupted = False
            competing = competing_waits = competing_task = None
            release_inspection = asyncio.Event()
            if recovery_case == "two_workers":
                inspected = asyncio.Event()
                original_inspect = _ExternalExecutionToWait.inspect_recovery
                first = True

                async def paused_inspect(self, session):
                    nonlocal first
                    native = await original_inspect(self, session)
                    if first:
                        first = False
                        inspected.set()
                        await release_inspection.wait()
                    return native

                monkeypatch.setattr(_ExternalExecutionToWait, "inspect_recovery", paused_inspect)
                competing_store = reopen()
                competing_waits = ExternalEventWaits(store=competing_store, access_policy=Policy())
                competing = app_for(competing_store)
                competing_task = asyncio.create_task(
                    SessionExternalWaitAdapter(competing, competing_waits).recover_to_wait(
                        registered, context=CONTEXT, inactive_for_seconds=0
                    )
                )
                await asyncio.wait_for(inspected.wait(), 30)
            if recovery_case == "lost_ack":

                async def lose_park_ack(self, invocation):
                    await original_park(self, invocation)
                    raise ConnectionError("Lost committed park acknowledgement")

                monkeypatch.setattr(_ExternalExecutionToWait, "park", lose_park_ack)
            if recovery_case == "cancel_race":
                original_cleanup_check = _ExternalExecutionToWait.recovery_cleanup_requested
                first_check = True

                async def cancel_after_routing(boundary, invocation):
                    nonlocal first_check
                    cleanup = await original_cleanup_check(boundary, invocation)
                    if first_check:
                        first_check = False
                        assert not cleanup
                        await restored_waits.cancel(
                            correlation, operation_key="cancel-during-recovery", context=CONTEXT
                        )
                    return cleanup

                monkeypatch.setattr(
                    _ExternalExecutionToWait, "recovery_cleanup_requested", cancel_after_routing
                )
                with pytest.raises(ExternalWaitUnavailable):
                    await adapter.recover_to_wait(
                        registered, context=CONTEXT, inactive_for_seconds=0
                    )
                assert len(provider.requests) == 1 and policy.calls == 1
                assert (await restored_waits.inspect(correlation, context=CONTEXT)).pending_handoff
                monkeypatch.setattr(
                    _ExternalExecutionToWait, "recovery_cleanup_requested", original_cleanup_check
                )
                settled = await adapter.service_wait(registered, context=CONTEXT)
                assert settled.wait.handoff == "excluded" and not settled.wait.pending_handoff
                assert len(provider.requests) == 1 and policy.calls == 1
                await restored.aclose()
                await restored_waits.aclose()
                return
            receipt = await adapter.recover_to_wait(
                registered, context=CONTEXT, inactive_for_seconds=0
            )
            monkeypatch.setattr(_ExternalExecutionToWait, "park", original_park)
            preparation_requests = 1 + int(recovery_case == "policy_continue")
            assert len(provider.requests) == preparation_requests
            assert policy.calls == 2 + int(recovery_case in {"policy_interrupt", "policy_continue"})
            native = await restored_store.load_continuation_ticket(
                receipt.session_id,
                registration_key=original_ticket.registration_key,
                session_instance_id=original_ticket.session_instance_id,
            )
            assert native.preparation == before.continuation
            assert native.ticket.state == "WAITING"
            assert native.recovery_writer is not None
            if competing_task is not None:
                after_winner = await restored_store.load(session_id)
                release_inspection.set()
                assert await asyncio.wait_for(competing_task, 30) == receipt
                assert await restored_store.load(session_id) == after_winner
                assert len(provider.requests) == 1
                await competing.aclose()
                await competing_waits.aclose()
            assert (
                await adapter.recover_to_wait(registered, context=CONTEXT, inactive_for_seconds=0)
                == receipt
            )
            await restored_waits.deliver(
                ExternalEventDelivery(
                    correlation=correlation, delivery_id="job-done", payload_json='{"done":true}'
                ),
                context=CONTEXT,
            )
            await adapter.service_wait(registered, context=CONTEXT)
            assert len(provider.requests) == preparation_requests + 1
            await adapter.service_wait(registered, context=CONTEXT)
            assert len(provider.requests) == preparation_requests + 1
            assert (await restored_waits.inspect(correlation, context=CONTEXT)).handoff == "settled"
            await restored.aclose()
            await restored_waits.aclose()

    asyncio.run(scenario())


@pytest.mark.skipif(
    os.name != "posix", reason="Real SIGKILL qualification uses POSIX process death."
)
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_process_death_before_parking_reconstructs_exact_wait(backend, tmp_path, request):
    _process_death_scenario(backend, tmp_path, request)


def _process_death_scenario(
    backend, tmp_path, request, boundary="park", cleanup=None, recovery_cycles=0
):
    completed_boundary = boundary in {"parked_before_release", "before_park_cleanup"}

    async def scenario():
        clock = [datetime.now(UTC)]
        async with stores(backend, tmp_path, request, clock) as (store, reopen):
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            registered = registration(correlation)
            await waits.register(registered, context=CONTEXT)
            session_id = "external-killed-" + uuid4().hex
            setup = {
                "backend": backend,
                "database": str(tmp_path / "waits.sqlite")
                if backend == "sqlite"
                else request.getfixturevalue("postgres_dsn"),
                "session_id": session_id,
                "registration": registered.model_dump_json(),
                "boundary": boundary,
                "repeated_recovery": bool(recovery_cycles),
            }
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "tests.recovery.external_wait_worker",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                process.stdin.write((json.dumps(setup) + "\n").encode())
                await process.stdin.drain()
                ready = await asyncio.wait_for(process.stdout.readline(), 30)
                if not ready:
                    _, diagnostic = await asyncio.wait_for(process.communicate(), 10)
                    pytest.fail("Worker terminated before its boundary:\n" + diagnostic.decode())
                assert json.loads(ready) == {
                    "ready": True,
                    "requests": 0 if boundary == "before_ticket" else 1,
                }
                if completed_boundary:
                    # A parked ticket alone does not authorize taking a live
                    # invocation. Its independently renewed lease still owns it.
                    live_store = reopen()
                    live_waits = ExternalEventWaits(store=live_store, access_policy=Policy())
                    live_app = CayuApp(
                        session_store=live_store,
                        enable_logging=False,
                        loop_policies=[RequestStopPolicy()],
                    )
                    live_provider = ScriptedModelProvider([])
                    live_app.register_provider(live_provider, default=True)
                    live_app.register_agent(AgentSpec(name="root", model="model"))
                    live_adapter = SessionExternalWaitAdapter(live_app, live_waits)
                    live_epoch = (await live_store.load(session_id)).run_epoch
                    assert (
                        await live_store.inspect_session_execution(session_id)
                    ).state == "executing"
                    if boundary == "before_park_cleanup":
                        with pytest.raises(ExternalWaitUnavailable):
                            await live_adapter.recover_to_wait(
                                registered, context=CONTEXT, inactive_for_seconds=0
                            )
                    else:
                        await live_adapter.recover_to_wait(
                            registered, context=CONTEXT, inactive_for_seconds=0
                        )
                    assert (await live_store.load(session_id)).run_epoch == live_epoch
                    assert not live_provider.requests
                    if cleanup is not None:
                        from cayu.sessions._session_continuation import ContinuationConflict

                        if cleanup == "cancel":
                            await live_waits.cancel(
                                correlation, operation_key="cancel-before-death", context=CONTEXT
                            )
                            with pytest.raises(ContinuationConflict):
                                await live_adapter.service_wait(registered, context=CONTEXT)
                        else:
                            await live_waits.deliver(
                                ExternalEventDelivery(
                                    correlation=correlation,
                                    delivery_id="done",
                                    payload_json='{"done":true}',
                                ),
                                context=CONTEXT,
                            )
                            with pytest.raises(ContinuationConflict):
                                await live_adapter.retire_execution(
                                    registered, operation_key="retire-before-death", context=CONTEXT
                                )
                        assert (await live_store.load(session_id)).run_epoch == live_epoch
                        assert not live_provider.requests
                    await live_app.aclose()
                    await live_waits.aclose()
                process.send_signal(signal.SIGKILL)
                assert await asyncio.wait_for(process.wait(), 10) == -signal.SIGKILL
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
            # Observe real lease expiry; do not rewrite a claim or reset the database.
            await asyncio.sleep(5.1 if completed_boundary else 0.7)
            clock[0] = datetime.now(UTC)
            restored_store = reopen()
            restored_waits = ExternalEventWaits(store=restored_store, access_policy=Policy())
            stop_policy = RequestStopPolicy()
            app = CayuApp(
                session_store=restored_store,
                enable_logging=False,
                loop_policies=[stop_policy] if completed_boundary or recovery_cycles else [],
            )
            final_tool = boundary == "final_tool_publication"
            tool = FinalTool()
            provider = ScriptedModelProvider(
                [
                    call()
                    if final_tool
                    else list(scripted_structured_output({"answer": "OK"}, id="answer"))
                    if boundary == "tool_publication"
                    else [
                        ModelStreamEvent.text_delta("Received"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ]
                ]
            )
            app.register_provider(provider, default=True)
            app.register_agent(
                AgentSpec(name="root", model="model"), tools=[tool] if final_tool else []
            )
            adapter = SessionExternalWaitAdapter(app, restored_waits)
            if recovery_cycles:
                from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
                from cayu.sessions._invocation_lifecycle import (
                    _invocation_lifecycle_receipt_ledger_from_checkpoint,
                )

                async def receipts():
                    checkpoint = await runtime_checkpoint_session_store(
                        restored_store
                    ).load_checkpoint(session_id)
                    return {
                        item.command_identity: item
                        for item in _invocation_lifecycle_receipt_ledger_from_checkpoint(
                            checkpoint
                        ).receipts
                    }

                initial = await receipts()
                creation = next(item for key, item in initial.items() if key.startswith("create:"))
                assert not any(key.startswith("release:") for key in initial)
                seen = set(initial)
                pruned = set()
                stop_policy.interrupted = True
                for _ in range(recovery_cycles):
                    before = (await restored_store.load(session_id)).run_epoch
                    with pytest.raises(ExternalWaitUnavailable):
                        await adapter.recover_to_wait(
                            registered, context=CONTEXT, inactive_for_seconds=0
                        )
                    assert (await restored_store.load(session_id)).run_epoch > before
                    retained = await restored_store._read_external_wait(
                        correlation.request.scope, correlation.request.correlation_key
                    )
                    ticket = retained.continuation.intent
                    native = await restored_store.load_continuation_ticket(
                        session_id,
                        registration_key=ticket.registration_key,
                        session_instance_id=ticket.session_instance_id,
                    )
                    assert native.ticket.state == "ARMING"
                    current = await receipts()
                    assert current[creation.command_identity] == creation
                    pruned.update(seen - current.keys())
                    seen.update(current)
                    assert not provider.requests
                assert pruned, "The regression must exercise real receipt-ledger compaction."
                assert stop_policy.calls == recovery_cycles
                await restored_waits.deliver(
                    ExternalEventDelivery(
                        correlation=correlation, delivery_id="done", payload_json='{"done":true}'
                    ),
                    context=CONTEXT,
                )
                await app.aclose()
                await restored_waits.aclose()

                # Reconstruction must authenticate retirement from durable
                # creation and latest release, not a cached invocation object.
                restored_store = reopen()
                restored_waits = ExternalEventWaits(store=restored_store, access_policy=Policy())
                app = CayuApp(session_store=restored_store, enable_logging=False)
                adapter = SessionExternalWaitAdapter(app, restored_waits)
                terminal = await adapter.retire_execution(
                    registered, operation_key="retire-compacted", context=CONTEXT
                )
                assert terminal.handoff == "excluded" and not terminal.pending_handoff
                assert terminal.outcome.kind == "event"
                assert (
                    await adapter.retire_execution(
                        registered, operation_key="retire-compacted", context=CONTEXT
                    )
                    == terminal
                )
                from cayu.sessions._session_continuation_store import (
                    pending_admission_receipt_identities,
                )

                checkpoint = await runtime_checkpoint_session_store(restored_store).load_checkpoint(
                    session_id
                )
                assert creation.command_identity not in pending_admission_receipt_identities(
                    await restored_store.load(session_id), checkpoint
                )
                assert not provider.requests
                assert (
                    sum(
                        event.type == EventType.MODEL_STARTED
                        for event in await restored_store.load_events(session_id)
                    )
                    == 1
                )
                await restored_store.delete_session(session_id)
                assert (
                    await adapter.retire_execution(
                        registered, operation_key="retire-compacted", context=CONTEXT
                    )
                    == terminal
                )
                await app.aclose()
                await restored_waits.aclose()
                await waits.aclose()
                return
            if boundary == "before_ticket":
                from cayu.sessions.recovery import IncompleteSessionRecoveryRequest

                await restored_waits.cancel(correlation, operation_key="cancel", context=CONTEXT)
                with pytest.raises(ExternalWaitUnavailable, match="native writer release"):
                    await adapter.exclude_prepared_execution(registered, context=CONTEXT)
                await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id=session_id, inactive_for_seconds=0)
                )
                terminal = await adapter.exclude_prepared_execution(registered, context=CONTEXT)
                assert terminal.execution_excluded and not terminal.pending_handoff
                assert terminal.outcome.kind == "cancelled"
                assert not provider.requests
                await restored_store.delete_session(session_id)
                assert (
                    await adapter.exclude_prepared_execution(registered, context=CONTEXT)
                    == terminal
                )
                await app.aclose()
                await restored_waits.aclose()
                await waits.aclose()
                return
            if cleanup is not None:
                from cayu.external_wait_host import ExternalWaitHost

                host = ExternalWaitHost(adapter, context=CONTEXT)
                page = await host.service_once(scope=correlation.request.scope, source="renderer")
                assert page.settled == (correlation.request.correlation_key,)
                settled = await adapter.service_wait(registered, context=CONTEXT)
                assert settled.wait.handoff == "excluded"
                assert not settled.wait.pending_handoff
                assert settled.wait.outcome.kind == (
                    "cancelled" if cleanup == "cancel" else "event"
                )
                assert await adapter.service_wait(registered, context=CONTEXT) == settled
                assert not provider.requests and stop_policy.calls == 0
                original_events = await restored_store.load_events(session_id)
                assert sum(event.type == EventType.MODEL_STARTED for event in original_events) == 1
                await restored_store.delete_session(session_id)
                assert await adapter.service_wait(registered, context=CONTEXT) == settled
                await app.aclose()
                await restored_waits.aclose()
                await waits.aclose()
                return
            receipt = await adapter.recover_to_wait(
                registered, context=CONTEXT, inactive_for_seconds=0
            )
            assert receipt.session_id == session_id
            assert not provider.requests
            assert tool.calls == 0
            if boundary == "parked_before_release":
                assert stop_policy.calls == 0
                original_events = await restored_store.load_events(session_id)
                assert sum(event.type == EventType.MODEL_STARTED for event in original_events) == 1
                assert (
                    sum(event.type == EventType.MODEL_COMPLETED for event in original_events) == 1
                )
                epoch = (await restored_store.load(session_id)).run_epoch
                assert (
                    await adapter.recover_to_wait(
                        registered, context=CONTEXT, inactive_for_seconds=0
                    )
                    == receipt
                )
                assert (await restored_store.load(session_id)).run_epoch == epoch
            clock[0] = datetime.now(UTC)
            await restored_waits.deliver(
                ExternalEventDelivery(
                    correlation=correlation, delivery_id="done", payload_json='{"done":true}'
                ),
                context=CONTEXT,
            )
            await adapter.service_wait(registered, context=CONTEXT)
            assert len(provider.requests) == 1
            await adapter.service_wait(registered, context=CONTEXT)
            assert len(provider.requests) == 1
            assert tool.calls == int(final_tool)
            if boundary == "parked_before_release":
                assert stop_policy.calls == 1
            await app.aclose()
            await restored_waits.aclose()
            await waits.aclose()

    asyncio.run(scenario())


@pytest.mark.skipif(os.name != "posix", reason="Requires real process death.")
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_process_death_retains_creation_through_recovery_compaction(
    backend, tmp_path, request, monkeypatch
):
    from cayu.sessions import _invocation_lifecycle

    # Exercise the native compactor without requiring dozens of recoveries.
    # The original CREATE plus the current REBIND/RELEASE still fit; older
    # recovered epochs must be pruned rather than destroying pending authority.
    monkeypatch.setattr(_invocation_lifecycle, "INVOCATION_LIFECYCLE_RECEIPT_LEDGER_MAX_ITEMS", 8)
    _process_death_scenario(backend, tmp_path, request, recovery_cycles=6)


@pytest.mark.skipif(os.name != "posix", reason="Requires real process death.")
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("boundary", ["tool_publication", "final_tool_publication"])
def test_process_death_before_tool_publication_recovers_whole_turn(
    backend, boundary, tmp_path, request
):
    _process_death_scenario(backend, tmp_path, request, boundary=boundary)


@pytest.mark.skipif(os.name != "posix", reason="Requires real process death.")
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_process_death_after_parking_before_writer_release(backend, tmp_path, request):
    _process_death_scenario(backend, tmp_path, request, boundary="parked_before_release")


@pytest.mark.skipif(
    os.name != "posix", reason="Real SIGKILL qualification uses POSIX process death."
)
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
@pytest.mark.parametrize("cleanup", ["cancel", "retire"])
@pytest.mark.parametrize("boundary", ["parked_before_release", "before_park_cleanup"])
def test_process_death_can_release_terminal_wait(backend, cleanup, boundary, tmp_path, request):
    _process_death_scenario(backend, tmp_path, request, boundary=boundary, cleanup=cleanup)


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("before_park", [False, True, "before_result", "before_ticket"])
def test_cancelled_park_recovers_failed_writer_release(
    backend, before_park, tmp_path, request, monkeypatch
):
    from cayu.external_wait_host import ExternalWaitHost
    from cayu.runtime._checkpoint_store import _RuntimeCheckpointSessionStore
    from cayu.sessions._invocation_lifecycle import ReleaseInvocationCommand

    async def scenario():
        async with stores(backend, tmp_path, request, [datetime.now(UTC)]) as (store, reopen):
            waits = ExternalEventWaits(store=store, access_policy=Policy())
            correlation = await waits.reserve_correlation(reservation(), context=CONTEXT)
            registered = registration(correlation)
            await waits.register(registered, context=CONTEXT)
            provider = ScriptedModelProvider(
                [[ModelStreamEvent.completed({"finish_reason": "stop"})]]
            )

            def application(native):
                app = CayuApp(session_store=native, enable_logging=False)
                app.register_provider(provider, default=True)
                app.register_agent(AgentSpec(name="root", model="model"))
                return app

            app = application(store)
            entered = asyncio.Event()
            original_park = _ExternalExecutionToWait.park
            original_apply = _RuntimeCheckpointSessionStore.apply_invocation_lifecycle_command
            failed_releases = []

            async def parked(boundary, invocation):
                if not before_park:
                    await original_park(boundary, invocation)
                entered.set()
                await asyncio.Event().wait()

            async def fail_release(native, command):
                if isinstance(command, ReleaseInvocationCommand):
                    failed_releases.append(command)
                    raise ConnectionError("Native writer release temporarily unavailable")
                return await original_apply(native, command)

            async def blocked_provider(_request):
                entered.set()
                await asyncio.Event().wait()
                yield ModelStreamEvent.completed({"finish_reason": "stop"})

            with monkeypatch.context() as patch:
                patch.setattr(_ExternalExecutionToWait, "park", parked)
                if before_park == "before_result":
                    patch.setattr(provider, "stream", blocked_provider)
                elif before_park == "before_ticket":
                    patch.setattr(_ExternalExecutionToWait, "prepare", parked)
                patch.setattr(
                    _RuntimeCheckpointSessionStore,
                    "apply_invocation_lifecycle_command",
                    fail_release,
                )
                running = asyncio.create_task(
                    SessionExternalWaitAdapter(app, waits).run_to_wait(
                        RunRequest(
                            agent_name="root",
                            session_id="failed-release-" + uuid4().hex,
                            messages=[Message.text("user", "wait")],
                        ),
                        registered,
                        context=CONTEXT,
                    )
                )
                await asyncio.wait_for(entered.wait(), 20)
                await waits.cancel(correlation, operation_key="cancel", context=CONTEXT)
                running.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await running
                assert running.cancelled() and running.cancelling() == 1
            assert failed_releases
            assert (await waits.inspect(correlation, context=CONTEXT)).pending_handoff
            await app.aclose()
            await waits.aclose()
            restored_store = reopen()
            restored_waits = ExternalEventWaits(store=restored_store, access_policy=Policy())
            restored = application(restored_store)
            adapter = SessionExternalWaitAdapter(restored, restored_waits)
            if before_park == "before_ticket":
                from cayu.sessions.recovery import IncompleteSessionRecoveryRequest

                with pytest.raises(ExternalWaitUnavailable, match="native writer release"):
                    await adapter.exclude_prepared_execution(registered, context=CONTEXT)
                retained = await restored_store._read_external_wait(
                    correlation.request.scope, correlation.request.correlation_key
                )
                await restored.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id=retained.execution.intent.session_id, inactive_for_seconds=0
                    )
                )
            result = await ExternalWaitHost(adapter, context=CONTEXT).service_once(
                scope=correlation.request.scope, source="renderer"
            )
            assert result.settled == (correlation.request.correlation_key,)
            assert len(provider.requests) == (
                0 if before_park in {"before_result", "before_ticket"} else 1
            )
            if before_park == "before_ticket":
                settled = await adapter.exclude_prepared_execution(registered, context=CONTEXT)
                assert settled.execution_excluded and not settled.pending_handoff
                await restored_store.delete_session(retained.execution.intent.session_id)
                assert (
                    await adapter.exclude_prepared_execution(registered, context=CONTEXT) == settled
                )
            else:
                settled = await adapter.service_wait(registered, context=CONTEXT)
                assert settled.wait.handoff == "excluded" and not settled.wait.pending_handoff
            await restored.aclose()
            await restored_waits.aclose()

    asyncio.run(scenario())


@pytest.mark.skipif(os.name != "posix", reason="Requires real process death.")
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_process_death_before_ticket_uses_native_release(backend, tmp_path, request):
    _process_death_scenario(backend, tmp_path, request, boundary="before_ticket")
