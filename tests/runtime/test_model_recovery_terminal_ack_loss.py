"""A lost model settlement acknowledgement still owes durable terminal evidence."""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta

import pytest
from tests.runtime.runtime_provider_support import VersionedFakeProvider
from tests.runtime.test_orphaned_session_reconciliation import (
    _application,
    _budget,
    _model_orphan,
    _orphan,
    _slot,
)
from tests.runtime.test_orphaned_session_reconciliation import database_url as database_url

from cayu import EventType, SessionStatus
from cayu.storage import open_application_stores


def _crash_after_model_settlement():
    import cayu.runtime._recovery_ownership as recovery_ownership

    # A real short claim expires after this recoverer dies; no stored state is edited.
    recovery_ownership._INCOMPLETE_RECOVERY_CLAIM_LEASE = timedelta(seconds=1)

    async def scenario():
        stores = open_application_stores(
            os.environ["CAYU_ORPHAN_TEST_DATABASE_URL"],
            sqlite_path=os.environ["CAYU_ORPHAN_TEST_STORE"],
            event_watchers=False,
        )
        app = _application(stores.session_store, environment=_slot(), budget_policy=_budget())
        fanout = app._event_writer.fan_out_persisted

        async def die_after_native_commit(events, *args, **kwargs):
            if any(event.type is EventType.INTERACTION_INTERRUPTED for event in events):
                assert (await stores.session_store.load("scan")).status is SessionStatus.INTERRUPTED
                os._exit(137)
            return await fanout(events, *args, **kwargs)

        app._event_writer.fan_out_persisted = die_after_native_commit
        await app.reconcile_orphaned_sessions(inactive_for_seconds=0)
        raise AssertionError("The recovery did not commit its model settlement")

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "restart_configuration", ["unchanged", "missing_environment", "runtime_changed"]
)
@pytest.mark.parametrize("failure", ["exception", "process_exit"])
def test_model_settlement_commit_before_fanout_is_repaired_on_restart(
    tmp_path, database_url, monkeypatch, failure, restart_configuration
):
    _model_orphan(tmp_path, database_url)
    if failure == "process_exit":
        _orphan(
            tmp_path / "cayu.db",
            "scan",
            database_url=database_url,
            entry="_crash_after_model_settlement",
            module=__name__,
        )

    async def scenario():
        stores = open_application_stores(
            database_url, sqlite_path=tmp_path / "cayu.db", event_watchers=False
        )
        provider = VersionedFakeProvider([])
        app = _application(
            stores.session_store,
            environment=_slot(),
            budget_policy=_budget(),
            provider=lambda _: provider,
        )
        first = await stores.session_store.load_events("scan")
        stage = await stores.session_store.load_active_model_completion_stage("scan")
        writer, committed_faults = app._event_writer, []
        fanout = writer.fan_out_persisted

        async def lost_after_model_commit(events, *args, **kwargs):
            if any(event.type is EventType.INTERACTION_INTERRUPTED for event in events):
                assert (await stores.session_store.load("scan")).status is SessionStatus.INTERRUPTED
                assert (
                    await stores.session_store.load_model_completion_stage_settlement(
                        "scan", stage.stage.stage_id
                    )
                    is not None
                )
                committed_faults.append(True)
                raise OSError("model settlement committed; fanout acknowledgement lost")
            return await fanout(events, *args, **kwargs)

        try:
            if failure == "exception":
                monkeypatch.setattr(writer, "fan_out_persisted", lost_after_model_commit)
                await app.reconcile_orphaned_sessions()
                assert len(committed_faults) == 1
            first = await stores.session_store.load_events("scan")
            assert sum(event.type is EventType.INTERACTION_INTERRUPTED for event in first) == 1
            if failure == "process_exit":
                assert sum(event.type is EventType.SESSION_INTERRUPTED for event in first) == 0
            budget_ids = {event.id for event in first if event.type is EventType.BUDGET_RECONCILED}
            assert budget_ids
        finally:
            monkeypatch.setattr(writer, "fan_out_persisted", fanout)
            await app.aclose()
            await stores.close()

        stores = open_application_stores(
            database_url, sqlite_path=tmp_path / "cayu.db", event_watchers=False
        )
        if restart_configuration == "runtime_changed":
            import cayu.runtime._execution_profile_continuation as continuation

            monkeypatch.setattr(continuation, "_runtime_version", lambda: "999.0.0")
        restarted = _application(
            stores.session_store,
            environment=None if restart_configuration == "missing_environment" else _slot(),
            budget_policy=_budget(),
            provider=lambda _: provider,
        )
        try:
            from cayu.sessions.recovery import _incomplete_recovery_claim_from_checkpoint

            async with asyncio.timeout(10):
                while True:
                    claim = _incomplete_recovery_claim_from_checkpoint(
                        await stores.session_store.load_checkpoint("scan")
                    )
                    if claim is None or claim[1] <= datetime.now(UTC):
                        break
                    await asyncio.sleep(0.02)
            await restarted.reconcile_orphaned_sessions()
            events = await stores.session_store.load_events("scan")
            terminal = [event for event in events if event.type is EventType.SESSION_INTERRUPTED]
            assert len(terminal) == 1
            assert terminal[0].payload["reason"] == "orphaned_session_reconciled"
            assert sum(event.type is EventType.INTERACTION_INTERRUPTED for event in events) == 1
            assert {
                event.id for event in events if event.type is EventType.BUDGET_RECONCILED
            } == budget_ids
            assert provider.requests == []
            checkpoint = await stores.session_store.load_checkpoint("scan")
            assert "incomplete_session_recovery_claim" not in checkpoint
            assert await restarted.reconcile_orphaned_sessions() == ()
            assert await stores.session_store.load_events("scan") == events
        finally:
            await restarted.aclose()
            await stores.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "conflicting_reason",
    [
        False,
        True,
        "interruption_type",
        "metadata",
        "interruption_request_id",
        "payload_authority",
        "nested_authority",
    ],
)
def test_claimed_model_lost_ack_rejects_conflicting_reason_and_claim(
    tmp_path, database_url, monkeypatch, conflicting_reason
):
    """Two admitted calls may race; the loser must match the complete owed outcome."""
    from cayu.sessions._invocation_lifecycle import SettleInvocationCommand
    from cayu.sessions._model_stage import SessionModelCompletionStageConflict
    from cayu.sessions._paired_writes import owed_decision_from_install_mutation
    from cayu.sessions.authority import SessionRunFenced

    _model_orphan(tmp_path, database_url)

    async def scenario():
        stores = open_application_stores(
            database_url, sqlite_path=tmp_path / "cayu.db", event_watchers=False
        )
        app = _application(stores.session_store, environment=_slot(), budget_policy=_budget())
        original_settle = app._manual_model_completion_recovery.recover_model_completion_stage
        native_apply = app._runtime_session_store.apply_invocation_lifecycle_command
        winner_ready, loser_ready, committed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        commands, outcomes = [], []
        winner_reason = "orphaned_session_reconciled"
        loser_reason = (
            "same-stage-different-reason" if conflicting_reason is True else winner_reason
        )
        winner_payload = loser_payload = None
        if isinstance(conflicting_reason, str):
            winner_payload = {
                "interruption_type": "operator_requested",
                "reason": winner_reason,
                "metadata": {"decision": "original"},
                "interruption_request_id": "operator-interrupt-original",
            }
            loser_payload = dict(winner_payload)
            if conflicting_reason not in {"payload_authority", "nested_authority"}:
                loser_payload[conflicting_reason] = {
                    "interruption_type": "runtime_interrupted",
                    "metadata": {"decision": "different"},
                    "interruption_request_id": "operator-interrupt-different",
                }[conflicting_reason]
        prepare = app._event_writer.prepare

        def changed_prepared_provenance(event):
            from cayu.events import Event, event_with_runtime_nested_payload_authority

            prepared = prepare(event)
            if (
                prepared.type is EventType.SESSION_INTERRUPTED
                and asyncio.current_task().get_name() == "model-settlement-loser"
            ):
                if conflicting_reason == "payload_authority":
                    return Event.model_validate(prepared.model_dump(mode="python"))
                if conflicting_reason == "nested_authority":
                    return event_with_runtime_nested_payload_authority(
                        prepared, ("metadata", "decision")
                    )
            return prepared

        monkeypatch.setattr(app._event_writer, "prepare", changed_prepared_provenance)

        async def race_native(command):
            if (
                not isinstance(command, SettleInvocationCommand)
                or command.transition.model_completion_stage_settlement is None
            ):
                return await native_apply(command)
            transition = command.transition
            decision = owed_decision_from_install_mutation(
                transition.checkpoint_mutation,
                event=transition.event,
                to_status=transition.to_status,
            )
            if asyncio.current_task().get_name() == "model-settlement-winner":
                assert decision.terminal_payload["reason"] == winner_reason
                commands.append(command)
                winner_ready.set()
                await loser_ready.wait()
                await native_apply(command)
                committed.set()
                raise OSError("exact model transition committed before response was lost")
            assert decision.terminal_payload["reason"] == loser_reason
            loser_ready.set()
            await committed.wait()
            return await native_apply(command)

        async def race_owner(
            request, *, invocation_context=None, interruption_reason=None, interruption_payload=None
        ):
            assert interruption_reason == winner_reason and invocation_context is not None
            calls = [
                asyncio.create_task(
                    original_settle(
                        request,
                        invocation_context=invocation_context,
                        interruption_reason=winner_reason,
                        interruption_payload=winner_payload,
                    ),
                    name="model-settlement-winner",
                ),
            ]
            await asyncio.wait_for(winner_ready.wait(), 10)
            calls.append(
                asyncio.create_task(
                    original_settle(
                        request,
                        invocation_context=invocation_context,
                        interruption_reason=loser_reason,
                        interruption_payload=loser_payload,
                    ),
                    name="model-settlement-loser",
                )
            )
            _, pending = await asyncio.wait(calls, timeout=15)
            for call in pending:
                call.cancel()
            results = await asyncio.gather(*calls, return_exceptions=True)
            outcomes.extend(results)
            assert not isinstance(results[0], BaseException), results
            assert results[0].replayed
            if conflicting_reason:
                assert isinstance(results[1], SessionModelCompletionStageConflict), results
            else:
                assert not isinstance(results[1], BaseException) and results[1].replayed, results
            return results[0]

        monkeypatch.setattr(
            app._runtime_session_store, "apply_invocation_lifecycle_command", race_native
        )
        monkeypatch.setattr(
            app._manual_model_completion_recovery, "recover_model_completion_stage", race_owner
        )
        try:
            results = await app.reconcile_orphaned_sessions()
            assert outcomes and not isinstance(outcomes[0], BaseException), outcomes
            if conflicting_reason:
                assert isinstance(outcomes[1], SessionModelCompletionStageConflict), outcomes
            else:
                assert not isinstance(outcomes[1], BaseException) and outcomes[1].replayed, outcomes
            assert all(result.status is SessionStatus.INTERRUPTED for result in results), results
            assert winner_ready.is_set() and len(commands) == 1
            command = commands[0]
            replay = await native_apply(command)
            assert replay.replayed
            before = (
                await stores.session_store.load("scan"),
                await stores.session_store.load_checkpoint("scan"),
                await stores.session_store.load_events("scan"),
            )
            with pytest.raises(SessionRunFenced):
                await native_apply(
                    command.model_copy(update={"recovery_claim_id": "different-claim"})
                )
            assert before == (
                await stores.session_store.load("scan"),
                await stores.session_store.load_checkpoint("scan"),
                await stores.session_store.load_events("scan"),
            )
        finally:
            await app.aclose()
            await stores.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("successor", [False, True])
@pytest.mark.parametrize("replay_path", ["lost_ack", "prior_present"])
def test_model_recovery_readback_rejects_a_superseding_native_claim(
    tmp_path, database_url, monkeypatch, replay_path, successor
):
    """A tentative successor claim fences an earlier acknowledgement readback."""
    import cayu.runtime._recovery_ownership as recovery_ownership
    from cayu.sessions._invocation_lifecycle import SettleInvocationCommand
    from cayu.sessions.authority import SessionRunFenced
    from cayu.sessions.recovery import _incomplete_recovery_claim_from_checkpoint

    _model_orphan(tmp_path, database_url)
    monkeypatch.setattr(
        recovery_ownership, "_INCOMPLETE_RECOVERY_CLAIM_LEASE", timedelta(seconds=1)
    )

    async def scenario():
        stores = open_application_stores(
            database_url, sqlite_path=tmp_path / "cayu.db", event_watchers=False
        )
        other = open_application_stores(
            database_url, sqlite_path=tmp_path / "cayu.db", event_watchers=False
        )
        app = _application(stores.session_store, environment=_slot(), budget_policy=_budget())
        native_apply = app._runtime_session_store.apply_invocation_lifecycle_command
        settle = app._manual_model_completion_recovery.recover_model_completion_stage
        observed = []

        async def reserve_successor(command):
            result = await native_apply(command)
            if not isinstance(command, SettleInvocationCommand) or (
                command.transition.model_completion_stage_settlement is None
            ):
                return result
            async with asyncio.timeout(10):
                while True:
                    current = _incomplete_recovery_claim_from_checkpoint(
                        await other.session_store.load_checkpoint("scan")
                    )
                    if current[1] <= datetime.now(UTC):
                        break
                    await asyncio.sleep(0.02)

            def claim_at_store_time(session, checkpoint, store_now):
                current = _incomplete_recovery_claim_from_checkpoint(checkpoint)
                assert current[0] == command.recovery_claim_id and current[1] <= store_now
                copied = dict(checkpoint)
                marker = dict(copied["incomplete_session_recovery_claim"])
                marker["claim_id"] = "native-successor-claim"
                marker["claim_expires_at"] = (store_now + timedelta(seconds=30)).isoformat()
                copied["incomplete_session_recovery_claim"] = marker
                return copied

            if not successor:
                if replay_path == "lost_ack":
                    raise OSError(
                        "model settlement acknowledgement lost after ordinary lease expiry"
                    )
                return result
            before = await other.session_store.load("scan")
            reserved = await other.session_store.reserve_stalled_run_recovery(
                "scan",
                statuses={SessionStatus.INTERRUPTED},
                inactive_for_seconds=None,
                checkpoint_transform=claim_at_store_time,
            )
            assert reserved.run_epoch == before.run_epoch
            if replay_path == "lost_ack":
                raise OSError("model settlement acknowledgement lost after successor reserved")
            return result

        async def inspect_owner(
            request, *, invocation_context=None, interruption_reason=None, interruption_payload=None
        ):
            try:
                result = await settle(
                    request,
                    invocation_context=invocation_context,
                    interruption_reason=interruption_reason,
                    interruption_payload=interruption_payload,
                )
                if replay_path == "prior_present":
                    result = await settle(
                        request,
                        invocation_context=invocation_context,
                        interruption_reason=interruption_reason,
                        interruption_payload=interruption_payload,
                    )
            except BaseException as error:
                observed.append(error)
                raise
            observed.append(result)
            return result

        monkeypatch.setattr(
            app._runtime_session_store, "apply_invocation_lifecycle_command", reserve_successor
        )
        monkeypatch.setattr(
            app._manual_model_completion_recovery, "recover_model_completion_stage", inspect_owner
        )
        try:
            await app.reconcile_orphaned_sessions()
            assert len(observed) == 1, observed
            if successor:
                assert isinstance(observed[0], SessionRunFenced), observed
                checkpoint = await stores.session_store.load_checkpoint("scan")
                assert (
                    _incomplete_recovery_claim_from_checkpoint(checkpoint)[0]
                    == "native-successor-claim"
                )
            else:
                assert not isinstance(observed[0], BaseException) and observed[0].replayed, observed
            assert (await stores.session_store.load("scan")).status is SessionStatus.INTERRUPTED
        finally:
            await app.aclose()
            await stores.close()
            await other.close()

    asyncio.run(scenario())
