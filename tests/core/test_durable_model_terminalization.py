"""Terminalization never reconstructs executable application registrations."""

import asyncio
import contextlib

import pytest
from tests.core.test_model_completion_recovery import (
    _RecordingProvider,
    _stage_in_flight_model_boundary,
)

from cayu import CayuApp, EventQuery, EventType
from cayu.runtime.sessions import (
    InMemorySessionStore,
    ModelCompletionManualRecoveryRequest,
    SessionStatus,
)
from cayu.storage.sqlite import SQLiteSessionStore


def _store_for_backend(backend, database, request):
    if backend == "memory":
        return InMemorySessionStore()
    if backend == "sqlite":
        return SQLiteSessionStore(database)
    from cayu import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    return PostgresSessionStore(
        request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.MIGRATE
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("status", [SessionStatus.FAILED, SessionStatus.INTERRUPTED])
def test_terminalization_without_registrations_publishes_one_outcome(
    tmp_path, backend, status, request
):
    async def exercise():
        if backend == "postgres":
            from cayu import PostgresSessionStore
            from cayu.storage.migrations import SchemaMode

            store = PostgresSessionStore(
                request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.MIGRATE
            )
        else:
            store = (
                InMemorySessionStore()
                if backend == "memory"
                else SQLiteSessionStore(tmp_path / "terminal.db")
            )
        running, _, stage = await _stage_in_flight_model_boundary(
            store,
            session_id="model-terminal-" + status.value,
            provider_name=_RecordingProvider.name,
            reservation_ids=(),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        recovery_request = ModelCompletionManualRecoveryRequest(
            session_id=running.id,
            stage_id=stage.stage_id,
            expected_run_epoch=running.run_epoch,
            terminal_status=status,
            terminalization_only=True,
            expected_session_instance_id=running.instance_id,
            inactive_for_seconds=0,
        )
        result = await app.recover_model_completion_stage(recovery_request)
        assert result.session.status is status
        assert await store.load_active_model_completion_stage(running.id) is None
        replay = await app.recover_model_completion_stage(recovery_request)
        assert replay.replayed
        assert replay.settlement == result.settlement
        events = await store.query_events(
            EventQuery(
                session_id=running.id,
                event_types={EventType.SESSION_FAILED, EventType.SESSION_INTERRUPTED},
            )
        )
        assert len(events) == 1
        if backend != "memory":
            await store.close()

    asyncio.run(exercise())


def test_recovery_plan_executes_terminalization_without_registrations(tmp_path):
    from cayu import (
        RecoveryDecision,
        RecoveryExecutionRequest,
        RecoveryItemExecutionStatus,
        RecoveryPlanAction,
        RecoveryPlanRequest,
        RecoveryPlanSelection,
        RecoveryRegistrationStatus,
    )

    async def exercise():
        store = SQLiteSessionStore(tmp_path / "plan.db")
        running, _, _ = await _stage_in_flight_model_boundary(
            store,
            session_id="model-plan",
            provider_name=_RecordingProvider.name,
            reservation_ids=(),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        plan = await app.plan_recovery(
            RecoveryPlanRequest(
                selection=RecoveryPlanSelection(session_ids=(running.id,), inactive_for_seconds=0)
            )
        )
        item = plan.items[0]
        assert item.registration.status is RecoveryRegistrationStatus.TERMINALIZATION_ONLY
        assert RecoveryPlanAction.MODEL_MARK_FAILED in item.allowed_actions
        request = RecoveryExecutionRequest(
            plan=plan,
            execution_id="terminalize-model",
            decisions=(
                RecoveryDecision(item_id=item.item_id, action=RecoveryPlanAction.MODEL_MARK_FAILED),
            ),
        )
        receipt = await app.execute_recovery(request)
        assert receipt.items[0].status is RecoveryItemExecutionStatus.EXECUTED, receipt
        assert receipt.items[0].final_session_status is SessionStatus.FAILED
        replay = await app.execute_recovery(request)
        assert replay.items[0].replayed
        await store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "mismatch",
    ["incarnation", "epoch", "stage", "dependent_checkpoint", "undispatched", "accounting"],
)
def test_terminalization_rejects_invalid_authority_without_mutation(mismatch):
    from cayu.runtime._recovery_coordinator import ModelCompletionManualRecoveryRequired
    from cayu.runtime.sessions import SessionRunFenced

    async def exercise():
        store = InMemorySessionStore()
        running, _, stage = await _stage_in_flight_model_boundary(
            store,
            session_id="model-reject",
            provider_name=_RecordingProvider.name,
            reservation_ids=("missing-accounting",) if mismatch == "accounting" else (),
            dispatched=mismatch != "undispatched",
        )
        if mismatch == "dependent_checkpoint":
            await store.transform_checkpoint(
                running.id,
                lambda session, checkpoint: {
                    **checkpoint,
                    "pending_workspace_work": {"unknown": True},
                },
            )
        request = ModelCompletionManualRecoveryRequest(
            session_id=running.id,
            stage_id="wrong" if mismatch == "stage" else stage.stage_id,
            expected_run_epoch=running.run_epoch + (mismatch == "epoch"),
            expected_session_instance_id="wrong"
            if mismatch == "incarnation"
            else running.instance_id,
            terminal_status=SessionStatus.FAILED,
            terminalization_only=True,
            inactive_for_seconds=0,
        )
        before = (
            await store.load(running.id),
            await store.load_checkpoint(running.id),
            await store.load_events(running.id),
        )
        with pytest.raises((SessionRunFenced, ModelCompletionManualRecoveryRequired)):
            await CayuApp(session_store=store, enable_logging=False).recover_model_completion_stage(
                request
            )
        assert before == (
            await store.load(running.id),
            await store.load_checkpoint(running.id),
            await store.load_events(running.id),
        )

    asyncio.run(exercise())


@pytest.mark.parametrize("ledger_backend", ["memory", "sqlite"])
def test_terminalization_conservatively_settles_budget_and_retries_partial_failure(
    tmp_path, ledger_backend
):
    from decimal import Decimal

    from cayu import (
        BudgetLimit,
        BudgetReservation,
        InMemoryBudgetLedger,
        ModelAttemptIdentity,
        ModelPrice,
        PriceBook,
        SQLiteBudgetLedger,
    )
    from cayu.runtime.budgets import (
        BudgetReservationRecoveryContext,
        budget_reservation_authority_sha256,
    )

    class PartialLedger(InMemoryBudgetLedger):
        def __init__(self):
            super().__init__(reservation_ttl_seconds=None)
            self.calls = 0

        async def reconcile(self, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("partial settlement")
            return await super().reconcile(**kwargs)

    async def exercise():
        store = InMemorySessionStore()
        ledger = (
            PartialLedger()
            if ledger_backend == "memory"
            else SQLiteBudgetLedger(tmp_path / "budget.db", reservation_ttl_seconds=None)
        )
        records = []
        for scope, key in [("session", None), ("agent", "assistant")]:
            reserved = await ledger.reserve(
                limit=BudgetLimit(
                    scope=scope,
                    key=key,
                    max_estimated_cost=Decimal("1"),
                    pricing=PriceBook(
                        prices=(
                            ModelPrice.fixed(
                                provider_name=_RecordingProvider.name,
                                model="fake-model",
                                input_per_million=Decimal("1"),
                                output_per_million=Decimal("0"),
                            ),
                        )
                    ),
                    reservation=BudgetReservation(max_input_tokens=1000000, max_output_tokens=0),
                ),
                session_id="budget-model",
                agent_name="assistant",
                provider_name=_RecordingProvider.name,
                model="fake-model",
                model_attempt_identity=ModelAttemptIdentity(
                    model_step_id="mstep_" + "4" * 32, model_attempt_id="matt_" + "5" * 32
                ),
                settlement_event_payload={"interaction_id": "interaction-budget-model"},
            )
            records.append(reserved.record)
        ids = tuple(r.reservation_id for r in records)
        running, _, stage = await _stage_in_flight_model_boundary(
            store,
            session_id="budget-model",
            provider_name=_RecordingProvider.name,
            reservation_ids=ids,
            budget_reservations=tuple(
                BudgetReservationRecoveryContext(
                    reservation_id=r.reservation_id,
                    budget_limit_id=r.budget_limit_id,
                    reservation_authority_sha256=budget_reservation_authority_sha256(r),
                )
                for r in records
            ),
        )
        await ledger.mark_dispatched(reservation_ids=ids, dispatch_id=stage.stage_id)
        app = CayuApp(session_store=store, budget_ledger=ledger, enable_logging=False)
        request = ModelCompletionManualRecoveryRequest(
            session_id=running.id,
            stage_id=stage.stage_id,
            expected_run_epoch=running.run_epoch,
            expected_session_instance_id=running.instance_id,
            terminal_status=SessionStatus.FAILED,
            terminalization_only=True,
            inactive_for_seconds=0,
        )
        if ledger_backend == "memory":
            with pytest.raises(RuntimeError, match="partial settlement"):
                await app.recover_model_completion_stage(request)
            assert await store.load_active_model_completion_stage(running.id) is not None
            assert (await store.load(running.id)).status is SessionStatus.RUNNING
        result = await app.recover_model_completion_stage(request)
        assert result.session.status is SessionStatus.FAILED
        for r in records:
            settled = await ledger.load_reservation(r.reservation_id)
            assert settled.status == "reconciled"
        again = await app.recover_model_completion_stage(request)
        assert again.replayed and again.settlement == result.settlement
        if ledger_backend == "sqlite":
            await ledger.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_fresh_process_opaque_dispatch_and_competing_terminalizers(tmp_path, backend, request):
    import json
    import subprocess
    import sys
    from pathlib import Path

    helper = Path(__file__).with_name("model_terminalization_process.py")
    database = (
        request.getfixturevalue("postgres_dsn") if backend == "postgres" else tmp_path / "opaque.db"
    )
    producer = subprocess.run(
        [sys.executable, str(helper), "produce", str(database)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    request_json = producer.stdout.strip()
    assert json.loads(request_json)["terminalization_only"]
    args = [sys.executable, str(helper), "recover", str(database), request_json]
    workers = [
        subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for _ in range(2)
    ]
    try:
        outcomes = []
        for worker in workers:
            out, err = worker.communicate(timeout=30)
            assert worker.returncode == 0, err
            outcomes.append(json.loads(out))
        assert any(result["status"] == "failed" for result in outcomes)
        replay = subprocess.run(args, capture_output=True, text=True, check=True, timeout=30)
        assert json.loads(replay.stdout) == {"status": "failed", "replayed": True}
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.kill()
                worker.wait(timeout=10)

    async def verify():
        store = _store_for_backend(backend, database, request)
        events = await store.query_events(
            EventQuery(
                session_id="opaque-model",
                event_types=(EventType.SESSION_FAILED, EventType.SESSION_INTERRUPTED),
            )
        )
        assert len(events) == 1
        assert await store.load_active_model_completion_stage("opaque-model") is None
        await store.close()

    asyncio.run(verify())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_terminalization_reconciles_commit_before_acknowledgement_loss(
    tmp_path, backend, request, monkeypatch
):

    async def exercise():
        store = _store_for_backend(backend, tmp_path / "ack.db", request)
        original = type(store).settle_session_invocation
        lost = False

        async def lose_ack(self, command):
            nonlocal lost
            result = await original(self, command)
            if command.terminalization_only and not lost:
                lost = True
                raise OSError("committed before acknowledgement")
            return result

        monkeypatch.setattr(type(store), "settle_session_invocation", lose_ack)
        running, _, stage = await _stage_in_flight_model_boundary(
            store, session_id="model-ack", provider_name=_RecordingProvider.name, reservation_ids=()
        )
        app = CayuApp(session_store=store, enable_logging=False)
        recovery_request = ModelCompletionManualRecoveryRequest(
            session_id=running.id,
            stage_id=stage.stage_id,
            expected_run_epoch=running.run_epoch,
            expected_session_instance_id=running.instance_id,
            terminal_status=SessionStatus.FAILED,
            terminalization_only=True,
            inactive_for_seconds=0,
        )
        result = await app.recover_model_completion_stage(recovery_request)
        assert lost and result.replayed
        assert result.session.status is SessionStatus.FAILED
        assert (
            await app.recover_model_completion_stage(recovery_request)
        ).settlement == result.settlement
        events = await store.query_events(
            EventQuery(session_id=running.id, event_types=(EventType.SESSION_FAILED,))
        )
        assert len(events) == 1
        if backend != "memory":
            await store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("authority", ["plan", "claim"])
def test_plan_owner_change_at_publication_cannot_terminalize(
    tmp_path, monkeypatch, backend, request, authority
):
    from cayu import (
        RecoveryDecision,
        RecoveryExecutionRequest,
        RecoveryPlanAction,
        RecoveryPlanRequest,
        RecoveryPlanSelection,
    )

    async def exercise():
        store = _store_for_backend(backend, tmp_path / "stale-plan.db", request)
        running, _, _ = await _stage_in_flight_model_boundary(
            store,
            session_id=f"stale-{authority}-model",
            provider_name=_RecordingProvider.name,
            reservation_ids=(),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        plan = await app.plan_recovery(
            RecoveryPlanRequest(
                selection=RecoveryPlanSelection(session_ids=(running.id,), inactive_for_seconds=0)
            )
        )
        original = type(store).settle_session_invocation
        changed = False

        async def lose_plan(self, command):
            nonlocal changed
            if command.terminalization_only:

                def replace_owner(_session, checkpoint):
                    marker = checkpoint["recovery_plan_execution"]
                    marker["ownership"]["generation"] += 1
                    return checkpoint

                if authority == "plan":
                    await self.transform_checkpoint(running.id, replace_owner)
                else:
                    command = command.model_copy(update={"recovery_claim_id": "stale-claim"})
                changed = True
            return await original(self, command)

        monkeypatch.setattr(type(store), "settle_session_invocation", lose_plan)
        execution = RecoveryExecutionRequest(
            plan=plan,
            execution_id="stale-model-execution",
            decisions=(
                RecoveryDecision(
                    item_id=plan.items[0].item_id, action=RecoveryPlanAction.MODEL_MARK_FAILED
                ),
            ),
        )
        with contextlib.suppress(RuntimeError):
            await app.execute_recovery(execution)
        assert changed
        assert (await store.load(running.id)).status is SessionStatus.RUNNING
        assert await store.load_active_model_completion_stage(running.id) is not None
        assert (
            await store.query_events(
                EventQuery(
                    session_id=running.id,
                    event_types=(EventType.SESSION_FAILED, EventType.SESSION_INTERRUPTED),
                )
            )
            == []
        )
        if backend != "memory":
            await store.close()

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "fault", ["interaction", "fingerprint", "reservation", "task", "compaction", "custom_store"]
)
def test_terminalization_rejects_tampered_or_unsupported_stage_before_claim(fault, monkeypatch):
    async def exercise():
        store = InMemorySessionStore()
        running, _, _ = await _stage_in_flight_model_boundary(
            store,
            session_id="tampered-model",
            provider_name=_RecordingProvider.name,
            reservation_ids=(),
        )
        active = await store.load_active_model_completion_stage(running.id)
        altered = active.stage.model_dump(mode="python")
        context = altered["intent"]["recovery_context"]
        if fault == "interaction":
            altered["intent"]["interaction_id"] = "another-interaction"
        elif fault == "fingerprint":
            context["execution_profile_fingerprint"] = "0" * 64
        elif fault == "reservation":
            altered["reservation_ids"] = ("unbound-reservation",)
        elif fault == "task":
            context["task_id"] = "dependent-task"
        elif fault == "compaction":
            altered["purpose"] = "context-compaction"
        else:
            monkeypatch.setattr(store, "durable_model_terminalization_version", None)
        bad_stage = active.stage.model_validate(altered)

        async def load_altered(_session_id):
            return active.model_copy(update={"stage": bad_stage})

        monkeypatch.setattr(store, "load_active_model_completion_stage", load_altered)
        before = await store.load_checkpoint(running.id)
        before_session = await store.load(running.id)
        with pytest.raises((ValueError, RuntimeError)):
            await CayuApp(session_store=store, enable_logging=False).recover_model_completion_stage(
                ModelCompletionManualRecoveryRequest(
                    session_id=running.id,
                    stage_id=active.stage.stage_id,
                    expected_run_epoch=running.run_epoch,
                    expected_session_instance_id=running.instance_id,
                    terminal_status=SessionStatus.FAILED,
                    terminalization_only=True,
                    inactive_for_seconds=0,
                )
            )
        assert await store.load(running.id) == before_session
        assert await store.load_checkpoint(running.id) == before

    asyncio.run(exercise())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("status", [SessionStatus.FAILED, SessionStatus.INTERRUPTED])
def test_replanned_terminalization_resumes_elected_disposition(
    tmp_path, monkeypatch, backend, status, request
):
    import contextvars

    from cayu import (
        RecoveryDecision,
        RecoveryExecutionRequest,
        RecoveryItemExecutionStatus,
        RecoveryPlanAction,
        RecoveryPlanRequest,
        RecoveryPlanSelection,
    )
    from cayu.runtime._invocation_terminal_decision import (
        invocation_terminal_decision_from_checkpoint,
        settled_invocation_terminal_decision_from_checkpoint,
    )

    async def exercise():
        database = tmp_path / "replan.db"
        store = _store_for_backend(backend, database, request)
        running, _, stage = await _stage_in_flight_model_boundary(
            store,
            session_id="replanned-model-" + status.value,
            provider_name=_RecordingProvider.name,
            reservation_ids=(),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        selection = RecoveryPlanRequest(
            selection=RecoveryPlanSelection(session_ids=(running.id,), inactive_for_seconds=0)
        )
        action = (
            RecoveryPlanAction.MODEL_MARK_FAILED
            if status is SessionStatus.FAILED
            else RecoveryPlanAction.MODEL_MARK_INTERRUPTED
        )

        async def execution_request(app, execution_id, selected_action):
            plan = await app.plan_recovery(selection)
            return RecoveryExecutionRequest(
                plan=plan,
                execution_id=execution_id,
                decisions=(
                    RecoveryDecision(item_id=plan.items[0].item_id, action=selected_action),
                ),
            )

        original = type(store).settle_session_invocation
        injected = False

        async def fail_once(self, command):
            nonlocal injected
            if command.terminalization_only and not injected:
                injected = True
                raise OSError("transient publication outage")
            return await original(self, command)

        monkeypatch.setattr(type(store), "settle_session_invocation", fail_once)
        first_request = await execution_request(app, "first-attempt", action)
        failed = await app.execute_recovery(first_request)
        assert injected and failed.items[0].error_code == "OSError"
        elected = invocation_terminal_decision_from_checkpoint(
            await store.load_checkpoint(running.id)
        )
        assert elected is not None
        assert (await store.load(running.id)).status is SessionStatus.RUNNING
        assert await store.load_active_model_completion_stage(running.id) is not None
        replay = await app.execute_recovery(first_request)
        assert replay.items[0].replayed
        assert replay.items[0].status is RecoveryItemExecutionStatus.FAILED

        # New store/app owner and empty task context cannot inherit the old claim.
        if backend != "memory":
            await store.close()
            store = _store_for_backend(backend, database, request)
        replacement = CayuApp(session_store=store, enable_logging=False)

        async def replan():
            current = await store.load(running.id)
            before = await store.load_checkpoint(running.id)
            with pytest.raises(RuntimeError, match="stale run epoch"):
                await replacement.recover_model_completion_stage(
                    ModelCompletionManualRecoveryRequest(
                        session_id=running.id,
                        stage_id=stage.stage_id,
                        expected_run_epoch=current.run_epoch + 1,
                        expected_session_instance_id=running.instance_id,
                        terminal_status=status,
                        terminalization_only=True,
                        inactive_for_seconds=0,
                    )
                )
            assert await store.load_checkpoint(running.id) == before
            assert await store.load(running.id) == current
            conflicting_action = (
                RecoveryPlanAction.MODEL_MARK_INTERRUPTED
                if status is SessionStatus.FAILED
                else RecoveryPlanAction.MODEL_MARK_FAILED
            )
            conflict = await replacement.execute_recovery(
                await execution_request(replacement, "conflicting-attempt", conflicting_action)
            )
            assert conflict.items[0].error_code == "SessionRunFenced"
            assert (
                invocation_terminal_decision_from_checkpoint(
                    await store.load_checkpoint(running.id)
                )
                == elected
            )
            retry_request = await execution_request(replacement, "replacement-attempt", action)
            assert retry_request.plan.items[0].run_epoch > first_request.plan.items[0].run_epoch
            result = await replacement.execute_recovery(retry_request)
            assert result.items[0].status is RecoveryItemExecutionStatus.EXECUTED, result
            assert result.items[0].final_session_status is status
            assert (await replacement.execute_recovery(retry_request)).items[0].replayed

        await asyncio.create_task(replan(), context=contextvars.Context())
        assert await store.load_active_model_completion_stage(running.id) is None
        assert await store.load_model_completion_stage_settlement(running.id, stage.stage_id)
        settled = settled_invocation_terminal_decision_from_checkpoint(
            await store.load_checkpoint(running.id)
        )
        assert settled == elected
        events = await store.query_events(
            EventQuery(
                session_id=running.id,
                event_types=(EventType.SESSION_FAILED, EventType.SESSION_INTERRUPTED),
            )
        )
        assert len(events) == 1 and events[0].event.id == elected.terminal_event_id
        if backend != "memory":
            await store.close()

    asyncio.run(exercise())
