from __future__ import annotations

import asyncio
import sqlite3
import threading
from datetime import UTC, datetime, timedelta

import pytest
from tests.core.task_invocation_fixtures import task_backed_session_invocation
from tests.core.test_verified_work_contracts import (
    _accepted_decision,
    _artifact_evidence,
    _claim_completion_verification,
    _contract,
    _result_reference,
    _task_result,
    _verifier_profile_fingerprint,
)
from tests.core.test_work_attempt_admission import _prepare_request
from tests.core.verified_worker_fixtures import (
    verified_work_postgres_dsn as verified_work_postgres_dsn,
)

from cayu.runtime.invocation import TaskExecutionSource
from cayu.runtime.invocation_release import InvocationReleaseEvidence
from cayu.runtime.tasks import (
    InMemoryTaskStore,
    TaskAggregateFilter,
    TaskClaimLost,
    TaskCreate,
    TaskStatus,
    task_create_with_runtime_invocation,
)
from cayu.runtime.work_attempt_admission import (
    AdmittedCompletionProposalRequest,
    WorkAttemptAdmissionActivate,
    WorkAttemptAdmissionConflict,
    WorkAttemptAdmissionState,
    WorkAttemptExecutionClaimLost,
    WorkAttemptExecutionClaimRequest,
    WorkAttemptExecutionEntryDisposition,
    WorkAttemptExecutionEntryRequest,
    WorkAttemptExecutionStopRequest,
    WorkAttemptRecoveryActivate,
    require_work_attempt_claim_result,
    require_work_attempt_execution_entry_result,
    require_work_attempt_execution_stop_result,
)
from cayu.runtime.work_attempt_lifecycle import (
    WorkAttemptLifecycleSettlement,
    WorkAttemptPreparationHold,
    work_attempt_admission_authority_sha256,
)
from cayu.runtime.work_attempt_semantics import WorkAttemptRunSemantics
from cayu.runtime.work_contracts import (
    CompletionDecisionApplicationRequest,
    CompletionProposalCreate,
    CompletionVerificationClaimRequest,
    TaskCompletionDecisionRequired,
    WorkCompletionConflict,
)
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresTaskStore
from cayu.storage.sqlite import SQLiteTaskStore


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def lifecycle_store(request, tmp_path):
    if request.param == "memory":
        return InMemoryTaskStore()
    if request.param == "sqlite":
        return SQLiteTaskStore(tmp_path / "lifecycle.sqlite")
    return PostgresTaskStore(
        request.getfixturevalue("verified_work_postgres_dsn"), schema_mode=SchemaMode.CREATE
    )


def _run_lifecycle_scenario(store, scenario):
    async def owned():
        try:
            await scenario()
        finally:
            if isinstance(store, (SQLiteTaskStore, PostgresTaskStore)):
                await store.close()

    asyncio.run(owned())


async def _lifecycle_clock_now(store):
    if isinstance(store, PostgresTaskStore):
        await store._ensure_ready()
        async with store._pool.connection() as connection, connection.cursor() as cursor:
            return await store._database_now(cursor)
    return datetime.now(UTC)


async def _advance_lifecycle_clock(store, clock, target):
    if isinstance(store, PostgresTaskStore):
        # Never substitute the application clock for PostgreSQL lease authority.
        async with asyncio.timeout(20):
            while await _lifecycle_clock_now(store) < target:
                await asyncio.sleep(0.05)
    else:
        clock[0] = target


async def _reopen_lifecycle_store(store, tmp_path, *, after_expiry=None):
    if isinstance(store, PostgresTaskStore):
        if after_expiry is not None:
            await _advance_lifecycle_clock(store, None, after_expiry)
        return PostgresTaskStore(store._pool.conninfo, schema_mode=SchemaMode.VALIDATE)
    if isinstance(store, SQLiteTaskStore):
        return SQLiteTaskStore(
            tmp_path / "lifecycle.sqlite",
            schema_mode=SchemaMode.VALIDATE,
            ownership_clock=(lambda: after_expiry) if after_expiry is not None else None,
        )
    raise ValueError("In-memory stores cannot be reopened.")


async def _active_settlement_fixture(store, suffix="", *, deadline=None, lease_seconds=300):
    """Store conformance authority; not a proof of runtime cleanup coverage."""
    contract = _contract()
    await store.publish_work_contract(contract)
    task = await store.create_task(
        TaskCreate(
            task_id=f"lifecycle-task{suffix}",
            type=f"work{suffix}",
            work_contract=contract.reference(),
        )
    )
    prepare = _prepare_request(
        task_id=task.id,
        session_id=f"lifecycle-session{suffix}",
        session_invocation=await task_backed_session_invocation(
            store, task.id, f"lifecycle-session{suffix}"
        ),
        admission_id=f"admission-{suffix or '1'}",
        attempt_id=f"attempt-{suffix or '1'}",
        claim_id=f"claim-{suffix or '1'}",
    ).model_copy(
        update={
            "run_semantics": WorkAttemptRunSemantics(max_steps=7, deadline_expires_at=deadline),
            "lease_seconds": lease_seconds,
        }
    )
    prepared = await store.prepare_work_attempt_admission(prepare)
    active = await store.activate_work_attempt_admission(
        WorkAttemptAdmissionActivate(
            admission_id=prepared.admission_id,
            claim_id=prepared.claim.claim_id,
            prepare_request_sha256=prepared.prepare_request_sha256,
            session_evidence_sha256="1" * 64,
        )
    )
    request = WorkAttemptLifecycleSettlement(
        settlement_id=f"lifecycle-stop{suffix}",
        task_id=task.id,
        admission_id=active.admission_id,
        expected_admission_sha256=work_attempt_admission_authority_sha256(active),
        release_evidence=InvocationReleaseEvidence(
            session_id=active.session_id,
            session_instance_id=active.session_invocation.session_instance_id,
            interaction_id=active.interaction_id,
            command_identity="release:lifecycle-session",
            command_sha256="2" * 64,
            record_sha256="3" * 64,
            profile_fingerprint=active.source_execution_profile_fingerprint,
            run_epoch=1,
            released_run_epoch=2,
        ),
        kind="runtime_stop",
        stop_reason="work_contract_elapsed_limit",
    )
    return active, request


@pytest.mark.parametrize("decision_first", [False, True])
def test_proposal_deadline_settlement_fences_late_verifier_work(
    lifecycle_store, decision_first, monkeypatch, tmp_path
):
    async def scenario():
        store = lifecycle_store
        now = [await _lifecycle_clock_now(store)]
        if not isinstance(store, PostgresTaskStore):
            monkeypatch.setattr(store, "_ownership_clock", lambda: now[0])
        expires_at = now[0] + timedelta(seconds=5)
        active, original = await _active_settlement_fixture(store, deadline=expires_at)
        await store.enter_work_attempt_execution(
            WorkAttemptExecutionEntryRequest(
                admission_id=active.admission_id,
                prepare_request_sha256=active.prepare_request_sha256,
                claim_id=active.claim.claim_id,
                worker_id=active.claim.worker_id,
                execution_owner_id=active.claim.execution_owner_id,
                generation=active.claim.generation,
                run_epoch=1,
            )
        )
        proposal = await store.submit_admitted_completion_proposal(
            AdmittedCompletionProposalRequest(
                admission_id=active.admission_id,
                claim_id=active.claim.claim_id,
                execution_owner_id=active.claim.execution_owner_id,
                generation=active.claim.generation,
                proposal=CompletionProposalCreate(
                    proposal_id="deadline-proposal",
                    attempt_id=active.attempt_id,
                    result=_result_reference(),
                    evidence_references=(_artifact_evidence(),),
                ),
            )
        )
        released = await store.load_work_attempt_admission(active.admission_id)
        request = original.model_copy(
            update={
                "kind": "proposal_deadline_stop",
                "proposal_id": proposal.proposal_id,
                "proposal_request_sha256": proposal.request_sha256,
                "expected_admission_sha256": work_attempt_admission_authority_sha256(released),
            }
        )
        claim_request = CompletionVerificationClaimRequest(
            claim_id="deadline-verifier-claim",
            proposal_id=proposal.proposal_id,
            worker_id="verifier",
            verifier=_contract().verifier,
            verifier_profile_fingerprint=_verifier_profile_fingerprint(),
        )
        await _claim_completion_verification(store, claim_request)
        decision_request = _accepted_decision(
            proposal_id=proposal.proposal_id,
            claim_id=claim_request.claim_id,
            worker_id=claim_request.worker_id,
        )
        with pytest.raises(WorkAttemptAdmissionConflict, match="expired authority"):
            await store.settle_work_attempt_lifecycle(request)
        await _advance_lifecycle_clock(store, now, expires_at)
        if decision_first:
            decision = await store.record_completion_decision(decision_request)
            with pytest.raises(WorkAttemptAdmissionConflict, match="another lifecycle outcome"):
                await store.settle_work_attempt_lifecycle(request)
            assert await store.load_work_attempt_lifecycle_receipt(active.admission_id) is None
            assert await store.load_completion_decision(decision.decision_id) == decision
            assert (await store.load_task(active.task_id)).status is TaskStatus.RUNNING
            return
        for field, value in (
            ("proposal_id", None),
            ("proposal_request_sha256", None),
            ("stop_reason", "work_contract_budget_limit"),
            ("kind", "runtime_stop"),
        ):
            with pytest.raises(ValueError):
                await store.settle_work_attempt_lifecycle(request.model_copy(update={field: value}))
        for field, value in (
            ("proposal_id", "foreign-proposal"),
            ("proposal_request_sha256", "f" * 64),
        ):
            with pytest.raises(WorkAttemptAdmissionConflict):
                await store.settle_work_attempt_lifecycle(request.model_copy(update={field: value}))
        assert await store.load_work_attempt_lifecycle_receipt(active.admission_id) is None
        assert (await store.load_task(active.task_id)).status is TaskStatus.RUNNING
        receipt = await store.settle_work_attempt_lifecycle(request)
        assert receipt.task.status is TaskStatus.NEEDS_ATTENTION
        assert receipt.task.status_reason == "work_contract_elapsed_limit"
        assert not receipt.retired_contract_binding
        assert await store.load_active_work_contract_task_for_session(active.session_id) is not None
        with pytest.raises(WorkCompletionConflict, match="live task session"):
            await store.renew_completion_verification_claim(claim_request)
        with pytest.raises(WorkCompletionConflict, match="live task session"):
            await store.record_completion_decision(decision_request)
        with pytest.raises(ValueError, match="typed non-success"):
            type(receipt).model_validate(
                receipt.model_copy(
                    update={
                        "task": receipt.task.model_copy(
                            update={"status_reason": "work_contract_budget_limit"}
                        )
                    }
                ).model_dump(mode="python")
            )
        assert await store.load_completion_decision_for_proposal(proposal.proposal_id) is None
        assert await store.settle_work_attempt_lifecycle(request) == receipt
        if isinstance(store, (SQLiteTaskStore, PostgresTaskStore)):
            reopened = await _reopen_lifecycle_store(store, tmp_path)
            try:
                assert await reopened.settle_work_attempt_lifecycle(request) == receipt
            finally:
                await reopened.close()

    _run_lifecycle_scenario(lifecycle_store, scenario)


@pytest.mark.parametrize("reason", ["budget_limit", "elapsed_limit"])
def test_limit_stop_binds_entry_and_rejects_conflicting_settlement(lifecycle_store, reason):
    async def scenario():
        store = lifecycle_store
        active, settlement = await _active_settlement_fixture(store)
        entry_request = WorkAttemptExecutionEntryRequest(
            admission_id=active.admission_id,
            prepare_request_sha256=active.prepare_request_sha256,
            claim_id=active.claim.claim_id,
            worker_id=active.claim.worker_id,
            execution_owner_id=active.claim.execution_owner_id,
            generation=active.claim.generation,
            run_epoch=1,
        )
        entered = (await store.enter_work_attempt_execution(entry_request)).admission
        request = WorkAttemptExecutionStopRequest(
            admission_id=entered.admission_id,
            prepare_request_sha256=entered.prepare_request_sha256,
            claim_id=entered.claim.claim_id,
            worker_id=entered.claim.worker_id,
            execution_owner_id=entered.claim.execution_owner_id,
            generation=entered.claim.generation,
            execution_entry=entered.execution_entry,
            reason=reason,
        )
        for field, value in (
            ("generation", True),
            ("generation", 2),
            ("reason", "workspace_finalization_recovery"),
            ("reason", "unknown"),
        ):
            with pytest.raises(ValueError):
                await store.record_work_attempt_execution_stop(
                    request.model_copy(update={field: value})
                )
        assert await store.load_work_attempt_admission(active.admission_id) == entered
        stopped = await store.record_work_attempt_execution_stop(request)
        assert require_work_attempt_execution_stop_result(stopped, entered, request) == stopped
        assert await store.record_work_attempt_execution_stop(request) == stopped
        for field, value in (
            ("claim_id", "foreign-claim"),
            ("worker_id", "foreign-worker"),
            ("execution_owner_id", "foreign-owner"),
            ("prepare_request_sha256", "f" * 64),
        ):
            with pytest.raises(WorkAttemptAdmissionConflict):
                await store.record_work_attempt_execution_stop(
                    request.model_copy(update={field: value})
                )
        settlement = settlement.model_copy(
            update={
                "expected_admission_sha256": work_attempt_admission_authority_sha256(stopped),
                "stop_reason": "work_contract_execution_failed",
            }
        )
        with pytest.raises(WorkAttemptAdmissionConflict, match="another lifecycle outcome"):
            await store.settle_work_attempt_lifecycle(settlement)
        assert await store.load_work_attempt_lifecycle_receipt(active.admission_id) is None
        assert (await store.load_task(active.task_id)).status is TaskStatus.RUNNING
        receipt = await store.settle_work_attempt_lifecycle(
            settlement.model_copy(update={"stop_reason": "work_contract_" + reason})
        )
        assert receipt.task.status is TaskStatus.NEEDS_ATTENTION
        assert receipt.task.status_reason == "work_contract_" + reason

    _run_lifecycle_scenario(lifecycle_store, scenario)


def test_execution_stop_is_immutable_and_fences_replacement_dispatch(
    lifecycle_store, monkeypatch, tmp_path
):
    async def scenario():
        store = lifecycle_store
        now = [await _lifecycle_clock_now(store)]
        if not isinstance(store, PostgresTaskStore):
            monkeypatch.setattr(store, "_ownership_clock", lambda: now[0])
        lease_seconds = 5 if isinstance(store, PostgresTaskStore) else 300
        active, _ = await _active_settlement_fixture(store, lease_seconds=lease_seconds)
        entry_request = WorkAttemptExecutionEntryRequest(
            admission_id=active.admission_id,
            prepare_request_sha256=active.prepare_request_sha256,
            claim_id=active.claim.claim_id,
            worker_id=active.claim.worker_id,
            execution_owner_id=active.claim.execution_owner_id,
            generation=1,
            run_epoch=1,
        )
        entered = (await store.enter_work_attempt_execution(entry_request)).admission
        await _advance_lifecycle_clock(
            store, now, entered.claim.lease_expires_at + timedelta(seconds=1)
        )
        claim_request = WorkAttemptExecutionClaimRequest(
            admission_id=active.admission_id,
            claim_id="stop-recovery-claim",
            worker_id="stop-recovery-worker",
            execution_owner_id="stop-recovery-process",
            generation=2,
            lease_seconds=lease_seconds,
        )
        recovering = await store.claim_work_attempt_recovery(claim_request)
        request = WorkAttemptExecutionStopRequest(
            admission_id=active.admission_id,
            prepare_request_sha256=active.prepare_request_sha256,
            claim_id=recovering.claim.claim_id,
            worker_id=recovering.claim.worker_id,
            execution_owner_id=recovering.claim.execution_owner_id,
            generation=2,
            execution_entry=entered.execution_entry,
            reason="workspace_finalization_recovery",
        )
        conflicts = [
            request.model_copy(update={field: value})
            for field, value in (
                ("prepare_request_sha256", "f" * 64),
                ("claim_id", "foreign-claim"),
                ("worker_id", "foreign-worker"),
                ("execution_owner_id", "foreign-process"),
                ("generation", 3),
                (
                    "execution_entry",
                    entered.execution_entry.model_copy(
                        update={"request": entry_request.model_copy(update={"run_epoch": 2})}
                    ),
                ),
            )
        ]
        for conflict in conflicts:
            with pytest.raises(WorkAttemptExecutionClaimLost):
                await store.record_work_attempt_execution_stop(conflict)
        assert await store.load_work_attempt_admission(active.admission_id) == recovering
        results = await asyncio.gather(
            *(store.record_work_attempt_execution_stop(request) for _ in range(8))
        )
        stopped = results[0]
        assert all(result == stopped for result in results)
        assert stopped.execution_stop.request == request
        assert require_work_attempt_execution_stop_result(stopped, recovering, request) == stopped
        assert (
            require_work_attempt_claim_result(
                stopped,
                recovering,
                claim_request,
                operation_name="Concurrent stop publication",
                allowed_states=frozenset({WorkAttemptAdmissionState.RECOVERING}),
                allowed_previous_states=frozenset({WorkAttemptAdmissionState.RECOVERING}),
            )
            == stopped
        )
        for conflict in conflicts:
            with pytest.raises(WorkAttemptAdmissionConflict):
                await store.record_work_attempt_execution_stop(conflict)
        with pytest.raises(RuntimeError, match="erased or replaced"):
            require_work_attempt_claim_result(
                stopped.model_copy(update={"execution_stop": None}),
                stopped,
                claim_request,
                operation_name="Dropped execution stop",
                allowed_states=frozenset({WorkAttemptAdmissionState.RECOVERING}),
                allowed_previous_states=frozenset({WorkAttemptAdmissionState.RECOVERING}),
            )
        activated = await store.activate_work_attempt_recovery(
            WorkAttemptRecoveryActivate(
                admission_id=active.admission_id,
                claim_id=claim_request.claim_id,
                generation=2,
                recovery_evidence_sha256="d" * 64,
            )
        )
        assert activated.execution_stop == stopped.execution_stop
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await store.submit_admitted_completion_proposal(
                AdmittedCompletionProposalRequest(
                    admission_id=activated.admission_id,
                    claim_id=activated.claim.claim_id,
                    execution_owner_id=activated.claim.execution_owner_id,
                    generation=2,
                    proposal=CompletionProposalCreate(
                        proposal_id="stopped-proposal",
                        attempt_id=activated.attempt_id,
                        result=_result_reference(),
                        evidence_references=(_artifact_evidence(),),
                    ),
                )
            )
        assert await store.load_completion_proposal("stopped-proposal") is None
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await store.enter_work_attempt_execution(
                entry_request.model_copy(
                    update={
                        "claim_id": claim_request.claim_id,
                        "worker_id": claim_request.worker_id,
                        "execution_owner_id": claim_request.execution_owner_id,
                        "generation": 2,
                        "run_epoch": 3,
                    }
                )
            )
        await _advance_lifecycle_clock(
            store, now, activated.claim.lease_expires_at + timedelta(seconds=1)
        )
        assert await store.record_work_attempt_execution_stop(request) == activated
        if isinstance(store, (SQLiteTaskStore, PostgresTaskStore)):
            reopened = await _reopen_lifecycle_store(store, tmp_path)
            try:
                assert await reopened.record_work_attempt_execution_stop(request) == activated
            finally:
                await reopened.close()

    _run_lifecycle_scenario(lifecycle_store, scenario)


def test_runtime_stop_is_receipted_non_success_and_exactly_replayable(lifecycle_store) -> None:
    async def scenario():
        store = lifecycle_store
        active, request = await _active_settlement_fixture(store)
        receipt = await store.settle_work_attempt_lifecycle(request)
        assert receipt.task.status is TaskStatus.NEEDS_ATTENTION
        assert receipt.task.status_reason == "work_contract_elapsed_limit"
        assert receipt.task.worker_id is None
        assert receipt.task.lease_expires_at is None
        assert not receipt.retired_contract_binding
        settled = await store.load_work_attempt_admission(active.admission_id)
        assert settled.state is WorkAttemptAdmissionState.RELEASED
        assert await store.load_work_attempt_lifecycle_receipt(active.admission_id) == receipt
        assert await store.settle_work_attempt_lifecycle(request) == receipt
        assert await store.load_active_work_contract_task_for_session(active.session_id) is not None
        changed = request.model_copy(update={"stop_reason": "work_contract_budget_limit"})
        with pytest.raises(WorkAttemptAdmissionConflict, match="receipt"):
            await store.settle_work_attempt_lifecycle(changed)
        assert await store.load_task(request.task_id) == receipt.task
        receipt.task.metadata["caller-change"] = True
        assert "caller-change" not in (await store.load_task(request.task_id)).metadata

    _run_lifecycle_scenario(lifecycle_store, scenario)


def test_execution_entry_elects_one_dispatch_and_replay_cannot_dispatch(lifecycle_store, tmp_path):
    async def scenario():
        store = lifecycle_store
        admission, _ = await _active_settlement_fixture(
            store, lease_seconds=5 if isinstance(store, PostgresTaskStore) else 300
        )
        claim = admission.claim
        request = WorkAttemptExecutionEntryRequest(
            admission_id=admission.admission_id,
            prepare_request_sha256=admission.prepare_request_sha256,
            claim_id=claim.claim_id,
            worker_id=claim.worker_id,
            execution_owner_id=claim.execution_owner_id,
            generation=claim.generation,
            run_epoch=1,
        )
        results = await asyncio.gather(
            *(store.enter_work_attempt_execution(request) for _ in range(8))
        )
        assert [item.disposition for item in results].count(
            WorkAttemptExecutionEntryDisposition.ENTERED
        ) == 1
        assert [item.disposition for item in results].count(
            WorkAttemptExecutionEntryDisposition.ALREADY_ENTERED
        ) == 7
        durable = await store.load_work_attempt_admission(admission.admission_id)
        assert durable.execution_entry.request == request
        for result in results:
            assert require_work_attempt_execution_entry_result(result, admission, request) == result
        elected = next(
            result
            for result in results
            if result.disposition is WorkAttemptExecutionEntryDisposition.ENTERED
        )
        with pytest.raises(RuntimeError, match="duplicate dispatch grant"):
            require_work_attempt_execution_entry_result(elected, durable, request)
        for field, value in (
            ("run_epoch", 2),
            ("claim_id", "different-claim"),
            ("execution_owner_id", "different-process"),
        ):
            with pytest.raises(RuntimeError, match="conflicting authority"):
                require_work_attempt_execution_entry_result(
                    elected, admission, request.model_copy(update={field: value})
                )
        with pytest.raises(TypeError, match="invalid result type"):
            require_work_attempt_execution_entry_result(
                elected.model_dump(mode="python"), admission, request
            )
        claim_request = WorkAttemptExecutionClaimRequest(
            admission_id=admission.admission_id,
            claim_id=claim.claim_id,
            worker_id=claim.worker_id,
            execution_owner_id=claim.execution_owner_id,
            generation=claim.generation,
            lease_seconds=int((claim.lease_expires_at - claim.claimed_at).total_seconds()),
        )
        # A claim lookup may race the entry CAS. Observing the entry does not
        # turn an otherwise exact claim replay into conflicting authority.
        assert (
            require_work_attempt_claim_result(
                await store.claim_work_attempt_recovery(claim_request),
                admission,
                claim_request,
                operation_name="Concurrent claim replay",
                allowed_states=frozenset({WorkAttemptAdmissionState.ACTIVE}),
                allowed_previous_states=frozenset({WorkAttemptAdmissionState.ACTIVE}),
            )
            == durable
        )
        if isinstance(store, (SQLiteTaskStore, PostgresTaskStore)):
            reopened = await _reopen_lifecycle_store(
                store,
                tmp_path,
                after_expiry=admission.claim.lease_expires_at + timedelta(seconds=1),
            )
            try:
                replayed = await reopened.enter_work_attempt_execution(request)
                assert replayed.disposition is WorkAttemptExecutionEntryDisposition.ALREADY_ENTERED
                assert replayed.admission == durable
            finally:
                await reopened.close()
        assert (
            await store.enter_work_attempt_execution(request)
        ).disposition is WorkAttemptExecutionEntryDisposition.ALREADY_ENTERED
        for field, value in (
            ("prepare_request_sha256", "f" * 64),
            ("claim_id", "different-claim"),
            ("worker_id", "different-worker"),
            ("execution_owner_id", "different-process"),
            ("generation", claim.generation + 1),
        ):
            with pytest.raises(WorkAttemptExecutionClaimLost):
                await store.enter_work_attempt_execution(request.model_copy(update={field: value}))
        with pytest.raises(WorkAttemptAdmissionConflict, match="original request"):
            await store.enter_work_attempt_execution(request.model_copy(update={"run_epoch": 2}))
        assert await store.load_work_attempt_admission(admission.admission_id) == durable
        results[0].admission.run_semantics.request_metadata["caller"] = True
        assert (
            "caller"
            not in (
                await store.load_work_attempt_admission(admission.admission_id)
            ).run_semantics.request_metadata
        )

    _run_lifecycle_scenario(lifecycle_store, scenario)


@pytest.mark.parametrize(
    "reason", ["work_contract_preparation_failed", "work_contract_preparation_timed_out"]
)
def test_preparation_hold_is_exact_receipted_and_replays_after_lease_expiry(
    reason, lifecycle_store, tmp_path, monkeypatch
):
    async def scenario():
        store = lifecycle_store
        now = [await _lifecycle_clock_now(store)]
        if not isinstance(store, PostgresTaskStore):
            monkeypatch.setattr(store, "_ownership_clock", lambda: now[0])
        contract = _contract()
        await store.publish_work_contract(contract)
        task = await store.create_task(
            TaskCreate(task_id="prepare-failure", type="work", work_contract=contract.reference())
        )
        claimed = await store.claim_task(
            "worker", lease_seconds=10 if isinstance(store, PostgresTaskStore) else 30
        )
        assert claimed.id == task.id
        request = WorkAttemptPreparationHold(
            hold_id="preparation-hold",
            task_id=task.id,
            contract=contract.reference(),
            worker_id="worker",
            lease_expires_at=claimed.lease_expires_at,
            reason=reason,
        )
        with pytest.raises(TaskClaimLost):
            await store.hold_work_attempt_preparation(
                request.model_copy(update={"worker_id": "stale-worker"})
            )
        assert await store.load_task(task.id) == claimed
        assert await store.load_work_attempt_preparation_hold_receipt(request.hold_id) is None
        if isinstance(store, SQLiteTaskStore):
            store._connection.execute(
                "CREATE TRIGGER reject_preparation_hold BEFORE INSERT ON cayu_work_attempt_preparation_holds "
                "BEGIN SELECT RAISE(ABORT, 'injected preparation receipt failure'); END"
            )
            with pytest.raises(
                sqlite3.IntegrityError, match="injected preparation receipt failure"
            ):
                await store.hold_work_attempt_preparation(request)
            assert await store.load_task(task.id) == claimed
            assert await store.load_work_attempt_preparation_hold_receipt(request.hold_id) is None
            store._connection.execute("DROP TRIGGER reject_preparation_hold")
        elif isinstance(store, PostgresTaskStore):
            from psycopg.errors import RaiseException

            async with store._pool.connection() as connection:
                await connection.execute(
                    "CREATE FUNCTION reject_preparation_hold() RETURNS trigger "
                    "LANGUAGE plpgsql AS $$ BEGIN "
                    "RAISE EXCEPTION 'injected preparation receipt failure'; END; $$"
                )
                await connection.execute(
                    "CREATE TRIGGER reject_preparation_hold BEFORE INSERT "
                    "ON cayu_work_attempt_preparation_holds FOR EACH ROW "
                    "EXECUTE FUNCTION reject_preparation_hold()"
                )
            try:
                with pytest.raises(RaiseException, match="injected preparation receipt failure"):
                    await store.hold_work_attempt_preparation(request)
                assert await store.load_task(task.id) == claimed
                assert (
                    await store.load_work_attempt_preparation_hold_receipt(request.hold_id) is None
                )
            finally:
                async with store._pool.connection() as connection:
                    await connection.execute(
                        "DROP TRIGGER reject_preparation_hold ON cayu_work_attempt_preparation_holds"
                    )
                    await connection.execute("DROP FUNCTION reject_preparation_hold()")
        receipt = await store.hold_work_attempt_preparation(request)
        assert receipt.task.status is TaskStatus.NEEDS_ATTENTION
        assert receipt.task.status_reason == reason
        assert receipt.task.status_payload == {}
        assert receipt.task.session_id is None
        assert receipt.task.worker_id is None
        assert await store.load_latest_work_attempt_admission(task.id) is None
        expires_at = claimed.lease_expires_at + timedelta(seconds=1)
        await _advance_lifecycle_clock(store, now, expires_at)

        async def verify_replay(replay_store):
            assert await replay_store.hold_work_attempt_preparation(request) == receipt
            other_reason = (
                "work_contract_preparation_timed_out"
                if reason == "work_contract_preparation_failed"
                else "work_contract_preparation_failed"
            )
            for field, value in (
                ("task_id", "another-task"),
                ("contract", contract.reference().model_copy(update={"version": 2})),
                ("worker_id", "another-worker"),
                ("lease_expires_at", request.lease_expires_at + timedelta(seconds=1)),
                ("reason", other_reason),
            ):
                with pytest.raises(WorkAttemptAdmissionConflict, match="receipt"):
                    await replay_store.hold_work_attempt_preparation(
                        request.model_copy(update={field: value})
                    )
            receipt.task.metadata["caller"] = True
            replay = await replay_store.load_work_attempt_preparation_hold_receipt(request.hold_id)
            assert "caller" not in replay.task.metadata
            assert "caller" not in (await replay_store.load_task(task.id)).metadata

        if isinstance(store, (SQLiteTaskStore, PostgresTaskStore)):
            reopened = await _reopen_lifecycle_store(store, tmp_path, after_expiry=expires_at)
            try:
                await verify_replay(reopened)
            finally:
                await reopened.close()
        else:
            await verify_replay(store)

    _run_lifecycle_scenario(lifecycle_store, scenario)


def test_unsettled_discovery_is_scoped_bounded_and_receipt_aware(lifecycle_store) -> None:
    async def scenario():
        store = lifecycle_store
        records = [await _active_settlement_fixture(store, suffix) for suffix in ("z", "a", "m")]
        first_page = await store.list_unsettled_work_attempt_admissions(limit=2)
        assert [item.admission_id for item in first_page] == ["admission-a", "admission-m"]
        second_page = await store.list_unsettled_work_attempt_admissions(
            limit=2, after=first_page[-1].admission_id
        )
        assert [item.admission_id for item in second_page] == ["admission-z"]
        assert await store.list_unsettled_work_attempt_admissions(after="admission-z") == []
        scoped = await store.list_unsettled_work_attempt_admissions(
            task_filter=TaskAggregateFilter(type="worka")
        )
        assert [item.admission_id for item in scoped] == ["admission-a"]
        await store.settle_work_attempt_lifecycle(records[1][1])
        assert [
            item.admission_id for item in await store.list_unsettled_work_attempt_admissions()
        ] == ["admission-m", "admission-z"]
        first_page[0].run_semantics.request_metadata["caller"] = True
        stored = await store.load_work_attempt_admission("admission-a")
        assert "caller" not in stored.run_semantics.request_metadata

    _run_lifecycle_scenario(lifecycle_store, scenario)


@pytest.mark.parametrize("limit", [True, False, 0, 1001, "2"])
def test_unsettled_discovery_rejects_invalid_bounds(lifecycle_store, limit) -> None:
    async def scenario():
        with pytest.raises(ValueError):
            await lifecycle_store.list_unsettled_work_attempt_admissions(limit=limit)

    _run_lifecycle_scenario(lifecycle_store, scenario)


@pytest.mark.parametrize("lifecycle_store", ["postgres"], indirect=True)
@pytest.mark.parametrize("cancel_requests", [1, 2])
def test_postgres_discovery_cancellation_quiesces_dispatched_read(
    lifecycle_store, cancel_requests
) -> None:
    async def scenario():
        import psycopg

        store = lifecycle_store
        active, request = await _active_settlement_fixture(store)
        owner = None
        async with await psycopg.AsyncConnection.connect(store._pool.conninfo) as blocker:
            try:
                await blocker.execute(
                    "LOCK TABLE cayu_work_attempt_admissions IN ACCESS EXCLUSIVE MODE"
                )
                owner = asyncio.create_task(store.list_unsettled_work_attempt_admissions())
                async with asyncio.timeout(10):
                    while True:
                        await blocker.execute("SELECT pg_stat_clear_snapshot()")
                        cursor = await blocker.execute(
                            "SELECT pid FROM pg_stat_activity "
                            "WHERE datname = current_database() AND pid <> pg_backend_pid() "
                            "AND state = 'active' AND wait_event_type = 'Lock' "
                            "AND pg_backend_pid() = ANY(pg_blocking_pids(pid)) "
                            "AND query LIKE %s",
                            ("SELECT admission.admission_id FROM cayu_work_attempt_admissions%",),
                        )
                        blocked = await cursor.fetchall()
                        if blocked:
                            assert len(blocked) == 1
                            reader_pid = blocked[0][0]
                            break
                        await asyncio.sleep(0.01)

                # Positive server-side dispatch evidence precedes the real
                # cancellation; no manually raised exception or driver wrapper.
                for _ in range(cancel_requests):
                    owner.cancel("stop dispatched discovery")
                assert owner.cancelling() == cancel_requests
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(owner), timeout=5)
                assert owner.cancelled()
                assert owner.cancelling() == cancel_requests

                # Keep the blocker held until the connection has rolled back or
                # closed. A cancelled Python waiter alone is not this evidence.
                async with asyncio.timeout(5):
                    while True:
                        await blocker.execute("SELECT pg_stat_clear_snapshot()")
                        cursor = await blocker.execute(
                            "SELECT pid FROM pg_stat_activity WHERE pid = %s "
                            "AND (state = 'active' OR xact_start IS NOT NULL)",
                            (reader_pid,),
                        )
                        if not await cursor.fetchall():
                            break
                        await asyncio.sleep(0.01)
                await blocker.rollback()
                assert await store.load_work_attempt_admission(active.admission_id) == active
                assert await store.load_work_attempt_lifecycle_receipt(active.admission_id) is None
                receipt = await store.settle_work_attempt_lifecycle(request)
                assert receipt.task.status is TaskStatus.NEEDS_ATTENTION
                assert await store.list_unsettled_work_attempt_admissions() == []
            finally:
                await blocker.rollback()
                if owner is not None:
                    if not owner.done():
                        owner.cancel()
                    await asyncio.wait_for(asyncio.gather(owner, return_exceptions=True), timeout=5)

    _run_lifecycle_scenario(lifecycle_store, scenario)


def test_sqlite_discovery_cancellation_retains_connection_until_reader_settles(
    tmp_path, monkeypatch
) -> None:
    async def scenario():
        store = SQLiteTaskStore(tmp_path / "discovery-cancellation.sqlite")
        entered = threading.Event()
        release = threading.Event()
        owner = follower = None
        try:
            active, request = await _active_settlement_fixture(store)
            original = store._load_work_attempt_admission_unlocked

            def blocked_load(admission_id):
                entered.set()
                if not release.wait(timeout=5):
                    raise TimeoutError("test did not release the dispatched discovery read")
                return original(admission_id)

            monkeypatch.setattr(store, "_load_work_attempt_admission_unlocked", blocked_load)
            owner = asyncio.create_task(store.list_unsettled_work_attempt_admissions())
            assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 2), timeout=3)
            owner.cancel("first cancellation")
            await asyncio.sleep(0)
            owner.cancel("second cancellation")
            follower = asyncio.create_task(store.settle_work_attempt_lifecycle(request))
            await asyncio.sleep(0)
            assert owner.cancelling() == 2
            assert not owner.done()
            assert store._lock.locked()
            assert not follower.done()
            release.set()
            with pytest.raises(asyncio.CancelledError, match="first cancellation"):
                await asyncio.wait_for(owner, timeout=3)
            assert owner.cancelled()
            assert owner.cancelling() == 2
            receipt = await asyncio.wait_for(follower, timeout=3)
            assert receipt.task.status is TaskStatus.NEEDS_ATTENTION
            assert await store.load_work_attempt_lifecycle_receipt(active.admission_id) == receipt
            assert await store.list_unsettled_work_attempt_admissions() == []
        finally:
            release.set()
            pending = [task for task in (owner, follower) if task is not None]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "field,value",
    [
        ("session_id", "other-session"),
        ("session_instance_id", "other-instance"),
        ("interaction_id", "other-interaction"),
        ("profile_fingerprint", "4" * 64),
    ],
)
def test_runtime_stop_rejects_conflicting_release_without_mutation(
    lifecycle_store, field, value
) -> None:
    async def scenario():
        store = lifecycle_store
        active, request = await _active_settlement_fixture(store)
        before = await store.load_task(request.task_id)
        changed = request.model_copy(
            update={"release_evidence": request.release_evidence.model_copy(update={field: value})}
        )
        with pytest.raises(WorkAttemptAdmissionConflict):
            await store.settle_work_attempt_lifecycle(changed)
        assert await store.load_task(request.task_id) == before
        assert await store.load_work_attempt_admission(active.admission_id) == active
        assert await store.load_work_attempt_lifecycle_receipt(active.admission_id) is None

    _run_lifecycle_scenario(lifecycle_store, scenario)


def test_sqlite_settlement_replays_after_restart_and_lease_expiry(tmp_path) -> None:
    async def scenario():
        path = tmp_path / "restart.sqlite"
        store = SQLiteTaskStore(path)
        try:
            active, request = await _active_settlement_fixture(store)
            receipt = await store.settle_work_attempt_lifecycle(request)
        finally:
            await store.close()
        restarted = SQLiteTaskStore(
            path,
            schema_mode=SchemaMode.VALIDATE,
            ownership_clock=lambda: active.claim.lease_expires_at + timedelta(hours=1),
        )
        try:
            assert await restarted.settle_work_attempt_lifecycle(request) == receipt
            assert (
                await restarted.load_work_attempt_lifecycle_receipt(active.admission_id) == receipt
            )
        finally:
            await restarted.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("lifecycle_store", ["sqlite", "postgres"], indirect=True)
def test_receipt_failure_rolls_back_task_and_admission(lifecycle_store, tmp_path) -> None:
    async def scenario():
        store = lifecycle_store
        active, request = await _active_settlement_fixture(store)
        before = await store.load_task(request.task_id)
        if isinstance(store, SQLiteTaskStore):
            store._connection.execute(
                "CREATE TRIGGER reject_lifecycle_receipt BEFORE INSERT ON cayu_work_attempt_lifecycle_receipts "
                "BEGIN SELECT RAISE(ABORT, 'injected receipt failure'); END"
            )
            failure_type = sqlite3.IntegrityError
        else:
            from psycopg.errors import RaiseException

            failure_type = RaiseException
            async with store._pool.connection() as connection:
                await connection.execute(
                    "CREATE FUNCTION reject_lifecycle_receipt() RETURNS trigger "
                    "LANGUAGE plpgsql AS $$ BEGIN "
                    "RAISE EXCEPTION 'injected receipt failure'; END; $$"
                )
                await connection.execute(
                    "CREATE TRIGGER reject_lifecycle_receipt BEFORE INSERT "
                    "ON cayu_work_attempt_lifecycle_receipts FOR EACH ROW "
                    "EXECUTE FUNCTION reject_lifecycle_receipt()"
                )
        try:
            with pytest.raises(failure_type, match="injected receipt failure"):
                await store.settle_work_attempt_lifecycle(request)
            assert await store.load_task(request.task_id) == before
            assert await store.load_work_attempt_admission(active.admission_id) == active
            assert await store.load_work_attempt_lifecycle_receipt(active.admission_id) is None
        finally:
            if isinstance(store, SQLiteTaskStore):
                store._connection.execute("DROP TRIGGER reject_lifecycle_receipt")
            else:
                async with store._pool.connection() as connection:
                    await connection.execute(
                        "DROP TRIGGER reject_lifecycle_receipt ON cayu_work_attempt_lifecycle_receipts"
                    )
                    await connection.execute("DROP FUNCTION reject_lifecycle_receipt()")
        receipt = await store.settle_work_attempt_lifecycle(request)
        assert receipt.task.status is TaskStatus.NEEDS_ATTENTION
        reopened = await _reopen_lifecycle_store(store, tmp_path)
        try:
            assert await reopened.settle_work_attempt_lifecycle(request) == receipt
        finally:
            await reopened.close()

    _run_lifecycle_scenario(lifecycle_store, scenario)


@pytest.mark.parametrize("populated", [False, True])
def test_revision_84_migration_requires_empty_pre_worker_admission_history(
    tmp_path, populated
) -> None:
    path = tmp_path / "revision83.sqlite"

    async def create():
        store = SQLiteTaskStore(path)
        try:
            await store.create_task(TaskCreate(task_id="ordinary", type="ordinary"))
            if populated:
                await _active_settlement_fixture(store)
        finally:
            await store.close()

    asyncio.run(create())
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE cayu_work_attempt_lifecycle_receipts")
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision = 84")
        connection.execute("PRAGMA user_version = 83")
    if populated:
        with pytest.raises(RuntimeError, match="cannot reconstruct executable settings"):
            SQLiteTaskStore(path, schema_mode=SchemaMode.MIGRATE)
        with sqlite3.connect(path) as connection:
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 83
            assert (
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE name = 'cayu_work_attempt_lifecycle_receipts'"
                ).fetchone()
                is None
            )
    else:

        async def migrate():
            store = SQLiteTaskStore(path, schema_mode=SchemaMode.MIGRATE)
            try:
                assert await store.load_task("ordinary") is not None
                assert store._connection.execute("PRAGMA user_version").fetchone()[0] == 84
            finally:
                await store.close()

        asyncio.run(migrate())


@pytest.mark.parametrize("with_sibling", [False, True])
def test_accepted_settlement_retires_only_its_own_contract_binding(
    lifecycle_store, with_sibling
) -> None:
    async def scenario():
        store = lifecycle_store
        active, stop_request = await _active_settlement_fixture(store)
        sibling = None
        if with_sibling:
            sibling = await store.create_task(
                task_create_with_runtime_invocation(
                    TaskCreate(
                        task_id="sibling",
                        type="work",
                        session_id=active.session_id,
                        work_contract=active.contract,
                    ),
                    source=TaskExecutionSource.SDK_TASK,
                    session_invocation=active.session_invocation,
                )
            )
        proposal = await store.submit_admitted_completion_proposal(
            AdmittedCompletionProposalRequest(
                admission_id=active.admission_id,
                claim_id=active.claim.claim_id,
                execution_owner_id=active.claim.execution_owner_id,
                generation=active.claim.generation,
                proposal=CompletionProposalCreate(
                    proposal_id="accepted-proposal",
                    attempt_id=active.attempt_id,
                    result=_result_reference(),
                    evidence_references=(_artifact_evidence(),),
                ),
            )
        )
        claim = await _claim_completion_verification(
            store,
            CompletionVerificationClaimRequest(
                claim_id="accepted-claim",
                proposal_id=proposal.proposal_id,
                worker_id="verifier",
                verifier=_contract().verifier,
                verifier_profile_fingerprint=_verifier_profile_fingerprint(),
            ),
        )
        decision = await store.record_completion_decision(
            _accepted_decision(
                proposal_id=proposal.proposal_id, claim_id=claim.claim_id, worker_id=claim.worker_id
            )
        )
        application = CompletionDecisionApplicationRequest(
            task_id=active.task_id,
            decision_id=decision.decision_id,
            idempotency_key="accepted-apply",
            result=_task_result(),
            result_reference=proposal.result,
        )
        completed = await store.apply_completion_decision(application)
        assert completed.status is TaskStatus.COMPLETED
        assert [
            item.admission_id for item in await store.list_unsettled_work_attempt_admissions()
        ] == [active.admission_id]
        assert await store.load_active_work_contract_task_for_session(active.session_id) is not None
        released = await store.load_work_attempt_admission(active.admission_id)
        request = WorkAttemptLifecycleSettlement(
            settlement_id="accepted-settle",
            task_id=active.task_id,
            admission_id=active.admission_id,
            expected_admission_sha256=work_attempt_admission_authority_sha256(released),
            release_evidence=stop_request.release_evidence,
            kind="decision_application",
            decision_id=decision.decision_id,
            application_idempotency_key=application.idempotency_key,
        )
        receipt = await store.settle_work_attempt_lifecycle(request)
        assert await store.list_unsettled_work_attempt_admissions() == []
        assert receipt.retired_contract_binding
        assert receipt.task == completed
        assert await store.settle_work_attempt_lifecycle(request) == receipt
        assert await store.apply_completion_decision(application) == completed
        if isinstance(store, InMemoryTaskStore):
            # Exercise the index-maintenance owner after receipt publication.
            store._store_task(completed)
        remaining = await store.load_active_work_contract_task_for_session(active.session_id)
        if sibling is not None:
            assert remaining.id == sibling.id
            with pytest.raises(TaskCompletionDecisionRequired):
                await store.admit_ordinary_session_execution(active.session_id)
        else:
            assert remaining is None
            await store.admit_ordinary_session_execution(active.session_id)

    _run_lifecycle_scenario(lifecycle_store, scenario)
