from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from tests.core.task_invocation_fixtures import task_backed_session_invocation
from tests.core.test_verified_work_contracts import (
    _artifact_evidence,
    _claim_completion_verification,
    _contract,
    _digest,
    _RecordingProvider,
    _rejected_decision,
    _result_reference,
    _verifier_profile_fingerprint,
)

from cayu import (
    AgentSpec,
    CayuApp,
    CompletionDecisionApplicationRequest,
    CompletionVerificationClaimRequest,
    EventQuery,
    EventType,
    Message,
    PostgresSessionStore,
    PostgresTaskStore,
    RunRequest,
    SessionStatus,
    Task,
    TaskClaimLost,
    TaskCreate,
    TaskStatus,
    TaskTerminalizationRequest,
    TaskTerminalKind,
    WorkAttemptAdmissionConflict,
    WorkAttemptCreate,
    WorkAttemptExecutionClaimLost,
    WorkAttemptExecutionRequest,
    WorkAttemptProposalRequest,
    WorkAttemptRecoveryRequest,
)
from cayu.runtime.sessions import INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY
from cayu.runtime.work_attempt_admission import (
    AdmittedCompletionProposalRequest,
    WorkAttemptAdmission,
    WorkAttemptAdmissionActivate,
    WorkAttemptAdmissionPrepare,
    WorkAttemptAdmissionState,
    WorkAttemptExecutionClaimRequest,
    WorkAttemptRecoveryActivate,
)
from cayu.runtime.work_contracts import CompletionProposalCreate
from cayu.storage.migrations import SchemaMode


class _PostgresAdmissionLockOrderStore(PostgresTaskStore):
    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self, dsn: str) -> None:
        super().__init__(dsn, schema_mode=SchemaMode.CREATE)
        self.ordinary_task_locked = asyncio.Event()
        self.release_ordinary_task_lock = asyncio.Event()
        self.renewal_admission_locked = asyncio.Event()

    async def _load_task_locked(self, cur: Any, task_id: str) -> Task:
        task = await super()._load_task_locked(cur, task_id)
        current = asyncio.current_task()
        if current is not None and current.get_name() == "ordinary-heartbeat":
            self.ordinary_task_locked.set()
            await self.release_ordinary_task_lock.wait()
        return task

    async def _load_work_attempt_admission_row(
        self,
        cur: Any,
        admission_id: str,
        *,
        for_update: bool = False,
    ) -> WorkAttemptAdmission | None:
        admission = await super()._load_work_attempt_admission_row(
            cur,
            admission_id,
            for_update=for_update,
        )
        if for_update:
            self.renewal_admission_locked.set()
        return admission


def test_postgres_admission_separates_scheduling_and_queue_lease_clocks(
    postgres_dsn: str,
) -> None:
    async def scenario() -> None:
        suffix = uuid4().hex
        verified_now = [datetime(2100, 1, 1, tzinfo=UTC)]
        store = PostgresTaskStore(
            postgres_dsn,
            clock=lambda: verified_now[0],
            schema_mode=SchemaMode.CREATE,
        )
        contract = _contract(contract_id=f"postgres-admission-clock-{suffix}")
        await store.publish_work_contract(contract)

        live_task = await store.create_task(
            TaskCreate(
                task_id=f"postgres-admission-live-clock-task-{suffix}",
                type="verified-work",
                available_at=datetime(2099, 1, 1, tzinfo=UTC),
                work_contract=contract.reference(),
            )
        )
        live_worker = f"postgres-admission-live-clock-worker-{suffix}"
        assert await store.claim_task(live_worker) is not None
        live_session_id = f"postgres-admission-live-clock-session-{suffix}"
        live = await store.prepare_work_attempt_admission(
            WorkAttemptAdmissionPrepare(
                admission_id=f"postgres-admission-live-clock-admission-{suffix}",
                claim_id=f"postgres-admission-live-clock-claim-{suffix}",
                attempt_id=f"postgres-admission-live-clock-attempt-{suffix}",
                task_id=live_task.id,
                session_id=live_session_id,
                interaction_id=f"postgres-admission-live-clock-interaction-{suffix}",
                worker_id=live_worker,
                execution_owner_id=f"postgres-admission-live-clock-owner-{suffix}",
                generation=1,
                lease_seconds=300,
                kind="initial",
                source_request_sha256=_digest(f"postgres-admission-live-source-{suffix}"),
                contract=contract.reference(),
                session_invocation=await task_backed_session_invocation(
                    store,
                    live_task.id,
                    live_session_id,
                ),
                source_execution_profile_fingerprint=_digest(
                    f"postgres-admission-live-profile-{suffix}"
                ),
            )
        )
        assert live.claim.claimed_at == verified_now[0]

        verified_now[0] = datetime(2000, 1, 1, tzinfo=UTC)
        expired_task = await store.create_task(
            TaskCreate(
                task_id=f"postgres-admission-expired-clock-task-{suffix}",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        expired_worker = f"postgres-admission-expired-clock-worker-{suffix}"
        assert await store.claim_task(expired_worker, lease_seconds=1) is not None
        await asyncio.sleep(1.1)
        expired_session_id = f"postgres-admission-expired-clock-session-{suffix}"
        expired_request = WorkAttemptAdmissionPrepare(
            admission_id=f"postgres-admission-expired-clock-admission-{suffix}",
            claim_id=f"postgres-admission-expired-clock-claim-{suffix}",
            attempt_id=f"postgres-admission-expired-clock-attempt-{suffix}",
            task_id=expired_task.id,
            session_id=expired_session_id,
            interaction_id=f"postgres-admission-expired-clock-interaction-{suffix}",
            worker_id=expired_worker,
            execution_owner_id=f"postgres-admission-expired-clock-owner-{suffix}",
            generation=1,
            lease_seconds=300,
            kind="initial",
            source_request_sha256=_digest(f"postgres-admission-expired-source-{suffix}"),
            contract=contract.reference(),
            session_invocation=await task_backed_session_invocation(
                store,
                expired_task.id,
                expired_session_id,
            ),
            source_execution_profile_fingerprint=_digest(
                f"postgres-admission-expired-profile-{suffix}"
            ),
        )
        with pytest.raises(TaskClaimLost, match="expired"):
            await store.prepare_work_attempt_admission(expired_request)
        untouched = await store.load_task(expired_task.id)
        assert untouched is not None
        assert untouched.status is TaskStatus.CLAIMED
        assert untouched.session_id is None
        assert await store.load_work_attempt_admission(expired_request.admission_id) is None
        await store.close()

    asyncio.run(scenario())


def test_postgres_cancelled_admission_is_quiescent_before_successor_retry(
    postgres_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        import psycopg
        from psycopg_pool import AsyncConnectionPool

        from cayu.storage import _postgres_verified_work as postgres_verified_work

        suffix = uuid4().hex
        application_name = f"cayu-admission-cancellation-{suffix}"
        monkeypatch.setattr(
            postgres_verified_work,
            "_POSTGRES_MUTATION_CANCELLATION_GRACE_SECONDS",
            0.0,
        )
        pool = AsyncConnectionPool(
            postgres_dsn,
            min_size=1,
            max_size=2,
            open=False,
            close_returns=True,
            kwargs={"application_name": application_name},
        )
        await pool.open()
        store = PostgresTaskStore(pool=pool, schema_mode=SchemaMode.CREATE)
        lock_connection = await psycopg.AsyncConnection.connect(postgres_dsn)
        caller: asyncio.Task[WorkAttemptAdmission] | None = None
        lock_held = False
        try:
            contract = _contract(contract_id=f"postgres-cancel-contract-{suffix}")
            await store.publish_work_contract(contract)
            task = await store.create_task(
                TaskCreate(
                    task_id=f"postgres-cancel-task-{suffix}",
                    type="verified-work",
                    work_contract=contract.reference(),
                )
            )
            session_id = f"postgres-cancel-session-{suffix}"
            request = WorkAttemptAdmissionPrepare(
                admission_id=f"postgres-cancel-admission-{suffix}",
                claim_id=f"postgres-cancel-claim-{suffix}",
                attempt_id=f"postgres-cancel-attempt-{suffix}",
                task_id=task.id,
                session_id=session_id,
                interaction_id=f"postgres-cancel-interaction-{suffix}",
                worker_id=f"postgres-cancel-worker-{suffix}",
                execution_owner_id=f"postgres-cancel-owner-{suffix}",
                generation=1,
                lease_seconds=300,
                kind="initial",
                source_request_sha256=_digest(f"postgres-cancel-source-{suffix}"),
                contract=contract.reference(),
                session_invocation=await task_backed_session_invocation(
                    store,
                    task.id,
                    session_id,
                ),
                source_execution_profile_fingerprint=_digest(f"postgres-cancel-profile-{suffix}"),
            )

            await lock_connection.execute(
                "LOCK TABLE cayu_work_attempt_admissions IN ACCESS EXCLUSIVE MODE"
            )
            lock_held = True
            caller = asyncio.create_task(
                store.prepare_work_attempt_admission(request),
                name="postgres-admission-cancellation-caller",
            )

            # Observe the real pool connection waiting inside the admission
            # transaction. This is positive dispatch evidence rather than a
            # scheduler sleep standing in for the ownership boundary.
            for _ in range(100):
                cursor = await lock_connection.execute(
                    "SELECT EXISTS ("
                    "SELECT 1 FROM pg_stat_activity "
                    "WHERE datname = current_database() "
                    "AND application_name = %s "
                    "AND wait_event_type = 'Lock'"
                    ")",
                    (application_name,),
                )
                row = await cursor.fetchone()
                if row == (True,):
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("Admission mutation did not reach the held database lock")

            caller.cancel("stop waiting after admission dispatch")
            assert caller.cancelling() == 1
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(caller), timeout=5)
            assert caller.cancelled()
            assert not any(
                pending.get_name() == "cayu-postgres-store-mutation"
                for pending in asyncio.all_tasks()
            )

            # The cancelled owner is already quiescent while the competing
            # transaction still holds the table lock. Releasing that lock must
            # not permit a delayed admission to publish.
            await lock_connection.commit()
            lock_held = False
            untouched = await store.load_task(task.id)
            assert untouched is not None
            assert untouched.status is TaskStatus.PENDING
            assert untouched.worker_id is None
            assert untouched.session_id is None
            assert await store.load_work_attempt_admission(request.admission_id) is None

            successor = await store.prepare_work_attempt_admission(request)
            assert successor.state is WorkAttemptAdmissionState.PREPARING
            assert successor.claim.worker_id == request.worker_id
        finally:
            if lock_held:
                await lock_connection.rollback()
            if caller is not None and not caller.done():
                caller.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await caller
            await lock_connection.close()
            await store.close()
            await pool.close()

    asyncio.run(scenario())


def test_postgres_work_attempt_admission_continuation_and_recovery(postgres_dsn) -> None:
    async def scenario() -> None:
        suffix = uuid4().hex
        first_store = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        second_store = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        contract = _contract(contract_id=f"postgres-admission-contract-{suffix}")
        task_id = f"postgres-admission-task-{suffix}"
        session_id = f"postgres-admission-session-{suffix}"
        await first_store.publish_work_contract(contract)
        task = await first_store.create_task(
            TaskCreate(
                task_id=task_id,
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        session_invocation = await task_backed_session_invocation(
            first_store,
            task.id,
            session_id,
        )

        initial_request = WorkAttemptAdmissionPrepare(
            admission_id=f"postgres-admission-1-{suffix}",
            claim_id=f"postgres-admission-claim-1-{suffix}",
            attempt_id=f"postgres-admission-attempt-1-{suffix}",
            task_id=task.id,
            session_id=session_id,
            interaction_id=f"postgres-admission-interaction-1-{suffix}",
            worker_id=f"postgres-admission-worker-1-{suffix}",
            execution_owner_id=f"postgres-admission-owner-1-{suffix}",
            generation=1,
            lease_seconds=1,
            kind="initial",
            source_request_sha256=_digest(f"postgres-source-request-1-{suffix}"),
            contract=contract.reference(),
            session_invocation=session_invocation,
            source_execution_profile_fingerprint=_digest(f"postgres-source-profile-{suffix}"),
        )
        left, right = await asyncio.gather(
            first_store.prepare_work_attempt_admission(initial_request),
            second_store.prepare_work_attempt_admission(
                initial_request.model_copy(
                    update={
                        "admission_id": f"postgres-conflict-admission-{suffix}",
                        "claim_id": f"postgres-conflict-claim-{suffix}",
                        "attempt_id": f"postgres-conflict-attempt-{suffix}",
                        "interaction_id": f"postgres-conflict-interaction-{suffix}",
                    }
                )
            ),
            return_exceptions=True,
        )
        outcomes = (left, right)
        prepared = next(item for item in outcomes if not isinstance(item, BaseException))
        failure = next(item for item in outcomes if isinstance(item, BaseException))
        assert isinstance(failure, WorkAttemptAdmissionConflict)
        assert prepared.state is WorkAttemptAdmissionState.PREPARING
        await asyncio.sleep(1.1)
        preparation_recovery = WorkAttemptExecutionClaimRequest(
            admission_id=prepared.admission_id,
            claim_id=f"postgres-preparation-recovery-claim-{suffix}",
            worker_id=f"postgres-preparation-recovery-worker-{suffix}",
            execution_owner_id=f"postgres-preparation-recovery-owner-{suffix}",
            generation=2,
            lease_seconds=300,
        )
        prepared = await first_store.claim_work_attempt_recovery(preparation_recovery)
        assert prepared.state is WorkAttemptAdmissionState.PREPARING
        assert prepared.claim.generation == 2
        assert (await second_store.claim_work_attempt_recovery(preparation_recovery)) == prepared
        active = await first_store.activate_work_attempt_admission(
            WorkAttemptAdmissionActivate(
                admission_id=prepared.admission_id,
                claim_id=prepared.claim.claim_id,
                prepare_request_sha256=prepared.prepare_request_sha256,
                session_evidence_sha256=_digest(f"postgres-session-evidence-1-{suffix}"),
            )
        )
        active_task = await second_store.load_task(task.id)
        assert active_task is not None
        assert active_task.session_instance_id == session_invocation.session_instance_id
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await second_store.pause_task(task.id, reason="ordinary-pause")
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await second_store.heartbeat(task.id, active.claim.worker_id)
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await second_store.complete_task(
                task.id,
                {"status": "ordinary-worker-completion"},
                worker_id=active.claim.worker_id,
            )
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await second_store.fail_task(
                task.id,
                {"message": "ordinary worker failure"},
                worker_id=active.claim.worker_id,
            )
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await second_store.terminalize_task(
                TaskTerminalizationRequest(
                    task_id=task.id,
                    worker_id=active.claim.worker_id,
                    idempotency_key=f"postgres-ordinary-terminalization-{suffix}",
                    kind=TaskTerminalKind.FAILED,
                    error={"message": "ordinary worker terminalization"},
                )
            )
        proposal_request = AdmittedCompletionProposalRequest(
            admission_id=active.admission_id,
            claim_id=active.claim.claim_id,
            execution_owner_id=active.claim.execution_owner_id,
            generation=active.claim.generation,
            proposal=CompletionProposalCreate(
                proposal_id=f"postgres-proposal-1-{suffix}",
                attempt_id=active.attempt_id,
                result=_result_reference(),
                evidence_references=(_artifact_evidence(),),
            ),
        )
        proposal = await first_store.submit_admitted_completion_proposal(proposal_request)
        assert await second_store.submit_admitted_completion_proposal(proposal_request) == proposal
        released_task = await second_store.load_task(task.id)
        assert released_task is not None
        assert released_task.session_instance_id == session_invocation.session_instance_id
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await second_store.fail_task(task.id, {"message": "stale failure"})
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await second_store.cancel_task(task.id, {"message": "stale cancellation"})
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await second_store.pause_task(task.id, reason="stale pause")
        with pytest.raises(TaskClaimLost):
            await second_store.heartbeat(task.id, active.claim.worker_id)
        with pytest.raises(TaskClaimLost):
            await second_store.release_task(task.id, active.claim.worker_id)
        with pytest.raises(TaskClaimLost):
            await second_store.release_attached_task_worker(task.id, active.claim.worker_id)
        with pytest.raises(TaskClaimLost):
            await second_store.complete_task(
                task.id,
                {"status": "stale-worker-completion"},
                worker_id=active.claim.worker_id,
            )
        with pytest.raises(TaskClaimLost):
            await second_store.fail_task(
                task.id,
                {"message": "stale worker failure"},
                worker_id=active.claim.worker_id,
            )
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await second_store.terminalize_task(
                TaskTerminalizationRequest(
                    task_id=task.id,
                    worker_id=active.claim.worker_id,
                    idempotency_key=f"postgres-stale-terminalization-{suffix}",
                    kind=TaskTerminalKind.FAILED,
                    error={"message": "stale worker terminalization"},
                )
            )
        assert await first_store.load_task(task.id) == released_task
        assert await second_store.submit_admitted_completion_proposal(proposal_request) == proposal
        verifier_claim = await _claim_completion_verification(
            first_store,
            CompletionVerificationClaimRequest(
                claim_id=f"postgres-verifier-claim-{suffix}",
                proposal_id=proposal.proposal_id,
                worker_id=f"postgres-verifier-worker-{suffix}",
                verifier=contract.verifier,
                verifier_profile_fingerprint=_verifier_profile_fingerprint(contract.verifier),
            ),
        )
        decision = await first_store.record_completion_decision(
            _rejected_decision(
                proposal_id=proposal.proposal_id,
                claim_id=verifier_claim.claim_id,
                worker_id=verifier_claim.worker_id,
                decision_id=f"postgres-rejected-decision-{suffix}",
            )
        )
        await first_store.apply_completion_decision(
            CompletionDecisionApplicationRequest(
                task_id=task.id,
                decision_id=decision.decision_id,
                idempotency_key=f"postgres-decision-application-{suffix}",
            )
        )
        bypass_attempt_id = f"postgres-direct-bypass-attempt-{suffix}"
        with pytest.raises(WorkAttemptAdmissionConflict, match="permanently governed"):
            await second_store.begin_work_attempt(
                WorkAttemptCreate(
                    attempt_id=bypass_attempt_id,
                    task_id=task.id,
                    session_id=session_id,
                    contract=contract.reference(),
                    execution_profile_fingerprint=(
                        initial_request.source_execution_profile_fingerprint
                    ),
                    worker_id=None,
                )
            )
        assert await first_store.load_work_attempt(bypass_attempt_id) is None

        continuation_request = WorkAttemptAdmissionPrepare(
            admission_id=f"postgres-admission-2-{suffix}",
            claim_id=f"postgres-admission-claim-2-{suffix}",
            attempt_id=f"postgres-admission-attempt-2-{suffix}",
            task_id=task.id,
            session_id=session_id,
            interaction_id=f"postgres-admission-interaction-2-{suffix}",
            worker_id=f"postgres-admission-worker-2-{suffix}",
            execution_owner_id=f"postgres-admission-owner-2-{suffix}",
            generation=1,
            lease_seconds=1,
            kind="continuation",
            predecessor_admission_id=active.admission_id,
            source_request_sha256=_digest(f"postgres-source-request-2-{suffix}"),
            contract=contract.reference(),
            session_invocation=session_invocation,
            source_execution_profile_fingerprint=(
                initial_request.source_execution_profile_fingerprint
            ),
        )
        continuation = await second_store.prepare_work_attempt_admission(continuation_request)
        assert continuation.continuation is not None
        assert continuation.continuation.prior_admission_id == active.admission_id
        assert continuation.continuation.decision == decision
        continued = await second_store.activate_work_attempt_admission(
            WorkAttemptAdmissionActivate(
                admission_id=continuation.admission_id,
                claim_id=continuation.claim.claim_id,
                prepare_request_sha256=continuation.prepare_request_sha256,
                session_evidence_sha256=_digest(f"postgres-session-evidence-2-{suffix}"),
            )
        )
        assert continued.attempt is not None
        assert continued.attempt.ordinal == 2
        assert await first_store.submit_admitted_completion_proposal(proposal_request) == proposal
        await asyncio.sleep(1.1)
        recovery_request = WorkAttemptExecutionClaimRequest(
            admission_id=continued.admission_id,
            claim_id=f"postgres-recovery-claim-{suffix}",
            worker_id=f"postgres-recovery-worker-{suffix}",
            execution_owner_id=f"postgres-recovery-owner-{suffix}",
            generation=2,
            lease_seconds=300,
        )
        recovering = await first_store.claim_work_attempt_recovery(recovery_request)
        assert recovering.state is WorkAttemptAdmissionState.RECOVERING
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await second_store.submit_admitted_completion_proposal(
                AdmittedCompletionProposalRequest(
                    admission_id=continued.admission_id,
                    claim_id=continued.claim.claim_id,
                    execution_owner_id=continued.claim.execution_owner_id,
                    generation=continued.claim.generation,
                    proposal=CompletionProposalCreate(
                        proposal_id=f"postgres-stale-proposal-{suffix}",
                        attempt_id=continued.attempt_id,
                        result=_result_reference(),
                        evidence_references=(_artifact_evidence(),),
                    ),
                )
            )
        recovery_activation = WorkAttemptRecoveryActivate(
            admission_id=recovering.admission_id,
            claim_id=recovering.claim.claim_id,
            generation=recovering.claim.generation,
            recovery_evidence_sha256=_digest(f"postgres-recovery-evidence-{suffix}"),
        )
        recovered = await first_store.activate_work_attempt_recovery(recovery_activation)
        assert recovered.state is WorkAttemptAdmissionState.ACTIVE
        assert (await second_store.claim_work_attempt_recovery(recovery_request)) == recovered

        drifted_session_instance_id = "00000000-0000-4000-8000-000000000099"
        async with first_store._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE cayu_tasks SET session_instance_id = %s WHERE id = %s",
                    (drifted_session_instance_id, task.id),
                )
            await conn.commit()
        with pytest.raises(
            WorkAttemptExecutionClaimLost,
            match="exact task-session authority",
        ):
            await second_store.activate_work_attempt_recovery(recovery_activation)
        async with first_store._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE cayu_tasks SET session_instance_id = %s WHERE id = %s",
                    (session_invocation.session_instance_id, task.id),
                )
            await conn.commit()

        recovered_proposal_request = AdmittedCompletionProposalRequest(
            admission_id=recovered.admission_id,
            claim_id=recovered.claim.claim_id,
            execution_owner_id=recovered.claim.execution_owner_id,
            generation=recovered.claim.generation,
            proposal=CompletionProposalCreate(
                proposal_id=f"postgres-recovered-proposal-{suffix}",
                attempt_id=recovered.attempt_id,
                result=_result_reference(),
                evidence_references=(_artifact_evidence(),),
            ),
        )
        recovered_proposal = await first_store.submit_admitted_completion_proposal(
            recovered_proposal_request
        )
        assert (
            await second_store.submit_admitted_completion_proposal(recovered_proposal_request)
            == recovered_proposal
        )
        await first_store.close()
        await second_store.close()

        import psycopg

        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "UPDATE cayu_work_attempt_admissions "
                    "SET admission_json = admission_json "
                    "#- '{continuation,prior_admission_id}' "
                    "WHERE admission_id = %s",
                    (continued.admission_id,),
                )
                await cursor.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 62")
            await connection.commit()

        migrated_store = PostgresTaskStore(
            postgres_dsn,
            schema_mode=SchemaMode.MIGRATE,
        )
        try:
            migrated_continuation = await migrated_store.load_work_attempt_admission(
                continued.admission_id
            )
            assert migrated_continuation is not None
            assert migrated_continuation.continuation is not None
            assert migrated_continuation.continuation.prior_admission_id == active.admission_id
        finally:
            await migrated_store.close()

    asyncio.run(scenario())


def test_postgres_public_first_crash_recovery_needs_no_direct_store_mutation(
    postgres_dsn: str,
) -> None:
    async def scenario() -> None:
        suffix = uuid4().hex
        source_sessions = PostgresSessionStore(
            postgres_dsn,
            schema_mode=SchemaMode.CREATE,
        )
        source_tasks = PostgresTaskStore(
            postgres_dsn,
            schema_mode=SchemaMode.CREATE,
        )
        replacement_sessions: PostgresSessionStore | None = None
        replacement_tasks: PostgresTaskStore | None = None
        try:
            source_app = CayuApp(
                session_store=source_sessions,
                task_store=source_tasks,
                enable_logging=False,
            )
            source_app.register_provider(_RecordingProvider(), default=True)
            source_app.register_agent(
                AgentSpec(
                    name="worker",
                    model="verified-work-test-model",
                    system_prompt="Preserve the exact PostgreSQL governed context.",
                )
            )
            contract = _contract(contract_id=f"postgres-first-crash-contract-{suffix}")
            await source_tasks.publish_work_contract(contract)
            task = await source_tasks.create_task(
                TaskCreate(
                    task_id=f"postgres-first-crash-task-{suffix}",
                    type="verified-work",
                    work_contract=contract.reference(),
                )
            )
            admitted = await source_app.admit_work_attempt(
                RunRequest(
                    agent_name="worker",
                    task_id=task.id,
                    session_id=f"postgres-first-crash-session-{suffix}",
                    messages=[Message.text("user", "Recover this crashed attempt.")],
                ),
                execution=WorkAttemptExecutionRequest(
                    admission_id=f"postgres-first-crash-admission-{suffix}",
                    claim_id=f"postgres-first-crash-claim-1-{suffix}",
                    attempt_id=f"postgres-first-crash-attempt-{suffix}",
                    interaction_id=f"postgres-first-crash-interaction-{suffix}",
                    worker_id=f"postgres-first-crash-worker-1-{suffix}",
                    generation=1,
                    lease_seconds=1,
                ),
            )
            source_session = await source_sessions.load(admitted.session_id)
            assert source_session is not None
            assert source_session.status is SessionStatus.RUNNING
            deferred = await source_sessions.load_deferred_interaction_input(admitted.session_id)
            assert deferred is not None
            source_messages = [Message.text("user", "Recover this crashed attempt.")]
            assert deferred.initial_transcript_messages == [
                Message.text("system", "Preserve the exact PostgreSQL governed context."),
                *source_messages,
            ]
            with pytest.raises(RuntimeError, match="authenticated projection"):
                await source_sessions.replace_initial_transcript_messages(
                    admitted.session_id,
                    source_messages,
                    [Message.text("system", "Forged replacement."), *source_messages],
                    interaction_id=admitted.interaction_id,
                )
            assert await source_sessions.load_transcript(admitted.session_id) == []

            await asyncio.sleep(1.1)
            replacement_sessions = PostgresSessionStore(
                postgres_dsn,
                schema_mode=SchemaMode.VALIDATE,
            )
            replacement_tasks = PostgresTaskStore(
                postgres_dsn,
                schema_mode=SchemaMode.VALIDATE,
            )
            replacement_app = CayuApp(
                session_store=replacement_sessions,
                task_store=replacement_tasks,
                enable_logging=False,
            )
            replacement_app.register_provider(_RecordingProvider(), default=True)
            replacement_app.register_agent(
                AgentSpec(
                    name="worker",
                    model="verified-work-test-model",
                    system_prompt="Preserve the exact PostgreSQL governed context.",
                )
            )
            recovered = await replacement_app.recover_work_attempt(
                WorkAttemptRecoveryRequest(
                    admission_id=admitted.admission_id,
                    claim_id=f"postgres-first-crash-claim-2-{suffix}",
                    worker_id=f"postgres-first-crash-worker-2-{suffix}",
                    generation=2,
                    lease_seconds=300,
                )
            )

            assert recovered.state is WorkAttemptAdmissionState.ACTIVE
            assert recovered.claim.generation == 2
            assert recovered.attempt == admitted.attempt
            recovered_session = await replacement_sessions.load(admitted.session_id)
            assert recovered_session is not None
            assert recovered_session.status is SessionStatus.RUNNING
            assert recovered_session.run_epoch == source_session.run_epoch + 3
            recovered_checkpoint = await replacement_sessions.load_checkpoint(admitted.session_id)
            assert recovered_checkpoint is not None
            assert INITIAL_TRANSCRIPT_PENDING_CHECKPOINT_KEY not in recovered_checkpoint
            assert (
                await replacement_sessions.load_deferred_interaction_input(admitted.session_id)
                is None
            )
            assert await replacement_sessions.load_transcript(admitted.session_id) == [
                Message.text("system", "Preserve the exact PostgreSQL governed context."),
                Message.text("user", "Recover this crashed attempt."),
            ]
            lifecycle = await replacement_sessions.query_events(
                EventQuery(session_id=admitted.session_id, limit=100)
            )
            assert (
                sum(record.event.type is EventType.INTERACTION_STARTED for record in lifecycle) == 1
            )
            assert not any(
                record.event.type
                in {
                    EventType.INTERACTION_COMPLETED,
                    EventType.INTERACTION_FAILED,
                    EventType.INTERACTION_INTERRUPTED,
                }
                for record in lifecycle
            )
            with pytest.raises(WorkAttemptExecutionClaimLost):
                await source_app.submit_work_attempt_proposal(
                    WorkAttemptProposalRequest(
                        admission_id=admitted.admission_id,
                        claim_id=admitted.claim.claim_id,
                        generation=admitted.claim.generation,
                        proposal=CompletionProposalCreate(
                            proposal_id=f"postgres-first-crash-stale-proposal-{suffix}",
                            attempt_id=admitted.attempt_id,
                            result=_result_reference(),
                            evidence_references=(_artifact_evidence(),),
                        ),
                    )
                )
        finally:
            if replacement_sessions is not None:
                await replacement_sessions.close()
            if replacement_tasks is not None:
                await replacement_tasks.close()
            await source_sessions.close()
            await source_tasks.close()

    asyncio.run(scenario())


def test_postgres_session_interaction_identity_conflict_is_typed(postgres_dsn: str) -> None:
    async def scenario() -> None:
        suffix = uuid4().hex
        first_store = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        second_store = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        contract = _contract(contract_id=f"postgres-interaction-contract-{suffix}")
        await first_store.publish_work_contract(contract)
        tasks = [
            await first_store.create_task(
                TaskCreate(
                    task_id=f"postgres-interaction-task-{suffix}-{index}",
                    type="verified-work",
                    work_contract=contract.reference(),
                )
            )
            for index in range(2)
        ]
        session_id = f"postgres-interaction-session-{suffix}"
        interaction_id = f"postgres-interaction-{suffix}"

        def request(index: int) -> WorkAttemptAdmissionPrepare:
            return WorkAttemptAdmissionPrepare(
                admission_id=f"postgres-interaction-admission-{suffix}-{index}",
                claim_id=f"postgres-interaction-claim-{suffix}-{index}",
                attempt_id=f"postgres-interaction-attempt-{suffix}-{index}",
                task_id=tasks[index].id,
                session_id=session_id,
                interaction_id=interaction_id,
                worker_id=f"postgres-interaction-worker-{suffix}-{index}",
                execution_owner_id=f"postgres-interaction-owner-{suffix}-{index}",
                generation=1,
                lease_seconds=300,
                kind="initial",
                source_request_sha256=_digest(
                    f"postgres-interaction-source-request-{suffix}-{index}"
                ),
                contract=contract.reference(),
                session_invocation=task_invocations[index],
                source_execution_profile_fingerprint=_digest(
                    f"postgres-interaction-profile-{suffix}-{index}"
                ),
            )

        task_invocations = [
            await task_backed_session_invocation(
                first_store,
                task.id,
                session_id,
            )
            for task in tasks
        ]
        await first_store.prepare_work_attempt_admission(request(0))
        with pytest.raises(WorkAttemptAdmissionConflict, match="Session interaction"):
            await second_store.prepare_work_attempt_admission(request(1))
        untouched = await first_store.load_task(tasks[1].id)
        assert untouched is not None
        assert untouched.status is TaskStatus.PENDING
        assert untouched.session_id is None
        assert untouched.worker_id is None
        await first_store.close()
        await second_store.close()

    asyncio.run(scenario())


def test_postgres_concurrent_admissions_have_one_unreleased_session_owner(
    postgres_dsn: str,
) -> None:
    async def scenario() -> None:
        suffix = uuid4().hex
        first_store = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        second_store = PostgresTaskStore(postgres_dsn, schema_mode=SchemaMode.VALIDATE)
        contract = _contract(contract_id=f"postgres-session-owner-{suffix}")
        await first_store.publish_work_contract(contract)
        tasks = [
            await first_store.create_task(
                TaskCreate(
                    task_id=f"postgres-session-owner-task-{suffix}-{index}",
                    type="verified-work",
                    work_contract=contract.reference(),
                )
            )
            for index in range(2)
        ]
        session_id = f"postgres-session-owner-session-{suffix}"
        invocations = [
            await task_backed_session_invocation(
                first_store,
                task.id,
                session_id,
            )
            for task in tasks
        ]

        def request(index: int) -> WorkAttemptAdmissionPrepare:
            return WorkAttemptAdmissionPrepare(
                admission_id=f"postgres-session-owner-admission-{suffix}-{index}",
                claim_id=f"postgres-session-owner-claim-{suffix}-{index}",
                attempt_id=f"postgres-session-owner-attempt-{suffix}-{index}",
                task_id=tasks[index].id,
                session_id=session_id,
                interaction_id=f"postgres-session-owner-interaction-{suffix}-{index}",
                worker_id=f"postgres-session-owner-worker-{suffix}-{index}",
                execution_owner_id=f"postgres-session-owner-process-{suffix}-{index}",
                generation=1,
                lease_seconds=300,
                kind="initial",
                source_request_sha256=_digest(f"postgres-session-owner-request-{suffix}-{index}"),
                contract=contract.reference(),
                session_invocation=invocations[index],
                source_execution_profile_fingerprint=_digest(
                    f"postgres-session-owner-profile-{suffix}-{index}"
                ),
            )

        outcomes = await asyncio.gather(
            first_store.prepare_work_attempt_admission(request(0)),
            second_store.prepare_work_attempt_admission(request(1)),
            return_exceptions=True,
        )
        admitted_indexes = [
            index for index, outcome in enumerate(outcomes) if type(outcome) is WorkAttemptAdmission
        ]
        rejected_indexes = [
            index
            for index, outcome in enumerate(outcomes)
            if isinstance(outcome, WorkAttemptAdmissionConflict)
        ]
        assert len(admitted_indexes) == 1
        assert len(rejected_indexes) == 1
        rejected_index = rejected_indexes[0]
        rejected = outcomes[rejected_index]
        assert "Session already has an unreleased" in str(rejected)
        untouched = await first_store.load_task(tasks[rejected_index].id)
        assert untouched is not None
        assert untouched.status is TaskStatus.PENDING
        assert untouched.session_id is None
        assert untouched.worker_id is None
        await first_store.close()
        await second_store.close()

    asyncio.run(scenario())


def test_postgres_ordinary_fence_and_claim_renewal_have_one_lock_order(
    postgres_dsn: str,
) -> None:
    async def scenario() -> None:
        suffix = uuid4().hex
        store = _PostgresAdmissionLockOrderStore(postgres_dsn)
        contract = _contract(contract_id=f"postgres-lock-order-contract-{suffix}")
        await store.publish_work_contract(contract)
        task = await store.create_task(
            TaskCreate(
                task_id=f"postgres-lock-order-task-{suffix}",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        session_id = f"postgres-lock-order-session-{suffix}"
        prepare = WorkAttemptAdmissionPrepare(
            admission_id=f"postgres-lock-order-admission-{suffix}",
            claim_id=f"postgres-lock-order-claim-{suffix}",
            attempt_id=f"postgres-lock-order-attempt-{suffix}",
            task_id=task.id,
            session_id=session_id,
            interaction_id=f"postgres-lock-order-interaction-{suffix}",
            worker_id=f"postgres-lock-order-worker-{suffix}",
            execution_owner_id=f"postgres-lock-order-owner-{suffix}",
            generation=1,
            lease_seconds=300,
            kind="initial",
            source_request_sha256=_digest(f"postgres-lock-order-source-{suffix}"),
            contract=contract.reference(),
            session_invocation=await task_backed_session_invocation(
                store,
                task.id,
                session_id,
            ),
            source_execution_profile_fingerprint=_digest(f"postgres-lock-order-profile-{suffix}"),
        )
        prepared = await store.prepare_work_attempt_admission(prepare)
        active = await store.activate_work_attempt_admission(
            WorkAttemptAdmissionActivate(
                admission_id=prepared.admission_id,
                claim_id=prepared.claim.claim_id,
                prepare_request_sha256=prepared.prepare_request_sha256,
                session_evidence_sha256=_digest(f"postgres-lock-order-evidence-{suffix}"),
            )
        )
        store.renewal_admission_locked.clear()

        ordinary = asyncio.create_task(
            store.heartbeat(task.id, active.claim.worker_id),
            name="ordinary-heartbeat",
        )
        await asyncio.wait_for(store.ordinary_task_locked.wait(), timeout=10)
        renewal = asyncio.create_task(
            store.renew_work_attempt_execution_claim(
                WorkAttemptExecutionClaimRequest(
                    admission_id=active.admission_id,
                    claim_id=active.claim.claim_id,
                    worker_id=active.claim.worker_id,
                    execution_owner_id=active.claim.execution_owner_id,
                    generation=active.claim.generation,
                    lease_seconds=600,
                )
            ),
            name="claim-renewal",
        )
        await asyncio.wait_for(store.renewal_admission_locked.wait(), timeout=10)
        store.release_ordinary_task_lock.set()

        ordinary_result, renewal_result = await asyncio.wait_for(
            asyncio.gather(ordinary, renewal, return_exceptions=True),
            timeout=10,
        )
        assert isinstance(ordinary_result, WorkAttemptExecutionClaimLost)
        assert isinstance(renewal_result, WorkAttemptAdmission)
        assert renewal_result.claim.lease_expires_at > active.claim.lease_expires_at
        durable_task = await store.load_task(task.id)
        assert durable_task is not None
        assert durable_task.lease_expires_at == renewal_result.claim.lease_expires_at
        await store.close()

    asyncio.run(scenario())
