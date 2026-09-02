from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pytest
from tests.core.completion_verifier_profile_fixtures import (
    prepare_test_completion_verifier_profile,
)
from tests.core.task_invocation_fixtures import (
    task_backed_session_invocation,
    unattributed_session_invocation_binding,
)

from cayu import (
    CayuApp,
    CompletionCriterionOutcome,
    CompletionDecision,
    CompletionDecisionApplicationRequest,
    CompletionDecisionCreate,
    CompletionGap,
    CompletionProposal,
    CompletionProposalCreate,
    CompletionResultReference,
    CompletionResultResolverRef,
    CompletionSatisfactionBasis,
    CompletionVerdict,
    CompletionVerificationClaim,
    CompletionVerificationClaimRequest,
    CompletionVerifierRef,
    CriterionOutcomeStatus,
    InMemoryTaskStore,
    Task,
    TaskCompletionDecisionRequired,
    TaskCreate,
    TaskQuery,
    TaskStatus,
    WorkAttempt,
    WorkAttemptCreate,
    WorkCompletionConflict,
    WorkContract,
    WorkContractDraft,
    WorkCriterion,
    completion_result_sha256,
    work_contract_from_draft,
)
from cayu.runtime import tasks as tasks_module
from cayu.storage import _postgres_verified_work as postgres_verified_work
from cayu.storage._postgres_verified_work import PostgresVerifiedWorkMixin
from cayu.vaults import SecretRedactor


def _allow_test_postgres_mutation_boundary(monkeypatch) -> None:
    """Let focused owner tests use small fakes instead of a live driver."""

    monkeypatch.setattr(
        postgres_verified_work,
        "_require_quiescent_postgres_mutation_pool",
        lambda _pool, *, allowed_configure=None: None,
    )
    monkeypatch.setattr(
        postgres_verified_work,
        "_require_quiescent_postgres_mutation_connection",
        lambda _connection: None,
    )


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _resolver() -> CompletionResultResolverRef:
    return CompletionResultResolverRef(
        resolver_id="deterministic-result-content",
        version="v1",
        configuration_fingerprint=_digest("deterministic-result-content-v1"),
    )


def _contract(*, contract_id: str) -> WorkContract:
    return work_contract_from_draft(
        WorkContractDraft(
            contract_id=contract_id,
            version=1,
            objective="Verify one deterministic result.",
            criteria=(
                WorkCriterion(
                    criterion_id="result",
                    ordinal=1,
                    description="The result satisfies the deterministic verifier.",
                ),
            ),
            verifier=CompletionVerifierRef(
                verifier_id="deterministic-result",
                version="v1",
                configuration_fingerprint=_digest("deterministic-result-v1"),
            ),
            result_resolver=_resolver(),
        )
    )


async def _prepare_decision(
    store: InMemoryTaskStore,
    contract: WorkContract,
    *,
    verdict: CompletionVerdict,
    suffix: str,
) -> tuple[
    Task,
    WorkAttempt,
    CompletionProposal,
    CompletionVerificationClaim,
    CompletionDecision,
    dict[str, object],
]:
    session_id = f"session:clock:{suffix}"
    task = await store.create_running_task(
        TaskCreate(
            task_id=f"task-{suffix}",
            type="verified-work",
            session_id=session_id,
            work_contract=contract.reference(),
        ),
        session_invocation=unattributed_session_invocation_binding(session_id),
    )
    attempt = await store.begin_work_attempt(
        WorkAttemptCreate(
            attempt_id=f"attempt-{suffix}",
            task_id=task.id,
            session_id=session_id,
            contract=contract.reference(),
            execution_profile_fingerprint=_digest(f"profile-{suffix}"),
        )
    )
    result: dict[str, object] = {"verified": True, "suffix": suffix}
    result_reference = CompletionResultReference(
        kind="task.result",
        reference_id=f"result:{suffix}",
        digest=completion_result_sha256(result),
    )
    proposal = await store.submit_completion_proposal(
        CompletionProposalCreate(
            proposal_id=f"proposal-{suffix}",
            attempt_id=attempt.attempt_id,
            result=result_reference,
        )
    )
    verifier_profile = await prepare_test_completion_verifier_profile(
        store,
        proposal.proposal_id,
    )
    claim = await store.claim_completion_verification(
        CompletionVerificationClaimRequest(
            claim_id=f"claim-{suffix}",
            proposal_id=proposal.proposal_id,
            worker_id=f"verifier-{suffix}",
            verifier=contract.verifier,
            verifier_profile_fingerprint=verifier_profile.profile.fingerprint,
        )
    )
    accepted = verdict is CompletionVerdict.ACCEPTED
    decision = await store.record_completion_decision(
        CompletionDecisionCreate(
            decision_id=f"decision-{suffix}",
            proposal_id=proposal.proposal_id,
            claim_id=claim.claim_id,
            worker_id=claim.worker_id,
            verifier=contract.verifier,
            verifier_profile_fingerprint=verifier_profile.profile.fingerprint,
            verdict=verdict,
            criterion_outcomes=(
                CompletionCriterionOutcome(
                    criterion_id="result",
                    status=(
                        CriterionOutcomeStatus.SATISFIED
                        if accepted
                        else CriterionOutcomeStatus.UNSATISFIED
                    ),
                    reason_code="result.verified" if accepted else "result.rejected",
                    satisfaction_basis=(
                        CompletionSatisfactionBasis.VERIFIER_ASSERTION if accepted else None
                    ),
                ),
            ),
            gaps=(
                ()
                if accepted
                else (
                    CompletionGap(
                        criterion_id="result",
                        code="result.rejected",
                    ),
                )
            ),
        )
    )
    return task, attempt, proposal, claim, decision, result


@pytest.mark.parametrize(
    "verification_time",
    [
        datetime(2001, 1, 1, tzinfo=UTC),
        datetime(2100, 1, 1, tzinfo=UTC),
    ],
    ids=("backward-skew", "forward-skew"),
)
@pytest.mark.parametrize(
    ("verdict", "expected_status"),
    [
        (CompletionVerdict.ACCEPTED, TaskStatus.COMPLETED),
        (CompletionVerdict.BLOCKED, TaskStatus.BLOCKED),
    ],
)
def test_decision_application_keeps_verification_and_task_clock_domains_separate(
    monkeypatch: pytest.MonkeyPatch,
    verification_time: datetime,
    verdict: CompletionVerdict,
    expected_status: TaskStatus,
) -> None:
    lifecycle_time = [datetime(2026, 8, 21, 10, 0, tzinfo=UTC)]
    verification_now = [verification_time]
    lifecycle_clock_fails = [False]
    verification_clock_fails = [False]

    class LifecycleDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            if lifecycle_clock_fails[0]:
                raise RuntimeError("lifecycle clock must not be consulted during replay")
            value = lifecycle_time[0]
            return value if tz is not None else value.replace(tzinfo=None)

    monkeypatch.setattr(tasks_module, "datetime", LifecycleDatetime)

    def verification_clock() -> datetime:
        if verification_clock_fails[0]:
            raise RuntimeError("verification clock must not be consulted during replay")
        return verification_now[0]

    async def scenario() -> None:
        store = InMemoryTaskStore(clock=verification_clock)
        contract = _contract(contract_id=f"clock-domain-{verdict.value}")
        await store.publish_work_contract(contract)
        task, attempt, proposal, claim, decision, result = await _prepare_decision(
            store,
            contract,
            verdict=verdict,
            suffix=f"{verdict.value}-{verification_time.year}",
        )

        assert attempt.started_at == verification_time
        assert proposal.proposed_at == verification_time
        assert claim.claimed_at == verification_time
        assert claim.lease_expires_at == verification_time + timedelta(seconds=300)
        assert decision.decided_at == verification_time
        assert task.updated_at == lifecycle_time[0]

        lifecycle_time[0] += timedelta(minutes=5)
        request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=decision.decision_id,
            idempotency_key=f"apply-{verdict.value}-{verification_time.year}",
            result=result if verdict is CompletionVerdict.ACCEPTED else None,
            result_reference=(proposal.result if verdict is CompletionVerdict.ACCEPTED else None),
        )
        applied = await store.apply_completion_decision(request)
        receipt = await store.load_completion_decision_application_receipt(
            task.id,
            request.idempotency_key,
        )

        assert applied.status is expected_status
        assert applied.updated_at == lifecycle_time[0]
        assert applied.updated_at >= task.updated_at
        assert applied.completed_at == (
            lifecycle_time[0] if verdict is CompletionVerdict.ACCEPTED else None
        )
        assert receipt is not None
        assert receipt.applied_at == lifecycle_time[0]
        assert receipt.task == applied

        original_receipt = receipt
        lifecycle_clock_fails[0] = True
        verification_clock_fails[0] = True
        assert await store.apply_completion_decision(request) == applied
        assert (
            await store.load_completion_decision_application_receipt(
                task.id,
                request.idempotency_key,
            )
            == original_receipt
        )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("verdict", "expected_status"),
    [
        (CompletionVerdict.ACCEPTED, TaskStatus.COMPLETED),
        (CompletionVerdict.BLOCKED, TaskStatus.BLOCKED),
    ],
)
def test_decision_application_clamps_regressed_lifecycle_wall_clock(
    monkeypatch: pytest.MonkeyPatch,
    verdict: CompletionVerdict,
    expected_status: TaskStatus,
) -> None:
    lifecycle_time = [datetime(2026, 8, 21, 10, 0, tzinfo=UTC)]

    class LifecycleDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = lifecycle_time[0]
            return value if tz is not None else value.replace(tzinfo=None)

    monkeypatch.setattr(tasks_module, "datetime", LifecycleDatetime)

    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract(contract_id=f"regressed-lifecycle-{verdict.value}")
        await store.publish_work_contract(contract)
        task, _, proposal, _, decision, result = await _prepare_decision(
            store,
            contract,
            verdict=verdict,
            suffix=f"regressed-lifecycle-{verdict.value}",
        )

        lifecycle_time[0] -= timedelta(hours=1)
        request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=decision.decision_id,
            idempotency_key=f"apply-regressed-lifecycle-{verdict.value}",
            result=result if verdict is CompletionVerdict.ACCEPTED else None,
            result_reference=(proposal.result if verdict is CompletionVerdict.ACCEPTED else None),
        )
        applied = await store.apply_completion_decision(request)
        receipt = await store.load_completion_decision_application_receipt(
            task.id,
            request.idempotency_key,
        )

        assert applied.status is expected_status
        assert applied.updated_at == task.updated_at
        assert applied.completed_at == (
            task.updated_at if verdict is CompletionVerdict.ACCEPTED else None
        )
        assert receipt is not None
        assert receipt.applied_at == task.updated_at
        assert receipt.task == applied

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "binding_path",
    ("create", "create-running", "start", "attach"),
)
def test_every_contracted_session_binding_path_publishes_authority(binding_path: str) -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract(contract_id=f"indexed-{binding_path}")
        await store.publish_work_contract(contract)
        session_id = f"session:indexed:{binding_path}"
        request = TaskCreate(
            task_id=f"indexed-task-{binding_path}",
            type="verified-work",
            session_id=session_id if binding_path in {"create", "create-running"} else None,
            work_contract=contract.reference(),
        )

        if binding_path == "create":
            task = await store.create_task(request)
        elif binding_path == "create-running":
            task = await store.create_running_task(
                request,
                session_invocation=unattributed_session_invocation_binding(session_id),
            )
        else:
            pending = await store.create_task(request)
            session_invocation = await task_backed_session_invocation(
                store,
                pending.id,
                session_id,
            )
            if binding_path == "start":
                task = await store.start_task(
                    pending.id,
                    session_id=session_id,
                    session_invocation=session_invocation,
                )
            else:
                claimed = await store.claim_task(
                    "index-worker",
                    TaskQuery(type=request.type),
                )
                assert claimed is not None
                task = await store.attach_task(
                    pending.id,
                    session_id=session_id,
                    session_invocation=session_invocation,
                    worker_id="index-worker",
                    lease_expires_at=pending.lease_expires_at,
                )

        assert task.session_id == session_id
        assert await store.load_active_work_contract_task_for_session(session_id) == task
        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware"):
            await store.admit_ordinary_session_execution(session_id)

    asyncio.run(scenario())


@pytest.mark.parametrize("first_operation", ("admission", "attachment"))
def test_claimed_contract_attachment_and_ordinary_admission_are_atomic(
    first_operation: str,
) -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract(contract_id="claimed-attachment-race")
        await store.publish_work_contract(contract)
        session_id = "session:claimed-attachment-race"
        pending = await store.create_task(
            TaskCreate(
                task_id="claimed-attachment-race-task",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        session_invocation = await task_backed_session_invocation(
            store,
            pending.id,
            session_id,
        )
        worker_id = "claimed-attachment-race-worker"
        claimed = await store.claim_task(worker_id, TaskQuery(type=pending.type))
        assert claimed is not None
        assert claimed.id == pending.id

        async def attach() -> Task:
            return await store.attach_task(
                claimed.id,
                session_id=session_id,
                session_invocation=session_invocation,
                worker_id=worker_id,
                lease_expires_at=claimed.lease_expires_at,
            )

        # Queue both production entrances behind the real store lock so the
        # test exercises contending operations rather than sequential coroutine
        # execution. ``asyncio.Lock`` grants the first waiter ownership first.
        await store._lock.acquire()
        try:
            if first_operation == "admission":
                admission_task = asyncio.create_task(
                    store.admit_ordinary_session_execution(session_id)
                )
                await asyncio.sleep(0)
                admission_waited = not admission_task.done()
                attachment_task = asyncio.create_task(attach())
                await asyncio.sleep(0)
                attachment_waited = not attachment_task.done()
            else:
                attachment_task = asyncio.create_task(attach())
                await asyncio.sleep(0)
                attachment_waited = not attachment_task.done()
                admission_task = asyncio.create_task(
                    store.admit_ordinary_session_execution(session_id)
                )
                await asyncio.sleep(0)
                admission_waited = not admission_task.done()
        finally:
            store._lock.release()

        admission_result, attachment_result = await asyncio.gather(
            admission_task,
            attachment_task,
            return_exceptions=True,
        )

        assert admission_waited
        assert attachment_waited
        stored = await store.load_task(claimed.id)
        assert stored is not None
        if first_operation == "admission":
            assert admission_result is None
            assert isinstance(attachment_result, WorkCompletionConflict)
            assert stored.status is TaskStatus.CLAIMED
            assert stored.session_id is None
            assert await store.load_active_work_contract_task_for_session(session_id) is None
            with pytest.raises(WorkCompletionConflict, match="prior ordinary session execution"):
                await store.attach_task(
                    claimed.id,
                    session_id=session_id,
                    session_invocation=session_invocation,
                    worker_id=worker_id,
                    lease_expires_at=claimed.lease_expires_at,
                )
        else:
            assert isinstance(admission_result, TaskCompletionDecisionRequired)
            assert not isinstance(attachment_result, BaseException)
            assert attachment_result == stored
            assert stored.status is TaskStatus.RUNNING
            assert stored.session_id == session_id
            assert await store.load_active_work_contract_task_for_session(session_id) == stored
            with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware"):
                await store.admit_ordinary_session_execution(session_id)

    asyncio.run(scenario())


def test_multiple_and_terminal_contracted_tasks_retain_session_authority() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        contract = _contract(contract_id="shared-terminal-authority")
        await store.publish_work_contract(contract)
        session_id = "session:shared-terminal-authority"
        tasks = [
            await store.create_task(
                TaskCreate(
                    task_id=f"shared-terminal-task-{ordinal}",
                    type="verified-work",
                    session_id=session_id,
                    work_contract=contract.reference(),
                )
            )
            for ordinal in range(2)
        ]

        assert set(store._contracted_task_ids_by_session[session_id]) == {task.id for task in tasks}
        selected = await store.load_active_work_contract_task_for_session(session_id)
        assert selected is not None
        assert selected.id in {task.id for task in tasks}

        for task in tasks:
            terminal = await store.cancel_task(task.id)
            assert terminal.status is TaskStatus.CANCELLED

        retained = await store.load_active_work_contract_task_for_session(session_id)
        assert retained is not None
        assert retained.status is TaskStatus.CANCELLED
        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware"):
            await store.admit_ordinary_session_execution(session_id)

    asyncio.run(scenario())


def test_contracted_session_authority_does_not_scan_unrelated_tasks() -> None:
    class ScanDetectingTaskDict(dict[str, Task]):
        values_call_count = 0

        def values(self):
            self.values_call_count += 1
            return super().values()

    async def scenario() -> None:
        store = InMemoryTaskStore()
        for ordinal in range(2_000):
            await store.create_task(
                TaskCreate(
                    task_id=f"unrelated-task-{ordinal}",
                    type="ordinary-work",
                )
            )
        contract = _contract(contract_id="bounded-session-lookup")
        await store.publish_work_contract(contract)
        session_id = "session:bounded-session-lookup"
        contracted = await store.create_task(
            TaskCreate(
                task_id="bounded-session-lookup-task",
                type="verified-work",
                session_id=session_id,
                work_contract=contract.reference(),
            )
        )

        tasks = ScanDetectingTaskDict(store._tasks)
        store._tasks = tasks
        assert await store.load_active_work_contract_task_for_session(session_id) == contracted
        with pytest.raises(TaskCompletionDecisionRequired, match="verifier-aware"):
            await store.admit_ordinary_session_execution(session_id)
        await store.admit_ordinary_session_execution("session:uncontracted")
        assert tasks.values_call_count == 0

    asyncio.run(scenario())


def test_postgres_task_store_rejects_unproven_mutation_connection_class() -> None:
    from psycopg import AsyncConnection
    from psycopg_pool import AsyncConnectionPool

    from cayu import PostgresTaskStore

    class CancellationResistantConnection(AsyncConnection):
        pass

    pool = AsyncConnectionPool(
        "",
        connection_class=CancellationResistantConnection,
        open=False,
    )

    with pytest.raises(
        TypeError,
        match="require the built-in psycopg AsyncConnection implementation",
    ):
        PostgresTaskStore(pool=pool)


def test_postgres_task_store_rejects_unproven_mutation_pool_subclass() -> None:
    from psycopg_pool import AsyncConnectionPool

    from cayu import PostgresTaskStore

    class CancellationResistantPool(AsyncConnectionPool):
        pass

    pool = CancellationResistantPool("", open=False)

    with pytest.raises(
        TypeError,
        match="require the built-in AsyncConnectionPool implementation",
    ):
        PostgresTaskStore(pool=pool)


@pytest.mark.parametrize(
    ("callback_option", "message"),
    [
        ("check", "pool check callback"),
        ("configure", "unverified pool configure callback"),
        ("reset", "pool reset callback"),
        ("reconnect_failed", "pool reconnect callback"),
    ],
)
def test_postgres_task_store_rejects_unproven_pool_callbacks(
    callback_option: str,
    message: str,
) -> None:
    from psycopg_pool import AsyncConnectionPool

    from cayu import PostgresTaskStore

    async def callback(_owner) -> None:
        await asyncio.Event().wait()

    pool = AsyncConnectionPool(
        "",
        open=False,
        **{callback_option: callback},
    )

    with pytest.raises(TypeError, match=message):
        PostgresTaskStore(pool=pool)


def test_postgres_task_store_rejects_connection_behavior_factories() -> None:
    from psycopg import AsyncCursor
    from psycopg_pool import AsyncConnectionPool

    from cayu import PostgresTaskStore

    class CancellationResistantCursor(AsyncCursor):
        pass

    pool = AsyncConnectionPool(
        "",
        open=False,
        kwargs={"cursor_factory": CancellationResistantCursor},
    )

    with pytest.raises(TypeError, match="connection behavior factories"):
        PostgresTaskStore(pool=pool)


def test_postgres_task_store_revalidates_pool_before_lazy_readiness() -> None:
    from psycopg_pool import AsyncConnectionPool

    from cayu import PostgresTaskStore

    pool = AsyncConnectionPool("", open=False)
    store = PostgresTaskStore(pool=pool)

    async def cancellation_resistant_check(_connection) -> None:
        await asyncio.Event().wait()

    pool._check = cancellation_resistant_check

    async def scenario() -> None:
        with pytest.raises(TypeError, match="pool check callback"):
            await store.load_task("pool-drift-must-fail-before-readiness")

    asyncio.run(scenario())


def test_postgres_task_store_accepts_static_nonbehavioral_pool_options() -> None:
    from psycopg_pool import AsyncConnectionPool

    from cayu import PostgresTaskStore

    pool = AsyncConnectionPool(
        "",
        open=False,
        kwargs={"prepare_threshold": None},
    )

    assert PostgresTaskStore(pool=pool)._pool is pool


def test_postgres_verified_work_rollback_failure_preserves_primary_and_fences_connection(
    monkeypatch,
) -> None:
    _allow_test_postgres_mutation_boundary(monkeypatch)
    primary = ValueError("primary mutation failure")
    rollback_failure = RuntimeError("rollback failure")

    class Cursor:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    class Connection:
        def __init__(self) -> None:
            self.closed = False
            self.pgconn = PgConnection(self)

        def cursor(self):
            return Cursor()

        async def commit(self) -> None:
            raise AssertionError("failing operation must not commit")

        async def rollback(self) -> None:
            raise rollback_failure

        async def close(self) -> None:
            raise AssertionError("rollback fencing must not call close()")

    class PgConnection:
        def __init__(self, connection: Connection) -> None:
            self.connection = connection
            self.finish_count = 0

        def finish(self) -> None:
            self.finish_count += 1
            self.connection.closed = True

    connection = Connection()

    class ConnectionContext:
        async def __aenter__(self):
            return connection

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    class Pool:
        def connection(self):
            return ConnectionContext()

    class Store(PostgresVerifiedWorkMixin):
        _pool = Pool()

    async def scenario() -> None:
        async def operation(conn, cursor) -> None:
            del conn, cursor
            raise primary

        with pytest.raises(BaseExceptionGroup) as captured:
            await Store()._run_verified_work_mutation(operation)

        assert captured.value.exceptions == (primary, rollback_failure)
        assert connection.closed is True
        assert connection.pgconn.finish_count == 1
        assert rollback_failure.__context__ is None

    asyncio.run(scenario())


def test_postgres_cancelled_mutation_closes_exact_connection_before_return(
    monkeypatch,
) -> None:
    _allow_test_postgres_mutation_boundary(monkeypatch)
    monkeypatch.setattr(
        postgres_verified_work,
        "_POSTGRES_MUTATION_CANCELLATION_GRACE_SECONDS",
        0.01,
    )

    async def scenario() -> None:
        operation_started = asyncio.Event()
        cancellation_received = asyncio.Event()
        connection_closed = asyncio.Event()

        class Cursor:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

        class Connection:
            def __init__(self) -> None:
                self.closed = False
                self.commit_count = 0
                self.rollback_count = 0
                self.pgconn = PgConnection(self)

            def cursor(self):
                return Cursor()

            async def commit(self) -> None:
                self.commit_count += 1

            async def rollback(self) -> None:
                self.rollback_count += 1

        class PgConnection:
            def __init__(self, connection: Connection) -> None:
                self.connection = connection

            def finish(self) -> None:
                self.connection.closed = True
                connection_closed.set()

        connection = Connection()

        class ConnectionContext:
            async def __aenter__(self):
                return connection

            async def __aexit__(self, exc_type, exc, traceback):
                return False

        class Pool:
            def connection(self):
                return ConnectionContext()

        class Store(PostgresVerifiedWorkMixin):
            _pool = Pool()

        async def operation(conn, cursor) -> None:
            del conn, cursor
            operation_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # Model a driver operation that cannot finish its cancellation
                # unwind until the exact checked-out connection is discarded.
                cancellation_received.set()
                await connection_closed.wait()
                raise

        caller = asyncio.create_task(Store()._run_verified_work_mutation(operation))
        await operation_started.wait()
        caller.cancel("stop blocked Postgres mutation")
        assert caller.cancelling() == 1
        await cancellation_received.wait()
        caller.cancel("repeat blocked Postgres mutation cancellation")
        assert caller.cancelling() == 2
        with pytest.raises(asyncio.CancelledError, match="stop blocked Postgres mutation"):
            await caller

        assert caller.cancelled()
        assert caller.cancelling() == 2
        assert connection.closed is True
        assert connection.commit_count == 0
        assert connection.rollback_count == 1
        assert not any(
            task.get_name() == "cayu-postgres-store-mutation" for task in asyncio.all_tasks()
        )

    asyncio.run(scenario())


def test_postgres_cancelled_mutation_preserves_owner_and_abort_failures(
    monkeypatch,
) -> None:
    _allow_test_postgres_mutation_boundary(monkeypatch)
    monkeypatch.setattr(
        postgres_verified_work,
        "_POSTGRES_MUTATION_CANCELLATION_GRACE_SECONDS",
        0.01,
    )

    primary_rollback_failure = RuntimeError("rollback after cancellation failed")
    abort_failure = OSError("physical abort reported failure")

    async def scenario() -> None:
        operation_started = asyncio.Event()
        connection_aborted = asyncio.Event()

        class Cursor:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

        class Connection:
            def __init__(self) -> None:
                self.closed = False
                self.pgconn = PgConnection(self)

            def cursor(self):
                return Cursor()

            async def commit(self) -> None:
                raise AssertionError("cancelled mutation must not commit")

            async def rollback(self) -> None:
                await connection_aborted.wait()
                raise primary_rollback_failure

            async def close(self) -> None:
                self.closed = True

        class PgConnection:
            def __init__(self, connection: Connection) -> None:
                self.connection = connection

            def finish(self) -> None:
                self.connection.closed = True
                connection_aborted.set()
                raise abort_failure

        connection = Connection()

        class ConnectionContext:
            async def __aenter__(self):
                return connection

            async def __aexit__(self, exc_type, exc, traceback):
                return False

        class Pool:
            def connection(self):
                return ConnectionContext()

        class Store(PostgresVerifiedWorkMixin):
            _pool = Pool()

        async def operation(conn, cursor) -> None:
            del conn, cursor
            operation_started.set()
            await asyncio.Event().wait()

        caller = asyncio.create_task(Store()._run_verified_work_mutation(operation))
        await operation_started.wait()
        caller.cancel("cancel with compound settlement failures")

        with pytest.raises(asyncio.CancelledError) as captured:
            await caller

        assert captured.value.args == ("cancel with compound settlement failures",)
        cause = captured.value.__cause__
        assert isinstance(cause, BaseExceptionGroup)
        assert len(cause.exceptions) == 2
        owner_failure, retained_abort_failure = cause.exceptions
        assert isinstance(owner_failure, BaseExceptionGroup)
        assert owner_failure.exceptions == (primary_rollback_failure,)
        assert retained_abort_failure is abort_failure
        assert connection.closed is True
        assert not any(
            task.get_name() == "cayu-postgres-store-mutation" for task in asyncio.all_tasks()
        )

    asyncio.run(scenario())


def test_public_boundary_preserves_redacted_postgres_cancellation_settlement_failures(
    monkeypatch,
) -> None:
    _allow_test_postgres_mutation_boundary(monkeypatch)
    monkeypatch.setattr(
        postgres_verified_work,
        "_POSTGRES_MUTATION_CANCELLATION_GRACE_SECONDS",
        0.01,
    )
    secret = "postgres.settlement.secret"

    async def scenario() -> None:
        operation_started = asyncio.Event()
        connection_aborted = asyncio.Event()

        class Cursor:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

        class Connection:
            def __init__(self) -> None:
                self.closed = False
                self.pgconn = PgConnection(self)

            def cursor(self):
                return Cursor()

            async def commit(self) -> None:
                raise AssertionError("cancelled mutation must not commit")

            async def rollback(self) -> None:
                await connection_aborted.wait()
                raise RuntimeError(f"rollback failed with {secret}")

            async def close(self) -> None:
                self.closed = True

        class PgConnection:
            def __init__(self, connection: Connection) -> None:
                self.connection = connection

            def finish(self) -> None:
                self.connection.closed = True
                connection_aborted.set()
                raise OSError(f"abort failed with {secret}")

        connection = Connection()

        class ConnectionContext:
            async def __aenter__(self):
                return connection

            async def __aexit__(self, exc_type, exc, traceback):
                return False

        class Pool:
            def connection(self):
                return ConnectionContext()

        class Store(PostgresVerifiedWorkMixin, InMemoryTaskStore):
            verified_work_mutations_are_cancellation_quiescent = True
            _pool = Pool()

            async def publish_work_contract(self, contract: WorkContract) -> WorkContract:
                async def operation(conn, cursor) -> WorkContract:
                    del conn, cursor
                    operation_started.set()
                    await asyncio.Event().wait()
                    return contract

                return await self._run_verified_work_mutation(operation)

        app = CayuApp(
            task_store=Store(),
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        draft = WorkContractDraft(
            contract_id="postgres-settlement-boundary-contract",
            version=1,
            objective="Preserve bounded settlement evidence.",
            criteria=(
                WorkCriterion(
                    criterion_id="settlement",
                    ordinal=1,
                    description="Settlement failures remain observable.",
                ),
            ),
            verifier=CompletionVerifierRef(
                verifier_id="deterministic-settlement",
                version="v1",
                configuration_fingerprint=_digest("deterministic-settlement-v1"),
            ),
            result_resolver=_resolver(),
        )
        caller = asyncio.create_task(app.create_work_contract(draft))
        await operation_started.wait()
        caller.cancel("cancel public Postgres mutation")

        with pytest.raises(asyncio.CancelledError) as captured:
            await caller

        assert caller.cancelled()
        assert captured.value.args == ("cancel public Postgres mutation",)
        cause = captured.value.__cause__
        assert isinstance(cause, BaseExceptionGroup)
        pending: list[BaseException] = [cause]
        leaves: list[BaseException] = []
        while pending:
            candidate = pending.pop()
            if isinstance(candidate, BaseExceptionGroup):
                pending.extend(reversed(candidate.exceptions))
            else:
                leaves.append(candidate)
        assert len(leaves) == 2
        assert all(secret not in str(leaf) and secret not in repr(leaf) for leaf in leaves)
        assert connection.closed is True

    asyncio.run(scenario())
