from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pytest
from tests.core.task_invocation_fixtures import (
    task_backed_session_invocation,
    unattributed_session_invocation_binding,
)

from cayu import (
    CompletionCriterionOutcome,
    CompletionDecision,
    CompletionDecisionApplicationRequest,
    CompletionDecisionCreate,
    CompletionGap,
    CompletionProposal,
    CompletionProposalCreate,
    CompletionResultReference,
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


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


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
    claim = await store.claim_completion_verification(
        CompletionVerificationClaimRequest(
            claim_id=f"claim-{suffix}",
            proposal_id=proposal.proposal_id,
            worker_id=f"verifier-{suffix}",
            verifier=contract.verifier,
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
