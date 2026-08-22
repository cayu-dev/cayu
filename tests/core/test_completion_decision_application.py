from __future__ import annotations

import asyncio
import warnings
from datetime import timedelta
from hashlib import sha256

import pytest
from tests.core.task_invocation_fixtures import unattributed_session_invocation_binding
from tests.provider_traceback_assertions import is_cayu_source_filename

from cayu import CayuApp
from cayu.runtime.invocation import InvocationOrigin, InvocationOriginTrust
from cayu.runtime.sessions import InMemorySessionStore
from cayu.runtime.tasks import (
    CompletionDecisionApplicationReceipt,
    InMemoryTaskStore,
    Task,
    TaskCreate,
    TaskStatus,
    TaskStore,
)
from cayu.runtime.work_contracts import (
    CompletionContinuationPolicy,
    CompletionCriterionOutcome,
    CompletionDecision,
    CompletionDecisionApplicationRequest,
    CompletionDecisionCreate,
    CompletionGap,
    CompletionProposalCreate,
    CompletionRejectionAction,
    CompletionResultReference,
    CompletionSatisfactionBasis,
    CompletionVerdict,
    CompletionVerificationClaim,
    CompletionVerificationClaimRequest,
    CompletionVerifierRef,
    CriterionOutcomeStatus,
    WorkAttemptCreate,
    WorkCompletionConflict,
    WorkContract,
    WorkContractDraft,
    WorkCriterion,
    completion_decision_request_sha256,
    completion_gap_fingerprint,
    completion_result_sha256,
    completion_verification_claim_authority_sha256,
    copy_completion_decision_application_request,
    work_contract_from_draft,
)
from cayu.storage.sqlite import SQLiteTaskStore
from cayu.vaults import SecretRedactor


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _verifier() -> CompletionVerifierRef:
    return CompletionVerifierRef(
        verifier_id="application-test",
        version="v1",
        configuration_fingerprint=_digest("application-test-v1"),
    )


def _contract(
    *,
    contract_id: str = "application-contract",
    rejection_action: CompletionRejectionAction = CompletionRejectionAction.CONTINUE,
    max_attempts: int = 3,
    max_repeated_gap_count: int = 2,
) -> WorkContract:
    return work_contract_from_draft(
        WorkContractDraft(
            contract_id=contract_id,
            version=1,
            objective="Publish an independently verified result.",
            criteria=(
                WorkCriterion(
                    criterion_id="ready",
                    ordinal=1,
                    description="The result is ready.",
                ),
            ),
            verifier=_verifier(),
            continuation_policy=CompletionContinuationPolicy(
                rejection_action=rejection_action,
                max_attempts=max_attempts,
                max_repeated_gap_count=max_repeated_gap_count,
            ),
        )
    )


def _result(suffix: str) -> dict[str, object]:
    return {"artifact_id": f"artifact:{suffix}"}


def _result_reference(suffix: str) -> CompletionResultReference:
    return CompletionResultReference(
        kind="session.output",
        reference_id=f"session:{suffix}",
        digest=completion_result_sha256(_result(suffix)),
    )


def _decision_request(
    *,
    verdict: CompletionVerdict,
    proposal_id: str,
    claim_id: str,
    decision_id: str,
) -> CompletionDecisionCreate:
    if verdict is CompletionVerdict.ACCEPTED:
        outcome = CompletionCriterionOutcome(
            criterion_id="ready",
            status=CriterionOutcomeStatus.SATISFIED,
            reason_code="ready.confirmed",
            satisfaction_basis=CompletionSatisfactionBasis.VERIFIER_ASSERTION,
        )
        gaps = ()
    else:
        outcome = CompletionCriterionOutcome(
            criterion_id="ready",
            status=CriterionOutcomeStatus.UNSATISFIED,
            reason_code="ready.missing",
        )
        gaps = (
            CompletionGap(
                criterion_id="ready",
                code="ready.missing",
                summary="The result is not ready.",
            ),
        )
    return CompletionDecisionCreate(
        decision_id=decision_id,
        proposal_id=proposal_id,
        claim_id=claim_id,
        worker_id="verifier-worker",
        verifier=_verifier(),
        verdict=verdict,
        criterion_outcomes=(outcome,),
        gaps=gaps,
    )


async def _persist_decision(
    store: TaskStore,
    *,
    task: Task,
    ordinal: int,
    verdict: CompletionVerdict,
    result_reference: CompletionResultReference | None = None,
) -> tuple[str, CompletionResultReference]:
    suffix = str(ordinal)
    attempt = await store.begin_work_attempt(
        WorkAttemptCreate(
            attempt_id=f"attempt-{suffix}",
            task_id=task.id,
            session_id=task.session_id or "missing-session",
            contract=task.work_contract or _contract().reference(),
            execution_profile_fingerprint=_digest(f"profile-{suffix}"),
        )
    )
    proposal_result = result_reference or _result_reference(suffix)
    proposal = await store.submit_completion_proposal(
        CompletionProposalCreate(
            proposal_id=f"proposal-{suffix}",
            attempt_id=attempt.attempt_id,
            result=proposal_result,
        )
    )
    claim = await store.claim_completion_verification(
        CompletionVerificationClaimRequest(
            claim_id=f"claim-{suffix}",
            proposal_id=proposal.proposal_id,
            worker_id="verifier-worker",
            verifier=_verifier(),
        )
    )
    decision = await store.record_completion_decision(
        _decision_request(
            verdict=verdict,
            proposal_id=proposal.proposal_id,
            claim_id=claim.claim_id,
            decision_id=f"decision-{suffix}",
        )
    )
    assert decision.claim_authority_sha256 == (
        completion_verification_claim_authority_sha256(claim)
    )
    return decision.decision_id, proposal_result


async def _running_task(
    store: TaskStore,
    *,
    contract: WorkContract | None = None,
    task_input: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> Task:
    contract = contract or _contract()
    await store.publish_work_contract(contract)
    return await store.create_running_task(
        TaskCreate(
            task_id="application-task",
            type="verified-work",
            session_id="session:application",
            input={} if task_input is None else task_input,
            metadata={} if metadata is None else metadata,
            work_contract=contract.reference(),
        ),
        session_invocation=unattributed_session_invocation_binding("session:application"),
    )


def _app(store: TaskStore) -> CayuApp:
    return CayuApp(
        session_store=InMemorySessionStore(),
        task_store=store,
        enable_logging=False,
    )


def _assert_secret_absent_from_cayu_error(error: BaseException, secret: str) -> None:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        assert secret not in str(current)
        assert secret not in repr(current)
        traceback = current.__traceback__
        while traceback is not None:
            if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
                retained = {
                    name: type(value).__name__
                    for name, value in traceback.tb_frame.f_locals.items()
                    if secret in repr(value)
                }
                assert retained == {}, (
                    traceback.tb_frame.f_code.co_filename,
                    traceback.tb_frame.f_code.co_name,
                    retained,
                )
            traceback = traceback.tb_next
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)


def test_app_applies_accepted_decision_and_exactly_replays_receipt() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = _app(store)
        task = await _running_task(store)
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )
        request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=decision_id,
            idempotency_key="application-accepted",
            result=_result("1"),
            result_reference=reference,
        )

        completed = await app.apply_completion_decision(request)
        replayed = await app.apply_completion_decision(request)

        assert completed.status is TaskStatus.COMPLETED
        assert completed.result == _result("1")
        assert replayed == completed

    asyncio.run(scenario())


def test_sqlite_app_applies_accepted_decision_and_exactly_replays_receipt() -> None:
    async def scenario() -> None:
        store = SQLiteTaskStore(":memory:")
        try:
            app = _app(store)
            task = await _running_task(store)
            decision_id, reference = await _persist_decision(
                store,
                task=task,
                ordinal=1,
                verdict=CompletionVerdict.ACCEPTED,
            )
            request = CompletionDecisionApplicationRequest(
                task_id=task.id,
                decision_id=decision_id,
                idempotency_key="sqlite-application-accepted",
                result=_result("1"),
                result_reference=reference,
            )

            completed = await app.apply_completion_decision(request)
            assert completed.status is TaskStatus.COMPLETED
            assert await app.apply_completion_decision(request) == completed
        finally:
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("verdict", "contract", "expected_status", "expected_reason"),
    [
        (
            CompletionVerdict.BLOCKED,
            _contract(),
            TaskStatus.BLOCKED,
            "work_contract_blocked",
        ),
        (
            CompletionVerdict.NEEDS_REVIEW,
            _contract(),
            TaskStatus.NEEDS_ATTENTION,
            "work_contract_needs_review",
        ),
        (
            CompletionVerdict.REJECTED,
            _contract(rejection_action=CompletionRejectionAction.INTERRUPT),
            TaskStatus.PAUSED,
            "work_contract_rejected",
        ),
        (
            CompletionVerdict.REJECTED,
            _contract(max_attempts=1),
            TaskStatus.NEEDS_ATTENTION,
            "work_contract_attempt_limit",
        ),
    ],
)
def test_app_applies_explicit_nonaccepted_transition_semantics(
    verdict: CompletionVerdict,
    contract: WorkContract,
    expected_status: TaskStatus,
    expected_reason: str,
) -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = _app(store)
        task = await _running_task(store, contract=contract)
        decision_id, _ = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=verdict,
        )

        applied = await app.apply_completion_decision(
            CompletionDecisionApplicationRequest(
                task_id=task.id,
                decision_id=decision_id,
                idempotency_key=f"application-{verdict.value}-{expected_reason}",
            )
        )

        assert applied.status is expected_status
        assert applied.status_reason == expected_reason
        assert applied.worker_id is None
        assert applied.lease_expires_at is None

    asyncio.run(scenario())


def test_app_applies_repeated_gap_limit_on_second_matching_rejection() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = _app(store)
        contract = _contract(max_repeated_gap_count=1)
        task = await _running_task(store, contract=contract)
        first_id, _ = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.REJECTED,
        )
        continued = await app.apply_completion_decision(
            CompletionDecisionApplicationRequest(
                task_id=task.id,
                decision_id=first_id,
                idempotency_key="application-first-gap",
            )
        )
        second_id, _ = await _persist_decision(
            store,
            task=continued,
            ordinal=2,
            verdict=CompletionVerdict.REJECTED,
        )

        held = await app.apply_completion_decision(
            CompletionDecisionApplicationRequest(
                task_id=task.id,
                decision_id=second_id,
                idempotency_key="application-repeated-gap",
            )
        )

        assert held.status is TaskStatus.NEEDS_ATTENTION
        assert held.status_reason == "work_contract_repeated_gap_limit"

    asyncio.run(scenario())


def test_app_replays_old_application_snapshot_after_later_decision() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = _app(store)
        task = await _running_task(store)
        rejected_id, _ = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.REJECTED,
        )
        rejected_request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=rejected_id,
            idempotency_key="application-rejected",
        )
        rejected_snapshot = await app.apply_completion_decision(rejected_request)
        accepted_id, accepted_reference = await _persist_decision(
            store,
            task=rejected_snapshot,
            ordinal=2,
            verdict=CompletionVerdict.ACCEPTED,
        )
        completed = await app.apply_completion_decision(
            CompletionDecisionApplicationRequest(
                task_id=task.id,
                decision_id=accepted_id,
                idempotency_key="application-accepted-later",
                result=_result("2"),
                result_reference=accepted_reference,
            )
        )

        replayed_rejection = await app.apply_completion_decision(rejected_request)

        assert rejected_snapshot.status is TaskStatus.RUNNING
        assert completed.status is TaskStatus.COMPLETED
        assert replayed_rejection == rejected_snapshot
        assert replayed_rejection != completed

    asyncio.run(scenario())


def test_app_rejects_conflicting_application_tuple_without_mutation() -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = _app(store)
        task = await _running_task(store)
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )
        request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=decision_id,
            idempotency_key="application-conflict",
            result=_result("1"),
            result_reference=reference,
        )
        await app.apply_completion_decision(request)

        with pytest.raises(WorkCompletionConflict, match="another request"):
            await app.apply_completion_decision(
                request.model_copy(
                    update={
                        "result": _result("other"),
                        "result_reference": _result_reference("other"),
                    }
                )
            )

        receipt = await store.load_completion_decision_application_receipt(
            task.id,
            request.idempotency_key,
        )
        assert receipt is not None
        assert receipt.task.result == _result("1")

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "changed_field",
    [
        "task_id",
        "decision_id",
        "idempotency_key",
        "result",
        "result_reference_kind",
        "result_reference_id",
        "result_reference_digest",
    ],
)
def test_app_exact_replay_rejects_each_changed_authority_field(
    changed_field: str,
) -> None:
    async def scenario() -> None:
        store = InMemoryTaskStore()
        app = _app(store)
        task = await _running_task(store)
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )
        request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=decision_id,
            idempotency_key="application-exact-tuple",
            result=_result("1"),
            result_reference=reference,
        )
        completed = await app.apply_completion_decision(request)
        changed: dict[str, object]
        if changed_field == "task_id":
            changed = {"task_id": "different-task"}
        elif changed_field == "decision_id":
            changed = {"decision_id": "different-decision"}
        elif changed_field == "idempotency_key":
            changed = {"idempotency_key": "different-application-key"}
        elif changed_field == "result":
            changed = {"result": _result("different")}
        else:
            reference_update = {
                "result_reference_kind": {"kind": "artifact.version"},
                "result_reference_id": {"reference_id": "artifact:different"},
                "result_reference_digest": {"digest": _digest("different-result")},
            }[changed_field]
            changed = {"result_reference": reference.model_copy(update=reference_update)}

        with pytest.raises(ValueError):
            await app.apply_completion_decision(request.model_copy(update=changed))

        receipt = await store.load_completion_decision_application_receipt(
            task.id,
            request.idempotency_key,
        )
        assert receipt is not None
        assert receipt.task == completed

    asyncio.run(scenario())


def test_app_reconciles_commit_then_raise_from_exact_receipt() -> None:
    class CommitThenRaiseStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def apply_completion_decision(
            self,
            request: CompletionDecisionApplicationRequest,
        ) -> Task:
            await super().apply_completion_decision(request)
            raise ConnectionError("application acknowledgement lost")

    async def scenario() -> None:
        store = CommitThenRaiseStore()
        app = _app(store)
        task = await _running_task(store)
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )

        completed = await app.apply_completion_decision(
            CompletionDecisionApplicationRequest(
                task_id=task.id,
                decision_id=decision_id,
                idempotency_key="application-ack-loss",
                result=_result("1"),
                result_reference=reference,
            )
        )

        assert completed.status is TaskStatus.COMPLETED
        assert completed.result == _result("1")

    asyncio.run(scenario())


def test_app_isolates_authoritative_request_from_store_mutation() -> None:
    class MutateAfterCommitStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def apply_completion_decision(
            self,
            request: CompletionDecisionApplicationRequest,
        ) -> Task:
            await super().apply_completion_decision(
                copy_completion_decision_application_request(request)
            )
            if request.result is None:  # pragma: no cover - construction invariant
                raise AssertionError("Accepted application result was not retained.")
            request.result["artifact_id"] = "artifact:extension-mutated"
            raise ConnectionError("application acknowledgement lost")

    async def scenario() -> None:
        store = MutateAfterCommitStore()
        app = _app(store)
        task = await _running_task(store)
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )
        request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=decision_id,
            idempotency_key="application-mutating-store",
            result=_result("1"),
            result_reference=reference,
        )

        completed = await app.apply_completion_decision(request)

        assert completed.status is TaskStatus.COMPLETED
        assert completed.result == _result("1")
        assert request.result == _result("1")

    asyncio.run(scenario())


@pytest.mark.parametrize("exact_replay", [False, True])
def test_app_rejects_secret_bearing_loaded_authority_before_mutation_or_replay(
    exact_replay: bool,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "application-loaded-authority-secret-canary"

    async def scenario() -> tuple[BaseException, Task]:
        store = InMemoryTaskStore()
        contract = _contract(contract_id=secret)
        task = await _running_task(store, contract=contract)
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=decision_id,
            idempotency_key="application-loaded-secret-authority",
            result=_result("1"),
            result_reference=reference,
        )
        if exact_replay:
            completed = await _app(store).apply_completion_decision(request)
            assert completed.status is TaskStatus.COMPLETED

        with pytest.raises(ValueError, match="public identity") as captured:
            await app.apply_completion_decision(request)

        receipt = await store.load_completion_decision_application_receipt(
            task.id,
            request.idempotency_key,
        )
        assert (receipt is not None) is exact_replay
        persisted = await store.load_task(task.id)
        assert persisted is not None
        return captured.value, persisted

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error, persisted = asyncio.run(scenario())

    assert persisted.status is (TaskStatus.COMPLETED if exact_replay else TaskStatus.RUNNING)
    _assert_secret_absent_from_cayu_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


@pytest.mark.parametrize("exact_replay", [False, True])
def test_app_detaches_sensitive_malformed_loaded_authority(
    exact_replay: bool,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "application-malformed-authority-secret-canary"

    class MalformedAuthorityStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.forged_decision: CompletionDecision | None = None
            self.application_calls = 0

        async def load_completion_decision(
            self,
            decision_id: str,
        ) -> CompletionDecision | None:
            if self.forged_decision is not None and self.forged_decision.decision_id == decision_id:
                return self.forged_decision.model_copy(deep=True)
            return await super().load_completion_decision(decision_id)

        async def load_completion_decision_for_proposal(
            self,
            proposal_id: str,
        ) -> CompletionDecision | None:
            if self.forged_decision is not None and self.forged_decision.proposal_id == proposal_id:
                return self.forged_decision.model_copy(deep=True)
            return await super().load_completion_decision_for_proposal(proposal_id)

        async def apply_completion_decision(
            self,
            request: CompletionDecisionApplicationRequest,
        ) -> Task:
            self.application_calls += 1
            return await super().apply_completion_decision(request)

    async def scenario() -> tuple[BaseException, Task, int]:
        store = MalformedAuthorityStore()
        task = await _running_task(store)
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )
        request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=decision_id,
            idempotency_key="application-malformed-authority",
            result=_result("1"),
            result_reference=reference,
        )
        if exact_replay:
            completed = await _app(store).apply_completion_decision(request)
            assert completed.status is TaskStatus.COMPLETED

        durable = await store.load_completion_decision(decision_id)
        assert durable is not None
        malformed_outcome = CompletionCriterionOutcome(
            criterion_id="outside-contract",
            status=CriterionOutcomeStatus.SATISFIED,
            reason_code="outside.confirmed",
            satisfaction_basis=CompletionSatisfactionBasis.VERIFIER_ASSERTION,
            summary=secret,
        )
        malformed_request = CompletionDecisionCreate(
            decision_id=durable.decision_id,
            proposal_id=durable.proposal_id,
            claim_id=durable.claim_id,
            worker_id=durable.worker_id,
            verifier=durable.verifier,
            verdict=durable.verdict,
            criterion_outcomes=(malformed_outcome,),
        )
        store.forged_decision = durable.model_copy(
            update={
                "criterion_outcomes": malformed_request.criterion_outcomes,
                "request_sha256": completion_decision_request_sha256(malformed_request),
                "gap_fingerprint": completion_gap_fingerprint(malformed_request),
            }
        )
        calls_before = store.application_calls
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )

        with pytest.raises(WorkCompletionConflict, match="invalid integrity") as captured:
            await app.apply_completion_decision(request)

        persisted = await store.load_task(task.id)
        assert persisted is not None
        return captured.value, persisted, store.application_calls - calls_before

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error, persisted, application_calls = asyncio.run(scenario())

    assert application_calls == 0
    assert persisted.status is (TaskStatus.COMPLETED if exact_replay else TaskStatus.RUNNING)
    _assert_secret_absent_from_cayu_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


@pytest.mark.parametrize("exact_replay", [False, True])
@pytest.mark.parametrize(
    "claim_variant",
    [
        "missing",
        "claim_id",
        "proposal_id",
        "worker_id",
        "execution_owner_id",
        "execution_timeout_seconds",
        "verifier",
        "attempt_number",
        "request_sha256",
        "claimed_at",
        "lease_expires_at",
        "decision_before_claim",
        "decision_at_expiry",
        "decision_after_expiry",
    ],
)
def test_app_requires_exact_durable_verification_claim_authority(
    exact_replay: bool,
    claim_variant: str,
) -> None:
    class ConflictingClaimStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.claim_variant: str | None = None
            self.application_calls = 0

        async def load_completion_verification_claim(
            self,
            proposal_id: str,
        ) -> CompletionVerificationClaim | None:
            claim = await super().load_completion_verification_claim(proposal_id)
            if claim is None or self.claim_variant is None:
                return claim
            if self.claim_variant == "missing":
                return None
            update: dict[str, object]
            if self.claim_variant == "claim_id":
                update = {"claim_id": "claim-substituted"}
            elif self.claim_variant == "proposal_id":
                update = {"proposal_id": "proposal-substituted"}
            elif self.claim_variant == "worker_id":
                update = {"worker_id": "verifier-worker-substituted"}
            elif self.claim_variant == "execution_owner_id":
                update = {"execution_owner_id": "execution-owner-substituted"}
            elif self.claim_variant == "execution_timeout_seconds":
                update = {"execution_timeout_seconds": 1.0}
            elif self.claim_variant == "verifier":
                update = {
                    "verifier": CompletionVerifierRef(
                        verifier_id="application-test-substituted",
                        version="v1",
                        configuration_fingerprint=_digest("application-test-substituted-v1"),
                    )
                }
            elif self.claim_variant == "attempt_number":
                update = {"attempt_number": claim.attempt_number + 1}
            elif self.claim_variant == "request_sha256":
                update = {"request_sha256": _digest("substituted-claim-request")}
            elif self.claim_variant == "claimed_at":
                update = {"claimed_at": claim.claimed_at - timedelta(seconds=1)}
            elif self.claim_variant == "lease_expires_at":
                update = {"lease_expires_at": claim.lease_expires_at + timedelta(seconds=1)}
            else:
                decision = await super().load_completion_decision_for_proposal(proposal_id)
                assert decision is not None
                if self.claim_variant == "decision_before_claim":
                    update = {
                        "claimed_at": decision.decided_at + timedelta(seconds=1),
                        "lease_expires_at": decision.decided_at + timedelta(seconds=2),
                    }
                elif self.claim_variant == "decision_at_expiry":
                    update = {
                        "claimed_at": decision.decided_at - timedelta(seconds=1),
                        "lease_expires_at": decision.decided_at,
                    }
                else:
                    update = {
                        "claimed_at": decision.decided_at - timedelta(seconds=2),
                        "lease_expires_at": decision.decided_at - timedelta(seconds=1),
                    }
            return claim.model_copy(update=update)

        async def apply_completion_decision(
            self,
            request: CompletionDecisionApplicationRequest,
        ) -> Task:
            self.application_calls += 1
            return await super().apply_completion_decision(request)

    async def scenario() -> tuple[Task, int]:
        store = ConflictingClaimStore()
        task = await _running_task(store)
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )
        request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=decision_id,
            idempotency_key="application-claim-authority",
            result=_result("1"),
            result_reference=reference,
        )
        if exact_replay:
            completed = await _app(store).apply_completion_decision(request)
            assert completed.status is TaskStatus.COMPLETED
        calls_before = store.application_calls
        store.claim_variant = claim_variant

        with pytest.raises(WorkCompletionConflict, match="verification-claim"):
            await _app(store).apply_completion_decision(request)

        persisted = await store.load_task(task.id)
        assert persisted is not None
        return persisted, store.application_calls - calls_before

    persisted, application_calls = asyncio.run(scenario())
    assert application_calls == 0
    assert persisted.status is (TaskStatus.COMPLETED if exact_replay else TaskStatus.RUNNING)


@pytest.mark.parametrize("exact_replay", [False, True])
@pytest.mark.parametrize("task_variant", ["id", "work_contract"])
def test_app_binds_loaded_task_authority_before_application_mutation(
    exact_replay: bool,
    task_variant: str,
) -> None:
    class ConflictingTaskAuthorityStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.forge_task_authority = False
            self.application_calls = 0

        async def load_task(self, task_id: str) -> Task | None:
            task = await super().load_task(task_id)
            if task is None or not self.forge_task_authority:
                return task
            if task_variant == "id":
                return task.model_copy(update={"id": "substituted-task-authority"})
            assert task.work_contract is not None
            substituted_contract = task.work_contract.model_copy(
                update={"contract_id": "substituted-contract-authority"}
            )
            return task.model_copy(update={"work_contract": substituted_contract})

        async def apply_completion_decision(
            self,
            request: CompletionDecisionApplicationRequest,
        ) -> Task:
            self.application_calls += 1
            return await super().apply_completion_decision(request)

    async def scenario() -> tuple[Task, CompletionDecisionApplicationReceipt | None, int]:
        store = ConflictingTaskAuthorityStore()
        task = await _running_task(store)
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )
        request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=decision_id,
            idempotency_key="application-conflicting-task-authority",
            result=_result("1"),
            result_reference=reference,
        )
        if exact_replay:
            completed = await _app(store).apply_completion_decision(request)
            assert completed.status is TaskStatus.COMPLETED
        calls_before = store.application_calls
        store.forge_task_authority = True

        with pytest.raises(WorkCompletionConflict, match="Durable task authority"):
            await _app(store).apply_completion_decision(request)

        store.forge_task_authority = False
        persisted = await store.load_task(task.id)
        assert persisted is not None
        receipt = await store.load_completion_decision_application_receipt(
            task.id,
            request.idempotency_key,
        )
        return persisted, receipt, store.application_calls - calls_before

    persisted, receipt, application_calls = asyncio.run(scenario())
    assert application_calls == 0
    assert persisted.status is (TaskStatus.COMPLETED if exact_replay else TaskStatus.RUNNING)
    assert (receipt is not None) is exact_replay


@pytest.mark.parametrize("exact_replay", [False, True])
def test_app_rejects_secret_bearing_claim_authority(
    exact_replay: bool,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "application-claim-authority-secret-canary"

    class SecretClaimStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.inject_secret = False

        async def load_completion_verification_claim(
            self,
            proposal_id: str,
        ) -> CompletionVerificationClaim | None:
            claim = await super().load_completion_verification_claim(proposal_id)
            if claim is None or not self.inject_secret:
                return claim
            return claim.model_copy(update={"execution_owner_id": secret})

    async def scenario() -> tuple[BaseException, Task]:
        store = SecretClaimStore()
        task = await _running_task(store)
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )
        request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=decision_id,
            idempotency_key="application-secret-claim-authority",
            result=_result("1"),
            result_reference=reference,
        )
        if exact_replay:
            completed = await _app(store).apply_completion_decision(request)
            assert completed.status is TaskStatus.COMPLETED
        store.inject_secret = True
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )

        with pytest.raises(ValueError, match="public identity") as captured:
            await app.apply_completion_decision(request)

        persisted = await store.load_task(task.id)
        assert persisted is not None
        return captured.value, persisted

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error, persisted = asyncio.run(scenario())

    assert persisted.status is (TaskStatus.COMPLETED if exact_replay else TaskStatus.RUNNING)
    _assert_secret_absent_from_cayu_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


@pytest.mark.parametrize("exact_replay", [False, True])
def test_app_rejects_secret_bearing_receipt_invocation_identity(
    exact_replay: bool,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "application-receipt-invocation-secret-canary"

    class SecretInvocationReceiptStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.forge_receipts = False

        def _forge_task(self, task: Task) -> Task:
            invocation = task.invocation.model_copy(
                update={
                    "origin": InvocationOrigin(
                        trust=InvocationOriginTrust.HOST_ASSERTED,
                        subject=secret,
                    )
                }
            )
            return task.model_copy(update={"invocation": invocation})

        async def apply_completion_decision(
            self,
            request: CompletionDecisionApplicationRequest,
        ) -> Task:
            task = await super().apply_completion_decision(request)
            return self._forge_task(task) if self.forge_receipts else task

        async def load_completion_decision_application_receipt(
            self,
            task_id: str,
            idempotency_key: str,
        ) -> CompletionDecisionApplicationReceipt | None:
            receipt = await super().load_completion_decision_application_receipt(
                task_id,
                idempotency_key,
            )
            if receipt is None or not self.forge_receipts:
                return receipt
            return receipt.model_copy(update={"task": self._forge_task(receipt.task)})

    async def scenario() -> tuple[BaseException, Task]:
        store = SecretInvocationReceiptStore()
        task = await _running_task(store)
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )
        request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=decision_id,
            idempotency_key="application-secret-receipt-invocation",
            result=_result("1"),
            result_reference=reference,
        )
        if exact_replay:
            completed = await _app(store).apply_completion_decision(request)
            assert completed.status is TaskStatus.COMPLETED
        store.forge_receipts = True
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )

        with pytest.raises(ValueError, match="public invocation identity") as captured:
            await app.apply_completion_decision(request)

        persisted = await store.load_task(task.id)
        assert persisted is not None
        return captured.value, persisted

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error, persisted = asyncio.run(scenario())

    assert persisted.status is TaskStatus.COMPLETED
    _assert_secret_absent_from_cayu_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_app_rejects_unproven_custom_store_before_application_mutation() -> None:
    class UnprovenApplicationStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.application_calls = 0

        async def apply_completion_decision(
            self,
            request: CompletionDecisionApplicationRequest,
        ) -> Task:
            self.application_calls += 1
            return await super().apply_completion_decision(request)

    async def scenario() -> None:
        store = UnprovenApplicationStore()
        app = _app(store)
        task = await _running_task(store)
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )

        with pytest.raises(NotImplementedError, match="cancellation-quiescent"):
            await app.apply_completion_decision(
                CompletionDecisionApplicationRequest(
                    task_id=task.id,
                    decision_id=decision_id,
                    idempotency_key="application-unproven-store",
                    result=_result("1"),
                    result_reference=reference,
                )
            )

        assert store.application_calls == 0
        persisted = await store.load_task(task.id)
        assert persisted is not None
        assert persisted.status is TaskStatus.RUNNING

    asyncio.run(scenario())


def test_app_preserves_cancellation_after_application_commit() -> None:
    class CommitThenWaitStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.committed = asyncio.Event()
            self.release = asyncio.Event()

        async def apply_completion_decision(
            self,
            request: CompletionDecisionApplicationRequest,
        ) -> Task:
            task = await super().apply_completion_decision(request)
            self.committed.set()
            await self.release.wait()
            return task

    async def scenario() -> None:
        store = CommitThenWaitStore()
        app = _app(store)
        task = await _running_task(store)
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )
        request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=decision_id,
            idempotency_key="application-cancelled",
            result=_result("1"),
            result_reference=reference,
        )
        owner = asyncio.create_task(app.apply_completion_decision(request))
        await store.committed.wait()
        owner.cancel("caller stopped waiting")
        store.release.set()

        with pytest.raises(asyncio.CancelledError):
            await owner
        assert owner.cancelled()
        assert owner.cancelling() == 1

        replayed = await app.apply_completion_decision(request)
        assert replayed.status is TaskStatus.COMPLETED

    asyncio.run(scenario())


def test_app_keeps_cancellation_scalar_when_application_settlement_fails_after_commit() -> None:
    class CommitThenCancellationFailureStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.committed = asyncio.Event()

        async def apply_completion_decision(
            self,
            request: CompletionDecisionApplicationRequest,
        ) -> Task:
            await super().apply_completion_decision(request)
            self.committed.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise TimeoutError("application cancellation settlement failed") from None
            raise AssertionError("unreachable")

    async def scenario() -> tuple[asyncio.CancelledError, Task, Task]:
        store = CommitThenCancellationFailureStore()
        app = _app(store)
        task = await _running_task(store)
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )
        request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=decision_id,
            idempotency_key="application-cancellation-settlement-failure",
            result=_result("1"),
            result_reference=reference,
        )
        owner = asyncio.create_task(app.apply_completion_decision(request))
        await store.committed.wait()
        owner.cancel("caller stopped application")

        with pytest.raises(asyncio.CancelledError) as captured:
            await owner
        assert owner.cancelled()
        assert owner.cancelling() == 1
        persisted = await store.load_task(task.id)
        assert persisted is not None
        replayed = await app.apply_completion_decision(request)
        return captured.value, persisted, replayed

    error, persisted, replayed = asyncio.run(scenario())
    assert type(error) is asyncio.CancelledError
    assert isinstance(error.__cause__, BaseExceptionGroup)
    leaves = list(error.__cause__.exceptions)
    assert [type(item) for item in leaves] == [TimeoutError]
    assert str(leaves[0]) == "application cancellation settlement failed"
    assert persisted.status is TaskStatus.COMPLETED
    assert replayed == persisted


def test_app_keeps_cancellation_scalar_when_receipt_readback_settlement_fails() -> None:
    class ReceiptReadbackCancellationFailureStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.receipt_lookups = 0
            self.readback_started = asyncio.Event()

        async def load_completion_decision_application_receipt(
            self,
            task_id: str,
            idempotency_key: str,
        ) -> CompletionDecisionApplicationReceipt | None:
            self.receipt_lookups += 1
            if self.receipt_lookups != 2:
                return await super().load_completion_decision_application_receipt(
                    task_id,
                    idempotency_key,
                )
            self.readback_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise TimeoutError("receipt readback cancellation settlement failed") from None
            raise AssertionError("unreachable")

    async def scenario() -> tuple[asyncio.CancelledError, Task, Task]:
        store = ReceiptReadbackCancellationFailureStore()
        app = _app(store)
        task = await _running_task(store)
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )
        request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=decision_id,
            idempotency_key="application-readback-cancellation-settlement-failure",
            result=_result("1"),
            result_reference=reference,
        )
        owner = asyncio.create_task(app.apply_completion_decision(request))
        await store.readback_started.wait()
        owner.cancel("caller stopped receipt readback")

        with pytest.raises(asyncio.CancelledError) as captured:
            await owner
        assert owner.cancelled()
        assert owner.cancelling() == 1
        persisted = await store.load_task(task.id)
        assert persisted is not None
        replayed = await app.apply_completion_decision(request)
        return captured.value, persisted, replayed

    error, persisted, replayed = asyncio.run(scenario())
    assert type(error) is asyncio.CancelledError
    assert isinstance(error.__cause__, BaseExceptionGroup)
    leaves = list(error.__cause__.exceptions)
    assert [type(item) for item in leaves] == [TimeoutError]
    assert str(leaves[0]) == "receipt readback cancellation settlement failed"
    assert persisted.status is TaskStatus.COMPLETED
    assert replayed == persisted


def test_app_keeps_cancellation_scalar_when_authority_lookup_settlement_fails() -> None:
    class AuthorityLookupCancellationFailureStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.decision_lookups = 0
            self.lookup_started = asyncio.Event()
            self.application_calls = 0

        async def load_completion_decision(self, decision_id: str):
            self.decision_lookups += 1
            if self.decision_lookups != 1:
                return await super().load_completion_decision(decision_id)
            self.lookup_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise TimeoutError("authority lookup cancellation settlement failed") from None
            raise AssertionError("unreachable")

        async def apply_completion_decision(
            self,
            request: CompletionDecisionApplicationRequest,
        ) -> Task:
            self.application_calls += 1
            return await super().apply_completion_decision(request)

    async def scenario() -> tuple[asyncio.CancelledError, Task, Task, int]:
        store = AuthorityLookupCancellationFailureStore()
        app = _app(store)
        task = await _running_task(store)
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )
        request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=decision_id,
            idempotency_key="application-authority-cancellation-settlement-failure",
            result=_result("1"),
            result_reference=reference,
        )
        owner = asyncio.create_task(app.apply_completion_decision(request))
        await store.lookup_started.wait()
        owner.cancel("caller stopped authority lookup")

        with pytest.raises(asyncio.CancelledError) as captured:
            await owner
        assert owner.cancelled()
        assert owner.cancelling() == 1
        persisted = await store.load_task(task.id)
        assert persisted is not None
        calls_after_cancellation = store.application_calls
        completed = await app.apply_completion_decision(request)
        return captured.value, persisted, completed, calls_after_cancellation

    error, persisted, completed, calls_after_cancellation = asyncio.run(scenario())
    assert type(error) is asyncio.CancelledError
    assert isinstance(error.__cause__, BaseExceptionGroup)
    leaves = list(error.__cause__.exceptions)
    assert [type(item) for item in leaves] == [TimeoutError]
    assert str(leaves[0]) == "authority lookup cancellation settlement failed"
    assert persisted.status is TaskStatus.RUNNING
    assert calls_after_cancellation == 0
    assert completed.status is TaskStatus.COMPLETED


def test_app_rejects_task_return_that_diverges_from_durable_receipt() -> None:
    class DivergentReturnStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def apply_completion_decision(
            self,
            request: CompletionDecisionApplicationRequest,
        ) -> Task:
            task = await super().apply_completion_decision(request)
            return task.model_copy(update={"status": TaskStatus.RUNNING})

    async def scenario() -> None:
        store = DivergentReturnStore()
        app = _app(store)
        task = await _running_task(store)
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )

        with pytest.raises(WorkCompletionConflict, match="conflicts with its application receipt"):
            await app.apply_completion_decision(
                CompletionDecisionApplicationRequest(
                    task_id=task.id,
                    decision_id=decision_id,
                    idempotency_key="application-divergent-return",
                    result=_result("1"),
                    result_reference=reference,
                )
            )

        persisted = await store.load_task(task.id)
        assert persisted is not None
        assert persisted.status is TaskStatus.COMPLETED

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("integer_value", "boolean_value"),
    [(1, True), (0, False)],
    ids=["true-is-not-one", "false-is-not-zero"],
)
def test_app_rejects_type_coercive_result_divergence_from_durable_receipt(
    integer_value: int,
    boolean_value: bool,
) -> None:
    class DivergentReturnStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def apply_completion_decision(
            self,
            request: CompletionDecisionApplicationRequest,
        ) -> Task:
            task = await super().apply_completion_decision(request)
            return task.model_copy(update={"result": {"nested": {"value": boolean_value}}})

    async def scenario() -> None:
        store = DivergentReturnStore()
        task = await _running_task(store)
        result = {"nested": {"value": integer_value}}
        reference = CompletionResultReference(
            kind="session.output",
            reference_id="session:numeric-return",
            digest=completion_result_sha256(result),
        )
        decision_id, _ = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
            result_reference=reference,
        )

        with pytest.raises(
            WorkCompletionConflict,
            match="conflicts with its application receipt",
        ):
            await _app(store).apply_completion_decision(
                CompletionDecisionApplicationRequest(
                    task_id=task.id,
                    decision_id=decision_id,
                    idempotency_key="application-type-coercive-return",
                    result=result,
                    result_reference=reference,
                )
            )

        persisted = await store.load_task(task.id)
        assert persisted is not None
        assert persisted.result is not None
        assert type(persisted.result["nested"]["value"]) is int

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("task_update", "message"),
    [
        (
            {"status": TaskStatus.RUNNING, "completed_at": None},
            "Accepted decision receipt",
        ),
        (
            {"session_id": "session:forged"},
            "work-attempt session",
        ),
    ],
)
def test_app_rejects_exact_receipt_with_semantically_forged_task_snapshot(
    task_update: dict[str, object],
    message: str,
) -> None:
    class ForgedReceiptStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.forge_receipts = False

        async def load_completion_decision_application_receipt(
            self,
            task_id: str,
            idempotency_key: str,
        ) -> CompletionDecisionApplicationReceipt | None:
            receipt = await super().load_completion_decision_application_receipt(
                task_id,
                idempotency_key,
            )
            if receipt is None or not self.forge_receipts:
                return receipt
            return receipt.model_copy(update={"task": receipt.task.model_copy(update=task_update)})

    async def scenario() -> None:
        store = ForgedReceiptStore()
        app = _app(store)
        task = await _running_task(store)
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )
        request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=decision_id,
            idempotency_key="application-forged-receipt",
            result=_result("1"),
            result_reference=reference,
        )
        completed = await app.apply_completion_decision(request)
        store.forge_receipts = True

        with pytest.raises(WorkCompletionConflict, match=message):
            await app.apply_completion_decision(request)

        persisted = await store.load_task(task.id)
        assert persisted == completed

    asyncio.run(scenario())


@pytest.mark.parametrize("reconciliation", [False, True])
@pytest.mark.parametrize(
    "field_name",
    [
        "invocation",
        "type",
        "parent_task_id",
        "available_at",
        "started_at",
        "input",
        "metadata",
    ],
)
def test_app_rejects_receipt_with_forged_immutable_task_authority(
    reconciliation: bool,
    field_name: str,
) -> None:
    class ForgedImmutableReceiptStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.forge_receipts = False
            self.raise_after_commit = False
            self.application_calls = 0

        async def apply_completion_decision(
            self,
            request: CompletionDecisionApplicationRequest,
        ) -> Task:
            self.application_calls += 1
            applied = await super().apply_completion_decision(request)
            if self.raise_after_commit:
                raise ConnectionError("application acknowledgement lost")
            return applied

        async def load_completion_decision_application_receipt(
            self,
            task_id: str,
            idempotency_key: str,
        ) -> CompletionDecisionApplicationReceipt | None:
            receipt = await super().load_completion_decision_application_receipt(
                task_id,
                idempotency_key,
            )
            if receipt is None or not self.forge_receipts:
                return receipt
            update: dict[str, object]
            if field_name == "invocation":
                update = {
                    "invocation": receipt.task.invocation.model_copy(
                        update={
                            "origin": InvocationOrigin(
                                trust=InvocationOriginTrust.HOST_ASSERTED,
                                subject="forged-safe-origin",
                            )
                        }
                    )
                }
            elif field_name == "type":
                update = {"type": "forged-safe-type"}
            elif field_name == "parent_task_id":
                update = {"parent_task_id": "forged-safe-parent"}
            elif field_name == "available_at":
                update = {"available_at": receipt.task.created_at + timedelta(days=1)}
            elif field_name == "started_at":
                assert receipt.task.started_at is not None
                update = {"started_at": receipt.task.started_at + timedelta(seconds=1)}
            elif field_name == "input":
                update = {"input": {"nested": {"value": True}}}
            else:
                update = {"metadata": {"nested": {"value": True}}}
            return receipt.model_copy(update={"task": receipt.task.model_copy(update=update)})

    async def scenario() -> tuple[WorkCompletionConflict, Task, int]:
        store = ForgedImmutableReceiptStore()
        task = await _running_task(
            store,
            task_input={"nested": {"value": 1}} if field_name == "input" else None,
            metadata={"nested": {"value": 1}} if field_name == "metadata" else None,
        )
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )
        request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=decision_id,
            idempotency_key="application-forged-immutable-authority",
            result=_result("1"),
            result_reference=reference,
        )
        if not reconciliation:
            completed = await _app(store).apply_completion_decision(request)
            assert completed.status is TaskStatus.COMPLETED
        store.forge_receipts = True
        store.raise_after_commit = reconciliation
        calls_before = store.application_calls

        with pytest.raises(
            WorkCompletionConflict,
            match="durable immutable task authority",
        ) as captured:
            await _app(store).apply_completion_decision(request)

        persisted = await store.load_task(task.id)
        assert persisted is not None
        return captured.value, persisted, store.application_calls - calls_before

    error, persisted, application_calls = asyncio.run(scenario())
    assert persisted.status is TaskStatus.COMPLETED
    assert application_calls == (1 if reconciliation else 0)
    if reconciliation:
        assert type(error.__cause__) is ConnectionError
        assert str(error.__cause__) == "application acknowledgement lost"
    else:
        assert error.__cause__ is None


@pytest.mark.parametrize("reconciliation", [False, True])
@pytest.mark.parametrize(
    ("integer_value", "boolean_value"),
    [(1, True), (0, False)],
    ids=["true-is-not-one", "false-is-not-zero"],
)
def test_app_rejects_type_coercive_receipt_result(
    reconciliation: bool,
    integer_value: int,
    boolean_value: bool,
) -> None:
    class ForgedNumericReceiptStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.forge_receipts = False
            self.raise_after_commit = False

        async def apply_completion_decision(
            self,
            request: CompletionDecisionApplicationRequest,
        ) -> Task:
            applied = await super().apply_completion_decision(request)
            if self.raise_after_commit:
                raise ConnectionError("application acknowledgement lost")
            return applied

        async def load_completion_decision_application_receipt(
            self,
            task_id: str,
            idempotency_key: str,
        ) -> CompletionDecisionApplicationReceipt | None:
            receipt = await super().load_completion_decision_application_receipt(
                task_id,
                idempotency_key,
            )
            if receipt is None or not self.forge_receipts:
                return receipt
            forged_result = {"nested": {"value": boolean_value}}
            return receipt.model_copy(
                update={"task": receipt.task.model_copy(update={"result": forged_result})}
            )

    async def scenario() -> tuple[WorkCompletionConflict, Task]:
        store = ForgedNumericReceiptStore()
        task = await _running_task(store)
        result = {"nested": {"value": integer_value}}
        reference = CompletionResultReference(
            kind="session.output",
            reference_id="session:numeric-receipt",
            digest=completion_result_sha256(result),
        )
        decision_id, _ = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
            result_reference=reference,
        )
        request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=decision_id,
            idempotency_key="application-type-coercive-receipt",
            result=result,
            result_reference=reference,
        )
        if not reconciliation:
            completed = await _app(store).apply_completion_decision(request)
            assert completed.status is TaskStatus.COMPLETED
        store.forge_receipts = True
        store.raise_after_commit = reconciliation

        with pytest.raises(
            WorkCompletionConflict,
            match="Accepted decision receipt",
        ) as captured:
            await _app(store).apply_completion_decision(request)

        persisted = await store.load_task(task.id)
        assert persisted is not None
        return captured.value, persisted

    error, persisted = asyncio.run(scenario())
    assert persisted.result is not None
    assert type(persisted.result["nested"]["value"]) is int
    if reconciliation:
        assert type(error.__cause__) is ConnectionError
        assert str(error.__cause__) == "application acknowledgement lost"
    else:
        assert error.__cause__ is None


def test_app_preserves_application_then_reconciliation_failure_order() -> None:
    class FailingApplicationStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.receipt_lookups = 0

        async def apply_completion_decision(
            self,
            request: CompletionDecisionApplicationRequest,
        ) -> Task:
            del request
            raise ConnectionError("application failed")

        async def load_completion_decision_application_receipt(
            self,
            task_id: str,
            idempotency_key: str,
        ):
            self.receipt_lookups += 1
            if self.receipt_lookups == 1:
                return await super().load_completion_decision_application_receipt(
                    task_id,
                    idempotency_key,
                )
            raise TimeoutError("receipt lookup failed")

    async def scenario() -> BaseExceptionGroup:
        store = FailingApplicationStore()
        app = _app(store)
        task = await _running_task(store)
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )
        with pytest.raises(BaseExceptionGroup) as captured:
            await app.apply_completion_decision(
                CompletionDecisionApplicationRequest(
                    task_id=task.id,
                    decision_id=decision_id,
                    idempotency_key="application-double-failure",
                    result=_result("1"),
                    result_reference=reference,
                )
            )
        return captured.value

    error = asyncio.run(scenario())
    assert [type(item) for item in error.exceptions] == [ConnectionError, TimeoutError]
    assert [str(item) for item in error.exceptions] == [
        "application failed",
        "receipt lookup failed",
    ]


def test_app_failure_reconciliation_does_not_retain_secret_task_authority(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "application-reconciliation-authority-secret-canary"

    class FailingApplicationStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def apply_completion_decision(
            self,
            request: CompletionDecisionApplicationRequest,
        ) -> Task:
            del request
            raise ConnectionError("application failed")

    async def scenario() -> tuple[ConnectionError, Task]:
        store = FailingApplicationStore()
        task = await _running_task(
            store,
            task_input={"private": {"value": secret}},
            metadata={"private": {"value": secret}},
        )
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )

        with pytest.raises(ConnectionError, match="application failed") as captured:
            await app.apply_completion_decision(
                CompletionDecisionApplicationRequest(
                    task_id=task.id,
                    decision_id=decision_id,
                    idempotency_key="application-secret-authority-failure",
                    result=_result("1"),
                    result_reference=reference,
                )
            )

        persisted = await store.load_task(task.id)
        assert persisted is not None
        return captured.value, persisted

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error, persisted = asyncio.run(scenario())

    assert persisted.status is TaskStatus.RUNNING
    _assert_secret_absent_from_cayu_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_app_preserves_application_failure_when_reconciled_receipt_is_rejected(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "application-reconciliation-receipt-secret-canary"

    class CommitThenForgeReceiptStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def apply_completion_decision(
            self,
            request: CompletionDecisionApplicationRequest,
        ) -> Task:
            await super().apply_completion_decision(request)
            raise ConnectionError("application acknowledgement lost")

        async def load_completion_decision_application_receipt(
            self,
            task_id: str,
            idempotency_key: str,
        ) -> CompletionDecisionApplicationReceipt | None:
            receipt = await super().load_completion_decision_application_receipt(
                task_id,
                idempotency_key,
            )
            if receipt is None:
                return None
            invocation = receipt.task.invocation.model_copy(
                update={
                    "origin": InvocationOrigin(
                        trust=InvocationOriginTrust.HOST_ASSERTED,
                        subject=secret,
                    )
                }
            )
            return receipt.model_copy(
                update={"task": receipt.task.model_copy(update={"invocation": invocation})}
            )

    async def scenario() -> tuple[ValueError, Task]:
        store = CommitThenForgeReceiptStore()
        task = await _running_task(
            store,
            task_input={"private": {"value": secret}},
            metadata={"private": {"value": secret}},
        )
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )

        with pytest.raises(ValueError, match="public invocation identity") as captured:
            await app.apply_completion_decision(
                CompletionDecisionApplicationRequest(
                    task_id=task.id,
                    decision_id=decision_id,
                    idempotency_key="application-reconciliation-receipt-rejected",
                    result=_result("1"),
                    result_reference=reference,
                )
            )

        persisted = await store.load_task(task.id)
        assert persisted is not None
        return captured.value, persisted

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error, persisted = asyncio.run(scenario())

    assert persisted.status is TaskStatus.COMPLETED
    assert type(error.__cause__) is ConnectionError
    assert str(error.__cause__) == "application acknowledgement lost"
    assert error.__cause__.__cause__ is None
    _assert_secret_absent_from_cayu_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_app_preserves_caller_cancellation_during_failure_reconciliation() -> None:
    class CancellationDuringReconciliationStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.receipt_lookups = 0
            self.reconciliation_started = asyncio.Event()

        async def apply_completion_decision(
            self,
            request: CompletionDecisionApplicationRequest,
        ) -> Task:
            del request
            raise ConnectionError("application failed")

        async def load_completion_decision_application_receipt(
            self,
            task_id: str,
            idempotency_key: str,
        ) -> CompletionDecisionApplicationReceipt | None:
            self.receipt_lookups += 1
            if self.receipt_lookups == 1:
                return await super().load_completion_decision_application_receipt(
                    task_id,
                    idempotency_key,
                )
            self.reconciliation_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    async def scenario() -> tuple[asyncio.CancelledError, Task]:
        store = CancellationDuringReconciliationStore()
        app = _app(store)
        task = await _running_task(store)
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )
        request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=decision_id,
            idempotency_key="application-reconciliation-cancelled",
            result=_result("1"),
            result_reference=reference,
        )
        owner = asyncio.create_task(app.apply_completion_decision(request))
        await store.reconciliation_started.wait()
        owner.cancel("caller stopped reconciliation")

        with pytest.raises(asyncio.CancelledError) as captured:
            await owner
        assert owner.cancelled()
        assert owner.cancelling() == 1
        persisted = await store.load_task(task.id)
        assert persisted is not None
        return captured.value, persisted

    error, persisted = asyncio.run(scenario())
    assert type(error) is asyncio.CancelledError
    assert isinstance(error.__cause__, ConnectionError)
    assert str(error.__cause__) == "application failed"
    assert persisted.status is TaskStatus.RUNNING


def test_app_keeps_cancellation_scalar_when_reconciliation_settlement_fails() -> None:
    class CancellationSettlementFailureStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.receipt_lookups = 0
            self.reconciliation_started = asyncio.Event()

        async def apply_completion_decision(
            self,
            request: CompletionDecisionApplicationRequest,
        ) -> Task:
            del request
            raise ConnectionError("application failed")

        async def load_completion_decision_application_receipt(
            self,
            task_id: str,
            idempotency_key: str,
        ) -> CompletionDecisionApplicationReceipt | None:
            self.receipt_lookups += 1
            if self.receipt_lookups == 1:
                return await super().load_completion_decision_application_receipt(
                    task_id,
                    idempotency_key,
                )
            self.reconciliation_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise TimeoutError("receipt cancellation settlement failed") from None
            raise AssertionError("unreachable")

    async def scenario() -> tuple[asyncio.CancelledError, Task]:
        store = CancellationSettlementFailureStore()
        app = _app(store)
        task = await _running_task(store)
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )
        request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=decision_id,
            idempotency_key="application-reconciliation-cancellation-settlement",
            result=_result("1"),
            result_reference=reference,
        )
        owner = asyncio.create_task(app.apply_completion_decision(request))
        await store.reconciliation_started.wait()
        owner.cancel("caller stopped reconciliation")

        with pytest.raises(asyncio.CancelledError) as captured:
            await owner
        assert owner.cancelled()
        assert owner.cancelling() == 1
        persisted = await store.load_task(task.id)
        assert persisted is not None
        return captured.value, persisted

    error, persisted = asyncio.run(scenario())
    assert type(error) is asyncio.CancelledError
    assert isinstance(error.__cause__, BaseExceptionGroup)
    leaves = list(error.__cause__.exceptions)
    assert [type(item) for item in leaves] == [ConnectionError, ExceptionGroup]
    reconciliation_group = leaves[1]
    assert isinstance(reconciliation_group, ExceptionGroup)
    assert [type(item) for item in reconciliation_group.exceptions] == [TimeoutError]
    assert not any(
        isinstance(item, asyncio.CancelledError)
        for item in [error.__cause__, *leaves, *reconciliation_group.exceptions]
    )
    assert persisted.status is TaskStatus.RUNNING


def test_app_preserves_process_control_during_failure_reconciliation() -> None:
    class ProcessControlDuringReconciliationStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.receipt_lookups = 0

        async def apply_completion_decision(
            self,
            request: CompletionDecisionApplicationRequest,
        ) -> Task:
            del request
            raise ConnectionError("application failed")

        async def load_completion_decision_application_receipt(
            self,
            task_id: str,
            idempotency_key: str,
        ) -> CompletionDecisionApplicationReceipt | None:
            self.receipt_lookups += 1
            if self.receipt_lookups == 1:
                return await super().load_completion_decision_application_receipt(
                    task_id,
                    idempotency_key,
                )
            raise SystemExit("worker stopped during reconciliation")

    async def scenario() -> tuple[SystemExit, Task]:
        store = ProcessControlDuringReconciliationStore()
        app = _app(store)
        task = await _running_task(store)
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )
        with pytest.raises(SystemExit) as captured:
            await app.apply_completion_decision(
                CompletionDecisionApplicationRequest(
                    task_id=task.id,
                    decision_id=decision_id,
                    idempotency_key="application-reconciliation-process-control",
                    result=_result("1"),
                    result_reference=reference,
                )
            )
        persisted = await store.load_task(task.id)
        assert persisted is not None
        return captured.value, persisted

    error, persisted = asyncio.run(scenario())
    assert type(error) is SystemExit
    assert str(error) == "worker stopped during reconciliation"
    assert isinstance(error.__cause__, ConnectionError)
    assert str(error.__cause__) == "application failed"
    assert persisted.status is TaskStatus.RUNNING


def test_app_does_not_reconcile_process_control_into_success() -> None:
    class FatalAfterCommitStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def apply_completion_decision(
            self,
            request: CompletionDecisionApplicationRequest,
        ) -> Task:
            await super().apply_completion_decision(request)
            raise SystemExit("worker shutdown")

    async def scenario() -> tuple[SystemExit, Task]:
        store = FatalAfterCommitStore()
        app = _app(store)
        task = await _running_task(store)
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )
        request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=decision_id,
            idempotency_key="application-process-control",
            result=_result("1"),
            result_reference=reference,
        )
        with pytest.raises(SystemExit) as captured:
            await app.apply_completion_decision(request)
        replayed = await app.apply_completion_decision(request)
        return captured.value, replayed

    error, replayed = asyncio.run(scenario())
    assert str(error) == "worker shutdown"
    assert replayed.status is TaskStatus.COMPLETED


def test_app_validation_does_not_expose_mutated_secret_result(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "application-result-secret-canary"

    async def scenario() -> BaseException:
        store = InMemoryTaskStore()
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        task = await _running_task(store)
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
        )
        request = CompletionDecisionApplicationRequest(
            task_id=task.id,
            decision_id=decision_id,
            idempotency_key="application-secret-validation",
            result=_result("1"),
            result_reference=reference,
        )
        if request.result is None:  # pragma: no cover - construction invariant
            raise AssertionError("Accepted application result was not retained.")
        request.result[secret] = object()
        with pytest.raises(ValueError) as captured:
            await app.apply_completion_decision(request)
        return captured.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error = asyncio.run(scenario())

    _assert_secret_absent_from_cayu_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_app_identity_rejection_does_not_retain_secret_request(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "application-identity-secret-canary"

    async def scenario() -> BaseException:
        store = InMemoryTaskStore()
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        with pytest.raises(ValueError, match="public identity") as captured:
            await app.apply_completion_decision(
                CompletionDecisionApplicationRequest(
                    task_id="application-task",
                    decision_id="decision-1",
                    idempotency_key=secret,
                )
            )
        return captured.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error = asyncio.run(scenario())

    _assert_secret_absent_from_cayu_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_app_post_commit_failure_does_not_retain_valid_secret_result(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "application-committed-result-secret-canary"
    result: dict[str, object] = {"credential": secret}
    result_reference = CompletionResultReference(
        kind="session.output",
        reference_id="session:secret-result",
        digest=completion_result_sha256(result),
    )

    class FailPostCommitReceiptLookupStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.receipt_lookups = 0

        async def load_completion_decision_application_receipt(
            self,
            task_id: str,
            idempotency_key: str,
        ) -> CompletionDecisionApplicationReceipt | None:
            self.receipt_lookups += 1
            if self.receipt_lookups == 1:
                return await super().load_completion_decision_application_receipt(
                    task_id,
                    idempotency_key,
                )
            raise ConnectionError("receipt lookup failed after commit")

    async def scenario() -> tuple[BaseException, Task]:
        store = FailPostCommitReceiptLookupStore()
        app = CayuApp(
            task_store=store,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        task = await _running_task(store)
        decision_id, reference = await _persist_decision(
            store,
            task=task,
            ordinal=1,
            verdict=CompletionVerdict.ACCEPTED,
            result_reference=result_reference,
        )
        with pytest.raises(ConnectionError) as captured:
            await app.apply_completion_decision(
                CompletionDecisionApplicationRequest(
                    task_id=task.id,
                    decision_id=decision_id,
                    idempotency_key="application-secret-post-commit-failure",
                    result=result,
                    result_reference=reference,
                )
            )
        persisted = await store.load_task(task.id)
        assert persisted is not None
        return captured.value, persisted

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error, persisted = asyncio.run(scenario())

    assert persisted.status is TaskStatus.COMPLETED
    assert persisted.result == result
    _assert_secret_absent_from_cayu_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)
