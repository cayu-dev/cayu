from __future__ import annotations

import asyncio
import json
import sqlite3
import traceback as traceback_module
import warnings
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from threading import Event as ThreadEvent
from threading import Thread
from typing import Any

import pytest
from tests.core.task_invocation_fixtures import (
    stored_session_invocation,
    task_backed_session_invocation,
)
from tests.core.test_verified_work_contracts import (
    _accepted_decision,
    _artifact_evidence,
    _claim_completion_verification,
    _contract,
    _digest,
    _RecordingProvider,
    _rejected_decision,
    _result_reference,
    _task_result,
    _verifier_profile_fingerprint,
)
from tests.provider_traceback_assertions import is_cayu_source_filename

from cayu import (
    AgentSpec,
    CayuApp,
    CompletionDecisionApplicationRequest,
    CompletionResultResolutionRequest,
    CompletionResultResolver,
    CompletionResultResolverRequest,
    CompletionVerificationClaimRequest,
    Event,
    EventType,
    ForkSessionRequest,
    InMemorySessionStore,
    InMemoryTaskStore,
    Message,
    ResumeRequest,
    RunRequest,
    SecretRedactor,
    Session,
    SessionIdentity,
    SessionStatus,
    SessionStore,
    SQLiteSessionStore,
    SQLiteTaskStore,
    Task,
    TaskClaimLost,
    TaskCompletionDecisionRequired,
    TaskCreate,
    TaskStatus,
    TaskStore,
    TaskTerminalizationRequest,
    TaskTerminalKind,
    WorkAttemptClaimRenewalRequest,
    WorkAttemptCreate,
    WorkAttemptExecutionRequest,
    WorkAttemptProposalRequest,
    WorkAttemptRecoveryRequest,
    WorkCompletionConflict,
    WorkEvidenceReference,
)
from cayu._validation import canonical_durable_json_bytes
from cayu.runtime import (
    CheckpointTransform,
    DeferredInteractionInput,
    EventQuery,
    EventRecord,
    EventSink,
    InMemoryEventSink,
)
from cayu.runtime.loop_policies import LoopPolicy
from cayu.runtime.work_attempt_admission import (
    WORK_ATTEMPT_RECOVERY_CHECKPOINT_KEY,
    AdmittedCompletionProposalRequest,
    WorkAttemptAdmission,
    WorkAttemptAdmissionActivate,
    WorkAttemptAdmissionConflict,
    WorkAttemptAdmissionPrepare,
    WorkAttemptAdmissionState,
    WorkAttemptExecutionClaim,
    WorkAttemptExecutionClaimLost,
    WorkAttemptExecutionClaimRequest,
    WorkAttemptRecoveryActivate,
    WorkAttemptRecoveryRequired,
    work_attempt_execution_claim_request_sha256,
)
from cayu.runtime.work_contracts import (
    WORK_COMPLETION_DECISION_MAX_BYTES,
    CompletionProposal,
    CompletionProposalCreate,
)
from cayu.storage import _sqlite_support as sqlite_support
from cayu.storage.migrations import SchemaMode


class _LoseFirstAdmissionPreparationAcknowledgement(InMemoryTaskStore):
    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self, *, clock) -> None:
        super().__init__(clock=clock)
        self._lose_prepare_ack = True

    async def prepare_work_attempt_admission(
        self,
        request: WorkAttemptAdmissionPrepare,
    ) -> WorkAttemptAdmission:
        admission = await super().prepare_work_attempt_admission(request)
        if self._lose_prepare_ack:
            self._lose_prepare_ack = False
            raise RuntimeError("injected admission preparation acknowledgement loss")
        return admission


class _AdmissionResultResolver(CompletionResultResolver):
    async def resolve(
        self,
        request: CompletionResultResolverRequest,
    ) -> dict[str, object]:
        del request
        return _task_result()


class _LoseFirstAdmissionActivationAcknowledgement(InMemoryTaskStore):
    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        super().__init__(clock=clock)
        self._lose_activation_ack = True

    async def activate_work_attempt_admission(
        self,
        request: WorkAttemptAdmissionActivate,
    ) -> WorkAttemptAdmission:
        admission = await super().activate_work_attempt_admission(request)
        if self._lose_activation_ack:
            self._lose_activation_ack = False
            raise RuntimeError("injected admission activation acknowledgement loss")
        return admission


class _LoseFirstRecoveryActivationAcknowledgement(InMemoryTaskStore):
    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self, *, clock) -> None:
        super().__init__(clock=clock)
        self._lose_recovery_ack = True

    async def activate_work_attempt_recovery(
        self,
        request: WorkAttemptRecoveryActivate,
    ) -> WorkAttemptAdmission:
        admission = await super().activate_work_attempt_recovery(request)
        if self._lose_recovery_ack:
            self._lose_recovery_ack = False
            raise RuntimeError("injected recovery activation acknowledgement loss")
        return admission


class _FailFirstRecoveryActivationBeforeMutation(InMemoryTaskStore):
    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self, *, clock) -> None:
        super().__init__(clock=clock)
        self._fail_recovery_activation = True

    async def activate_work_attempt_recovery(
        self,
        request: WorkAttemptRecoveryActivate,
    ) -> WorkAttemptAdmission:
        if self._fail_recovery_activation:
            self._fail_recovery_activation = False
            raise RuntimeError("injected pre-mutation recovery activation failure")
        return await super().activate_work_attempt_recovery(request)


class _SecretBearingHistoricalClaimStore(_FailFirstRecoveryActivationBeforeMutation):
    """Substitute one historical recovery claim after its session transition."""

    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self, secret: str, *, clock) -> None:
        super().__init__(clock=clock)
        self._secret = secret
        self.forge_historical_claim = False

    async def load_work_attempt_execution_claim(
        self,
        claim_id: str,
    ) -> WorkAttemptExecutionClaim | None:
        claim = await super().load_work_attempt_execution_claim(claim_id)
        if claim is None or not self.forge_historical_claim:
            return claim
        return claim.model_copy(update={"admission_id": self._secret})


class _FailFirstSQLiteRecoveryActivationBeforeMutation(SQLiteTaskStore):
    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self, path, *, clock) -> None:
        super().__init__(path, clock=clock)
        self._fail_recovery_activation = True

    async def activate_work_attempt_recovery(
        self,
        request: WorkAttemptRecoveryActivate,
    ) -> WorkAttemptAdmission:
        if self._fail_recovery_activation:
            self._fail_recovery_activation = False
            raise RuntimeError("injected pre-mutation recovery activation failure")
        return await super().activate_work_attempt_recovery(request)


class _ConflictingWorkAttemptResultStore(InMemoryTaskStore):
    """Commit the built-in mutation, then forge one otherwise-valid return value."""

    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        super().__init__(clock=clock)
        self.forge_prepare = False
        self.forge_prepare_sha256: str | None = None
        self.forge_activation = False
        self.forge_renewal = False
        self.forge_renewal_lease = False
        self.forge_recovery: str | None = None
        self.forge_proposal = False

    async def prepare_work_attempt_admission(
        self,
        request: WorkAttemptAdmissionPrepare,
    ) -> WorkAttemptAdmission:
        admission = await super().prepare_work_attempt_admission(request)
        if self.forge_prepare or self.forge_prepare_sha256 is not None:
            return admission.model_copy(
                update={
                    "prepare_request_sha256": (
                        self.forge_prepare_sha256 or _digest("forged-prepare-result")
                    )
                }
            )
        return admission

    async def activate_work_attempt_admission(
        self,
        request: WorkAttemptAdmissionActivate,
    ) -> WorkAttemptAdmission:
        admission = await super().activate_work_attempt_admission(request)
        if self.forge_activation:
            return admission.model_copy(
                update={"source_request_sha256": _digest("forged-activation-result")}
            )
        return admission

    async def renew_work_attempt_execution_claim(
        self,
        request: WorkAttemptExecutionClaimRequest,
    ) -> WorkAttemptAdmission:
        previous = await self.load_work_attempt_admission(request.admission_id)
        assert previous is not None
        admission = await super().renew_work_attempt_execution_claim(request)
        if self.forge_renewal_lease:
            return admission.model_copy(
                update={
                    "claim": admission.claim.model_copy(
                        update={
                            "lease_expires_at": previous.claim.lease_expires_at
                            + timedelta(seconds=request.lease_seconds + 1)
                        }
                    )
                }
            )
        if self.forge_renewal:
            return admission.model_copy(
                update={"source_request_sha256": _digest("forged-renewal-result")}
            )
        return admission

    async def claim_work_attempt_recovery(
        self,
        request: WorkAttemptExecutionClaimRequest,
    ) -> WorkAttemptAdmission:
        previous = await self.load_work_attempt_admission(request.admission_id)
        assert previous is not None
        store_request = request
        if self.forge_recovery == "skipped_generation":
            store_request = request.model_copy(update={"generation": previous.claim.generation + 1})
        admission = await super().claim_work_attempt_recovery(store_request)
        if self.forge_recovery is None:
            return admission
        claim = admission.claim
        if self.forge_recovery == "early_replacement":
            claimed_at = previous.claim.lease_expires_at - timedelta(microseconds=1)
            claim = claim.model_copy(
                update={
                    "claimed_at": claimed_at,
                    "lease_expires_at": claimed_at + timedelta(seconds=request.lease_seconds),
                }
            )
        elif self.forge_recovery == "overlong_replacement":
            claim = claim.model_copy(
                update={"lease_expires_at": claim.lease_expires_at + timedelta(seconds=1)}
            )
        elif self.forge_recovery == "skipped_generation":
            claim = claim.model_copy(
                update={
                    "generation": request.generation,
                    "request_sha256": work_attempt_execution_claim_request_sha256(request),
                }
            )
        elif self.forge_recovery == "premature_activation":
            return admission.model_copy(
                update={
                    "state": WorkAttemptAdmissionState.ACTIVE,
                    "recovery_evidence_sha256": _digest("forged-recovery-evidence"),
                }
            )
        else:
            raise AssertionError(f"Unknown recovery forgery: {self.forge_recovery}")
        return admission.model_copy(update={"claim": claim})

    async def submit_admitted_completion_proposal(
        self,
        request: AdmittedCompletionProposalRequest,
    ) -> CompletionProposal:
        proposal = await super().submit_admitted_completion_proposal(request)
        if self.forge_proposal:
            return proposal.model_copy(update={"result": _result_reference("forged-result")})
        return proposal


class _SecretBearingTaskResultStore(InMemoryTaskStore):
    """Return a valid task model whose requested identity was substituted."""

    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self, secret: str, *, clock) -> None:
        super().__init__(clock=clock)
        self._secret = secret
        self.forge_task_result = False

    async def load_task(self, task_id: str) -> Task | None:
        task = await super().load_task(task_id)
        if task is None or not self.forge_task_result:
            return task
        return task.model_copy(update={"id": self._secret})


class _SecretBearingSessionCreationResult(InMemorySessionStore):
    """Commit session creation, then substitute a secret-bearing return identity."""

    def __init__(self, secret: str) -> None:
        super().__init__()
        self._secret = secret

    async def create(
        self,
        request: RunRequest,
        *,
        identity: SessionIdentity,
        interaction_started_event: Event | None = None,
        interaction_source_messages: list[Message] | None = None,
        checkpoint_transform: CheckpointTransform | None = None,
    ) -> Session:
        session = await super().create(
            request,
            identity=identity,
            interaction_started_event=interaction_started_event,
            interaction_source_messages=interaction_source_messages,
            checkpoint_transform=checkpoint_transform,
        )
        return session.model_copy(update={"id": self._secret})


class _SecretBearingCheckpointResult(InMemorySessionStore):
    """Return a secret-bearing checkpoint after committing session creation."""

    def __init__(self, secret: str) -> None:
        super().__init__()
        self._secret = secret

    async def load_checkpoint(self, session_id: str) -> dict[str, object] | None:
        checkpoint = await super().load_checkpoint(session_id)
        if checkpoint is None:
            return {"extension_diagnostic": self._secret}
        return {**checkpoint, "extension_diagnostic": self._secret}


class _FailWorkAttemptDeferredInputRead(InMemorySessionStore):
    """Raise one secret-bearing authority-read failure after session publication."""

    def __init__(self, secret: str) -> None:
        super().__init__()
        self._secret = secret

    async def load_deferred_interaction_input(
        self,
        session_id: str,
    ) -> DeferredInteractionInput | None:
        del session_id
        raise RuntimeError(self._secret)


class _BlockingWorkAttemptEventSink(EventSink):
    """Retain one real sink delivery until the test releases its effect."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.started.set()
        await self.release.wait()
        self.events.append(event.model_copy(deep=True))


class _LeaseAdvancingWorkAttemptSessionStore(InMemorySessionStore):
    """Spend more than one task-store lease during selected event lookups."""

    def __init__(self, now: list[datetime]) -> None:
        super().__init__()
        self._now = now
        self.delay_next_handoff = False

    async def query_events(self, query: EventQuery | None = None) -> list[EventRecord]:
        if (
            self.delay_next_handoff
            and query is not None
            and query.event_type is EventType.INTERACTION_STARTED
        ):
            self.delay_next_handoff = False
            await asyncio.sleep(0.2)
            self._now[0] += timedelta(seconds=0.6)
            await asyncio.sleep(0.2)
            self._now[0] += timedelta(seconds=0.6)
        return await super().query_events(query)


class _PartialWorkAttemptTaskStore(InMemoryTaskStore):
    """Advertise the feature while implementing only its first mutation."""

    supports_work_attempt_admission = True
    verified_work_mutations_are_cancellation_quiescent = True
    activate_work_attempt_admission = TaskStore.activate_work_attempt_admission

    def __init__(self) -> None:
        super().__init__()
        self.prepare_calls = 0

    async def prepare_work_attempt_admission(
        self,
        request: WorkAttemptAdmissionPrepare,
    ) -> WorkAttemptAdmission:
        self.prepare_calls += 1
        return await super().prepare_work_attempt_admission(request)


class _RacingWorkAttemptClaimStore(InMemoryTaskStore):
    """Pause one claim after the runtime's lookup while a peer activates it."""

    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self, *, clock) -> None:
        super().__init__(clock=clock)
        self.lose_next_prepare_ack = False
        self.block_next_claim = False
        self.claim_started = asyncio.Event()
        self.release_claim = asyncio.Event()

    async def prepare_work_attempt_admission(
        self,
        request: WorkAttemptAdmissionPrepare,
    ) -> WorkAttemptAdmission:
        admission = await super().prepare_work_attempt_admission(request)
        if self.lose_next_prepare_ack:
            self.lose_next_prepare_ack = False
            raise RuntimeError("injected preparation acknowledgement loss")
        return admission

    async def claim_work_attempt_recovery(
        self,
        request: WorkAttemptExecutionClaimRequest,
    ) -> WorkAttemptAdmission:
        if self.block_next_claim:
            self.block_next_claim = False
            self.claim_started.set()
            await self.release_claim.wait()
        return await super().claim_work_attempt_recovery(request)


class _RacingWorkAttemptRenewalStore(InMemoryTaskStore):
    """Pause one renewal after the runtime has captured its prior authority."""

    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self, *, clock) -> None:
        super().__init__(clock=clock)
        self.block_next_renewal = False
        self.renewal_started = asyncio.Event()
        self.release_renewal = asyncio.Event()

    async def renew_work_attempt_execution_claim(
        self,
        request: WorkAttemptExecutionClaimRequest,
    ) -> WorkAttemptAdmission:
        if self.block_next_renewal:
            self.block_next_renewal = False
            self.renewal_started.set()
            await self.release_renewal.wait()
        return await super().renew_work_attempt_execution_claim(request)


class _RacingWorkAttemptProposalStore(InMemoryTaskStore):
    """Pause a restarted proposal retry after its active-authority lookup."""

    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self, *, clock) -> None:
        super().__init__(clock=clock)
        self.block_next_proposal = False
        self.proposal_started = asyncio.Event()
        self.release_proposal = asyncio.Event()

    async def submit_admitted_completion_proposal(
        self,
        request: AdmittedCompletionProposalRequest,
    ) -> CompletionProposal:
        if self.block_next_proposal:
            self.block_next_proposal = False
            self.proposal_started.set()
            await self.release_proposal.wait()
        return await super().submit_admitted_completion_proposal(request)


class _IneligibleWorkAttemptTransitionStore(InMemoryTaskStore):
    """Return validly shaped authority from a state the runtime must reject."""

    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self, *, clock) -> None:
        super().__init__(clock=clock)
        self.forge_ineligible_renewal = False
        self.forge_ineligible_recovery = False

    async def renew_work_attempt_execution_claim(
        self,
        request: WorkAttemptExecutionClaimRequest,
    ) -> WorkAttemptAdmission:
        if not self.forge_ineligible_renewal:
            return await super().renew_work_attempt_execution_claim(request)
        admission = await self.load_work_attempt_admission(request.admission_id)
        assert admission is not None
        return admission.model_copy(update={"state": WorkAttemptAdmissionState.ACTIVE})

    async def claim_work_attempt_recovery(
        self,
        request: WorkAttemptExecutionClaimRequest,
    ) -> WorkAttemptAdmission:
        if not self.forge_ineligible_recovery:
            return await super().claim_work_attempt_recovery(request)
        admission = await self.load_work_attempt_admission(request.admission_id)
        assert admission is not None
        claimed_at = max(self._clock(), admission.claim.lease_expires_at)
        claim = admission.claim.model_copy(
            update={
                "claim_id": request.claim_id,
                "worker_id": request.worker_id,
                "execution_owner_id": request.execution_owner_id,
                "generation": request.generation,
                "request_sha256": work_attempt_execution_claim_request_sha256(request),
                "claimed_at": claimed_at,
                "lease_expires_at": claimed_at + timedelta(seconds=request.lease_seconds),
            }
        )
        return admission.model_copy(
            update={
                "state": WorkAttemptAdmissionState.RECOVERING,
                "claim": claim,
                "recovery_evidence_sha256": None,
            }
        )


class _BlockFirstWorkAttemptSessionCreation(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.create_started = asyncio.Event()
        self.release_create = asyncio.Event()
        self._block_first_create = True

    async def create(
        self,
        request: RunRequest,
        *,
        identity: SessionIdentity,
        interaction_started_event: Event | None = None,
        interaction_source_messages: list[Message] | None = None,
        checkpoint_transform: CheckpointTransform | None = None,
    ) -> Session:
        if self._block_first_create:
            self._block_first_create = False
            self.create_started.set()
            await self.release_create.wait()
        return await super().create(
            request,
            identity=identity,
            interaction_started_event=interaction_started_event,
            interaction_source_messages=interaction_source_messages,
            checkpoint_transform=checkpoint_transform,
        )


class _BlockFirstWorkAttemptRecoveryTransition(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.transition_started = asyncio.Event()
        self.release_transition = asyncio.Event()
        self._block_first_transition = True

    async def transition_status_and_checkpoint(
        self,
        session_id: str,
        **kwargs,
    ) -> Session:
        if self._block_first_transition:
            self._block_first_transition = False
            self.transition_started.set()
            await self.release_transition.wait()
        return await super().transition_status_and_checkpoint(session_id, **kwargs)


class _BlockFirstWorkAttemptSettlementFence(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.fence_committed = asyncio.Event()
        self.release_fence = asyncio.Event()
        self._block_first_fence = True

    async def fence_run_and_transform_checkpoint(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        checkpoint_transform: CheckpointTransform,
    ) -> Session:
        fenced = await super().fence_run_and_transform_checkpoint(
            session_id,
            statuses=statuses,
            checkpoint_transform=checkpoint_transform,
        )
        if self._block_first_fence:
            self._block_first_fence = False
            self.fence_committed.set()
            await self.release_fence.wait()
        return fenced


class _BlockNextWorkAttemptCheckpointLoad(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.load_started = asyncio.Event()
        self.release_load = asyncio.Event()
        self.block_next_load = False

    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        if self.block_next_load:
            self.block_next_load = False
            self.load_started.set()
            await self.release_load.wait()
        return await super().load_checkpoint(session_id)


class _FailOnceWorkAttemptRunFenceRelease(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next_release = False

    async def release_run_fence(self, session_id: str) -> None:
        if self.fail_next_release:
            self.fail_next_release = False
            raise RuntimeError("transient work-attempt run fence release failure")
        await super().release_run_fence(session_id)


class _BlockFirstWorkAttemptRunFenceRelease(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.release_started = asyncio.Event()
        self.release_cleanup = asyncio.Event()
        self.release_calls = 0

    async def release_run_fence(self, session_id: str) -> None:
        self.release_calls += 1
        if self.release_calls == 1:
            self.release_started.set()
            await self.release_cleanup.wait()
        await super().release_run_fence(session_id)


class _FailWorkAttemptSettlementFence(InMemorySessionStore):
    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    async def fence_run_and_transform_checkpoint(
        self,
        session_id: str,
        *,
        statuses: set[SessionStatus],
        checkpoint_transform: CheckpointTransform,
    ) -> Session:
        del session_id, statuses, checkpoint_transform
        raise RuntimeError(self._message)


class _BlockFirstWorkAttemptContinuationAdmission(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.admission_started = asyncio.Event()
        self.release_admission = asyncio.Event()
        self._block_first_admission = True

    async def admit_session_invocation(self, session_id: str, **kwargs):
        if self._block_first_admission:
            self._block_first_admission = False
            self.admission_started.set()
            await self.release_admission.wait()
        return await super().admit_session_invocation(session_id, **kwargs)


class _CancelFirstWorkAttemptSessionCreation(InMemorySessionStore):
    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    async def create(
        self,
        request: RunRequest,
        *,
        identity: SessionIdentity,
        interaction_started_event: Event | None = None,
        interaction_source_messages: list[Message] | None = None,
        checkpoint_transform: CheckpointTransform | None = None,
    ) -> Session:
        del (
            request,
            identity,
            interaction_started_event,
            interaction_source_messages,
            checkpoint_transform,
        )
        raise asyncio.CancelledError(self._message)


class _FailCancelledWorkAttemptSessionCreation(InMemorySessionStore):
    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message
        self.create_started = asyncio.Event()
        self.release_create = asyncio.Event()

    async def create(
        self,
        request: RunRequest,
        *,
        identity: SessionIdentity,
        interaction_started_event: Event | None = None,
        interaction_source_messages: list[Message] | None = None,
        checkpoint_transform: CheckpointTransform | None = None,
    ) -> Session:
        del (
            request,
            identity,
            interaction_started_event,
            interaction_source_messages,
            checkpoint_transform,
        )
        self.create_started.set()
        await self.release_create.wait()
        raise RuntimeError(self._message)


class _FailWorkAttemptSessionCreation(InMemorySessionStore):
    def __init__(self, message: str) -> None:
        super().__init__()
        self._message = message

    async def create(
        self,
        request: RunRequest,
        *,
        identity: SessionIdentity,
        interaction_started_event: Event | None = None,
        interaction_source_messages: list[Message] | None = None,
        checkpoint_transform: CheckpointTransform | None = None,
    ) -> Session:
        del (
            request,
            identity,
            interaction_started_event,
            interaction_source_messages,
            checkpoint_transform,
        )
        raise RuntimeError(self._message)


def _assert_secret_absent_from_work_attempt_error(error: BaseException, secret: str) -> None:
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        assert secret not in str(current)
        assert secret not in repr(current)
        assert secret not in "".join(traceback_module.format_exception(current))
        traceback = current.__traceback__
        while traceback is not None:
            if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
                assert all(
                    secret not in repr(value) for value in traceback.tb_frame.f_locals.values()
                )
            traceback = traceback.tb_next
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)


def _work_attempt_failure_leaf_types(
    error: BaseException,
) -> tuple[type[BaseException], ...]:
    pending = [error]
    leaves: list[type[BaseException]] = []
    while pending:
        current = pending.pop()
        if isinstance(current, BaseExceptionGroup):
            pending.extend(reversed(current.exceptions))
        else:
            leaves.append(type(current))
    return tuple(leaves)


def _prepare_request(
    *,
    task_id: str,
    session_id: str,
    session_invocation,
    admission_id: str = "admission-1",
    attempt_id: str = "attempt-1",
    claim_id: str = "execution-claim-1",
    worker_id: str = "worker-1",
    execution_owner_id: str = "runtime-owner-1",
) -> WorkAttemptAdmissionPrepare:
    return WorkAttemptAdmissionPrepare(
        admission_id=admission_id,
        claim_id=claim_id,
        attempt_id=attempt_id,
        task_id=task_id,
        session_id=session_id,
        interaction_id=f"interaction:{attempt_id}",
        worker_id=worker_id,
        execution_owner_id=execution_owner_id,
        generation=1,
        lease_seconds=300,
        kind="initial",
        source_request_sha256=_digest("source-request"),
        contract=_contract().reference(),
        session_invocation=session_invocation,
        source_execution_profile_fingerprint=_digest("worker-profile"),
    )


async def _configured_public_initial_admission(
    *,
    prefix: str,
    sessions: SessionStore,
    tasks: TaskStore,
    redactor: SecretRedactor,
    agent_system_prompt: str | None = None,
) -> tuple[CayuApp, RunRequest, WorkAttemptExecutionRequest]:
    app = CayuApp(
        session_store=sessions,
        task_store=tasks,
        secret_redactor=redactor,
        enable_logging=False,
    )
    app.register_provider(_RecordingProvider(), default=True)
    app.register_agent(
        AgentSpec(
            name="worker",
            model="verified-work-test-model",
            system_prompt=agent_system_prompt,
        )
    )
    contract = _contract(contract_id=f"{prefix}-contract")
    await tasks.publish_work_contract(contract)
    task = await tasks.create_task(
        TaskCreate(
            task_id=f"{prefix}-task",
            type="verified-work",
            work_contract=contract.reference(),
        )
    )
    return (
        app,
        RunRequest(
            agent_name="worker",
            task_id=task.id,
            session_id=f"{prefix}-session",
            messages=[Message.text("user", "Admit this exact governed attempt.")],
        ),
        WorkAttemptExecutionRequest(
            admission_id=f"{prefix}-admission",
            claim_id=f"{prefix}-claim",
            attempt_id=f"{prefix}-attempt",
            interaction_id=f"{prefix}-interaction",
            worker_id=f"{prefix}-worker",
            generation=1,
            lease_seconds=300,
        ),
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_execution_claim_identity_is_global_and_immutable(
    backend: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        store = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "claim-identity.db")
        )
        contract = _contract(contract_id=f"claim-identity-{backend}")
        await store.publish_work_contract(contract)
        tasks = [
            await store.create_task(
                TaskCreate(
                    task_id=f"claim-identity-task-{backend}-{index}",
                    type="verified-work",
                    work_contract=contract.reference(),
                )
            )
            for index in range(2)
        ]
        requests = []
        for index, task in enumerate(tasks):
            session_id = f"claim-identity-session-{backend}-{index}"
            requests.append(
                _prepare_request(
                    task_id=task.id,
                    session_id=session_id,
                    session_invocation=await task_backed_session_invocation(
                        store,
                        task.id,
                        session_id,
                    ),
                    admission_id=f"claim-identity-admission-{backend}-{index}",
                    attempt_id=f"claim-identity-attempt-{backend}-{index}",
                    claim_id=f"shared-claim-identity-{backend}",
                    worker_id=f"claim-identity-worker-{backend}-{index}",
                    execution_owner_id=f"claim-identity-owner-{backend}-{index}",
                ).model_copy(update={"contract": contract.reference()})
            )

        await store.prepare_work_attempt_admission(requests[0])
        with pytest.raises(WorkAttemptAdmissionConflict, match="claim identity"):
            await store.prepare_work_attempt_admission(requests[1])
        untouched = await store.load_task(tasks[1].id)
        assert untouched is not None
        assert untouched.status is TaskStatus.PENDING
        assert untouched.session_id is None
        assert untouched.worker_id is None

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_session_interaction_identity_is_global_and_immutable(
    backend: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        store = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "session-interaction-identity.db")
        )
        contract = _contract(contract_id=f"session-interaction-contract-{backend}")
        await store.publish_work_contract(contract)
        tasks = [
            await store.create_task(
                TaskCreate(
                    task_id=f"session-interaction-task-{backend}-{index}",
                    type="verified-work",
                    work_contract=contract.reference(),
                )
            )
            for index in range(2)
        ]
        session_id = f"shared-session-interaction-session-{backend}"
        requests = [
            _prepare_request(
                task_id=task.id,
                session_id=session_id,
                session_invocation=await task_backed_session_invocation(
                    store,
                    task.id,
                    session_id,
                ),
                admission_id=f"session-interaction-admission-{backend}-{index}",
                attempt_id=f"session-interaction-attempt-{backend}-{index}",
                claim_id=f"session-interaction-claim-{backend}-{index}",
                worker_id=f"session-interaction-worker-{backend}-{index}",
                execution_owner_id=f"session-interaction-owner-{backend}-{index}",
            ).model_copy(
                update={
                    "contract": contract.reference(),
                    "interaction_id": f"shared-session-interaction-{backend}",
                }
            )
            for index, task in enumerate(tasks)
        ]

        await store.prepare_work_attempt_admission(requests[0])
        with pytest.raises(WorkAttemptAdmissionConflict, match="Session interaction"):
            await store.prepare_work_attempt_admission(requests[1])
        untouched = await store.load_task(tasks[1].id)
        assert untouched is not None
        assert untouched.status is TaskStatus.PENDING
        assert untouched.session_id is None
        assert untouched.worker_id is None

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_one_unreleased_admission_owns_a_session_across_interactions(
    backend: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        sessions = _BlockFirstWorkAttemptSessionCreation()
        tasks = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "session-admission-owner.db")
        )
        provider = _RecordingProvider()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        contract = _contract(contract_id=f"session-admission-owner-{backend}")
        await tasks.publish_work_contract(contract)
        task_records = [
            await tasks.create_task(
                TaskCreate(
                    task_id=f"session-admission-owner-task-{backend}-{index}",
                    type="verified-work",
                    work_contract=contract.reference(),
                )
            )
            for index in range(2)
        ]
        session_id = f"session-admission-owner-session-{backend}"

        def run(index: int) -> RunRequest:
            return RunRequest(
                agent_name="worker",
                task_id=task_records[index].id,
                session_id=session_id,
                messages=[Message.text("user", f"attempt {index}")],
            )

        def execution(index: int) -> WorkAttemptExecutionRequest:
            return WorkAttemptExecutionRequest(
                admission_id=f"session-admission-owner-{backend}-{index}",
                claim_id=f"session-admission-owner-claim-{backend}-{index}",
                attempt_id=f"session-admission-owner-attempt-{backend}-{index}",
                interaction_id=f"session-admission-owner-interaction-{backend}-{index}",
                worker_id=f"session-admission-owner-worker-{backend}-{index}",
                generation=1,
                lease_seconds=300,
            )

        first = asyncio.create_task(app.admit_work_attempt(run(0), execution=execution(0)))
        await asyncio.wait_for(sessions.create_started.wait(), timeout=10)
        with pytest.raises(
            WorkAttemptAdmissionConflict,
            match="Session already has an unreleased",
        ):
            await app.admit_work_attempt(run(1), execution=execution(1))
        sessions.release_create.set()
        admitted = await first
        assert admitted.state is WorkAttemptAdmissionState.ACTIVE
        assert admitted.session_id == session_id
        untouched = await tasks.load_task(task_records[1].id)
        assert untouched is not None
        assert untouched.status is TaskStatus.PENDING
        assert untouched.session_id is None
        assert untouched.worker_id is None
        assert await tasks.load_work_attempt_admission(execution(1).admission_id) is None
        assert provider.requests == []
        if isinstance(tasks, SQLiteTaskStore):
            await tasks.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_admission_keeps_queue_lease_time_separate_from_verified_work_clock(
    backend: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        verified_now = datetime(2100, 1, 1, tzinfo=UTC)
        store = (
            InMemoryTaskStore(clock=lambda: verified_now)
            if backend == "memory"
            else SQLiteTaskStore(
                tmp_path / "admission-queue-clock.db",
                clock=lambda: verified_now,
            )
        )
        contract = _contract(contract_id=f"admission-queue-clock-{backend}")
        await store.publish_work_contract(contract)
        task = await store.create_task(
            TaskCreate(
                task_id=f"admission-queue-clock-task-{backend}",
                type="verified-work",
                available_at=datetime(2099, 1, 1, tzinfo=UTC),
                work_contract=contract.reference(),
            )
        )
        worker_id = f"admission-queue-clock-worker-{backend}"
        claimed = await store.claim_task(worker_id)
        assert claimed is not None
        session_id = f"admission-queue-clock-session-{backend}"
        request = _prepare_request(
            task_id=task.id,
            session_id=session_id,
            session_invocation=await task_backed_session_invocation(
                store,
                task.id,
                session_id,
            ),
            admission_id=f"admission-queue-clock-admission-{backend}",
            attempt_id=f"admission-queue-clock-attempt-{backend}",
            claim_id=f"admission-queue-clock-claim-{backend}",
            worker_id=worker_id,
        ).model_copy(update={"contract": contract.reference()})
        prepared = await store.prepare_work_attempt_admission(request)
        assert prepared.claim.claimed_at == verified_now
        assert prepared.claim.lease_expires_at > prepared.claim.claimed_at
        if isinstance(store, SQLiteTaskStore):
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_admission_rejects_expired_queue_lease_despite_behind_verified_work_clock(
    backend: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        verified_now = datetime(2000, 1, 1, tzinfo=UTC)
        store = (
            InMemoryTaskStore(clock=lambda: verified_now)
            if backend == "memory"
            else SQLiteTaskStore(
                tmp_path / "expired-admission-queue-clock.db",
                clock=lambda: verified_now,
            )
        )
        contract = _contract(contract_id=f"expired-admission-queue-clock-{backend}")
        await store.publish_work_contract(contract)
        task = await store.create_task(
            TaskCreate(
                task_id=f"expired-admission-queue-clock-task-{backend}",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        worker_id = f"expired-admission-queue-clock-worker-{backend}"
        claimed = await store.claim_task(worker_id)
        assert claimed is not None
        expired_at = datetime.now(UTC) - timedelta(seconds=1)
        if isinstance(store, InMemoryTaskStore):
            async with store._lock:
                current = store._require_task(task.id)
                store._store_task(current.model_copy(update={"lease_expires_at": expired_at}))
        else:
            async with store._lock:
                with store._connection:
                    store._connection.execute(
                        "UPDATE cayu_tasks SET lease_expires_at = ? WHERE id = ?",
                        (sqlite_support.format_datetime(expired_at), task.id),
                    )
        session_id = f"expired-admission-queue-clock-session-{backend}"
        request = _prepare_request(
            task_id=task.id,
            session_id=session_id,
            session_invocation=await task_backed_session_invocation(
                store,
                task.id,
                session_id,
            ),
            admission_id=f"expired-admission-queue-clock-admission-{backend}",
            attempt_id=f"expired-admission-queue-clock-attempt-{backend}",
            claim_id=f"expired-admission-queue-clock-claim-{backend}",
            worker_id=worker_id,
        ).model_copy(update={"contract": contract.reference()})
        with pytest.raises(TaskClaimLost, match="expired"):
            await store.prepare_work_attempt_admission(request)
        untouched = await store.load_task(task.id)
        assert untouched is not None
        assert untouched.worker_id == worker_id
        assert await store.load_work_attempt_admission(request.admission_id) is None
        if isinstance(store, SQLiteTaskStore):
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_work_attempt_admission_is_exact_and_claim_fenced(
    backend: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        store = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "admission.db")
        )
        contract = _contract()
        await store.publish_work_contract(contract)
        task = await store.create_task(
            TaskCreate(
                task_id="contract-task",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        session_id = "contract-session"
        request = _prepare_request(
            task_id=task.id,
            session_id=session_id,
            session_invocation=await task_backed_session_invocation(
                store,
                task.id,
                session_id,
            ),
        )

        with pytest.raises(WorkAttemptAdmissionConflict, match="Continuation admission"):
            await store.prepare_work_attempt_admission(
                request.model_copy(
                    update={
                        "kind": "continuation",
                        "predecessor_admission_id": "missing-predecessor-admission",
                    }
                )
            )
        untouched = await store.load_task(task.id)
        assert untouched is not None
        assert untouched.status is TaskStatus.PENDING

        prepared = await store.prepare_work_attempt_admission(request)
        assert prepared.state is WorkAttemptAdmissionState.PREPARING
        assert prepared.attempt is None
        assert await store.prepare_work_attempt_admission(request) == prepared
        reserved = await store.load_task(task.id)
        assert reserved is not None
        assert reserved.status is TaskStatus.RUNNING
        assert reserved.session_id == session_id
        assert reserved.session_instance_id == request.session_invocation.session_instance_id
        assert reserved.worker_id == request.worker_id

        with pytest.raises(WorkAttemptAdmissionConflict, match="runtime-owned"):
            await store.begin_work_attempt(
                WorkAttemptCreate(
                    attempt_id="bypass-attempt",
                    task_id=task.id,
                    session_id=session_id,
                    contract=contract.reference(),
                    execution_profile_fingerprint=_digest("worker-profile"),
                    worker_id=request.worker_id,
                )
            )

        with pytest.raises(WorkAttemptAdmissionConflict, match="another request"):
            await store.prepare_work_attempt_admission(
                request.model_copy(update={"interaction_id": "conflicting-interaction"})
            )

        activation = WorkAttemptAdmissionActivate(
            admission_id=request.admission_id,
            claim_id=request.claim_id,
            prepare_request_sha256=prepared.prepare_request_sha256,
            session_evidence_sha256=_digest("session-admission-evidence"),
        )
        admitted = await store.activate_work_attempt_admission(activation)
        assert admitted.state is WorkAttemptAdmissionState.ACTIVE
        assert admitted.attempt is not None
        assert admitted.attempt.ordinal == 1
        assert await store.activate_work_attempt_admission(activation) == admitted

        with pytest.raises(WorkAttemptExecutionClaimLost):
            await store.heartbeat(task.id, request.worker_id)
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await store.release_attached_task_worker(task.id, request.worker_id)
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await store.pause_task(task.id, reason="ordinary-pause")
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await store.block_task(task.id, reason="ordinary-block")
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await store.mark_task_needs_attention(task.id, reason="ordinary-attention")
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await store.resume_task(task.id)
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await store.cancel_task(task.id)
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await store.complete_task(task.id, _task_result(), worker_id=request.worker_id)
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await store.fail_task(
                task.id,
                {"message": "ordinary worker failure"},
                worker_id=request.worker_id,
            )
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await store.terminalize_task(
                TaskTerminalizationRequest(
                    task_id=task.id,
                    worker_id=request.worker_id,
                    idempotency_key="ordinary-terminalization",
                    kind=TaskTerminalKind.FAILED,
                    error={"message": "stale failure"},
                )
            )

        with pytest.raises(WorkAttemptExecutionClaimLost):
            await store.submit_admitted_completion_proposal(
                AdmittedCompletionProposalRequest(
                    admission_id=admitted.admission_id,
                    claim_id=admitted.claim.claim_id,
                    execution_owner_id="conflicting-runtime-owner",
                    generation=admitted.claim.generation,
                    proposal=CompletionProposalCreate(
                        proposal_id="wrong-owner-proposal",
                        attempt_id=admitted.attempt_id,
                        result=_result_reference(),
                        evidence_references=(_artifact_evidence(),),
                    ),
                )
            )

        renewal = WorkAttemptExecutionClaimRequest(
            admission_id=admitted.admission_id,
            claim_id=admitted.claim.claim_id,
            worker_id=admitted.claim.worker_id,
            execution_owner_id=admitted.claim.execution_owner_id,
            generation=admitted.claim.generation,
            lease_seconds=301,
        )
        renewed = await store.renew_work_attempt_execution_claim(renewal)
        assert renewed.claim.claimed_at == admitted.claim.claimed_at
        assert renewed.claim.lease_expires_at >= admitted.claim.lease_expires_at

        proposal_request = AdmittedCompletionProposalRequest(
            admission_id=admitted.admission_id,
            claim_id=admitted.claim.claim_id,
            execution_owner_id=admitted.claim.execution_owner_id,
            generation=admitted.claim.generation,
            proposal=CompletionProposalCreate(
                proposal_id="proposal-1",
                attempt_id=admitted.attempt_id,
                result=_result_reference(),
                evidence_references=(_artifact_evidence(),),
            ),
        )
        proposal = await store.submit_admitted_completion_proposal(proposal_request)
        assert proposal.attempt_id == admitted.attempt_id
        assert await store.submit_admitted_completion_proposal(proposal_request) == proposal
        with pytest.raises(WorkCompletionConflict, match="proposal replay"):
            await store.submit_admitted_completion_proposal(
                proposal_request.model_copy(
                    update={
                        "proposal": proposal_request.proposal.model_copy(
                            update={"result": _result_reference("changed")}
                        )
                    }
                )
            )
        released = await store.load_work_attempt_admission(admitted.admission_id)
        assert released is not None
        assert released.state is WorkAttemptAdmissionState.RELEASED
        released_task = await store.load_task(task.id)
        assert released_task is not None
        assert released_task.session_instance_id == request.session_invocation.session_instance_id
        assert released_task.worker_id is None
        assert released_task.lease_expires_at is None

        with pytest.raises(WorkAttemptExecutionClaimLost):
            await store.complete_task(task.id, _task_result())
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await store.fail_task(task.id, {"message": "stale failure"})
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await store.cancel_task(task.id, {"message": "stale cancellation"})
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await store.pause_task(task.id, reason="stale pause")
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await store.block_task(task.id, reason="stale block")
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await store.mark_task_needs_attention(task.id, reason="stale attention")
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await store.resume_task(task.id)
        with pytest.raises(TaskClaimLost):
            await store.complete_task(task.id, _task_result(), worker_id=request.worker_id)
        with pytest.raises(TaskClaimLost):
            await store.fail_task(
                task.id,
                {"message": "stale worker failure"},
                worker_id=request.worker_id,
            )
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await store.terminalize_task(
                TaskTerminalizationRequest(
                    task_id=task.id,
                    worker_id=request.worker_id,
                    idempotency_key="post-proposal-terminalization",
                    kind=TaskTerminalKind.FAILED,
                    error={"message": "stale terminalization"},
                )
            )
        with pytest.raises(TaskClaimLost):
            await store.heartbeat(task.id, request.worker_id)
        with pytest.raises(TaskClaimLost):
            await store.release_task(task.id, request.worker_id)
        with pytest.raises(TaskClaimLost):
            await store.release_attached_task_worker(task.id, request.worker_id)
        assert await store.load_task(task.id) == released_task
        assert await store.submit_admitted_completion_proposal(proposal_request) == proposal

        with pytest.raises(WorkAttemptExecutionClaimLost):
            await store.renew_work_attempt_execution_claim(renewal)

    asyncio.run(scenario())


def test_public_admission_persists_incarnation_through_result_resolution() -> None:
    async def scenario() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        app, run, execution = await _configured_public_initial_admission(
            prefix="result-incarnation",
            sessions=sessions,
            tasks=tasks,
            redactor=SecretRedactor(),
        )
        contract = _contract(contract_id="result-incarnation-contract")

        admitted = await app.admit_work_attempt(run, execution=execution)
        session = await sessions.load(admitted.session_id)
        task = await tasks.load_task(admitted.task_id)
        assert session is not None
        assert task is not None
        assert task.session_instance_id == session.instance_id
        assert task.session_instance_id == admitted.session_invocation.session_instance_id

        proposal = await app.submit_work_attempt_proposal(
            WorkAttemptProposalRequest(
                admission_id=admitted.admission_id,
                claim_id=admitted.claim.claim_id,
                generation=admitted.claim.generation,
                proposal=CompletionProposalCreate(
                    proposal_id="result-incarnation-proposal",
                    attempt_id=admitted.attempt_id,
                    result=_result_reference(),
                    evidence_references=(_artifact_evidence(),),
                ),
            )
        )
        verifier_claim = await _claim_completion_verification(
            tasks,
            CompletionVerificationClaimRequest(
                claim_id="result-incarnation-verifier-claim",
                proposal_id=proposal.proposal_id,
                worker_id="result-incarnation-verifier-worker",
                verifier=contract.verifier,
                verifier_profile_fingerprint=_verifier_profile_fingerprint(contract.verifier),
            ),
        )
        decision = await tasks.record_completion_decision(
            _accepted_decision(
                proposal_id=proposal.proposal_id,
                claim_id=verifier_claim.claim_id,
                worker_id=verifier_claim.worker_id,
            ).model_copy(update={"decision_id": "result-incarnation-decision"})
        )
        app.register_completion_result_resolver(
            contract.result_resolver,
            _AdmissionResultResolver(),
        )

        completed = await app.resolve_completion_result(
            CompletionResultResolutionRequest(
                task_id=task.id,
                decision_id=decision.decision_id,
                idempotency_key="result-incarnation-resolution",
            )
        )

        assert completed.status is TaskStatus.COMPLETED
        assert completed.result == _task_result()
        assert completed.session_instance_id == session.instance_id

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("phase", ["prepare", "active-replay"])
def test_admission_activation_rejects_task_session_incarnation_drift(
    backend: str,
    phase: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        store = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / f"incarnation-drift-{phase}.db")
        )
        contract = _contract(contract_id=f"incarnation-drift-{backend}-{phase}")
        await store.publish_work_contract(contract)
        task = await store.create_task(
            TaskCreate(
                task_id=f"incarnation-drift-task-{backend}-{phase}",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        session_id = f"incarnation-drift-session-{backend}-{phase}"
        request = _prepare_request(
            task_id=task.id,
            session_id=session_id,
            session_invocation=await task_backed_session_invocation(
                store,
                task.id,
                session_id,
            ),
            admission_id=f"incarnation-drift-admission-{backend}-{phase}",
            attempt_id=f"incarnation-drift-attempt-{backend}-{phase}",
            claim_id=f"incarnation-drift-claim-{backend}-{phase}",
        ).model_copy(update={"contract": contract.reference()})
        prepared = await store.prepare_work_attempt_admission(request)
        activation = WorkAttemptAdmissionActivate(
            admission_id=prepared.admission_id,
            claim_id=prepared.claim.claim_id,
            prepare_request_sha256=prepared.prepare_request_sha256,
            session_evidence_sha256=_digest(f"incarnation-drift-evidence-{phase}"),
        )
        if phase == "active-replay":
            assert (
                await store.activate_work_attempt_admission(activation)
            ).state is WorkAttemptAdmissionState.ACTIVE

        attached = await store.load_task(task.id)
        assert attached is not None
        drifted = attached.model_copy(
            update={"session_instance_id": "00000000-0000-4000-8000-000000000099"}
        )
        if isinstance(store, InMemoryTaskStore):
            store._store_task(drifted)
        else:
            with store._connection:
                store._connection.execute(
                    "UPDATE cayu_tasks SET session_instance_id = ? WHERE id = ?",
                    (drifted.session_instance_id, task.id),
                )

        with pytest.raises(
            WorkAttemptExecutionClaimLost,
            match="exact task-session authority",
        ):
            await store.activate_work_attempt_admission(activation)

        unchanged = await store.load_work_attempt_admission(prepared.admission_id)
        assert unchanged is not None
        assert unchanged.state is (
            WorkAttemptAdmissionState.PREPARING
            if phase == "prepare"
            else WorkAttemptAdmissionState.ACTIVE
        )
        if isinstance(store, SQLiteTaskStore):
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_cancelled_admission_waiter_cannot_publish_delayed_authority(
    backend: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        store = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "cancelled-admission.db")
        )
        contract = _contract(contract_id=f"cancelled-admission-{backend}")
        await store.publish_work_contract(contract)
        task = await store.create_task(
            TaskCreate(
                task_id=f"cancelled-admission-task-{backend}",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        session_id = f"cancelled-admission-session-{backend}"
        request = _prepare_request(
            task_id=task.id,
            session_id=session_id,
            session_invocation=await task_backed_session_invocation(
                store,
                task.id,
                session_id,
            ),
            admission_id=f"cancelled-admission-{backend}",
            attempt_id=f"cancelled-attempt-{backend}",
            claim_id=f"cancelled-claim-{backend}",
        ).model_copy(update={"contract": contract.reference()})

        await store._lock.acquire()
        pending = asyncio.create_task(store.prepare_work_attempt_admission(request))
        await asyncio.sleep(0)
        pending.cancel()
        assert pending.cancelling() == 1
        store._lock.release()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert pending.cancelled()
        assert await store.load_work_attempt_admission(request.admission_id) is None
        unchanged = await store.load_task(task.id)
        assert unchanged is not None
        assert unchanged.status is TaskStatus.PENDING
        assert unchanged.worker_id is None

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_expired_preparation_owner_is_replaced_before_publication(
    backend: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        store = (
            InMemoryTaskStore(clock=lambda: now[0])
            if backend == "memory"
            else SQLiteTaskStore(
                tmp_path / "prepared-replacement.db",
                clock=lambda: now[0],
            )
        )
        contract = _contract(contract_id=f"prepared-replacement-{backend}")
        await store.publish_work_contract(contract)
        task = await store.create_task(
            TaskCreate(
                task_id=f"prepared-replacement-task-{backend}",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        session_id = f"prepared-replacement-session-{backend}"
        prepare = _prepare_request(
            task_id=task.id,
            session_id=session_id,
            session_invocation=await task_backed_session_invocation(
                store,
                task.id,
                session_id,
            ),
            admission_id=f"prepared-replacement-admission-{backend}",
            attempt_id=f"prepared-replacement-attempt-{backend}",
            claim_id=f"prepared-replacement-claim-1-{backend}",
        ).model_copy(
            update={
                "contract": contract.reference(),
                "lease_seconds": 1,
            }
        )
        prepared = await store.prepare_work_attempt_admission(prepare)
        now[0] += timedelta(seconds=2)
        with pytest.raises(WorkAttemptExecutionClaimLost, match="expired"):
            await store.claim_work_attempt_recovery(
                WorkAttemptExecutionClaimRequest(
                    admission_id=prepared.admission_id,
                    claim_id=prepared.claim.claim_id,
                    worker_id=prepared.claim.worker_id,
                    execution_owner_id=prepared.claim.execution_owner_id,
                    generation=prepared.claim.generation,
                    lease_seconds=1,
                )
            )
        with pytest.raises(WorkAttemptAdmissionConflict, match="claim identity"):
            await store.claim_work_attempt_recovery(
                WorkAttemptExecutionClaimRequest(
                    admission_id=prepared.admission_id,
                    claim_id=prepared.claim.claim_id,
                    worker_id=f"prepared-replacement-worker-reused-{backend}",
                    execution_owner_id=f"prepared-replacement-owner-reused-{backend}",
                    generation=2,
                    lease_seconds=300,
                )
            )
        replacement_request = WorkAttemptExecutionClaimRequest(
            admission_id=prepared.admission_id,
            claim_id=f"prepared-replacement-claim-2-{backend}",
            worker_id=f"prepared-replacement-worker-2-{backend}",
            execution_owner_id=f"prepared-replacement-owner-2-{backend}",
            generation=2,
            lease_seconds=300,
        )
        replaced = await store.claim_work_attempt_recovery(replacement_request)
        assert replaced.state is WorkAttemptAdmissionState.PREPARING
        assert replaced.attempt is None
        assert replaced.claim.generation == 2
        assert await store.claim_work_attempt_recovery(replacement_request) == replaced
        active = await store.activate_work_attempt_admission(
            WorkAttemptAdmissionActivate(
                admission_id=replaced.admission_id,
                claim_id=replaced.claim.claim_id,
                prepare_request_sha256=replaced.prepare_request_sha256,
                session_evidence_sha256=_digest(f"prepared-replacement-evidence-{backend}"),
            )
        )
        assert active.state is WorkAttemptAdmissionState.ACTIVE
        assert active.attempt is not None

    asyncio.run(scenario())


@pytest.mark.parametrize("ordinary_operation", ["heartbeat", "cancel"])
def test_sqlite_ordinary_task_mutation_cannot_cross_admission_publication(
    ordinary_operation: str,
    tmp_path,
) -> None:
    path = tmp_path / f"sqlite-admission-race-{ordinary_operation}.db"

    async def scenario() -> None:
        store = SQLiteTaskStore(path)
        contract = _contract(contract_id=f"sqlite-race-{ordinary_operation}")
        await store.publish_work_contract(contract)
        task = await store.create_task(
            TaskCreate(
                task_id=f"sqlite-race-task-{ordinary_operation}",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        worker_id = f"sqlite-race-worker-{ordinary_operation}"
        claimed = await store.claim_task(worker_id)
        assert claimed is not None
        assert claimed.id == task.id
        session_id = f"sqlite-race-session-{ordinary_operation}"
        prepare = _prepare_request(
            task_id=task.id,
            session_id=session_id,
            session_invocation=await task_backed_session_invocation(
                store,
                task.id,
                session_id,
            ),
            admission_id=f"sqlite-race-admission-{ordinary_operation}",
            attempt_id=f"sqlite-race-attempt-{ordinary_operation}",
            claim_id=f"sqlite-race-claim-{ordinary_operation}",
            worker_id=worker_id,
            execution_owner_id=f"sqlite-race-owner-{ordinary_operation}",
        ).model_copy(update={"contract": contract.reference()})

        publication_done = ThreadEvent()
        publication_started = False
        publication_errors: list[BaseException] = []
        published: list[WorkAttemptAdmission] = []
        publication_thread: Thread | None = None

        def publish_admission() -> None:
            async def publish() -> None:
                competing = SQLiteTaskStore(path, schema_mode=SchemaMode.VALIDATE)
                try:
                    published.append(await competing.prepare_work_attempt_admission(prepare))
                except BaseException as exc:
                    publication_errors.append(exc)
                finally:
                    await competing.close()

            try:
                asyncio.run(publish())
            finally:
                publication_done.set()

        expected_update_fragment = (
            "SET lease_expires_at" if ordinary_operation == "heartbeat" else "SET status_reason ="
        )

        def interleave_after_ordinary_preflight(statement: str) -> None:
            nonlocal publication_started, publication_thread
            normalized = " ".join(statement.split())
            if (
                publication_started
                or not normalized.startswith("UPDATE cayu_tasks")
                or expected_update_fragment not in normalized
            ):
                return
            publication_started = True
            publication_thread = Thread(target=publish_admission, daemon=True)
            publication_thread.start()
            if not publication_done.wait(timeout=10):
                raise RuntimeError("Competing admission publication did not settle.")

        store._connection.set_trace_callback(interleave_after_ordinary_preflight)
        try:
            with pytest.raises(WorkAttemptExecutionClaimLost):
                if ordinary_operation == "heartbeat":
                    await store.heartbeat(task.id, worker_id)
                else:
                    await store.cancel_task(task.id)
        finally:
            store._connection.set_trace_callback(None)
            if publication_thread is not None:
                publication_thread.join(timeout=10)

        assert publication_started
        assert publication_errors == []
        assert len(published) == 1
        admission = published[0]
        assert admission.state is WorkAttemptAdmissionState.PREPARING
        durable_task = await store.load_task(task.id)
        assert durable_task is not None
        assert durable_task.status is TaskStatus.RUNNING
        assert durable_task.session_id == admission.session_id
        assert durable_task.worker_id == admission.claim.worker_id
        assert durable_task.lease_expires_at == admission.claim.lease_expires_at
        await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_work_attempt_recovery_requires_expiry_and_positive_activation(
    backend: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        store = (
            InMemoryTaskStore(clock=lambda: now[0])
            if backend == "memory"
            else SQLiteTaskStore(
                tmp_path / "recovery-admission.db",
                clock=lambda: now[0],
            )
        )
        contract = _contract()
        await store.publish_work_contract(contract)
        task = await store.create_task(
            TaskCreate(
                task_id="recover-contract-task",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        session_id = "recover-contract-session"
        request = _prepare_request(
            task_id=task.id,
            session_id=session_id,
            session_invocation=await task_backed_session_invocation(
                store,
                task.id,
                session_id,
            ),
            admission_id="recover-admission",
            attempt_id="recover-attempt",
            claim_id="recover-claim-1",
        )
        prepared = await store.prepare_work_attempt_admission(request)
        active = await store.activate_work_attempt_admission(
            WorkAttemptAdmissionActivate(
                admission_id=request.admission_id,
                claim_id=request.claim_id,
                prepare_request_sha256=prepared.prepare_request_sha256,
                session_evidence_sha256=_digest("initial-session-evidence"),
            )
        )
        replacement_request = WorkAttemptExecutionClaimRequest(
            admission_id=active.admission_id,
            claim_id="recover-claim-2",
            worker_id="worker-2",
            execution_owner_id="runtime-owner-2",
            generation=2,
            lease_seconds=1,
        )
        with pytest.raises(WorkAttemptExecutionClaimLost, match="still owns"):
            await store.claim_work_attempt_recovery(replacement_request)

        now[0] += timedelta(seconds=301)
        recovering = await store.claim_work_attempt_recovery(replacement_request)
        assert recovering.state is WorkAttemptAdmissionState.RECOVERING
        assert recovering.attempt == active.attempt
        assert recovering.interaction_id == active.interaction_id
        now[0] += timedelta(seconds=2)
        with pytest.raises(WorkAttemptExecutionClaimLost, match="recovery claim expired"):
            await store.claim_work_attempt_recovery(replacement_request)
        final_replacement_request = WorkAttemptExecutionClaimRequest(
            admission_id=active.admission_id,
            claim_id="recover-claim-3",
            worker_id="worker-3",
            execution_owner_id="runtime-owner-3",
            generation=3,
            lease_seconds=300,
        )
        recovering = await store.claim_work_attempt_recovery(final_replacement_request)
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await store.activate_work_attempt_recovery(
                WorkAttemptRecoveryActivate(
                    admission_id=recovering.admission_id,
                    claim_id="recover-claim-2",
                    generation=2,
                    recovery_evidence_sha256=_digest("stale-session-recovery"),
                )
            )

        recovery_activation = WorkAttemptRecoveryActivate(
            admission_id=recovering.admission_id,
            claim_id=recovering.claim.claim_id,
            generation=3,
            recovery_evidence_sha256=_digest("quiescent-session-recovery"),
        )
        recovered = await store.activate_work_attempt_recovery(recovery_activation)
        assert recovered.state is WorkAttemptAdmissionState.ACTIVE
        assert recovered.claim.generation == 3
        assert recovered.attempt_id == active.attempt_id

        attached = await store.load_task(task.id)
        assert attached is not None
        drifted_session_instance_id = "00000000-0000-4000-8000-000000000099"
        if isinstance(store, InMemoryTaskStore):
            store._store_task(
                attached.model_copy(update={"session_instance_id": drifted_session_instance_id})
            )
        else:
            with store._connection:
                store._connection.execute(
                    "UPDATE cayu_tasks SET session_instance_id = ? WHERE id = ?",
                    (drifted_session_instance_id, task.id),
                )
        with pytest.raises(
            WorkAttemptExecutionClaimLost,
            match="exact task-session authority",
        ):
            await store.activate_work_attempt_recovery(recovery_activation)
        if isinstance(store, InMemoryTaskStore):
            store._store_task(attached)
        else:
            with store._connection:
                store._connection.execute(
                    "UPDATE cayu_tasks SET session_instance_id = ? WHERE id = ?",
                    (attached.session_instance_id, task.id),
                )

        recovered_proposal_request = AdmittedCompletionProposalRequest(
            admission_id=recovered.admission_id,
            claim_id=recovered.claim.claim_id,
            execution_owner_id=recovered.claim.execution_owner_id,
            generation=recovered.claim.generation,
            proposal=CompletionProposalCreate(
                proposal_id=f"recovered-proposal-{backend}",
                attempt_id=recovered.attempt_id,
                result=_result_reference(),
                evidence_references=(_artifact_evidence(),),
            ),
        )
        recovered_proposal = await store.submit_admitted_completion_proposal(
            recovered_proposal_request
        )
        assert (
            await store.submit_admitted_completion_proposal(recovered_proposal_request)
            == recovered_proposal
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("fault", ["prepare", "activation"])
def test_public_admission_rejects_conflicting_extension_receipts(fault: str) -> None:
    async def scenario() -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        sessions = InMemorySessionStore()
        tasks = _ConflictingWorkAttemptResultStore(clock=lambda: now)
        setattr(tasks, f"forge_{fault}", True)
        sink = InMemoryEventSink()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            event_sinks=(sink,),
            enable_logging=False,
        )
        app.register_provider(_RecordingProvider(), default=True)
        app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        contract = _contract(contract_id=f"conflicting-{fault}-contract")
        await tasks.publish_work_contract(contract)
        task = await tasks.create_task(
            TaskCreate(
                task_id=f"conflicting-{fault}-task",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        run = RunRequest(
            agent_name="worker",
            task_id=task.id,
            session_id=f"conflicting-{fault}-session",
            messages=[Message.text("user", "Prepare exact governed work.")],
        )
        execution = WorkAttemptExecutionRequest(
            admission_id=f"conflicting-{fault}-admission",
            claim_id=f"conflicting-{fault}-claim",
            attempt_id=f"conflicting-{fault}-attempt",
            interaction_id=f"conflicting-{fault}-interaction",
            worker_id=f"conflicting-{fault}-worker",
            generation=1,
            lease_seconds=300,
        )

        with pytest.raises(RuntimeError, match="returned conflicting authority"):
            await app.admit_work_attempt(run, execution=execution)

        durable = await tasks.load_work_attempt_admission(execution.admission_id)
        assert durable is not None
        if fault == "prepare":
            assert durable.prepare_request_sha256 != _digest("forged-prepare-result")
        else:
            assert durable.source_request_sha256 != _digest("forged-activation-result")
        assert durable.state is (
            WorkAttemptAdmissionState.PREPARING
            if fault == "prepare"
            else WorkAttemptAdmissionState.ACTIVE
        )
        setattr(tasks, f"forge_{fault}", False)
        admitted = await app.admit_work_attempt(run, execution=execution)
        assert admitted == await tasks.load_work_attempt_admission(execution.admission_id)
        assert admitted.state is WorkAttemptAdmissionState.ACTIVE
        assert len(sink.events) == 1

    asyncio.run(scenario())


def test_public_renewal_and_proposal_reject_conflicting_extension_receipts() -> None:
    async def scenario() -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        sessions = InMemorySessionStore()
        tasks = _ConflictingWorkAttemptResultStore(clock=lambda: now)
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
        )
        app.register_provider(_RecordingProvider(), default=True)
        app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        contract = _contract(contract_id="conflicting-public-result-contract")
        await tasks.publish_work_contract(contract)
        task = await tasks.create_task(
            TaskCreate(
                task_id="conflicting-public-result-task",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        admitted = await app.admit_work_attempt(
            RunRequest(
                agent_name="worker",
                task_id=task.id,
                session_id="conflicting-public-result-session",
                messages=[Message.text("user", "Produce the governed result.")],
            ),
            execution=WorkAttemptExecutionRequest(
                admission_id="conflicting-public-result-admission",
                claim_id="conflicting-public-result-claim",
                attempt_id="conflicting-public-result-attempt",
                interaction_id="conflicting-public-result-interaction",
                worker_id="conflicting-public-result-worker",
                generation=1,
                lease_seconds=300,
            ),
        )

        renewal = WorkAttemptClaimRenewalRequest(
            admission_id=admitted.admission_id,
            claim_id=admitted.claim.claim_id,
            worker_id=admitted.claim.worker_id,
            generation=admitted.claim.generation,
            lease_seconds=301,
        )
        tasks.forge_renewal_lease = True
        with pytest.raises(RuntimeError, match="conflicting renewed authority"):
            await app.renew_work_attempt_claim(renewal)
        tasks.forge_renewal_lease = False
        tasks.forge_renewal = True
        with pytest.raises(RuntimeError, match="returned conflicting authority"):
            await app.renew_work_attempt_claim(renewal)
        durable_renewal = await tasks.load_work_attempt_admission(admitted.admission_id)
        assert durable_renewal is not None
        tasks.forge_renewal = False
        assert await app.renew_work_attempt_claim(renewal) == durable_renewal

        proposal_request = WorkAttemptProposalRequest(
            admission_id=admitted.admission_id,
            claim_id=admitted.claim.claim_id,
            generation=admitted.claim.generation,
            proposal=CompletionProposalCreate(
                proposal_id="conflicting-public-result-proposal",
                attempt_id=admitted.attempt_id,
                result=_result_reference("expected-result"),
                evidence_references=(_artifact_evidence(),),
            ),
        )
        tasks.forge_proposal = True
        with pytest.raises(RuntimeError, match="conflicting completion proposal"):
            await app.submit_work_attempt_proposal(proposal_request)
        tasks.forge_proposal = False
        proposal = await app.submit_work_attempt_proposal(proposal_request)
        assert proposal.result == proposal_request.proposal.result
        assert proposal.evidence_references == proposal_request.proposal.evidence_references

    asyncio.run(scenario())


def test_public_renewal_accepts_a_concurrent_extension_past_its_stale_snapshot() -> None:
    async def scenario() -> None:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        sessions = InMemorySessionStore()
        tasks = _RacingWorkAttemptRenewalStore(clock=lambda: now[0])
        app, run, execution = await _configured_public_initial_admission(
            prefix="concurrent-renewal",
            sessions=sessions,
            tasks=tasks,
            redactor=SecretRedactor(),
        )
        execution = execution.model_copy(update={"lease_seconds": 1})
        admitted = await app.admit_work_attempt(run, execution=execution)
        renewal = WorkAttemptClaimRenewalRequest(
            admission_id=admitted.admission_id,
            claim_id=admitted.claim.claim_id,
            worker_id=admitted.claim.worker_id,
            generation=admitted.claim.generation,
            lease_seconds=2,
        )

        tasks.block_next_renewal = True
        stale_renewal = asyncio.create_task(app.renew_work_attempt_claim(renewal))
        await tasks.renewal_started.wait()
        now[0] += timedelta(milliseconds=500)
        first = await app.renew_work_attempt_claim(renewal)
        now[0] += timedelta(seconds=1)
        tasks.release_renewal.set()
        second = await stale_renewal

        durable = await tasks.load_work_attempt_admission(admitted.admission_id)
        assert durable is not None
        assert first.claim.lease_expires_at < second.claim.lease_expires_at
        assert second == durable

    asyncio.run(scenario())


def test_restarted_proposal_retry_replays_when_release_wins_after_lookup() -> None:
    async def scenario() -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        sessions = InMemorySessionStore()
        tasks = _RacingWorkAttemptProposalStore(clock=lambda: now)
        owner_app, run, execution = await _configured_public_initial_admission(
            prefix="concurrent-proposal-replay",
            sessions=sessions,
            tasks=tasks,
            redactor=SecretRedactor(),
        )
        admitted = await owner_app.admit_work_attempt(run, execution=execution)
        restarted_app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
        )
        proposal_request = WorkAttemptProposalRequest(
            admission_id=admitted.admission_id,
            claim_id=admitted.claim.claim_id,
            generation=admitted.claim.generation,
            proposal=CompletionProposalCreate(
                proposal_id="concurrent-proposal-replay-proposal",
                attempt_id=admitted.attempt_id,
                result=_result_reference(),
                evidence_references=(_artifact_evidence(),),
            ),
        )

        tasks.block_next_proposal = True
        stale_retry = asyncio.create_task(
            restarted_app.submit_work_attempt_proposal(proposal_request)
        )
        await tasks.proposal_started.wait()
        winner = await owner_app.submit_work_attempt_proposal(proposal_request)
        tasks.release_proposal.set()
        replay = await stale_retry

        assert replay == winner
        released = await tasks.load_work_attempt_admission(admitted.admission_id)
        assert released is not None
        assert released.state is WorkAttemptAdmissionState.RELEASED

    asyncio.run(scenario())


def test_malformed_public_work_attempt_requests_are_secret_safe_across_diagnostics(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "malformed-public-work-attempt-secret"

    async def scenario() -> list[BaseException]:
        app = CayuApp(
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        run = RunRequest(
            agent_name="worker",
            task_id="malformed-public-task",
            session_id="malformed-public-session",
            messages=[Message.text("user", "Validate this request.")],
        )
        execution = WorkAttemptExecutionRequest(
            admission_id="malformed-public-admission",
            claim_id="malformed-public-claim",
            attempt_id="malformed-public-attempt",
            interaction_id="malformed-public-interaction",
            worker_id="malformed-public-worker",
            generation=1,
            lease_seconds=300,
        )
        renewal = WorkAttemptClaimRenewalRequest(
            admission_id="malformed-public-admission",
            claim_id="malformed-public-claim",
            worker_id="malformed-public-worker",
            generation=1,
            lease_seconds=300,
        )
        recovery = WorkAttemptRecoveryRequest(
            admission_id="malformed-public-admission",
            claim_id="malformed-public-recovery-claim",
            worker_id="malformed-public-recovery-worker",
            generation=2,
            lease_seconds=300,
        )
        proposal = WorkAttemptProposalRequest(
            admission_id="malformed-public-admission",
            claim_id="malformed-public-claim",
            generation=1,
            proposal=CompletionProposalCreate(
                proposal_id="malformed-public-proposal",
                attempt_id="malformed-public-attempt",
                result=_result_reference(),
                evidence_references=(_artifact_evidence(),),
            ),
        )

        errors: list[BaseException] = []
        cases = (
            (run.model_copy(), execution.model_copy()),
            (run.model_copy(), execution.model_copy()),
        )
        object.__setattr__(cases[0][0], "messages", [secret])
        object.__setattr__(cases[1][1], "generation", secret)
        object.__setattr__(renewal, "generation", secret)
        object.__setattr__(recovery, "generation", secret)
        object.__setattr__(proposal, "generation", secret)
        operations = (
            app.admit_work_attempt(cases[0][0], execution=cases[0][1]),
            app.admit_work_attempt(cases[1][0], execution=cases[1][1]),
            app.renew_work_attempt_claim(renewal),
            app.recover_work_attempt(recovery),
            app.submit_work_attempt_proposal(proposal),
        )
        for operation in operations:
            try:
                await operation
            except (TypeError, ValueError, RuntimeError) as error:
                errors.append(error)
            else:  # pragma: no cover - safety assertion
                raise AssertionError("Malformed work-attempt request was accepted.")
        return errors

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        errors = asyncio.run(scenario())

    assert len(errors) == 5
    for error in errors:
        _assert_secret_absent_from_work_attempt_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


@pytest.mark.parametrize("result_kind", ["task", "session", "checkpoint"])
def test_secret_bearing_work_attempt_store_results_are_diagnostic_safe(
    result_kind: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = f"secret-bearing-{result_kind}-result"

    async def scenario() -> BaseException:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        sessions = (
            _SecretBearingSessionCreationResult(secret)
            if result_kind == "session"
            else (
                _SecretBearingCheckpointResult(secret)
                if result_kind == "checkpoint"
                else InMemorySessionStore()
            )
        )
        tasks = (
            _SecretBearingTaskResultStore(secret, clock=lambda: now)
            if result_kind == "task"
            else InMemoryTaskStore(clock=lambda: now)
        )
        app, run, execution = await _configured_public_initial_admission(
            prefix=f"secret-bearing-{result_kind}",
            sessions=sessions,
            tasks=tasks,
            redactor=SecretRedactor(secret),
        )
        if isinstance(tasks, _SecretBearingTaskResultStore):
            tasks.forge_task_result = True
        with pytest.raises(RuntimeError) as captured:
            await app.admit_work_attempt(run, execution=execution)
        return captured.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error = asyncio.run(scenario())

    _assert_secret_absent_from_work_attempt_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_secret_bearing_historical_claim_is_diagnostic_safe_through_recovery(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "secret-bearing-historical-claim-result"

    async def scenario() -> BaseException:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        sessions = InMemorySessionStore()
        tasks = _SecretBearingHistoricalClaimStore(secret, clock=lambda: now[0])
        app, run, execution = await _configured_public_initial_admission(
            prefix="secret-bearing-historical-claim",
            sessions=sessions,
            tasks=tasks,
            redactor=SecretRedactor(secret),
        )
        execution = execution.model_copy(update={"lease_seconds": 1})
        admitted = await app.admit_work_attempt(run, execution=execution)
        await sessions.update_status(admitted.session_id, SessionStatus.INTERRUPTED)
        await sessions.release_run_fence(admitted.session_id)
        now[0] += timedelta(seconds=2)
        with pytest.raises(RuntimeError, match="pre-mutation recovery activation failure"):
            await app.recover_work_attempt(
                WorkAttemptRecoveryRequest(
                    admission_id=admitted.admission_id,
                    claim_id="secret-bearing-historical-claim-2",
                    worker_id="secret-bearing-historical-worker-2",
                    generation=2,
                    lease_seconds=1,
                )
            )

        now[0] += timedelta(seconds=2)
        tasks.forge_historical_claim = True
        replacement = CayuApp(
            session_store=sessions,
            task_store=tasks,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        with pytest.raises(WorkAttemptRecoveryRequired) as captured:
            await replacement.recover_work_attempt(
                WorkAttemptRecoveryRequest(
                    admission_id=admitted.admission_id,
                    claim_id="secret-bearing-historical-claim-3",
                    worker_id="secret-bearing-historical-worker-3",
                    generation=3,
                    lease_seconds=300,
                )
            )
        return captured.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error = asyncio.run(scenario())

    _assert_secret_absent_from_work_attempt_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_public_admission_exact_retry_accepts_concurrent_activation() -> None:
    async def scenario() -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        sessions = InMemorySessionStore()
        tasks = _RacingWorkAttemptClaimStore(clock=lambda: now)
        app, run, execution = await _configured_public_initial_admission(
            prefix="concurrent-admission-replay",
            sessions=sessions,
            tasks=tasks,
            redactor=SecretRedactor(),
        )
        tasks.lose_next_prepare_ack = True
        with pytest.raises(RuntimeError, match="preparation acknowledgement loss"):
            await app.admit_work_attempt(run, execution=execution)

        tasks.block_next_claim = True
        stale_retry = asyncio.create_task(app.admit_work_attempt(run, execution=execution))
        await tasks.claim_started.wait()
        try:
            winner = await app.admit_work_attempt(run, execution=execution)
        finally:
            tasks.release_claim.set()
        replay = await stale_retry

        assert winner.state is WorkAttemptAdmissionState.ACTIVE
        assert replay == winner
        assert replay == await tasks.load_work_attempt_admission(execution.admission_id)

    asyncio.run(scenario())


def test_public_recovery_exact_retry_accepts_concurrent_activation() -> None:
    async def scenario() -> None:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        sessions = InMemorySessionStore()
        tasks = _RacingWorkAttemptClaimStore(clock=lambda: now[0])
        app, run, execution = await _configured_public_initial_admission(
            prefix="concurrent-recovery-replay",
            sessions=sessions,
            tasks=tasks,
            redactor=SecretRedactor(),
        )
        execution = execution.model_copy(update={"lease_seconds": 1})
        admitted = await app.admit_work_attempt(run, execution=execution)
        await sessions.update_status(admitted.session_id, SessionStatus.INTERRUPTED)
        await sessions.release_run_fence(admitted.session_id)
        now[0] += timedelta(seconds=2)
        recovery = WorkAttemptRecoveryRequest(
            admission_id=admitted.admission_id,
            claim_id="concurrent-recovery-replay-claim-2",
            worker_id="concurrent-recovery-replay-worker-2",
            generation=2,
            lease_seconds=300,
        )

        tasks.block_next_claim = True
        stale_retry = asyncio.create_task(app.recover_work_attempt(recovery))
        await tasks.claim_started.wait()
        try:
            winner = await app.recover_work_attempt(recovery)
        finally:
            tasks.release_claim.set()
        replay = await stale_retry

        assert winner.state is WorkAttemptAdmissionState.ACTIVE
        assert replay == winner
        assert replay == await tasks.load_work_attempt_admission(admitted.admission_id)

    asyncio.run(scenario())


def test_public_claim_mutations_reject_released_extension_authority() -> None:
    async def scenario() -> None:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        sessions = InMemorySessionStore()
        tasks = _IneligibleWorkAttemptTransitionStore(clock=lambda: now[0])
        app, run, execution = await _configured_public_initial_admission(
            prefix="released-extension-authority",
            sessions=sessions,
            tasks=tasks,
            redactor=SecretRedactor(),
        )
        execution = execution.model_copy(update={"lease_seconds": 1})
        admitted = await app.admit_work_attempt(run, execution=execution)
        await app.submit_work_attempt_proposal(
            WorkAttemptProposalRequest(
                admission_id=admitted.admission_id,
                claim_id=admitted.claim.claim_id,
                generation=admitted.claim.generation,
                proposal=CompletionProposalCreate(
                    proposal_id="released-extension-authority-proposal",
                    attempt_id=admitted.attempt_id,
                    result=_result_reference(),
                    evidence_references=(_artifact_evidence(),),
                ),
            )
        )
        released = await tasks.load_work_attempt_admission(admitted.admission_id)
        assert released is not None
        assert released.state is WorkAttemptAdmissionState.RELEASED
        source_session = await sessions.load(admitted.session_id)
        assert source_session is not None

        tasks.forge_ineligible_renewal = True
        with pytest.raises(RuntimeError, match="conflicting authority"):
            await app.renew_work_attempt_claim(
                WorkAttemptClaimRenewalRequest(
                    admission_id=admitted.admission_id,
                    claim_id=admitted.claim.claim_id,
                    worker_id=admitted.claim.worker_id,
                    generation=admitted.claim.generation,
                    lease_seconds=300,
                )
            )
        tasks.forge_ineligible_renewal = False

        now[0] += timedelta(seconds=2)
        tasks.forge_ineligible_recovery = True
        with pytest.raises(RuntimeError, match="conflicting authority"):
            await app.recover_work_attempt(
                WorkAttemptRecoveryRequest(
                    admission_id=admitted.admission_id,
                    claim_id="released-extension-authority-claim-2",
                    worker_id="released-extension-authority-worker-2",
                    generation=2,
                    lease_seconds=300,
                )
            )

        assert await tasks.load_work_attempt_admission(admitted.admission_id) == released
        current_session = await sessions.load(admitted.session_id)
        assert current_session is not None
        assert current_session.status is source_session.status
        assert current_session.run_epoch == source_session.run_epoch

    asyncio.run(scenario())


def test_conflicting_extension_result_is_secret_safe_across_diagnostics(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = _digest("secret-bearing-work-attempt-result")

    async def scenario() -> BaseException:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        sessions = InMemorySessionStore()
        tasks = _ConflictingWorkAttemptResultStore(clock=lambda: now)
        tasks.forge_prepare_sha256 = secret
        app, run, execution = await _configured_public_initial_admission(
            prefix="secret-extension-result",
            sessions=sessions,
            tasks=tasks,
            redactor=SecretRedactor(secret),
        )
        with pytest.raises(RuntimeError, match="conflicting authority") as captured:
            await app.admit_work_attempt(run, execution=execution)
        return captured.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error = asyncio.run(scenario())

    _assert_secret_absent_from_work_attempt_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_session_authority_read_failure_is_detached_and_secret_safe(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "work-attempt-session-read-secret"

    async def scenario() -> BaseException:
        sessions = _FailWorkAttemptDeferredInputRead(secret)
        tasks = InMemoryTaskStore()
        app, run, execution = await _configured_public_initial_admission(
            prefix="failed-session-authority-read",
            sessions=sessions,
            tasks=tasks,
            redactor=SecretRedactor(secret),
        )
        with pytest.raises(RuntimeError) as captured:
            await app.admit_work_attempt(run, execution=execution)
        prepared = await tasks.load_work_attempt_admission(execution.admission_id)
        assert prepared is not None
        assert prepared.state is WorkAttemptAdmissionState.PREPARING
        assert await sessions.load(run.session_id) is not None
        return captured.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error = asyncio.run(scenario())

    _assert_secret_absent_from_work_attempt_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_partial_task_store_capability_fails_before_any_admission_mutation() -> None:
    async def scenario() -> None:
        sessions = InMemorySessionStore()
        tasks = _PartialWorkAttemptTaskStore()
        app = CayuApp(session_store=sessions, task_store=tasks, enable_logging=False)
        app.register_provider(_RecordingProvider(), default=True)
        app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        with pytest.raises(NotImplementedError, match="complete work-attempt admission contract"):
            await app.admit_work_attempt(
                RunRequest(
                    agent_name="worker",
                    task_id="partial-capability-task",
                    session_id="partial-capability-session",
                    messages=[Message.text("user", "Do not partially mutate admission state.")],
                ),
                execution=WorkAttemptExecutionRequest(
                    admission_id="partial-capability-admission",
                    claim_id="partial-capability-claim",
                    attempt_id="partial-capability-attempt",
                    interaction_id="partial-capability-interaction",
                    worker_id="partial-capability-worker",
                    generation=1,
                    lease_seconds=300,
                ),
            )
        assert tasks.prepare_calls == 0
        assert await sessions.load("partial-capability-session") is None

    asyncio.run(scenario())


def test_cancelled_interaction_fanout_settles_before_exact_retry_returns() -> None:
    async def scenario() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        sink = _BlockingWorkAttemptEventSink()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            event_sinks=(sink,),
            enable_logging=False,
        )
        app.register_provider(_RecordingProvider(), default=True)
        app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        contract = _contract(contract_id="cancelled-fanout-contract")
        await tasks.publish_work_contract(contract)
        task = await tasks.create_task(
            TaskCreate(
                task_id="cancelled-fanout-task",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        run = RunRequest(
            agent_name="worker",
            task_id=task.id,
            session_id="cancelled-fanout-session",
            messages=[Message.text("user", "Deliver admission before returning authority.")],
        )
        execution = WorkAttemptExecutionRequest(
            admission_id="cancelled-fanout-admission",
            claim_id="cancelled-fanout-claim",
            attempt_id="cancelled-fanout-attempt",
            interaction_id="cancelled-fanout-interaction",
            worker_id="cancelled-fanout-worker",
            generation=1,
            lease_seconds=300,
        )
        owner = asyncio.create_task(app.admit_work_attempt(run, execution=execution))
        await sink.started.wait()
        owner.cancel("cancel while interaction event delivery is in flight")
        assert owner.cancelling() == 1
        await asyncio.sleep(0)
        assert not owner.done()

        retry = asyncio.create_task(app.admit_work_attempt(run, execution=execution))
        await asyncio.sleep(0.05)
        assert not retry.done()
        assert sink.events == []

        sink.release.set()
        replay = await retry
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert owner.cancelled()
        assert owner.cancelling() == 1
        assert replay.state is WorkAttemptAdmissionState.ACTIVE
        assert len(sink.events) == 1
        records = await sessions.query_events(
            EventQuery(
                session_id=replay.session_id,
                interaction_id=replay.interaction_id,
                event_type=EventType.INTERACTION_STARTED,
                limit=2,
            )
        )
        assert len(records) == 1
        delivered = await sessions.get_persisted_event_side_effect_delivery(
            session_id=records[0].event.session_id,
            event_id=records[0].event.id,
        )
        assert delivered is not None
        assert delivered.status.value == "delivered"

    asyncio.run(scenario())


def test_public_handoff_keeps_initial_replay_and_recovery_claims_live() -> None:
    async def scenario() -> None:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        sessions = _LeaseAdvancingWorkAttemptSessionStore(now)
        tasks = InMemoryTaskStore(clock=lambda: now[0])
        sink = InMemoryEventSink()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            event_sinks=(sink,),
            enable_logging=False,
        )
        app.register_provider(_RecordingProvider(), default=True)
        app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        contract = _contract(contract_id="live-handoff-contract")
        await tasks.publish_work_contract(contract)
        task = await tasks.create_task(
            TaskCreate(
                task_id="live-handoff-task",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        run = RunRequest(
            agent_name="worker",
            task_id=task.id,
            session_id="live-handoff-session",
            messages=[Message.text("user", "Keep the admission lease live during delivery.")],
        )
        execution = WorkAttemptExecutionRequest(
            admission_id="live-handoff-admission",
            claim_id="live-handoff-claim-1",
            attempt_id="live-handoff-attempt",
            interaction_id="live-handoff-interaction",
            worker_id="live-handoff-worker-1",
            generation=1,
            lease_seconds=1,
        )

        sessions.delay_next_handoff = True
        admitted = await app.admit_work_attempt(run, execution=execution)
        assert admitted.state is WorkAttemptAdmissionState.ACTIVE
        assert admitted.claim.lease_expires_at > now[0]
        assert len(sink.events) == 1

        sessions.delay_next_handoff = True
        replay = await app.admit_work_attempt(run, execution=execution)
        assert replay.state is WorkAttemptAdmissionState.ACTIVE
        assert replay.claim.lease_expires_at > now[0]
        assert replay.claim.lease_expires_at > admitted.claim.lease_expires_at
        assert len(sink.events) == 1

        await sessions.update_status(replay.session_id, SessionStatus.INTERRUPTED)
        await sessions.release_run_fence(replay.session_id)
        now[0] = replay.claim.lease_expires_at + timedelta(microseconds=1)
        sessions.delay_next_handoff = True
        recovered = await app.recover_work_attempt(
            WorkAttemptRecoveryRequest(
                admission_id=replay.admission_id,
                claim_id="live-handoff-claim-2",
                worker_id="live-handoff-worker-2",
                generation=2,
                lease_seconds=1,
            )
        )
        assert recovered.state is WorkAttemptAdmissionState.ACTIVE
        assert recovered.claim.generation == 2
        assert recovered.claim.lease_expires_at > now[0]
        assert len(sink.events) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_public_initial_admission_activates_authority_before_any_provider_work(
    backend: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        sessions = InMemorySessionStore()
        tasks = (
            InMemoryTaskStore()
            if backend == "memory"
            else SQLiteTaskStore(tmp_path / "public-admission.db")
        )
        provider = _RecordingProvider()
        sink = InMemoryEventSink()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            event_sinks=(sink,),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        contract = _contract()
        await tasks.publish_work_contract(contract)
        task = await tasks.create_task(
            TaskCreate(
                task_id="public-admission-task",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        opaque_loop_policy = LoopPolicy()
        run = RunRequest(
            agent_name="worker",
            task_id=task.id,
            session_id="public-admission-session",
            messages=[Message.text("user", "Perform the governed work.")],
            loop_policies=(opaque_loop_policy,),
        )
        execution = WorkAttemptExecutionRequest(
            admission_id="public-admission",
            claim_id="public-admission-claim",
            attempt_id="public-admission-attempt",
            interaction_id="public-admission-interaction",
            worker_id="public-admission-worker",
            generation=1,
            lease_seconds=300,
        )

        with pytest.raises(ValueError, match="requires RunRequest.task_id"):
            await app.admit_work_attempt(
                run.model_copy(update={"task_id": None}),
                execution=execution,
            )
        with pytest.raises(ValueError, match="caller-stable RunRequest.session_id"):
            await app.admit_work_attempt(
                run.model_copy(update={"session_id": None}),
                execution=execution,
            )
        assert await tasks.load_work_attempt_admission(execution.admission_id) is None
        assert provider.requests == []

        admitted = await app.admit_work_attempt(run, execution=execution)
        assert admitted.state is WorkAttemptAdmissionState.ACTIVE
        assert admitted.attempt is not None
        assert admitted.task_id == task.id
        assert admitted.session_id == run.session_id
        assert admitted.interaction_id == execution.interaction_id
        assert provider.requests == []
        session = await sessions.load(admitted.session_id)
        assert session is not None
        assert session.status.value == "running"
        assert session.run_epoch == 1
        assert len(sink.events) == 1
        assert sink.events[0].type is EventType.INTERACTION_STARTED
        assert sink.events[0].interaction_id == admitted.interaction_id

        deferred = await sessions.load_deferred_interaction_input(admitted.session_id)
        assert deferred is not None
        await sessions.replace_initial_transcript_messages(
            admitted.session_id,
            deferred.source_messages,
            deferred.source_messages,
            interaction_id=admitted.interaction_id,
        )

        renewed = await app.renew_work_attempt_claim(
            WorkAttemptClaimRenewalRequest(
                admission_id=admitted.admission_id,
                claim_id=admitted.claim.claim_id,
                worker_id=admitted.claim.worker_id,
                generation=admitted.claim.generation,
                lease_seconds=301,
            )
        )
        assert renewed.claim.claimed_at == admitted.claim.claimed_at
        assert renewed.claim.request_sha256 == admitted.claim.request_sha256
        assert renewed.claim.lease_expires_at >= admitted.claim.lease_expires_at

        replay = await app.admit_work_attempt(run, execution=execution)
        assert replay == renewed
        assert len(sink.events) == 1
        assert provider.requests == []
        with pytest.raises(WorkCompletionConflict, match="identity conflicts"):
            await app.admit_work_attempt(
                run.model_copy(update={"loop_policies": (LoopPolicy(),)}),
                execution=execution,
            )
        assert provider.requests == []
        with pytest.raises(WorkCompletionConflict, match="conflicts"):
            await app.admit_work_attempt(
                run.model_copy(
                    update={"messages": [Message.text("user", "Conflicting governed work.")]}
                ),
                execution=execution,
            )
        proposal_request = WorkAttemptProposalRequest(
            admission_id=admitted.admission_id,
            claim_id=admitted.claim.claim_id,
            generation=admitted.claim.generation,
            proposal=CompletionProposalCreate(
                proposal_id=f"public-admission-proposal-{backend}",
                attempt_id=admitted.attempt_id,
                result=_result_reference(),
                evidence_references=(_artifact_evidence(),),
            ),
        )
        proposal = await app.submit_work_attempt_proposal(proposal_request)
        restarted_app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
        )
        assert await restarted_app.submit_work_attempt_proposal(proposal_request) == proposal
        await sessions.update_status(admitted.session_id, SessionStatus.COMPLETED)
        await sessions.release_run_fence(admitted.session_id)
        with pytest.raises(TaskCompletionDecisionRequired):
            _ = [
                event
                async for event in app.fork_session(
                    ForkSessionRequest(
                        source_session_id=admitted.session_id,
                        session_id=f"public-admission-fork-{backend}",
                    )
                )
            ]
        assert await sessions.load(f"public-admission-fork-{backend}") is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("fault", "generation"),
    [
        ("early_replacement", 2),
        ("overlong_replacement", 2),
        ("skipped_generation", 3),
        ("premature_activation", 2),
    ],
)
def test_public_recovery_rejects_conflicting_extension_claim_authority(
    fault: str,
    generation: int,
) -> None:
    async def scenario() -> None:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        sessions = InMemorySessionStore()
        tasks = _ConflictingWorkAttemptResultStore(clock=lambda: now[0])
        app, run, execution = await _configured_public_initial_admission(
            prefix=f"conflicting-recovery-{fault}",
            sessions=sessions,
            tasks=tasks,
            redactor=SecretRedactor(),
        )
        execution = execution.model_copy(update={"lease_seconds": 1})
        admitted = await app.admit_work_attempt(run, execution=execution)
        source_session = await sessions.load(admitted.session_id)
        assert source_session is not None
        now[0] += timedelta(seconds=2)
        tasks.forge_recovery = fault

        expected_failure = (
            "immutable session authority"
            if fault == "premature_activation"
            else "conflicting replacement authority"
        )
        with pytest.raises(RuntimeError, match=expected_failure):
            await app.recover_work_attempt(
                WorkAttemptRecoveryRequest(
                    admission_id=admitted.admission_id,
                    claim_id=f"conflicting-recovery-{fault}-claim-2",
                    worker_id=f"conflicting-recovery-{fault}-worker-2",
                    generation=generation,
                    lease_seconds=300,
                )
            )

        current_session = await sessions.load(admitted.session_id)
        assert current_session is not None
        assert current_session.status is source_session.status
        assert current_session.run_epoch == source_session.run_epoch

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_public_first_crash_recovery_needs_no_direct_store_mutation(
    backend: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        sessions: SessionStore = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "first-crash-sessions.sqlite")
        )
        tasks: TaskStore = (
            InMemoryTaskStore(clock=lambda: now[0])
            if backend == "memory"
            else SQLiteTaskStore(
                tmp_path / "first-crash-tasks.sqlite",
                clock=lambda: now[0],
            )
        )
        app, run, execution = await _configured_public_initial_admission(
            prefix=f"first-crash-{backend}",
            sessions=sessions,
            tasks=tasks,
            redactor=SecretRedactor(),
            agent_system_prompt="Preserve this exact governed system context.",
        )
        execution = execution.model_copy(update={"lease_seconds": 1})
        admitted = await app.admit_work_attempt(run, execution=execution)
        source_session = await sessions.load(admitted.session_id)
        assert source_session is not None
        assert source_session.status is SessionStatus.RUNNING
        deferred = await sessions.load_deferred_interaction_input(admitted.session_id)
        assert deferred is not None
        assert deferred.initial_transcript_messages == [
            Message.text("system", "Preserve this exact governed system context."),
            *run.messages,
        ]
        with pytest.raises(RuntimeError, match="authenticated projection"):
            await sessions.replace_initial_transcript_messages(
                admitted.session_id,
                run.messages,
                [Message.text("system", "Forged replacement."), *run.messages],
                interaction_id=admitted.interaction_id,
            )
        assert await sessions.load_transcript(admitted.session_id) == []

        now[0] += timedelta(seconds=2)
        replacement = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
        )
        replacement.register_provider(_RecordingProvider(), default=True)
        replacement.register_agent(
            AgentSpec(
                name="worker",
                model="verified-work-test-model",
                system_prompt="Preserve this exact governed system context.",
            )
        )
        recovered = await replacement.recover_work_attempt(
            WorkAttemptRecoveryRequest(
                admission_id=admitted.admission_id,
                claim_id=f"first-crash-{backend}-claim-2",
                worker_id=f"first-crash-{backend}-worker-2",
                generation=2,
                lease_seconds=300,
            )
        )

        assert recovered.state is WorkAttemptAdmissionState.ACTIVE
        assert recovered.claim.generation == 2
        assert recovered.attempt == admitted.attempt
        recovered_session = await sessions.load(admitted.session_id)
        assert recovered_session is not None
        assert recovered_session.status is SessionStatus.RUNNING
        assert recovered_session.run_epoch == source_session.run_epoch + 3
        recovered_checkpoint = await sessions.load_checkpoint(admitted.session_id)
        assert recovered_checkpoint is not None
        assert "initial_transcript_pending" not in recovered_checkpoint
        assert await sessions.load_deferred_interaction_input(admitted.session_id) is None
        assert await sessions.load_transcript(admitted.session_id) == [
            Message.text("system", "Preserve this exact governed system context."),
            *run.messages,
        ]
        lifecycle = await sessions.query_events(
            EventQuery(session_id=admitted.session_id, limit=100)
        )
        assert sum(record.event.type is EventType.INTERACTION_STARTED for record in lifecycle) == 1
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
            await app.submit_work_attempt_proposal(
                WorkAttemptProposalRequest(
                    admission_id=admitted.admission_id,
                    claim_id=admitted.claim.claim_id,
                    generation=admitted.claim.generation,
                    proposal=CompletionProposalCreate(
                        proposal_id=f"first-crash-{backend}-stale-proposal",
                        attempt_id=admitted.attempt_id,
                        result=_result_reference(),
                        evidence_references=(_artifact_evidence(),),
                    ),
                )
            )

        if backend == "sqlite":
            await sessions.close()
            await tasks.close()

    asyncio.run(scenario())


def test_migrated_source_only_initial_input_keeps_recovery_fenced(tmp_path) -> None:
    async def scenario() -> None:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        session_path = tmp_path / "migrated-source-only-session.sqlite"
        task_path = tmp_path / "migrated-source-only-task.sqlite"
        sessions = SQLiteSessionStore(session_path)
        tasks = SQLiteTaskStore(task_path, clock=lambda: now[0])
        source, run, execution = await _configured_public_initial_admission(
            prefix="migrated-source-only",
            sessions=sessions,
            tasks=tasks,
            redactor=SecretRedactor(),
            agent_system_prompt="Authenticated prefix that revision 61 did not retain.",
        )
        execution = execution.model_copy(update={"lease_seconds": 1})
        admitted = await source.admit_work_attempt(run, execution=execution)
        source_session = await sessions.load(admitted.session_id)
        assert source_session is not None
        await sessions.close()
        await tasks.close()

        session_connection = sqlite3.connect(session_path)
        try:
            payload = json.loads(
                session_connection.execute(
                    "SELECT source_messages_json FROM cayu_deferred_interaction_inputs "
                    "WHERE session_id = ?",
                    (admitted.session_id,),
                ).fetchone()[0]
            )
            session_connection.execute(
                "UPDATE cayu_deferred_interaction_inputs SET source_messages_json = ? "
                "WHERE session_id = ?",
                (
                    sqlite_support.json_dumps(payload["source_messages"]),
                    admitted.session_id,
                ),
            )
            session_connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 62")
            session_connection.execute("PRAGMA user_version = 61")
            session_connection.commit()
        finally:
            session_connection.close()

        task_connection = sqlite3.connect(task_path)
        try:
            task_connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 62")
            task_connection.execute("PRAGMA user_version = 61")
            task_connection.commit()
        finally:
            task_connection.close()

        migrated_sessions = SQLiteSessionStore(
            session_path,
            schema_mode=SchemaMode.MIGRATE,
        )
        migrated_tasks = SQLiteTaskStore(
            task_path,
            clock=lambda: now[0],
            schema_mode=SchemaMode.MIGRATE,
        )
        try:
            now[0] += timedelta(seconds=2)
            replacement = CayuApp(
                session_store=migrated_sessions,
                task_store=migrated_tasks,
                enable_logging=False,
            )
            replacement.register_provider(_RecordingProvider(), default=True)
            replacement.register_agent(
                AgentSpec(
                    name="worker",
                    model="verified-work-test-model",
                    system_prompt=("Authenticated prefix that revision 61 did not retain."),
                )
            )
            recovery = WorkAttemptRecoveryRequest(
                admission_id=admitted.admission_id,
                claim_id="migrated-source-only-claim-2",
                worker_id="migrated-source-only-worker-2",
                generation=2,
                lease_seconds=300,
            )
            for _attempt in range(2):
                with pytest.raises(
                    WorkAttemptRecoveryRequired,
                    match="authenticated complete initial transcript",
                ):
                    await replacement.recover_work_attempt(recovery)

            fenced = await migrated_tasks.load_work_attempt_admission(admitted.admission_id)
            assert fenced is not None
            assert fenced.state is WorkAttemptAdmissionState.RECOVERING
            current_session = await migrated_sessions.load(admitted.session_id)
            assert current_session is not None
            assert current_session.status is SessionStatus.RUNNING
            assert current_session.run_epoch == source_session.run_epoch
            deferred = await migrated_sessions.load_deferred_interaction_input(admitted.session_id)
            assert deferred is not None
            assert deferred.initial_transcript_messages is None
            assert await migrated_sessions.load_transcript(admitted.session_id) == []
        finally:
            await migrated_sessions.close()
            await migrated_tasks.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_first_crash_settlement_rejects_a_claim_that_expires_before_handoff(
    backend: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        sessions = _BlockNextWorkAttemptCheckpointLoad()
        tasks: TaskStore = (
            InMemoryTaskStore(clock=lambda: now[0])
            if backend == "memory"
            else SQLiteTaskStore(
                tmp_path / "expired-settlement-claim.sqlite",
                clock=lambda: now[0],
            )
        )
        source_app, run, execution = await _configured_public_initial_admission(
            prefix=f"expired-settlement-claim-{backend}",
            sessions=sessions,
            tasks=tasks,
            redactor=SecretRedactor(),
        )
        execution = execution.model_copy(update={"lease_seconds": 1})
        admitted = await source_app.admit_work_attempt(run, execution=execution)
        source_session = await sessions.load(admitted.session_id)
        assert source_session is not None
        now[0] += timedelta(seconds=2)

        replacement = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
        )
        replacement.register_provider(_RecordingProvider(), default=True)
        replacement.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        generation_two = WorkAttemptRecoveryRequest(
            admission_id=admitted.admission_id,
            claim_id=f"expired-settlement-claim-{backend}-claim-2",
            worker_id=f"expired-settlement-claim-{backend}-worker-2",
            generation=2,
            lease_seconds=1,
        )
        sessions.block_next_load = True
        pending = asyncio.create_task(replacement.recover_work_attempt(generation_two))
        await asyncio.wait_for(sessions.load_started.wait(), timeout=10)
        now[0] += timedelta(seconds=2)
        sessions.release_load.set()
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await pending

        assert await sessions.load(admitted.session_id) == source_session
        expired = await tasks.load_work_attempt_admission(admitted.admission_id)
        assert expired is not None
        assert expired.state is WorkAttemptAdmissionState.RECOVERING
        assert expired.claim.generation == 2
        recovered = await replacement.recover_work_attempt(
            WorkAttemptRecoveryRequest(
                admission_id=admitted.admission_id,
                claim_id=f"expired-settlement-claim-{backend}-claim-3",
                worker_id=f"expired-settlement-claim-{backend}-worker-3",
                generation=3,
                lease_seconds=300,
            )
        )
        assert recovered.state is WorkAttemptAdmissionState.ACTIVE
        assert recovered.claim.generation == 3

        if backend == "sqlite":
            await tasks.close()

    asyncio.run(scenario())


def test_first_crash_settlement_cleanup_failure_is_exactly_retryable() -> None:
    async def scenario() -> None:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        sessions = _FailOnceWorkAttemptRunFenceRelease()
        tasks = InMemoryTaskStore(clock=lambda: now[0])
        source_app, run, execution = await _configured_public_initial_admission(
            prefix="retry-first-crash-cleanup",
            sessions=sessions,
            tasks=tasks,
            redactor=SecretRedactor(),
        )
        execution = execution.model_copy(update={"lease_seconds": 1})
        admitted = await source_app.admit_work_attempt(run, execution=execution)
        now[0] += timedelta(seconds=2)
        replacement = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
        )
        replacement.register_provider(_RecordingProvider(), default=True)
        replacement.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        recovery = WorkAttemptRecoveryRequest(
            admission_id=admitted.admission_id,
            claim_id="retry-first-crash-cleanup-claim-2",
            worker_id="retry-first-crash-cleanup-worker-2",
            generation=2,
            lease_seconds=300,
        )

        sessions.fail_next_release = True
        with pytest.raises(RuntimeError, match="run fence release failure"):
            await replacement.recover_work_attempt(recovery)
        interrupted = await sessions.load(admitted.session_id)
        assert interrupted is not None
        assert interrupted.status is SessionStatus.INTERRUPTED
        recovering = await tasks.load_work_attempt_admission(admitted.admission_id)
        assert recovering is not None
        assert recovering.state is WorkAttemptAdmissionState.RECOVERING

        recovered = await replacement.recover_work_attempt(recovery)
        assert recovered.state is WorkAttemptAdmissionState.ACTIVE
        assert recovered.claim == recovering.claim
        current = await sessions.load(admitted.session_id)
        assert current is not None
        assert current.status is SessionStatus.RUNNING

    asyncio.run(scenario())


def test_concurrent_exact_first_crash_retry_is_bounded_until_cleanup_settles() -> None:
    async def scenario() -> None:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        sessions = _BlockFirstWorkAttemptRunFenceRelease()
        tasks = InMemoryTaskStore(clock=lambda: now[0])
        source_app, run, execution = await _configured_public_initial_admission(
            prefix="concurrent-first-crash-retry",
            sessions=sessions,
            tasks=tasks,
            redactor=SecretRedactor(),
        )
        execution = execution.model_copy(update={"lease_seconds": 1})
        admitted = await source_app.admit_work_attempt(run, execution=execution)
        now[0] += timedelta(seconds=2)
        replacement = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
        )
        replacement.register_provider(_RecordingProvider(), default=True)
        replacement.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        recovery = WorkAttemptRecoveryRequest(
            admission_id=admitted.admission_id,
            claim_id="concurrent-first-crash-retry-claim-2",
            worker_id="concurrent-first-crash-retry-worker-2",
            generation=2,
            lease_seconds=300,
        )

        owner = asyncio.create_task(replacement.recover_work_attempt(recovery))
        await asyncio.wait_for(sessions.release_started.wait(), timeout=10)
        with pytest.raises(
            WorkAttemptRecoveryRequired,
            match="still owned by another recovery",
        ):
            await asyncio.wait_for(replacement.recover_work_attempt(recovery), timeout=10)
        assert sessions.release_calls == 1
        assert not owner.done()

        sessions.release_cleanup.set()
        recovered = await owner
        assert recovered.state is WorkAttemptAdmissionState.ACTIVE
        assert await replacement.recover_work_attempt(recovery) == recovered
        assert sessions.release_calls == 1

    asyncio.run(scenario())


def test_first_crash_settlement_extension_failure_is_diagnostic_safe(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "secret-bearing-work-attempt-settlement-failure"

    async def scenario() -> BaseException:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        sessions = _FailWorkAttemptSettlementFence(secret)
        tasks = InMemoryTaskStore(clock=lambda: now[0])
        source_app, run, execution = await _configured_public_initial_admission(
            prefix="secret-settlement-failure",
            sessions=sessions,
            tasks=tasks,
            redactor=SecretRedactor(secret),
        )
        execution = execution.model_copy(update={"lease_seconds": 1})
        admitted = await source_app.admit_work_attempt(run, execution=execution)
        now[0] += timedelta(seconds=2)
        replacement = CayuApp(
            session_store=sessions,
            task_store=tasks,
            secret_redactor=SecretRedactor(secret),
            enable_logging=False,
        )
        replacement.register_provider(_RecordingProvider(), default=True)
        replacement.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        with pytest.raises(RuntimeError) as captured:
            await replacement.recover_work_attempt(
                WorkAttemptRecoveryRequest(
                    admission_id=admitted.admission_id,
                    claim_id="secret-settlement-failure-claim-2",
                    worker_id="secret-settlement-failure-worker-2",
                    generation=2,
                    lease_seconds=300,
                )
            )
        return captured.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error = asyncio.run(scenario())

    _assert_secret_absent_from_work_attempt_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_cancelled_first_crash_settlement_is_quiescent_and_exactly_retryable() -> None:
    async def scenario() -> None:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        sessions = _BlockFirstWorkAttemptSettlementFence()
        tasks = InMemoryTaskStore(clock=lambda: now[0])
        source_app, run, execution = await _configured_public_initial_admission(
            prefix="cancelled-first-crash",
            sessions=sessions,
            tasks=tasks,
            redactor=SecretRedactor(),
        )
        execution = execution.model_copy(update={"lease_seconds": 1})
        admitted = await source_app.admit_work_attempt(run, execution=execution)
        now[0] += timedelta(seconds=2)
        replacement = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
        )
        replacement.register_provider(_RecordingProvider(), default=True)
        replacement.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        recovery = WorkAttemptRecoveryRequest(
            admission_id=admitted.admission_id,
            claim_id="cancelled-first-crash-claim-2",
            worker_id="cancelled-first-crash-worker-2",
            generation=2,
            lease_seconds=300,
        )

        pending = asyncio.create_task(replacement.recover_work_attempt(recovery))
        await asyncio.wait_for(sessions.fence_committed.wait(), timeout=10)
        pending.cancel("stop after predecessor settlement fence committed")
        assert pending.cancelling() == 1
        await asyncio.sleep(0)
        assert not pending.done()
        sessions.release_fence.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert pending.cancelled()

        recovering = await tasks.load_work_attempt_admission(admitted.admission_id)
        assert recovering is not None
        assert recovering.state is WorkAttemptAdmissionState.RECOVERING
        recovered = await replacement.recover_work_attempt(recovery)
        assert recovered.state is WorkAttemptAdmissionState.ACTIVE
        assert recovered.claim == recovering.claim
        lifecycle = await sessions.query_events(
            EventQuery(session_id=admitted.session_id, limit=100)
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

    asyncio.run(scenario())


def test_public_recovery_fences_stale_generation_and_replays_exact_receipt() -> None:
    async def scenario() -> None:
        # Task-store lease time is intentionally behind the independently-owned
        # session-store clock. Recovery authority must not compare these clocks.
        now = [datetime(2020, 1, 1, tzinfo=UTC)]
        sessions = InMemorySessionStore()
        tasks = _LoseFirstRecoveryActivationAcknowledgement(clock=lambda: now[0])
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
        )
        app.register_provider(_RecordingProvider(), default=True)
        app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        contract = _contract(contract_id="recovery-admission-contract")
        await tasks.publish_work_contract(contract)
        task = await tasks.create_task(
            TaskCreate(
                task_id="recovery-admission-task",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        first = await app.admit_work_attempt(
            RunRequest(
                agent_name="worker",
                task_id=task.id,
                session_id="recovery-admission-session",
                messages=[Message.text("user", "Continue after a worker crash.")],
            ),
            execution=WorkAttemptExecutionRequest(
                admission_id="recovery-admission",
                claim_id="recovery-admission-claim-1",
                attempt_id="recovery-admission-attempt",
                interaction_id="recovery-admission-interaction",
                worker_id="recovery-admission-worker-1",
                generation=1,
                lease_seconds=1,
            ),
        )
        prior_epoch = (await sessions.load(first.session_id)).run_epoch
        now[0] += timedelta(seconds=2)
        recovery_request = WorkAttemptRecoveryRequest(
            admission_id=first.admission_id,
            claim_id="recovery-admission-claim-2",
            worker_id="recovery-admission-worker-2",
            generation=2,
            lease_seconds=300,
        )
        restarted_app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
        )
        restarted_app.register_provider(_RecordingProvider(), default=True)
        restarted_app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        failed_recovery = asyncio.create_task(restarted_app.recover_work_attempt(recovery_request))
        with pytest.raises(RuntimeError, match="recovery activation acknowledgement loss"):
            await failed_recovery
        committed = await tasks.load_work_attempt_admission(first.admission_id)
        assert committed is not None
        assert committed.state is WorkAttemptAdmissionState.ACTIVE
        committed_epoch = (await sessions.load(first.session_id)).run_epoch
        recovered = await restarted_app.recover_work_attempt(recovery_request)
        assert recovered.state is WorkAttemptAdmissionState.ACTIVE
        assert recovered.claim.generation == 2
        assert recovered.attempt == first.attempt
        recovered_session = await sessions.load(first.session_id)
        assert recovered_session is not None
        assert recovered_session.status is SessionStatus.RUNNING
        assert recovered_session.run_epoch == prior_epoch + 3
        assert recovered_session.run_epoch == committed_epoch
        assert await sessions.load_deferred_interaction_input(first.session_id) is None
        assert await restarted_app.recover_work_attempt(recovery_request) == recovered

        with pytest.raises(WorkAttemptExecutionClaimLost):
            await app.submit_work_attempt_proposal(
                WorkAttemptProposalRequest(
                    admission_id=first.admission_id,
                    claim_id=first.claim.claim_id,
                    generation=first.claim.generation,
                    proposal=CompletionProposalCreate(
                        proposal_id="stale-recovery-proposal",
                        attempt_id=first.attempt_id,
                        result=_result_reference(),
                        evidence_references=(_artifact_evidence(),),
                    ),
                )
            )
        await sessions.release_run_fence(first.session_id)
        assert (await sessions.load(first.session_id)).run_epoch == recovered_session.run_epoch + 1

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_recovery_reconciles_predecessor_session_transition_after_activation_crash(
    backend: str,
    tmp_path,
) -> None:
    async def scenario() -> None:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        sessions = InMemorySessionStore()
        tasks = (
            _FailFirstRecoveryActivationBeforeMutation(clock=lambda: now[0])
            if backend == "memory"
            else _FailFirstSQLiteRecoveryActivationBeforeMutation(
                tmp_path / "predecessor-recovery.db",
                clock=lambda: now[0],
            )
        )
        first_app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
        )
        first_app.register_provider(_RecordingProvider(), default=True)
        first_app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        contract = _contract(contract_id=f"predecessor-recovery-{backend}")
        await tasks.publish_work_contract(contract)
        task = await tasks.create_task(
            TaskCreate(
                task_id=f"predecessor-recovery-task-{backend}",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        admitted = await first_app.admit_work_attempt(
            RunRequest(
                agent_name="worker",
                task_id=task.id,
                session_id=f"predecessor-recovery-session-{backend}",
                messages=[Message.text("user", "Recover an interrupted activation.")],
            ),
            execution=WorkAttemptExecutionRequest(
                admission_id=f"predecessor-recovery-admission-{backend}",
                claim_id=f"predecessor-recovery-claim-1-{backend}",
                attempt_id=f"predecessor-recovery-attempt-{backend}",
                interaction_id=f"predecessor-recovery-interaction-{backend}",
                worker_id=f"predecessor-recovery-worker-1-{backend}",
                generation=1,
                lease_seconds=1,
            ),
        )
        await sessions.update_status(admitted.session_id, SessionStatus.INTERRUPTED)
        await sessions.release_run_fence(admitted.session_id)
        now[0] += timedelta(seconds=2)
        generation_two = WorkAttemptRecoveryRequest(
            admission_id=admitted.admission_id,
            claim_id=f"predecessor-recovery-claim-2-{backend}",
            worker_id=f"predecessor-recovery-worker-2-{backend}",
            generation=2,
            lease_seconds=1,
        )
        with pytest.raises(RuntimeError, match="pre-mutation recovery activation failure"):
            await first_app.recover_work_attempt(generation_two)
        interrupted_activation = await tasks.load_work_attempt_admission(admitted.admission_id)
        assert interrupted_activation is not None
        assert interrupted_activation.state is WorkAttemptAdmissionState.RECOVERING
        assert interrupted_activation.claim.generation == 2
        transitioned_session = await sessions.load(admitted.session_id)
        assert transitioned_session is not None
        assert transitioned_session.status is SessionStatus.RUNNING

        now[0] += timedelta(seconds=2)
        replacement_app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
        )
        replacement_app.register_provider(_RecordingProvider(), default=True)
        replacement_app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        recovered = await replacement_app.recover_work_attempt(
            WorkAttemptRecoveryRequest(
                admission_id=admitted.admission_id,
                claim_id=f"predecessor-recovery-claim-3-{backend}",
                worker_id=f"predecessor-recovery-worker-3-{backend}",
                generation=3,
                lease_seconds=300,
            )
        )
        assert recovered.state is WorkAttemptAdmissionState.ACTIVE
        assert recovered.claim.generation == 3
        assert recovered.attempt == admitted.attempt
        recovered_session = await sessions.load(admitted.session_id)
        assert recovered_session is not None
        assert recovered_session.status is SessionStatus.RUNNING
        assert recovered_session.run_epoch == transitioned_session.run_epoch + 1
        durable_task = await tasks.load_task(task.id)
        assert durable_task is not None
        assert durable_task.worker_id == recovered.claim.worker_id
        assert durable_task.lease_expires_at == recovered.claim.lease_expires_at
        with pytest.raises(WorkAttemptExecutionClaimLost):
            await first_app.submit_work_attempt_proposal(
                WorkAttemptProposalRequest(
                    admission_id=admitted.admission_id,
                    claim_id=generation_two.claim_id,
                    generation=generation_two.generation,
                    proposal=CompletionProposalCreate(
                        proposal_id=f"stale-predecessor-proposal-{backend}",
                        attempt_id=admitted.attempt_id,
                        result=_result_reference(),
                        evidence_references=(_artifact_evidence(),),
                    ),
                )
            )
        if isinstance(tasks, SQLiteTaskStore):
            await tasks.close()

    asyncio.run(scenario())


def test_public_recovery_rejects_session_mutation_after_claim_under_reverse_clock_skew() -> None:
    async def scenario() -> None:
        # This task-store clock is intentionally ahead of the session store.
        # The concurrent session mutation below would evade the former
        # last_activity_at > claimed_at comparison.
        now = [datetime(2100, 1, 1, tzinfo=UTC)]
        sessions = _BlockFirstWorkAttemptRecoveryTransition()
        tasks = InMemoryTaskStore(clock=lambda: now[0])
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
        )
        app.register_provider(_RecordingProvider(), default=True)
        app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        contract = _contract(contract_id="recovery-session-snapshot-contract")
        await tasks.publish_work_contract(contract)
        task = await tasks.create_task(
            TaskCreate(
                task_id="recovery-session-snapshot-task",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        admitted = await app.admit_work_attempt(
            RunRequest(
                agent_name="worker",
                task_id=task.id,
                session_id="recovery-session-snapshot-session",
                messages=[Message.text("user", "Recover only an unchanged session.")],
            ),
            execution=WorkAttemptExecutionRequest(
                admission_id="recovery-session-snapshot-admission",
                claim_id="recovery-session-snapshot-claim-1",
                attempt_id="recovery-session-snapshot-attempt",
                interaction_id="recovery-session-snapshot-interaction",
                worker_id="recovery-session-snapshot-worker-1",
                generation=1,
                lease_seconds=1,
            ),
        )
        await sessions.update_status(admitted.session_id, SessionStatus.INTERRUPTED)
        await sessions.release_run_fence(admitted.session_id)
        now[0] += timedelta(seconds=2)
        recovery = asyncio.create_task(
            app.recover_work_attempt(
                WorkAttemptRecoveryRequest(
                    admission_id=admitted.admission_id,
                    claim_id="recovery-session-snapshot-claim-2",
                    worker_id="recovery-session-snapshot-worker-2",
                    generation=2,
                    lease_seconds=300,
                )
            )
        )
        await asyncio.wait_for(sessions.transition_started.wait(), timeout=10)

        # Deliver the competing mutation through the real store entrance while
        # recovery is paused after its post-claim read and before its atomic CAS.
        await sessions.update_status(admitted.session_id, SessionStatus.INTERRUPTED)
        sessions.release_transition.set()
        with pytest.raises(
            WorkAttemptRecoveryRequired,
            match="changed after recovery was claimed",
        ):
            await recovery

        durable_session = await sessions.load(admitted.session_id)
        assert durable_session is not None
        assert durable_session.status is SessionStatus.INTERRUPTED
        recovering = await tasks.load_work_attempt_admission(admitted.admission_id)
        assert recovering is not None
        assert recovering.state is WorkAttemptAdmissionState.RECOVERING
        assert recovering.claim.generation == 2

    asyncio.run(scenario())


def test_cancelled_recovery_waits_for_dispatched_session_mutation_and_reconciles() -> None:
    async def scenario() -> None:
        now = [datetime.now(UTC)]
        sessions = _BlockFirstWorkAttemptRecoveryTransition()
        tasks = InMemoryTaskStore(clock=lambda: now[0])
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
        )
        app.register_provider(_RecordingProvider(), default=True)
        app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        contract = _contract(contract_id="cancelled-recovery-contract")
        await tasks.publish_work_contract(contract)
        task = await tasks.create_task(
            TaskCreate(
                task_id="cancelled-recovery-task",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        admitted = await app.admit_work_attempt(
            RunRequest(
                agent_name="worker",
                task_id=task.id,
                session_id="cancelled-recovery-session",
                messages=[Message.text("user", "Recover after the old worker settles.")],
            ),
            execution=WorkAttemptExecutionRequest(
                admission_id="cancelled-recovery-admission",
                claim_id="cancelled-recovery-claim-1",
                attempt_id="cancelled-recovery-attempt",
                interaction_id="cancelled-recovery-interaction",
                worker_id="cancelled-recovery-worker-1",
                generation=1,
                lease_seconds=1,
            ),
        )
        await sessions.update_status(admitted.session_id, SessionStatus.INTERRUPTED)
        await sessions.release_run_fence(admitted.session_id)
        now[0] += timedelta(seconds=2)
        recovery_request = WorkAttemptRecoveryRequest(
            admission_id=admitted.admission_id,
            claim_id="cancelled-recovery-claim-2",
            worker_id="cancelled-recovery-worker-2",
            generation=2,
            lease_seconds=300,
        )
        pending = asyncio.create_task(app.recover_work_attempt(recovery_request))
        await sessions.transition_started.wait()
        pending.cancel()
        assert pending.cancelling() == 1
        await asyncio.sleep(0)
        assert not pending.done()
        sessions.release_transition.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert pending.cancelled()

        recovering = await tasks.load_work_attempt_admission(admitted.admission_id)
        assert recovering is not None
        assert recovering.state is WorkAttemptAdmissionState.RECOVERING
        recovered_session = await sessions.load(admitted.session_id)
        assert recovered_session is not None
        assert recovered_session.status is SessionStatus.RUNNING

        recovered = await app.recover_work_attempt(recovery_request)
        assert recovered.state is WorkAttemptAdmissionState.ACTIVE
        assert recovered.claim == recovering.claim
        assert (await sessions.load(admitted.session_id)).run_epoch == recovered_session.run_epoch

    asyncio.run(scenario())


def test_admission_activation_acknowledgement_loss_rebuilds_local_fences() -> None:
    async def scenario() -> None:
        sessions = InMemorySessionStore()
        tasks = _LoseFirstAdmissionActivationAcknowledgement()
        provider = _RecordingProvider()
        sink = InMemoryEventSink()
        contract = _contract(contract_id="activation-ack-contract")
        await tasks.publish_work_contract(contract)
        task = await tasks.create_task(
            TaskCreate(
                task_id="activation-ack-task",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            event_sinks=(sink,),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        run = RunRequest(
            agent_name="worker",
            task_id=task.id,
            session_id="activation-ack-session",
            messages=[Message.text("user", "Reconcile activation acknowledgement loss.")],
        )
        execution = WorkAttemptExecutionRequest(
            admission_id="activation-ack-admission",
            claim_id="activation-ack-claim",
            attempt_id="activation-ack-attempt",
            interaction_id="activation-ack-interaction",
            worker_id="activation-ack-worker",
            generation=1,
            lease_seconds=300,
        )

        failed_admission = asyncio.create_task(app.admit_work_attempt(run, execution=execution))
        with pytest.raises(RuntimeError, match="activation acknowledgement loss"):
            await failed_admission
        committed = await tasks.load_work_attempt_admission(execution.admission_id)
        assert committed is not None
        assert committed.state is WorkAttemptAdmissionState.ACTIVE
        assert sink.events == []
        committed_session = await sessions.load(run.session_id)
        assert committed_session is not None

        replay = await app.admit_work_attempt(run, execution=execution)
        assert replay == committed
        assert len(sink.events) == 1
        assert sink.events[0].type is EventType.INTERACTION_STARTED
        assert sink.events[0].interaction_id == committed.interaction_id
        assert provider.requests == []
        await sessions.release_run_fence(run.session_id)
        released_session = await sessions.load(run.session_id)
        assert released_session is not None
        assert released_session.run_epoch == committed_session.run_epoch + 1

    asyncio.run(scenario())


def test_recovery_completes_interaction_fanout_after_activation_acknowledgement_loss() -> None:
    async def scenario() -> None:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        sessions = InMemorySessionStore()
        tasks = _LoseFirstAdmissionActivationAcknowledgement(clock=lambda: now[0])
        sink = InMemoryEventSink()
        contract = _contract(contract_id="recovery-fanout-contract")
        await tasks.publish_work_contract(contract)
        task = await tasks.create_task(
            TaskCreate(
                task_id="recovery-fanout-task",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            event_sinks=(sink,),
            enable_logging=False,
        )
        app.register_provider(_RecordingProvider(), default=True)
        app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        execution = WorkAttemptExecutionRequest(
            admission_id="recovery-fanout-admission",
            claim_id="recovery-fanout-claim-1",
            attempt_id="recovery-fanout-attempt",
            interaction_id="recovery-fanout-interaction",
            worker_id="recovery-fanout-worker-1",
            generation=1,
            lease_seconds=1,
        )
        with pytest.raises(RuntimeError, match="activation acknowledgement loss"):
            await app.admit_work_attempt(
                RunRequest(
                    agent_name="worker",
                    task_id=task.id,
                    session_id="recovery-fanout-session",
                    messages=[Message.text("user", "Recover the interaction handoff.")],
                ),
                execution=execution,
            )
        committed = await tasks.load_work_attempt_admission(execution.admission_id)
        assert committed is not None
        assert committed.state is WorkAttemptAdmissionState.ACTIVE
        assert sink.events == []

        await sessions.update_status(committed.session_id, SessionStatus.INTERRUPTED)
        await sessions.release_run_fence(committed.session_id)
        now[0] += timedelta(seconds=2)
        recovered_app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            event_sinks=(sink,),
            enable_logging=False,
        )
        recovery = WorkAttemptRecoveryRequest(
            admission_id=committed.admission_id,
            claim_id="recovery-fanout-claim-2",
            worker_id="recovery-fanout-worker-2",
            generation=2,
            lease_seconds=300,
        )
        recovered = await recovered_app.recover_work_attempt(recovery)
        assert recovered.state is WorkAttemptAdmissionState.ACTIVE
        assert recovered.claim.generation == 2
        assert len(sink.events) == 1
        assert sink.events[0].type is EventType.INTERACTION_STARTED
        assert sink.events[0].interaction_id == recovered.interaction_id
        assert await recovered_app.recover_work_attempt(recovery) == recovered
        assert len(sink.events) == 1

    asyncio.run(scenario())


def test_prepared_admission_is_reclaimed_after_process_acknowledgement_loss() -> None:
    async def scenario() -> None:
        now = [datetime.now(UTC)]
        sessions = InMemorySessionStore()
        tasks = _LoseFirstAdmissionPreparationAcknowledgement(clock=lambda: now[0])
        contract = _contract(contract_id="prepared-recovery-contract")
        await tasks.publish_work_contract(contract)
        task = await tasks.create_task(
            TaskCreate(
                task_id="prepared-recovery-task",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        run = RunRequest(
            agent_name="worker",
            task_id=task.id,
            session_id="prepared-recovery-session",
            messages=[Message.text("user", "Recover the exact prepared attempt.")],
        )

        first_app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
        )
        first_app.register_provider(_RecordingProvider(), default=True)
        first_app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        with pytest.raises(RuntimeError, match="acknowledgement loss"):
            await first_app.admit_work_attempt(
                run,
                execution=WorkAttemptExecutionRequest(
                    admission_id="prepared-recovery-admission",
                    claim_id="prepared-recovery-claim-1",
                    attempt_id="prepared-recovery-attempt",
                    interaction_id="prepared-recovery-interaction",
                    worker_id="prepared-recovery-worker-1",
                    generation=1,
                    lease_seconds=1,
                ),
            )
        prepared = await tasks.load_work_attempt_admission("prepared-recovery-admission")
        assert prepared is not None
        assert prepared.state is WorkAttemptAdmissionState.PREPARING
        assert await sessions.load(run.session_id) is None

        now[0] += timedelta(seconds=2)
        replacement_app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
        )
        replacement_app.register_provider(_RecordingProvider(), default=True)
        replacement_app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        active = await replacement_app.admit_work_attempt(
            run,
            execution=WorkAttemptExecutionRequest(
                admission_id="prepared-recovery-admission",
                claim_id="prepared-recovery-claim-2",
                attempt_id="prepared-recovery-attempt",
                interaction_id="prepared-recovery-interaction",
                worker_id="prepared-recovery-worker-2",
                generation=2,
                lease_seconds=300,
            ),
        )
        assert active.state is WorkAttemptAdmissionState.ACTIVE
        assert active.claim.generation == 2
        assert active.claim.claim_id == "prepared-recovery-claim-2"
        assert active.attempt is not None
        assert (await sessions.load(run.session_id)).status is SessionStatus.RUNNING
        current_task = await tasks.load_task(task.id)
        assert current_task is not None
        assert current_task.worker_id == "prepared-recovery-worker-2"

    asyncio.run(scenario())


def test_child_only_session_cancellation_is_a_safe_operational_failure(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "child-session-cancellation-secret"

    async def scenario() -> BaseException:
        sessions = _CancelFirstWorkAttemptSessionCreation(secret)
        tasks = InMemoryTaskStore()
        app, run, execution = await _configured_public_initial_admission(
            prefix="child-session-cancellation",
            sessions=sessions,
            tasks=tasks,
            redactor=SecretRedactor(secret),
        )
        run = run.model_copy(update={"messages": [Message.text("user", f"Keep {secret} private.")]})
        session_id = run.session_id
        assert session_id is not None
        owner = asyncio.create_task(app.admit_work_attempt(run, execution=execution))
        with pytest.raises(RuntimeError, match="cancelled without caller cancellation") as captured:
            await owner
        assert not owner.cancelled()
        assert owner.cancelling() == 0
        assert await sessions.load(session_id) is None
        prepared = await tasks.load_work_attempt_admission(execution.admission_id)
        assert prepared is not None
        assert prepared.state is WorkAttemptAdmissionState.PREPARING
        return captured.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error = asyncio.run(scenario())

    _assert_secret_absent_from_work_attempt_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_caller_cancellation_detaches_failed_session_mutation_evidence(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "cancelled-session-failure-secret"

    async def scenario() -> tuple[asyncio.CancelledError, asyncio.Task[WorkAttemptAdmission]]:
        sessions = _FailCancelledWorkAttemptSessionCreation(secret)
        tasks = InMemoryTaskStore()
        app, run, execution = await _configured_public_initial_admission(
            prefix="cancelled-session-failure",
            sessions=sessions,
            tasks=tasks,
            redactor=SecretRedactor(secret),
        )
        run = run.model_copy(update={"messages": [Message.text("user", f"Keep {secret} private.")]})
        session_id = run.session_id
        assert session_id is not None
        owner = asyncio.create_task(app.admit_work_attempt(run, execution=execution))
        await sessions.create_started.wait()
        owner.cancel("stop waiting for the failed session mutation")
        assert owner.cancelling() == 1
        await asyncio.sleep(0)
        assert not owner.done()
        sessions.release_create.set()
        with pytest.raises(asyncio.CancelledError) as captured:
            await owner
        assert owner.cancelled()
        assert owner.cancelling() == 1
        assert isinstance(captured.value.__cause__, RuntimeError)
        assert await sessions.load(session_id) is None
        prepared = await tasks.load_work_attempt_admission(execution.admission_id)
        assert prepared is not None
        assert prepared.state is WorkAttemptAdmissionState.PREPARING
        return captured.value, owner

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        cancellation, _owner = asyncio.run(scenario())

    _assert_secret_absent_from_work_attempt_error(cancellation, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_session_creation_failure_detaches_request_and_extension_secrets(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request_secret = "session-create-request-payload-secret"
    extension_secret = "session-create-extension-failure-secret"

    async def scenario() -> BaseException:
        sessions = _FailWorkAttemptSessionCreation(extension_secret)
        tasks = InMemoryTaskStore()
        app, run, execution = await _configured_public_initial_admission(
            prefix="failed-session-create",
            sessions=sessions,
            tasks=tasks,
            redactor=SecretRedactor([request_secret, extension_secret]),
        )
        run = run.model_copy(
            update={"messages": [Message.text("user", f"Keep {request_secret} private.")]}
        )
        session_id = run.session_id
        assert session_id is not None
        with pytest.raises(RuntimeError) as captured:
            await app.admit_work_attempt(run, execution=execution)
        assert await sessions.load(session_id) is None
        prepared = await tasks.load_work_attempt_admission(execution.admission_id)
        assert prepared is not None
        assert prepared.state is WorkAttemptAdmissionState.PREPARING
        return captured.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error = asyncio.run(scenario())

    for secret in (request_secret, extension_secret):
        _assert_secret_absent_from_work_attempt_error(error, secret)
        assert secret not in caplog.text
    captured = capsys.readouterr()
    assert request_secret not in captured.out
    assert request_secret not in captured.err
    assert extension_secret not in captured.out
    assert extension_secret not in captured.err
    assert all(
        request_secret not in str(item.message) and extension_secret not in str(item.message)
        for item in caught_warnings
    )


@pytest.mark.parametrize(
    "failure_type",
    (
        WorkAttemptAdmissionConflict,
        WorkAttemptExecutionClaimLost,
        WorkAttemptRecoveryRequired,
    ),
)
def test_grouped_admission_failure_fallback_preserves_public_classification(
    failure_type: type[Exception],
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = f"| {failure_type.__name__}: token"

    class GroupedAdmissionFailureStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        async def prepare_work_attempt_admission(
            self,
            request: WorkAttemptAdmissionPrepare,
        ) -> WorkAttemptAdmission:
            del request
            raise ExceptionGroup(
                "grouped admission failure",
                [failure_type("token")],
            )

    async def scenario() -> BaseExceptionGroup:
        sessions = InMemorySessionStore()
        tasks = GroupedAdmissionFailureStore()
        app, run, execution = await _configured_public_initial_admission(
            prefix="grouped-admission-failure",
            sessions=sessions,
            tasks=tasks,
            redactor=SecretRedactor(secret),
        )
        session_id = run.session_id
        assert session_id is not None
        with pytest.raises(BaseExceptionGroup) as captured:
            await app.admit_work_attempt(run, execution=execution)
        assert await sessions.load(session_id) is None
        assert await tasks.load_work_attempt_admission(execution.admission_id) is None
        return captured.value

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        error = asyncio.run(scenario())

    assert _work_attempt_failure_leaf_types(error) == (failure_type,)
    _assert_secret_absent_from_work_attempt_error(error, secret)
    captured = capsys.readouterr()
    assert secret not in caplog.text
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in str(item.message) for item in caught_warnings)


def test_cancelled_public_preparation_is_quiescent_and_reclaimable() -> None:
    async def scenario() -> None:
        now = [datetime.now(UTC)]
        sessions = _BlockFirstWorkAttemptSessionCreation()
        tasks = InMemoryTaskStore(clock=lambda: now[0])
        contract = _contract(contract_id="cancelled-preparation-contract")
        await tasks.publish_work_contract(contract)
        task = await tasks.create_task(
            TaskCreate(
                task_id="cancelled-preparation-task",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        provider = _RecordingProvider()
        first_app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
        )
        first_app.register_provider(provider, default=True)
        first_app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        run = RunRequest(
            agent_name="worker",
            task_id=task.id,
            session_id="cancelled-preparation-session",
            messages=[Message.text("user", "Cancel before session publication.")],
        )
        pending = asyncio.create_task(
            first_app.admit_work_attempt(
                run,
                execution=WorkAttemptExecutionRequest(
                    admission_id="cancelled-preparation-admission",
                    claim_id="cancelled-preparation-claim-1",
                    attempt_id="cancelled-preparation-attempt",
                    interaction_id="cancelled-preparation-interaction",
                    worker_id="cancelled-preparation-worker-1",
                    generation=1,
                    lease_seconds=1,
                ),
            )
        )
        await sessions.create_started.wait()
        pending.cancel()
        assert pending.cancelling() == 1
        await asyncio.sleep(0)
        assert not pending.done()
        sessions.release_create.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert pending.cancelled()
        assert provider.requests == []
        committed_session = await sessions.load(run.session_id)
        assert committed_session is not None
        assert committed_session.status is SessionStatus.RUNNING
        prepared = await tasks.load_work_attempt_admission("cancelled-preparation-admission")
        assert prepared is not None
        assert prepared.state is WorkAttemptAdmissionState.PREPARING
        assert prepared.claim.generation == 1

        now[0] += timedelta(seconds=2)
        replacement_app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
        )
        replacement_app.register_provider(provider, default=True)
        replacement_app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        active = await replacement_app.admit_work_attempt(
            run,
            execution=WorkAttemptExecutionRequest(
                admission_id="cancelled-preparation-admission",
                claim_id="cancelled-preparation-claim-2",
                attempt_id="cancelled-preparation-attempt",
                interaction_id="cancelled-preparation-interaction",
                worker_id="cancelled-preparation-worker-2",
                generation=2,
                lease_seconds=300,
            ),
        )
        assert active.state is WorkAttemptAdmissionState.ACTIVE
        assert active.claim.generation == 2
        assert (await sessions.load(run.session_id)).status is SessionStatus.RUNNING
        assert provider.requests == []

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_public_rejected_continue_admission_preserves_contract_and_adds_interaction(
    backend: str,
    tmp_path,
) -> None:
    request_secret = f"continuation-request-payload-secret-{backend}"

    async def scenario() -> None:
        now = [datetime(2026, 1, 1, tzinfo=UTC)]
        sessions = _BlockFirstWorkAttemptContinuationAdmission()
        tasks = (
            InMemoryTaskStore(clock=lambda: now[0])
            if backend == "memory"
            else SQLiteTaskStore(
                tmp_path / "continuation-admission.db",
                clock=lambda: now[0],
            )
        )
        provider = _RecordingProvider()
        sink = InMemoryEventSink()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            event_sinks=(sink,),
            secret_redactor=SecretRedactor(request_secret),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker", model="verified-work-test-model"))
        contract = _contract(contract_id="continuation-admission-contract")
        await tasks.publish_work_contract(contract)
        task = await tasks.create_task(
            TaskCreate(
                task_id="continuation-admission-task",
                type="verified-work",
                work_contract=contract.reference(),
            )
        )
        first_execution = WorkAttemptExecutionRequest(
            admission_id="continuation-admission-1",
            claim_id="continuation-execution-claim-1",
            attempt_id="continuation-attempt-1",
            interaction_id="continuation-interaction-1",
            worker_id="continuation-worker-1",
            generation=1,
            lease_seconds=1,
        )
        first = await app.admit_work_attempt(
            RunRequest(
                agent_name="worker",
                task_id=task.id,
                session_id="continuation-admission-session",
                messages=[Message.text("user", "Create the first result.")],
            ),
            execution=first_execution,
        )
        now[0] += timedelta(seconds=2)
        first = await app.recover_work_attempt(
            WorkAttemptRecoveryRequest(
                admission_id=first.admission_id,
                claim_id="continuation-execution-claim-1-recovered",
                worker_id="continuation-worker-1-recovered",
                generation=2,
                lease_seconds=300,
            )
        )
        first_checkpoint = await sessions.load_checkpoint(first.session_id)
        assert first_checkpoint is not None
        assert "initial_transcript_pending" not in first_checkpoint
        assert WORK_ATTEMPT_RECOVERY_CHECKPOINT_KEY in first_checkpoint
        first_proposal_request = WorkAttemptProposalRequest(
            admission_id=first.admission_id,
            claim_id=first.claim.claim_id,
            generation=first.claim.generation,
            proposal=CompletionProposalCreate(
                proposal_id="continuation-proposal-1",
                attempt_id=first.attempt_id,
                result=_result_reference(),
                evidence_references=(_artifact_evidence(),),
            ),
        )
        proposal = await app.submit_work_attempt_proposal(first_proposal_request)
        assert await sessions.load_deferred_interaction_input(first.session_id) is None
        await sessions.update_status(first.session_id, SessionStatus.COMPLETED)
        await sessions.release_run_fence(first.session_id)
        verifier_claim = await _claim_completion_verification(
            tasks,
            CompletionVerificationClaimRequest(
                claim_id="continuation-verifier-claim",
                proposal_id=proposal.proposal_id,
                worker_id="continuation-verifier-worker",
                verifier=contract.verifier,
                verifier_profile_fingerprint=_verifier_profile_fingerprint(contract.verifier),
            ),
        )
        large_evidence = tuple(
            WorkEvidenceReference(
                kind="artifact.version",
                reference_id=f"large-evidence:{index:03d}:" + ("x" * 1800),
                requirement_id="artifact",
                version=str(index),
                digest=_digest(f"large-evidence-{index}"),
            )
            for index in range(220)
        )
        decision = await tasks.record_completion_decision(
            _rejected_decision(
                proposal_id=proposal.proposal_id,
                claim_id=verifier_claim.claim_id,
                worker_id=verifier_claim.worker_id,
                decision_id="continuation-rejected-decision",
            ).model_copy(
                update={
                    "evidence_references": (
                        _artifact_evidence(),
                        *large_evidence,
                    )
                }
            )
        )
        decision_bytes = canonical_durable_json_bytes(
            decision.model_dump(mode="json", warnings=False),
            "large_continuation_decision",
        )
        assert 128 * 1024 < len(decision_bytes) <= WORK_COMPLETION_DECISION_MAX_BYTES
        await tasks.apply_completion_decision(
            CompletionDecisionApplicationRequest(
                task_id=task.id,
                decision_id=decision.decision_id,
                idempotency_key="continuation-rejected-application",
            )
        )
        sibling_task = await app.create_task(
            TaskCreate(
                task_id=f"continuation-sibling-task-{backend}",
                type="verified-work",
                session_id=first.session_id,
                work_contract=contract.reference(),
            )
        )
        sibling_task = await tasks.start_task(
            sibling_task.id,
            session_id=first.session_id,
            session_invocation=await stored_session_invocation(sessions, first.session_id),
        )
        assert sibling_task.status is TaskStatus.RUNNING
        with pytest.raises(WorkAttemptAdmissionConflict, match="permanently governed"):
            await tasks.begin_work_attempt(
                WorkAttemptCreate(
                    attempt_id="continuation-direct-bypass-attempt",
                    task_id=task.id,
                    session_id=first.session_id,
                    contract=contract.reference(),
                    execution_profile_fingerprint=first.source_execution_profile_fingerprint,
                    worker_id=None,
                )
            )
        assert await tasks.load_work_attempt("continuation-direct-bypass-attempt") is None

        second_execution = WorkAttemptExecutionRequest(
            admission_id="continuation-admission-2",
            claim_id="continuation-execution-claim-2",
            attempt_id="continuation-attempt-2",
            interaction_id="continuation-interaction-2",
            worker_id="continuation-worker-2",
            task_id=task.id,
            predecessor_admission_id=first.admission_id,
            generation=1,
            lease_seconds=1,
        )
        continuation_request = ResumeRequest(
            session_id=first.session_id,
            messages=[
                Message.text(
                    "user",
                    f"Resolve the reported gaps while keeping {request_secret} private.",
                )
            ],
        )
        continuation_checkpoint = await sessions.load_checkpoint(first.session_id)
        assert continuation_checkpoint is not None
        foreign_checkpoint = {
            **continuation_checkpoint,
            WORK_ATTEMPT_RECOVERY_CHECKPOINT_KEY: {
                "admission_id": "foreign-recovery-admission",
                "claim_id": "foreign-recovery-claim",
                "generation": 1,
                "request_sha256": _digest("foreign-recovery-request"),
            },
        }
        await sessions.checkpoint(first.session_id, foreign_checkpoint)
        with pytest.raises(
            WorkAttemptRecoveryRequired,
            match="belongs to another predecessor",
        ):
            await app.admit_work_attempt(
                continuation_request,
                execution=second_execution,
            )
        assert await tasks.load_work_attempt_admission(second_execution.admission_id) is None
        malformed_checkpoint = {
            **continuation_checkpoint,
            WORK_ATTEMPT_RECOVERY_CHECKPOINT_KEY: {
                "admission_id": first.admission_id,
            },
        }
        await sessions.checkpoint(first.session_id, malformed_checkpoint)
        with pytest.raises(
            WorkAttemptRecoveryRequired,
            match="malformed durable authority",
        ):
            await app.admit_work_attempt(
                continuation_request,
                execution=second_execution,
            )
        assert await tasks.load_work_attempt_admission(second_execution.admission_id) is None
        await sessions.checkpoint(first.session_id, continuation_checkpoint)
        wrong_selection = second_execution.model_copy(
            update={
                "admission_id": "continuation-wrong-task-admission",
                "claim_id": "continuation-wrong-task-claim",
                "attempt_id": "continuation-wrong-task-attempt",
                "interaction_id": "continuation-wrong-task-interaction",
                "task_id": sibling_task.id,
            }
        )
        with pytest.raises(
            WorkAttemptAdmissionConflict,
            match="released task-session authority",
        ):
            await app.admit_work_attempt(
                continuation_request,
                execution=wrong_selection,
            )
        assert await tasks.load_work_attempt_admission(wrong_selection.admission_id) is None
        missing_predecessor = second_execution.model_copy(
            update={
                "admission_id": "continuation-missing-predecessor-admission",
                "claim_id": "continuation-missing-predecessor-claim",
                "attempt_id": "continuation-missing-predecessor-attempt",
                "interaction_id": "continuation-missing-predecessor-interaction",
                "predecessor_admission_id": "missing-predecessor-admission",
            }
        )
        with pytest.raises(
            WorkAttemptAdmissionConflict,
            match="predecessor admission is missing",
        ):
            await app.admit_work_attempt(
                continuation_request,
                execution=missing_predecessor,
            )
        assert await tasks.load_work_attempt_admission(missing_predecessor.admission_id) is None
        pending_continuation = asyncio.create_task(
            app.admit_work_attempt(
                continuation_request,
                execution=second_execution,
            )
        )
        await sessions.admission_started.wait()
        pending_continuation.cancel()
        assert pending_continuation.cancelling() == 1
        await asyncio.sleep(0)
        assert not pending_continuation.done()
        sessions.release_admission.set()
        with pytest.raises(asyncio.CancelledError) as captured_cancellation:
            await pending_continuation
        assert pending_continuation.cancelled()
        _assert_secret_absent_from_work_attempt_error(
            captured_cancellation.value,
            request_secret,
        )
        prepared_continuation = await tasks.load_work_attempt_admission(
            second_execution.admission_id
        )
        assert prepared_continuation is not None
        assert prepared_continuation.state is WorkAttemptAdmissionState.PREPARING

        second = await app.admit_work_attempt(continuation_request, execution=second_execution)
        assert second.state is WorkAttemptAdmissionState.ACTIVE
        assert second.attempt is not None
        assert second.attempt.ordinal == 2
        assert second.session_id == first.session_id
        assert second.interaction_id != first.interaction_id
        assert second.contract == first.contract == contract.reference()
        assert second.continuation is not None
        assert second.continuation.decision == decision
        assert second.continuation.gaps == decision.gaps
        second_checkpoint = await sessions.load_checkpoint(second.session_id)
        assert second_checkpoint is not None
        assert WORK_ATTEMPT_RECOVERY_CHECKPOINT_KEY not in second_checkpoint
        assert [
            event.interaction_id
            for event in sink.events
            if event.type is EventType.INTERACTION_STARTED
        ] == [
            first.interaction_id,
            second.interaction_id,
        ]
        await sessions.update_status(second.session_id, SessionStatus.RUNNING)
        assert (
            await app.admit_work_attempt(continuation_request, execution=second_execution) == second
        )
        assert sum(event.type is EventType.INTERACTION_STARTED for event in sink.events) == 2
        assert provider.requests == []
        now[0] += timedelta(seconds=2)
        recovered_second = await app.recover_work_attempt(
            WorkAttemptRecoveryRequest(
                admission_id=second.admission_id,
                claim_id="continuation-execution-claim-2-recovered",
                worker_id="continuation-worker-2-recovered",
                generation=2,
                lease_seconds=300,
            )
        )
        assert recovered_second.state is WorkAttemptAdmissionState.ACTIVE
        assert recovered_second.claim.generation == 2
        assert recovered_second.attempt == second.attempt
        restarted_app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            enable_logging=False,
        )
        assert await restarted_app.submit_work_attempt_proposal(first_proposal_request) == proposal
        with pytest.raises(ValueError, match="accepts new messages only"):
            await app.admit_work_attempt(
                continuation_request.model_copy(update={"loop_policies": (LoopPolicy(),)}),
                execution=second_execution,
            )
        assert provider.requests == []
        if backend == "sqlite":
            await tasks.close()
            connection = sqlite3.connect(tmp_path / "continuation-admission.db")
            try:
                payload = json.loads(
                    connection.execute(
                        "SELECT admission_json FROM cayu_work_attempt_admissions "
                        "WHERE admission_id = ?",
                        (second.admission_id,),
                    ).fetchone()[0]
                )
                payload["continuation"].pop("prior_admission_id")
                connection.execute(
                    "UPDATE cayu_work_attempt_admissions SET admission_json = ? "
                    "WHERE admission_id = ?",
                    (sqlite_support.json_dumps(payload), second.admission_id),
                )
                connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 62")
                connection.execute("PRAGMA user_version = 61")
                connection.commit()
            finally:
                connection.close()
            migrated = SQLiteTaskStore(
                tmp_path / "continuation-admission.db",
                schema_mode=SchemaMode.MIGRATE,
            )
            try:
                migrated_second = await migrated.load_work_attempt_admission(second.admission_id)
                assert migrated_second is not None
                assert migrated_second.continuation is not None
                assert migrated_second.continuation.prior_admission_id == first.admission_id
            finally:
                await migrated.close()

    asyncio.run(scenario())


def test_sqlite_revision_61_migrates_and_validates_admission_authority(
    tmp_path,
) -> None:
    path = tmp_path / "revision-61-work-attempt.sqlite"

    async def create() -> None:
        store = SQLiteTaskStore(path)
        try:
            await store.create_task(TaskCreate(task_id="pre-61-task", type="ordinary"))
        finally:
            await store.close()

    asyncio.run(create())
    connection = sqlite3.connect(path)
    try:
        connection.execute("DROP TABLE cayu_work_attempt_execution_claims")
        connection.execute("DROP TABLE cayu_work_attempt_admissions")
        connection.execute("DELETE FROM cayu_schema_migrations WHERE revision >= 61")
        connection.execute("PRAGMA user_version = 60")
        connection.commit()
    finally:
        connection.close()

    async def migrate() -> None:
        store = SQLiteTaskStore(path, schema_mode=SchemaMode.MIGRATE)
        try:
            assert await store.load_task("pre-61-task") is not None
        finally:
            await store.close()

    asyncio.run(migrate())
    connection = sqlite3.connect(path)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name IN (?, ?)",
                (
                    "cayu_work_attempt_admissions",
                    "cayu_work_attempt_execution_claims",
                ),
            )
        }
        assert tables == {
            "cayu_work_attempt_admissions",
            "cayu_work_attempt_execution_claims",
        }
        connection.execute("DROP INDEX idx_cayu_work_attempt_claim_current")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(RuntimeError, match="work-attempt admission schema"):
        SQLiteTaskStore(path, schema_mode=SchemaMode.VALIDATE)
