"""Runtime-owned accepted-result resolution and application boundary."""

from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from time import monotonic
from typing import Any, cast
from uuid import uuid4

from cayu._exception_groups import exception_tree_contains
from cayu._task_wait import (
    CapturedAwaitableOutcome,
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    consume_pending_task_cancellation,
    restore_task_cancellation_requests,
)
from cayu._validation import canonical_durable_json_bytes
from cayu.core.events import (
    Event,
    EventType,
    event_with_runtime_envelope_authority,
    event_with_runtime_generated_id,
    event_with_runtime_payload_authority,
)
from cayu.runtime._completion_decision_application_coordinator import (
    CompletionDecisionApplicationCoordinator,
    _CompletionDecisionApplicationNotCommitted,
)
from cayu.runtime._diagnostics import (
    credential_safe_runtime_exception,
    credential_safe_runtime_exception_group,
    exception_diagnostic,
)
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._task_store_operation_boundary import (
    capture_sensitive_validation,
    raise_task_store_operation_failure,
)
from cayu.runtime.completion_result_resolvers import (
    CompletionResultResolutionRequest,
    CompletionResultResolver,
    CompletionResultResolverExecutionError,
    CompletionResultResolverRequest,
    CompletionResultResolverUnavailable,
    CompletionResultUnavailable,
    copy_completion_result_resolution_request,
)
from cayu.runtime.sessions import (
    Session,
    SessionStore,
    _complete_completion_result_event_publication,
    _release_completion_result_event_publication,
    _renew_completion_result_event_publication,
    _reserve_completion_result_event_publication,
)
from cayu.runtime.tasks import CompletionDecisionApplicationReceipt, Task
from cayu.runtime.work_contracts import (
    CompletionDecision,
    CompletionDecisionApplicationRequest,
    CompletionProposal,
    CompletionResultResolverRef,
    WorkAttempt,
    WorkCompletionConflict,
    WorkContract,
    completion_decision_application_request_sha256,
)
from cayu.runtime.workspace_observation_recovery import (
    retain_workspace_observation_pending_cancellation_requests,
)
from cayu.vaults import SecretRedactor

_MAX_ACTIVE_RESULT_RESOLVERS = 64
_PROCESS_CONTROL_SIGNALS = (GeneratorExit, KeyboardInterrupt, SystemExit)
_PUBLICATION_OWNER_ID_PREFIX = "completion-result-owner:v1:"
_PUBLICATION_OWNER_LEASE_SECONDS = 360.0
_PUBLICATION_OWNER_HEARTBEAT_SECONDS = 30.0
_PUBLICATION_RELEASE_SETTLEMENT_SECONDS = 5.0


@dataclass(slots=True)
class _SingleFlightLock:
    lock: asyncio.Lock
    users: int = 0

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.users = 0


@dataclass(frozen=True, slots=True)
class _ResolvedEventPublicationAuthority:
    session_id: str
    publication_id: str
    authority_sha256: str
    source_session_instance_id: str


@dataclass(frozen=True, slots=True)
class _PublicationOwner:
    owner_id: str


@dataclass(slots=True)
class _PublicationLeaseDeadline:
    monotonic: float


@dataclass(slots=True)
class _PublicationHeartbeat:
    stop: asyncio.Event
    task: asyncio.Task[None]
    lease_deadline: _PublicationLeaseDeadline
    ownership_lost: asyncio.Future[BaseException]
    transferred: bool = False
    observed_failure_id: int | None = None


@dataclass(slots=True)
class _ResolutionCapacityLease:
    reservation: object | None = None
    transferred: bool = False


@dataclass(slots=True)
class _ApplicationSettlementEvidence:
    not_committed: bool = False


def _resolver_key(reference: CompletionResultResolverRef) -> tuple[str, str, str]:
    return (
        reference.resolver_id,
        reference.version,
        reference.configuration_fingerprint,
    )


class CompletionResultResolverCoordinator:
    """Resolve one accepted result and apply it through exact durable receipts."""

    def __init__(
        self,
        *,
        application_coordinator: CompletionDecisionApplicationCoordinator,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        secret_redactor: SecretRedactor,
    ) -> None:
        if not isinstance(
            application_coordinator,
            CompletionDecisionApplicationCoordinator,
        ):
            raise TypeError(
                "Result resolution requires a CompletionDecisionApplicationCoordinator."
            )
        if not isinstance(event_writer, RuntimeEventWriter):
            raise TypeError("Result resolution requires a RuntimeEventWriter.")
        if not callable(getattr(session_store, "publish_checkpoint_and_events", None)):
            raise TypeError("Result resolution requires a SessionStore.")
        if not isinstance(secret_redactor, SecretRedactor):
            raise TypeError("Result resolution requires a SecretRedactor.")
        self._application_coordinator = application_coordinator
        self._session_store = session_store
        self._event_writer = event_writer
        self._secret_redactor = secret_redactor
        self._process_id = os.getpid()
        self._resolvers: dict[tuple[str, str, str], CompletionResultResolver] = {}
        self._locks: dict[str, _SingleFlightLock] = {}
        self._adapter_tasks: set[asyncio.Task[CapturedAwaitableOutcome[dict[str, object]]]] = set()
        self._resolution_capacity_reservations: set[object] = set()
        self._draining_adapter_tasks: dict[
            str,
            asyncio.Task[CapturedAwaitableOutcome[None]],
        ] = {}

    def _ensure_process_local_generation(self) -> None:
        process_id = os.getpid()
        if process_id == self._process_id:
            return
        if (
            self._locks
            or self._adapter_tasks
            or self._resolution_capacity_reservations
            or self._draining_adapter_tasks
        ):
            raise self._safe_execution_error(
                "Result resolver coordinator inherited active execution state across a "
                "process boundary; rebuild the application in this worker."
            ) from None
        self._process_id = process_id

    def register(
        self,
        reference: CompletionResultResolverRef,
        resolver: CompletionResultResolver,
    ) -> CompletionResultResolverRef:
        self._ensure_process_local_generation()
        if type(reference) is not CompletionResultResolverRef:
            del reference, resolver
            raise TypeError("Result resolver registration requires a CompletionResultResolverRef.")
        if not isinstance(resolver, CompletionResultResolver):
            del reference, resolver
            raise TypeError("Result resolver registration requires a CompletionResultResolver.")
        validation = capture_sensitive_validation(
            lambda value=reference: CompletionResultResolverRef(
                resolver_id=value.resolver_id,
                version=value.version,
                configuration_fingerprint=value.configuration_fingerprint,
            ),
            operation_name="Completion result resolver reference validation",
            redactor=self._secret_redactor,
        )
        del reference
        if validation.failure is not None:
            del resolver
            raise_task_store_operation_failure(validation.failure)
        copied = validation.result
        del validation
        if copied is None:
            del resolver
            raise ValueError("Completion result resolver reference is invalid.") from None
        if any(
            self._secret_redactor.redact_text(value) != value
            for value in (
                copied.resolver_id,
                copied.version,
                copied.configuration_fingerprint,
            )
        ):
            del copied, resolver
            raise ValueError(
                "Completion result resolver identity contains a workload secret and cannot "
                "be registered as durable authority."
            ) from None
        key = _resolver_key(copied)
        if key in self._resolvers:
            del copied, key, resolver
            raise credential_safe_runtime_exception(
                ValueError,
                "Completion result resolver identity is already registered.",
                redactor=self._secret_redactor,
                fallback_message="Completion result resolver registration conflict.",
            ) from None
        self._resolvers[key] = resolver
        del resolver
        return copied

    async def resolve(self, request: CompletionResultResolutionRequest) -> Task:
        self._ensure_process_local_generation()
        validation = capture_sensitive_validation(
            lambda value=request: copy_completion_result_resolution_request(value),
            operation_name="Completion result resolution request validation",
            redactor=self._secret_redactor,
        )
        del request
        if validation.failure is not None:
            raise_task_store_operation_failure(validation.failure)
        copied = validation.result
        del validation
        if copied is None:
            raise ValueError("Completion result resolution request is invalid.") from None
        if any(
            self._secret_redactor.redact_text(value) != value
            for value in (copied.task_id, copied.decision_id, copied.idempotency_key)
        ):
            del copied
            raise ValueError(
                "Completion result resolution contains a workload secret in public identity."
            ) from None

        key = copied.decision_id
        entry = self._locks.get(key)
        if entry is None:
            entry = _SingleFlightLock()
            self._locks[key] = entry
        entry.users += 1
        try:
            current_task = asyncio.current_task()
            cancellation_baseline = 0 if current_task is None else current_task.cancelling()
            safe_cancellation = None
            cancellation_failure = None
            try:
                await entry.lock.acquire()
            except asyncio.CancelledError as cancellation:
                current_requests = 0 if current_task is None else current_task.cancelling()
                if current_requests > cancellation_baseline:
                    safe_cancellation = self._redeliver_safe_caller_cancellation(
                        cancellation,
                        preserve_requests=cancellation_baseline,
                    )
                else:
                    cancellation_failure = self._safe_dependency_cancellation_failure(
                        cancellation,
                        "Completion result resolution ownership wait was cancelled without "
                        "caller cancellation.",
                    )
                del cancellation
            if cancellation_failure is not None:
                del copied
                raise_task_store_operation_failure(cancellation_failure)
            if safe_cancellation is not None:
                del copied
                raise safe_cancellation
            try:
                capacity_lease = _ResolutionCapacityLease()
                try:
                    cancellation_baseline = 0 if current_task is None else current_task.cancelling()
                    safe_cancellation = None
                    cancellation_failure = None
                    try:
                        return await self._resolve_locked(copied, capacity_lease)
                    except asyncio.CancelledError as cancellation:
                        current_requests = 0 if current_task is None else current_task.cancelling()
                        if current_requests > cancellation_baseline:
                            safe_cancellation = self._redeliver_safe_caller_cancellation(
                                cancellation,
                                preserve_requests=cancellation_baseline,
                            )
                        else:
                            cancellation_failure = self._safe_dependency_cancellation_failure(
                                cancellation,
                                "Completion result resolution dependency was cancelled "
                                "without caller cancellation.",
                            )
                        del cancellation
                    if cancellation_failure is not None:
                        del copied
                        raise_task_store_operation_failure(cancellation_failure)
                    assert safe_cancellation is not None
                    del copied
                    raise safe_cancellation
                finally:
                    if not capacity_lease.transferred:
                        self._release_resolution_capacity(capacity_lease)
            finally:
                entry.lock.release()
        finally:
            entry.users -= 1
            if entry.users == 0 and self._locks.get(key) is entry:
                self._locks.pop(key, None)

    async def _resolve_locked(
        self,
        request: CompletionResultResolutionRequest,
        capacity_lease: _ResolutionCapacityLease,
    ) -> Task:
        receipt = None
        decision = None
        claim = None
        proposal = None
        attempt = None
        contract = None
        verifier_profile = None
        authority_task = None
        resolver = None
        adapter_request_validation = None
        adapter_request = None
        result = None
        application_validation = None
        application_request = None
        task = None
        final_receipt = None
        publication_authority = None
        publication_owner = None
        publication_reserved = False
        publication_heartbeat = None
        application_started = False
        application_settlement = _ApplicationSettlementEvidence()
        prepared_event = None
        failure: BaseException | None = None
        try:
            receipt = await self._application_coordinator.load_result_resolution_receipt(
                task_id=request.task_id,
                idempotency_key=request.idempotency_key,
            )
            (
                decision,
                claim,
                proposal,
                attempt,
                contract,
                verifier_profile,
                authority_task,
            ) = await self._application_coordinator.load_result_resolution_authority(
                task_id=request.task_id,
                decision_id=request.decision_id,
            )

            if receipt is None:
                resolver = self._resolvers.get(_resolver_key(contract.result_resolver))
                if resolver is None:
                    raise credential_safe_runtime_exception(
                        CompletionResultResolverUnavailable,
                        "The exact completion result resolver required by the work contract "
                        "is not registered.",
                        redactor=self._secret_redactor,
                        fallback_message=(
                            "The required completion result resolver is unavailable."
                        ),
                    ) from None
                self._reserve_resolution_capacity(capacity_lease)

            publication_authority = self._resolved_event_publication_authority(
                request=request,
                decision=decision,
                proposal=proposal,
                attempt=attempt,
                contract=contract,
                authority_task=authority_task,
            )
            publication_owner = self._new_publication_owner()

            if receipt is None:
                await self._require_resolver_not_draining(request.decision_id)
            publication_deadline = await self._reserve_resolved_event_publication(
                publication_authority,
                publication_owner,
            )
            publication_reserved = True
            if monotonic() >= publication_deadline:
                raise self._safe_execution_error(
                    "Completion-result publication ownership acknowledgement consumed its lease."
                ) from None
            publication_heartbeat = self._start_publication_heartbeat(
                publication_authority,
                publication_owner,
                claim_deadline_monotonic=publication_deadline,
            )

            if receipt is None:
                adapter_request_validation = capture_sensitive_validation(
                    lambda resolver_contract=contract, work_attempt=attempt, completion_proposal=proposal, completion_decision=decision: (
                        CompletionResultResolverRequest(
                            contract=resolver_contract,
                            attempt=work_attempt,
                            proposal=completion_proposal,
                            decision=completion_decision,
                            result_reference=completion_proposal.result,
                        )
                    ),
                    operation_name="Completion result resolver context validation",
                    redactor=self._secret_redactor,
                )
                if adapter_request_validation.failure is not None:
                    raise_task_store_operation_failure(adapter_request_validation.failure)
                adapter_request = adapter_request_validation.result
                adapter_request_validation = None
                if adapter_request is None:
                    raise WorkCompletionConflict(
                        "Durable completion result resolver context is invalid."
                    ) from None
                if resolver is None:
                    raise CompletionResultResolverUnavailable(
                        "The required completion result resolver is unavailable."
                    ) from None
                result = await self._invoke_resolver(
                    resolver,
                    adapter_request,
                    decision_id=request.decision_id,
                    timeout_seconds=request.execution_timeout_seconds,
                    publication_authority=publication_authority,
                    publication_owner=publication_owner,
                    publication_heartbeat=publication_heartbeat,
                    capacity_lease=capacity_lease,
                )
            else:
                if receipt.task_id != request.task_id or receipt.decision_id != request.decision_id:
                    raise WorkCompletionConflict(
                        "Completion result resolution identity is already bound to another "
                        "decision."
                    ) from None
                result = receipt.task.result
                if type(result) is not dict:
                    raise WorkCompletionConflict(
                        "Accepted decision application receipt contains no task result."
                    ) from None

            application_validation = capture_sensitive_validation(
                lambda value=result, resolution_request=request, result_reference=proposal.result: (
                    CompletionDecisionApplicationRequest(
                        task_id=resolution_request.task_id,
                        decision_id=resolution_request.decision_id,
                        idempotency_key=resolution_request.idempotency_key,
                        result=value,
                        result_reference=result_reference,
                    )
                ),
                operation_name="Resolved completion result validation",
                redactor=self._secret_redactor,
            )
            result = None
            if application_validation.failure is not None:
                raise_task_store_operation_failure(application_validation.failure)
            application_request = application_validation.result
            application_validation = None
            if application_request is None:
                raise self._safe_execution_error(
                    "Completion result resolver returned invalid result content."
                ) from None

            await self._require_publication_heartbeat_healthy(publication_heartbeat)
            await self._renew_resolved_event_publication(
                publication_authority,
                publication_owner,
                lease_deadline=publication_heartbeat.lease_deadline,
                ownership_lost=publication_heartbeat.ownership_lost,
            )
            await self._require_publication_heartbeat_healthy(publication_heartbeat)
            application_request_sha256 = completion_decision_application_request_sha256(
                application_request
            )
            application_started = True
            task = await self._apply_with_publication_ownership(
                application_request,
                publication_heartbeat,
                settlement_evidence=application_settlement,
            )
            application_request = None
            final_receipt = await self._application_coordinator.load_result_resolution_receipt(
                task_id=request.task_id,
                idempotency_key=request.idempotency_key,
            )
            if final_receipt is None or final_receipt.request_sha256 != application_request_sha256:
                raise WorkCompletionConflict(
                    "Completion result application has no exact durable receipt."
                ) from None
            prepared_event = self._prepare_resolved_event(
                request=request,
                receipt=final_receipt,
                session_id=attempt.session_id,
                contract=contract,
                result_kind=proposal.result.kind,
                result_reference_id=proposal.result.reference_id,
                result_digest=proposal.result.digest,
            )
            if await self._event_writer.is_exact_persisted(prepared_event):
                await self._complete_resolved_event_publication(
                    publication_authority,
                    publication_owner,
                    require_present=False,
                )
                publication_reserved = False
                await self._stop_publication_heartbeat(
                    publication_heartbeat,
                    publication_completed=True,
                )
                publication_heartbeat = None
                await self._event_writer.fan_out_persisted([prepared_event])
                return task
            await self._publish_resolved_event(
                prepared_event=prepared_event,
                publication_authority=publication_authority,
                publication_owner=publication_owner,
            )
            publication_reserved = False
            await self._stop_publication_heartbeat(
                publication_heartbeat,
                publication_completed=True,
            )
            publication_heartbeat = None
            return task
        except BaseException as caught:
            failure = caught

        assert failure is not None
        heartbeat_failure = await self._stop_publication_heartbeat(
            publication_heartbeat,
            publication_completed=False,
        )
        publication_heartbeat = None
        cleanup_authoritative: BaseException | None = None
        cleanup_cause: BaseException | None = None
        if (
            isinstance(failure, Exception)
            and publication_reserved
            and (
                not application_started
                or isinstance(failure, _CompletionDecisionApplicationNotCommitted)
                or application_settlement.not_committed
            )
            and request.decision_id not in self._draining_adapter_tasks
            and publication_authority is not None
            and publication_owner is not None
        ):
            release_task = asyncio.create_task(
                capture_awaitable_outcome(
                    lambda authority=publication_authority, owner=publication_owner: (
                        self._release_publication_owner_with_heartbeat(
                            authority,
                            owner,
                        )
                    )
                ),
                name="cayu-completion-result-publication-release",
            )
            release_outcome = await await_shielded_task_outcome(
                release_task,
                timeout_s=_PUBLICATION_RELEASE_SETTLEMENT_SECONDS,
                timeout_after_cancellation_s=0,
            )
            if not release_task.done():
                self._retain_publication_settlement(
                    request.decision_id,
                    release_task,
                    capacity_lease=capacity_lease,
                )
            release_error = release_outcome.error
            captured_release = release_outcome.result
            if release_error is None and captured_release is not None:
                release_error = captured_release.error
            if (
                release_outcome.timed_out
                and release_outcome.cancellation is None
                and release_error is None
            ):
                release_error = TimeoutError(
                    "Completion-result publication cleanup remains in progress."
                )
            if (
                release_error is None
                and captured_release is None
                and release_outcome.cancellation is None
            ):
                release_error = RuntimeError(
                    "Completion-result event publication reservation cleanup returned no outcome."
                )
            safe_cleanup_cancellation = None
            if release_outcome.cancellation is not None:
                cancellation = release_outcome.cancellation
                safe_cleanup_cancellation = self._safe_caller_cancellation(cancellation)
                consumed = release_outcome.cancellation_requests_consumed
                restore_task_cancellation_requests(
                    consumed,
                    cancellation=safe_cleanup_cancellation,
                )
                retain_workspace_observation_pending_cancellation_requests(
                    safe_cleanup_cancellation,
                    max(consumed, 1),
                )
                del cancellation
            if release_error is not None and exception_tree_contains(
                release_error,
                _PROCESS_CONTROL_SIGNALS,
            ):
                cleanup_authoritative = self._detached_process_control_failure(release_error)
                primary_evidence = self._detached_cleanup_failure(failure)
                cleanup_cause = (
                    primary_evidence
                    if safe_cleanup_cancellation is None
                    else self._safe_cleanup_failure_group(
                        "Completion-result resolution failed while cleanup was cancelled.",
                        primary_evidence,
                        safe_cleanup_cancellation,
                    )
                )
            elif safe_cleanup_cancellation is not None:
                cleanup_authoritative = safe_cleanup_cancellation
                primary_evidence = self._detached_cleanup_failure(failure)
                cleanup_cause = (
                    primary_evidence
                    if release_error is None
                    else self._safe_cleanup_failure_group(
                        "Completion-result resolution and reservation cleanup failed.",
                        primary_evidence,
                        self._detached_cleanup_failure(release_error),
                    )
                )
            elif release_error is not None:
                cleanup_cause = self._detached_cleanup_failure(release_error)
            del release_task, release_outcome, release_error, captured_release

        if heartbeat_failure is not None:
            if exception_tree_contains(heartbeat_failure, _PROCESS_CONTROL_SIGNALS):
                safe_heartbeat_failure = self._detached_process_control_failure(heartbeat_failure)
                prior_authoritative = cleanup_authoritative
                cleanup_authoritative = safe_heartbeat_failure
                heartbeat_evidence = self._detached_cleanup_failure(failure)
                related_failures = [heartbeat_evidence]
                if prior_authoritative is not None:
                    related_failures.append(prior_authoritative)
                if cleanup_cause is not None:
                    related_failures.append(cleanup_cause)
                cleanup_cause = (
                    related_failures[0]
                    if len(related_failures) == 1
                    else self._safe_cleanup_failure_group(
                        "Completion-result resolution reported multiple cleanup signals.",
                        *related_failures,
                    )
                )
            else:
                safe_heartbeat_failure = self._detached_cleanup_failure(heartbeat_failure)
                cleanup_cause = (
                    safe_heartbeat_failure
                    if cleanup_cause is None
                    else self._safe_cleanup_failure_group(
                        "Completion-result ownership and reservation cleanup failed.",
                        safe_heartbeat_failure,
                        cleanup_cause,
                    )
                )

        del request, receipt, decision, claim, proposal, attempt, contract
        del verifier_profile, authority_task
        del resolver, adapter_request_validation, adapter_request, result
        del application_validation, application_request, task, final_receipt, prepared_event
        del application_settlement
        del publication_authority, publication_owner, heartbeat_failure
        if cleanup_authoritative is not None:
            raise cleanup_authoritative from cleanup_cause
        if cleanup_cause is not None:
            raise failure from cleanup_cause
        raise failure

    async def _invoke_resolver(
        self,
        resolver: CompletionResultResolver,
        request: CompletionResultResolverRequest,
        *,
        decision_id: str,
        timeout_seconds: float,
        publication_authority: _ResolvedEventPublicationAuthority,
        publication_owner: _PublicationOwner,
        publication_heartbeat: _PublicationHeartbeat,
        capacity_lease: _ResolutionCapacityLease,
    ) -> dict[str, object]:
        task = asyncio.create_task(
            capture_awaitable_outcome(
                lambda adapter=resolver, value=request: adapter.resolve(value)
            ),
            name="cayu-completion-result-resolver",
        )
        del resolver, request
        self._adapter_tasks.add(task)
        task.add_done_callback(self._adapter_task_settled)
        shielded = await await_shielded_task_outcome(
            task,
            timeout_s=timeout_seconds,
            timeout_after_cancellation_s=0,
        )
        if shielded.cancellation is not None:
            cancellation = shielded.cancellation
            safe_cancellation = self._safe_caller_cancellation(cancellation)
            consumed = shielded.cancellation_requests_consumed
            captured = shielded.result
            fatal = None if captured is None else captured.error
            if fatal is not None and exception_tree_contains(fatal, _PROCESS_CONTROL_SIGNALS):
                safe_fatal = self._detached_process_control_failure(fatal)
                restore_task_cancellation_requests(consumed, cancellation=safe_cancellation)
                self._adapter_tasks.discard(task)
                del cancellation, fatal, captured, shielded, task
                raise safe_fatal from safe_cancellation
            self._retain_draining(
                decision_id,
                task,
                publication_authority=publication_authority,
                publication_owner=publication_owner,
                publication_heartbeat=publication_heartbeat,
                capacity_lease=capacity_lease,
            )
            publication_heartbeat.transferred = True
            restore_task_cancellation_requests(consumed, cancellation=safe_cancellation)
            retain_workspace_observation_pending_cancellation_requests(
                safe_cancellation,
                max(consumed, 1),
            )
            del cancellation, fatal, captured, shielded, task
            raise safe_cancellation from None
        if shielded.timed_out:
            self._retain_draining(
                decision_id,
                task,
                publication_authority=publication_authority,
                publication_owner=publication_owner,
                publication_heartbeat=publication_heartbeat,
                capacity_lease=capacity_lease,
            )
            publication_heartbeat.transferred = True
            raise self._safe_execution_error(
                "Completion result resolver exceeded its bounded execution timeout."
            ) from None

        self._adapter_tasks.discard(task)
        captured = shielded.result
        task_failure = shielded.error
        del shielded, task
        if task_failure is not None:
            if exception_tree_contains(task_failure, _PROCESS_CONTROL_SIGNALS):
                safe_failure = self._detached_process_control_failure(task_failure)
                del task_failure, captured
                raise safe_failure
            del task_failure, captured
            raise self._safe_execution_error(
                "Completion result resolver task failed before returning an outcome."
            ) from None
        if captured is None:
            raise self._safe_execution_error(
                "Completion result resolver returned no captured outcome."
            ) from None
        if captured.error is not None:
            failure = captured.error
            if exception_tree_contains(failure, _PROCESS_CONTROL_SIGNALS):
                safe_failure = self._detached_process_control_failure(failure)
                del captured, failure
                raise safe_failure
            if isinstance(failure, CompletionResultUnavailable):
                del captured, failure
                raise credential_safe_runtime_exception(
                    CompletionResultUnavailable,
                    "The accepted completion result is unavailable from its exact resolver.",
                    redactor=self._secret_redactor,
                    fallback_message="The accepted completion result is unavailable.",
                ) from None
            if isinstance(failure, asyncio.CancelledError):
                del captured, failure
                raise self._safe_execution_error(
                    "Completion result resolver was cancelled without caller cancellation."
                ) from None
            del captured, failure
            raise credential_safe_runtime_exception(
                CompletionResultResolverExecutionError,
                "Completion result resolver execution failed.",
                redactor=self._secret_redactor,
                fallback_message="Completion result resolver execution failed.",
            ) from None
        if type(captured.result) is not dict:
            del captured
            raise self._safe_execution_error(
                "Completion result resolver must return a JSON object."
            ) from None
        return captured.result

    async def _apply_with_publication_ownership(
        self,
        request: CompletionDecisionApplicationRequest,
        heartbeat: _PublicationHeartbeat,
        *,
        settlement_evidence: _ApplicationSettlementEvidence,
    ) -> Task:
        application_task = asyncio.create_task(
            self._application_coordinator.apply(request),
            name="cayu-completion-result-application",
        )
        del request
        try:
            completed, _pending = await asyncio.wait(
                (
                    application_task,
                    heartbeat.ownership_lost,
                    heartbeat.task,
                ),
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError as cancellation:
            application_task.cancel("Completion result application caller was cancelled.")
            settlement = await await_shielded_task_outcome(
                application_task,
                cancellation=cancellation,
                timeout_s=None,
                timeout_after_cancellation_s=None,
            )
            safe_cancellation = self._safe_caller_cancellation(cancellation)
            if settlement.error is not None and not isinstance(
                settlement.error,
                asyncio.CancelledError,
            ):
                if exception_tree_contains(settlement.error, _PROCESS_CONTROL_SIGNALS):
                    raise self._detached_process_control_failure(
                        settlement.error
                    ) from safe_cancellation
                safe_cancellation.__cause__ = self._detached_cleanup_failure(settlement.error)
                safe_cancellation.__suppress_context__ = True
            retain_workspace_observation_pending_cancellation_requests(
                safe_cancellation,
                max(settlement.cancellation_requests_consumed, 1),
            )
            restore_task_cancellation_requests(
                settlement.cancellation_requests_consumed,
                cancellation=safe_cancellation,
            )
            raise safe_cancellation from safe_cancellation.__cause__

        if heartbeat.ownership_lost in completed or heartbeat.task in completed:
            if heartbeat.ownership_lost.done():
                raw_ownership_failure = heartbeat.ownership_lost.result()
                heartbeat.observed_failure_id = id(raw_ownership_failure)
                ownership_failure = self._detached_cleanup_failure(raw_ownership_failure)
            else:
                try:
                    heartbeat.task.result()
                except BaseException as failure:
                    heartbeat.observed_failure_id = id(failure)
                    ownership_failure = self._detached_cleanup_failure(failure)
                else:
                    ownership_failure = self._safe_execution_error(
                        "Completion-result publication ownership ended during application."
                    )
            application_task.cancel("Completion result application lost publication ownership.")
            settlement = await await_shielded_task_outcome(
                application_task,
                timeout_s=None,
                timeout_after_cancellation_s=None,
            )
            if settlement.error is not None and exception_tree_contains(
                settlement.error,
                (_CompletionDecisionApplicationNotCommitted,),
            ):
                settlement_evidence.not_committed = True
            if settlement.cancellation is not None:
                safe_cancellation = self._safe_caller_cancellation(settlement.cancellation)
                safe_cancellation.__cause__ = ownership_failure
                safe_cancellation.__suppress_context__ = True
                retain_workspace_observation_pending_cancellation_requests(
                    safe_cancellation,
                    max(settlement.cancellation_requests_consumed, 1),
                )
                restore_task_cancellation_requests(
                    settlement.cancellation_requests_consumed,
                    cancellation=safe_cancellation,
                )
                raise safe_cancellation from ownership_failure
            if settlement.error is not None and not isinstance(
                settlement.error,
                asyncio.CancelledError,
            ):
                if exception_tree_contains(settlement.error, _PROCESS_CONTROL_SIGNALS):
                    raise self._detached_process_control_failure(
                        settlement.error
                    ) from ownership_failure
                raise ownership_failure from self._detached_cleanup_failure(settlement.error)
            raise ownership_failure from None

        return application_task.result()

    async def _require_resolver_not_draining(self, decision_id: str) -> None:
        settlement = self._draining_adapter_tasks.get(decision_id)
        if settlement is not None and not settlement.done():
            raise self._safe_execution_error(
                "The prior exact completion result resolver execution is still draining."
            ) from None
        if settlement is None:
            return
        self._draining_adapter_tasks.pop(decision_id, None)
        captured = settlement.result()
        if captured.error is not None:
            failure = captured.error
            if exception_tree_contains(failure, _PROCESS_CONTROL_SIGNALS):
                safe_failure = self._detached_process_control_failure(failure)
            else:
                safe_failure = self._detached_cleanup_failure(failure)
            del captured, settlement, failure
            raise safe_failure from None

    def _safe_execution_error(self, message: str) -> CompletionResultResolverExecutionError:
        return credential_safe_runtime_exception(
            CompletionResultResolverExecutionError,
            message,
            redactor=self._secret_redactor,
            fallback_message="Completion result resolver execution failed.",
        )

    def _safe_dependency_cancellation_failure(
        self,
        cancellation: asyncio.CancelledError,
        message: str,
    ) -> CompletionResultResolverExecutionError:
        failure = self._safe_execution_error(message)
        carried_cause = cancellation.__cause__
        if carried_cause is not None:
            failure.__cause__ = self._detached_cleanup_failure(carried_cause)
            failure.__suppress_context__ = True
        return failure

    def _safe_caller_cancellation(
        self,
        cancellation: asyncio.CancelledError,
    ) -> asyncio.CancelledError:
        diagnostic = exception_diagnostic(
            cancellation,
            empty_message="completion result resolution cancelled",
            nonportable_message=(
                "Completion result resolution cancellation had a non-portable diagnostic."
            ),
            redactor=self._secret_redactor,
        )
        return credential_safe_runtime_exception(
            asyncio.CancelledError,
            diagnostic.message,
            redactor=self._secret_redactor,
            fallback_message="Completion result resolution cancellation was redacted.",
        )

    def _redeliver_safe_caller_cancellation(
        self,
        cancellation: asyncio.CancelledError,
        *,
        preserve_requests: int,
    ) -> asyncio.CancelledError:
        safe = self._safe_caller_cancellation(cancellation)
        carried_cause = cancellation.__cause__
        if carried_cause is not None:
            safe.__cause__ = self._detached_cleanup_failure(carried_cause)
            safe.__suppress_context__ = True
        current_task = asyncio.current_task()
        current_requests = 0 if current_task is None else current_task.cancelling()
        consumed = max(current_requests - preserve_requests, 0)
        consume_pending_task_cancellation(
            cancellation,
            preserve_requests=preserve_requests,
        )
        restore_task_cancellation_requests(consumed, cancellation=safe)
        if consumed:
            retain_workspace_observation_pending_cancellation_requests(safe, consumed)
        return safe

    def _detached_process_control_failure(self, error: BaseException) -> BaseException:
        if isinstance(error, BaseExceptionGroup):
            return credential_safe_runtime_exception_group(
                error,
                group_message="Completion result resolver reported multiple failures.",
                leaf_mapper=self._detached_process_control_failure,
                invalid_leaf_factory=lambda: self._safe_execution_error(
                    "Completion result resolver reported invalid failure evidence."
                ),
                truncated_leaf_factory=lambda: self._safe_execution_error(
                    "Additional completion result resolver failures were omitted."
                ),
                fallback_leaf_mapper=lambda _leaf: self._safe_execution_error(
                    "Completion result resolver execution failed."
                ),
                redactor=self._secret_redactor,
            )
        for signal_type in _PROCESS_CONTROL_SIGNALS:
            if isinstance(error, signal_type):
                return credential_safe_runtime_exception(
                    signal_type,
                    "Completion result resolver raised a process-control signal.",
                    redactor=self._secret_redactor,
                    fallback_message=(
                        "Completion result resolver process-control diagnostic was redacted."
                    ),
                )
        return self._safe_execution_error("Completion result resolver execution failed.")

    def _detached_cleanup_failure(self, error: BaseException) -> BaseException:
        """Detach one cleanup-related signal from extension traceback and payload state."""

        if isinstance(error, BaseExceptionGroup):
            return credential_safe_runtime_exception_group(
                error,
                group_message="Completion result resolution reported multiple failures.",
                leaf_mapper=self._detached_cleanup_failure,
                invalid_leaf_factory=lambda: self._safe_execution_error(
                    "Completion result resolution reported invalid failure evidence."
                ),
                truncated_leaf_factory=lambda: self._safe_execution_error(
                    "Additional completion result resolution failures were omitted."
                ),
                fallback_leaf_mapper=lambda _leaf: self._safe_execution_error(
                    "Completion result resolution failed."
                ),
                redactor=self._secret_redactor,
            )
        if isinstance(error, _PROCESS_CONTROL_SIGNALS):
            return self._detached_process_control_failure(error)
        if isinstance(error, asyncio.CancelledError):
            return self._safe_caller_cancellation(error)
        diagnostic = exception_diagnostic(
            error,
            empty_message="completion result resolution failed",
            nonportable_message=(
                "Completion result resolution failed with a non-portable diagnostic."
            ),
            redactor=self._secret_redactor,
        )
        safe_type: type[BaseException]
        if type(error) in {
            CompletionResultResolverExecutionError,
            CompletionResultResolverUnavailable,
            CompletionResultUnavailable,
            WorkCompletionConflict,
            ConnectionError,
            TimeoutError,
            KeyError,
            ValueError,
            TypeError,
            NotImplementedError,
            RuntimeError,
        }:
            safe_type = type(error)
        else:
            safe_type = CompletionResultResolverExecutionError
        return credential_safe_runtime_exception(
            safe_type,
            diagnostic.message,
            redactor=self._secret_redactor,
            fallback_message="Completion result resolution failure diagnostic was redacted.",
        )

    def _safe_cleanup_failure_group(
        self,
        message: str,
        *failures: BaseException,
    ) -> BaseExceptionGroup:
        ordered: list[BaseException] = []
        for failure in failures:
            if all(failure is not existing for existing in ordered):
                ordered.append(failure)
        group: BaseExceptionGroup = (
            ExceptionGroup(message, cast("list[Exception]", ordered))
            if all(isinstance(failure, Exception) for failure in ordered)
            else BaseExceptionGroup(message, ordered)
        )
        return credential_safe_runtime_exception_group(
            group,
            group_message=message,
            leaf_mapper=lambda leaf: leaf,
            invalid_leaf_factory=lambda: self._safe_execution_error(
                "Completion result cleanup reported invalid failure evidence."
            ),
            truncated_leaf_factory=lambda: self._safe_execution_error(
                "Additional completion result cleanup failures were omitted."
            ),
            fallback_leaf_mapper=lambda leaf: self._detached_cleanup_failure(leaf),
            redactor=self._secret_redactor,
        )

    def _retain_draining(
        self,
        decision_id: str,
        task: asyncio.Task[CapturedAwaitableOutcome[dict[str, object]]],
        *,
        publication_authority: _ResolvedEventPublicationAuthority,
        publication_owner: _PublicationOwner,
        publication_heartbeat: _PublicationHeartbeat,
        capacity_lease: _ResolutionCapacityLease,
    ) -> None:
        settlement = asyncio.create_task(
            capture_awaitable_outcome(
                lambda: self._settle_draining_resolver(
                    task,
                    publication_authority=publication_authority,
                    publication_owner=publication_owner,
                    publication_heartbeat=publication_heartbeat,
                )
            ),
            name="cayu-completion-result-resolver-settlement",
        )
        self._draining_adapter_tasks[decision_id] = settlement
        capacity_lease.transferred = True
        settlement.add_done_callback(
            lambda completed, key=decision_id, lease=capacity_lease: (
                self._publication_settlement_completed(
                    key,
                    completed,
                    capacity_lease=lease,
                )
            )
        )

    def _retain_publication_settlement(
        self,
        decision_id: str,
        settlement: asyncio.Task[CapturedAwaitableOutcome[None]],
        *,
        capacity_lease: _ResolutionCapacityLease,
    ) -> None:
        self._draining_adapter_tasks[decision_id] = settlement
        capacity_lease.transferred = True
        settlement.add_done_callback(
            lambda completed, key=decision_id, lease=capacity_lease: (
                self._publication_settlement_completed(
                    key,
                    completed,
                    capacity_lease=lease,
                )
            )
        )

    async def _settle_draining_resolver(
        self,
        task: asyncio.Task[CapturedAwaitableOutcome[dict[str, object]]],
        *,
        publication_authority: _ResolvedEventPublicationAuthority,
        publication_owner: _PublicationOwner,
        publication_heartbeat: _PublicationHeartbeat,
    ) -> None:
        resolver_failure: BaseException | None = None
        release_failure: BaseException | None = None
        try:
            captured = await task
            if captured.error is not None and exception_tree_contains(
                captured.error,
                _PROCESS_CONTROL_SIGNALS,
            ):
                resolver_failure = self._detached_process_control_failure(captured.error)
            try:
                await self._release_resolved_event_publication(
                    publication_authority,
                    publication_owner,
                    require_present=False,
                )
            except BaseException as error:
                release_failure = self._detached_cleanup_failure(error)
        finally:
            publication_heartbeat.stop.set()
            heartbeat_failure = await self._stop_publication_heartbeat(
                publication_heartbeat,
                publication_completed=False,
                transferred_owner=True,
            )
            if heartbeat_failure is not None and release_failure is None:
                release_failure = self._detached_cleanup_failure(heartbeat_failure)
        if resolver_failure is not None:
            if release_failure is not None:
                raise resolver_failure from release_failure
            raise resolver_failure
        if release_failure is not None:
            raise release_failure

    async def _heartbeat_publication_owner(
        self,
        authority: _ResolvedEventPublicationAuthority,
        owner_id: str,
        stop: asyncio.Event,
        lease_deadline: _PublicationLeaseDeadline,
        ownership_lost: asyncio.Future[BaseException],
    ) -> None:
        while True:
            remaining = lease_deadline.monotonic - monotonic()
            if remaining <= 0:
                failure = self._safe_execution_error(
                    "Completion-result publication ownership acknowledgement expired."
                )
                self._record_publication_ownership_loss(ownership_lost, failure)
                raise failure from None
            try:
                await asyncio.wait_for(
                    stop.wait(),
                    timeout=min(
                        _PUBLICATION_OWNER_HEARTBEAT_SECONDS,
                        remaining / 2.0,
                    ),
                )
            except TimeoutError:
                pass
            else:
                return
            renewal_started_monotonic = monotonic()
            renewal_task = asyncio.create_task(
                self._session_store._publish_completion_result_event_publication(
                    authority.session_id,
                    checkpoint_transform=lambda _session, checkpoint, store_now: (
                        _renew_completion_result_event_publication(
                            checkpoint,
                            publication_id=authority.publication_id,
                            authority_sha256=authority.authority_sha256,
                            owner_id=owner_id,
                            owner_expires_at=store_now
                            + timedelta(seconds=_PUBLICATION_OWNER_LEASE_SECONDS),
                            now=store_now,
                        )
                    ),
                    events=[],
                ),
                name="cayu-completion-result-publication-renewal",
            )
            while True:
                renewal_outcome = await await_shielded_task_outcome(
                    renewal_task,
                    timeout_s=max(0.0, lease_deadline.monotonic - monotonic()),
                )
                if not renewal_outcome.timed_out or monotonic() >= lease_deadline.monotonic:
                    break
            if renewal_outcome.cancellation is not None:
                settlement = await await_shielded_task_outcome(
                    renewal_task,
                    cancellation=renewal_outcome.cancellation,
                    timeout_s=None,
                    timeout_after_cancellation_s=None,
                )
                if settlement.error is not None:
                    renewal_outcome.cancellation.__cause__ = self._detached_cleanup_failure(
                        settlement.error
                    )
                    renewal_outcome.cancellation.__suppress_context__ = True
                raise renewal_outcome.cancellation
            if renewal_outcome.timed_out:
                failure = self._safe_execution_error(
                    "Completion-result publication ownership renewal was not acknowledged "
                    "before its local lease deadline."
                )
                self._record_publication_ownership_loss(ownership_lost, failure)
                settlement = await await_shielded_task_outcome(
                    renewal_task,
                    timeout_s=None,
                    timeout_after_cancellation_s=None,
                )
                if settlement.cancellation is not None:
                    safe_cancellation = self._safe_caller_cancellation(settlement.cancellation)
                    safe_cancellation.__cause__ = failure
                    safe_cancellation.__suppress_context__ = True
                    retain_workspace_observation_pending_cancellation_requests(
                        safe_cancellation,
                        max(settlement.cancellation_requests_consumed, 1),
                    )
                    restore_task_cancellation_requests(
                        settlement.cancellation_requests_consumed,
                        cancellation=safe_cancellation,
                    )
                    raise safe_cancellation
                if settlement.error is not None:
                    if exception_tree_contains(
                        settlement.error,
                        _PROCESS_CONTROL_SIGNALS,
                    ):
                        raise self._detached_process_control_failure(settlement.error) from failure
                    raise failure from self._detached_cleanup_failure(settlement.error)
                raise failure from None
            if renewal_outcome.error is not None:
                self._record_publication_ownership_loss(
                    ownership_lost,
                    renewal_outcome.error,
                )
                raise renewal_outcome.error
            self._acknowledge_publication_renewal_deadline(
                renewal_started_monotonic,
                lease_deadline=lease_deadline,
                ownership_lost=ownership_lost,
            )

    def _acknowledge_publication_renewal_deadline(
        self,
        renewal_started_monotonic: float,
        *,
        lease_deadline: _PublicationLeaseDeadline | None = None,
        ownership_lost: asyncio.Future[BaseException] | None = None,
    ) -> float:
        acknowledged_deadline = renewal_started_monotonic + _PUBLICATION_OWNER_LEASE_SECONDS
        if lease_deadline is not None:
            lease_deadline.monotonic = max(
                lease_deadline.monotonic,
                acknowledged_deadline,
            )
            acknowledged_deadline = lease_deadline.monotonic
        if monotonic() >= acknowledged_deadline:
            failure = self._safe_execution_error(
                "Completion-result publication ownership renewal acknowledgement "
                "consumed its lease."
            )
            if ownership_lost is not None:
                self._record_publication_ownership_loss(ownership_lost, failure)
            raise failure from None
        return acknowledged_deadline

    @staticmethod
    def _record_publication_ownership_loss(
        ownership_lost: asyncio.Future[BaseException],
        failure: BaseException,
    ) -> None:
        if not ownership_lost.done():
            ownership_lost.set_result(failure)

    def _start_publication_heartbeat(
        self,
        authority: _ResolvedEventPublicationAuthority,
        owner: _PublicationOwner,
        *,
        claim_deadline_monotonic: float,
    ) -> _PublicationHeartbeat:
        stop = asyncio.Event()
        lease_deadline = _PublicationLeaseDeadline(claim_deadline_monotonic)
        ownership_lost: asyncio.Future[BaseException] = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(
            self._heartbeat_publication_owner(
                authority,
                owner.owner_id,
                stop,
                lease_deadline,
                ownership_lost,
            ),
            name="cayu-completion-result-publication-heartbeat",
        )
        return _PublicationHeartbeat(
            stop=stop,
            task=task,
            lease_deadline=lease_deadline,
            ownership_lost=ownership_lost,
        )

    async def _require_publication_heartbeat_healthy(
        self,
        heartbeat: _PublicationHeartbeat,
    ) -> None:
        if heartbeat.ownership_lost.done():
            failure = heartbeat.ownership_lost.result()
            heartbeat.observed_failure_id = id(failure)
            raise self._detached_cleanup_failure(failure) from None
        if not heartbeat.task.done():
            return
        try:
            await heartbeat.task
        except BaseException as error:
            raise self._detached_cleanup_failure(error) from None
        raise self._safe_execution_error(
            "Completion-result publication ownership ended before application."
        ) from None

    async def _stop_publication_heartbeat(
        self,
        heartbeat: _PublicationHeartbeat | None,
        *,
        publication_completed: bool,
        transferred_owner: bool = False,
    ) -> BaseException | None:
        if heartbeat is None or (heartbeat.transferred and not transferred_owner):
            return None
        heartbeat.stop.set()
        try:
            await heartbeat.task
        except BaseException as error:
            if publication_completed:
                return None
            if id(error) == heartbeat.observed_failure_id:
                return error.__cause__
            return error
        return None

    async def _release_publication_owner_with_heartbeat(
        self,
        authority: _ResolvedEventPublicationAuthority,
        owner: _PublicationOwner,
    ) -> None:
        claim_deadline_monotonic = await self._renew_resolved_event_publication(
            authority,
            owner,
        )
        heartbeat = self._start_publication_heartbeat(
            authority,
            owner,
            claim_deadline_monotonic=claim_deadline_monotonic,
        )
        release_failure: BaseException | None = None
        try:
            await self._release_resolved_event_publication(
                authority,
                owner,
                require_present=False,
            )
        except BaseException as error:
            release_failure = error
        finally:
            heartbeat.stop.set()
            heartbeat_failure = await self._stop_publication_heartbeat(
                heartbeat,
                publication_completed=False,
            )
        if release_failure is not None:
            if heartbeat_failure is not None:
                raise release_failure from self._detached_cleanup_failure(heartbeat_failure)
            raise release_failure
        if heartbeat_failure is not None:
            raise heartbeat_failure

    def _publication_settlement_completed(
        self,
        decision_id: str,
        completed: asyncio.Task[CapturedAwaitableOutcome[None]],
        *,
        capacity_lease: _ResolutionCapacityLease,
    ) -> None:
        self._release_resolution_capacity(capacity_lease)
        if self._draining_adapter_tasks.get(decision_id) is not completed:
            return
        try:
            captured = completed.result()
        except BaseException:
            return
        if captured.error is None:
            self._draining_adapter_tasks.pop(decision_id, None)

    def _reserve_resolution_capacity(self, lease: _ResolutionCapacityLease) -> None:
        if lease.reservation is not None:
            return
        if len(self._resolution_capacity_reservations) >= _MAX_ACTIVE_RESULT_RESOLVERS:
            raise self._safe_execution_error(
                "Completion result resolver execution capacity is exhausted."
            ) from None
        reservation = object()
        self._resolution_capacity_reservations.add(reservation)
        lease.reservation = reservation

    def _release_resolution_capacity(self, lease: _ResolutionCapacityLease) -> None:
        reservation = lease.reservation
        if reservation is not None:
            self._resolution_capacity_reservations.discard(reservation)

    def _resolved_event_publication_authority(
        self,
        *,
        request: CompletionResultResolutionRequest,
        decision: CompletionDecision,
        proposal: CompletionProposal,
        attempt: WorkAttempt,
        contract: WorkContract,
        authority_task: Task,
    ) -> _ResolvedEventPublicationAuthority:
        if authority_task.session_id != attempt.session_id:
            raise WorkCompletionConflict(
                "Completion result publication conflicts with its task session authority."
            ) from None
        if authority_task.session_instance_id is None:
            raise WorkCompletionConflict(
                "Completion result publication has no exact session-instance authority."
            ) from None
        task_root_session_id = authority_task.invocation.root_session_id
        if task_root_session_id not in {None, attempt.session_id}:
            raise WorkCompletionConflict(
                "Completion result publication conflicts with task invocation authority."
            ) from None
        encoded = canonical_durable_json_bytes(
            {
                "schema_version": 1,
                "task_id": request.task_id,
                "decision_id": request.decision_id,
                "idempotency_key": request.idempotency_key,
                "decision_request_sha256": decision.request_sha256,
                "proposal_id": proposal.proposal_id,
                "attempt_id": attempt.attempt_id,
                "session_id": attempt.session_id,
                "source_session_instance_id": authority_task.session_instance_id,
                "contract_id": contract.contract_id,
                "contract_version": contract.version,
                "contract_fingerprint": contract.fingerprint,
                "resolver_id": contract.result_resolver.resolver_id,
                "resolver_version": contract.result_resolver.version,
                "resolver_configuration_fingerprint": (
                    contract.result_resolver.configuration_fingerprint
                ),
                "result_kind": proposal.result.kind,
                "result_reference_id": proposal.result.reference_id,
                "result_digest": proposal.result.digest,
            },
            "completion_result_event_publication_authority",
        )
        authority_sha256 = sha256(encoded).hexdigest()
        return _ResolvedEventPublicationAuthority(
            session_id=attempt.session_id,
            publication_id=f"completion-result-publication:v1:{authority_sha256}",
            authority_sha256=authority_sha256,
            source_session_instance_id=authority_task.session_instance_id,
        )

    @staticmethod
    def _new_publication_owner() -> _PublicationOwner:
        owner_token = f"{uuid4().hex}{uuid4().hex}"
        return _PublicationOwner(
            owner_id=f"{_PUBLICATION_OWNER_ID_PREFIX}{owner_token}",
        )

    async def _reserve_resolved_event_publication(
        self,
        authority: _ResolvedEventPublicationAuthority,
        owner: _PublicationOwner,
    ) -> float:
        capability = getattr(
            self._session_store,
            "_supports_completion_result_event_publication_reservation_protocol",
            None,
        )
        if not callable(capability) or capability() is not True:
            raise NotImplementedError(
                "Completion result resolution requires a SessionStore that owns "
                "publication reservation, checkpoint replacement, and deletion fencing."
            ) from None
        reservation_started_monotonic = monotonic()
        await self._session_store._publish_completion_result_event_publication(
            authority.session_id,
            checkpoint_transform=lambda session, checkpoint, store_now: (
                self._reserve_resolved_event_publication_checkpoint(
                    session,
                    checkpoint,
                    authority=authority,
                    owner=owner,
                    now=store_now,
                )
            ),
            events=[],
        )
        claim_deadline_monotonic = reservation_started_monotonic + _PUBLICATION_OWNER_LEASE_SECONDS
        return claim_deadline_monotonic

    @staticmethod
    def _reserve_resolved_event_publication_checkpoint(
        session: Session,
        checkpoint: dict[str, Any] | None,
        *,
        authority: _ResolvedEventPublicationAuthority,
        owner: _PublicationOwner,
        now: datetime,
    ) -> dict[str, Any]:
        if session.instance_id != authority.source_session_instance_id:
            raise WorkCompletionConflict(
                "Completion result publication source session instance changed."
            ) from None
        return _reserve_completion_result_event_publication(
            checkpoint,
            publication_id=authority.publication_id,
            authority_sha256=authority.authority_sha256,
            owner_id=owner.owner_id,
            owner_expires_at=now + timedelta(seconds=_PUBLICATION_OWNER_LEASE_SECONDS),
            now=now,
        )

    async def _release_resolved_event_publication(
        self,
        authority: _ResolvedEventPublicationAuthority,
        owner: _PublicationOwner,
        *,
        require_present: bool,
    ) -> None:
        await self._session_store._publish_completion_result_event_publication(
            authority.session_id,
            checkpoint_transform=lambda _session, checkpoint, store_now: (
                _release_completion_result_event_publication(
                    checkpoint,
                    publication_id=authority.publication_id,
                    authority_sha256=authority.authority_sha256,
                    owner_id=owner.owner_id,
                    require_present=require_present,
                    now=store_now,
                )
            ),
            events=[],
        )

    async def _renew_resolved_event_publication(
        self,
        authority: _ResolvedEventPublicationAuthority,
        owner: _PublicationOwner,
        *,
        lease_deadline: _PublicationLeaseDeadline | None = None,
        ownership_lost: asyncio.Future[BaseException] | None = None,
    ) -> float:
        renewal_started_monotonic = monotonic()
        await self._session_store._publish_completion_result_event_publication(
            authority.session_id,
            checkpoint_transform=lambda _session, checkpoint, store_now: (
                _renew_completion_result_event_publication(
                    checkpoint,
                    publication_id=authority.publication_id,
                    authority_sha256=authority.authority_sha256,
                    owner_id=owner.owner_id,
                    owner_expires_at=store_now
                    + timedelta(seconds=_PUBLICATION_OWNER_LEASE_SECONDS),
                    now=store_now,
                )
            ),
            events=[],
        )
        return self._acknowledge_publication_renewal_deadline(
            renewal_started_monotonic,
            lease_deadline=lease_deadline,
            ownership_lost=ownership_lost,
        )

    async def _complete_resolved_event_publication(
        self,
        authority: _ResolvedEventPublicationAuthority,
        owner: _PublicationOwner,
        *,
        require_present: bool,
    ) -> None:
        await self._session_store._publish_completion_result_event_publication(
            authority.session_id,
            checkpoint_transform=lambda _session, checkpoint, store_now: (
                _complete_completion_result_event_publication(
                    checkpoint,
                    publication_id=authority.publication_id,
                    authority_sha256=authority.authority_sha256,
                    owner_id=owner.owner_id,
                    require_present=require_present,
                    now=store_now,
                )
            ),
            events=[],
        )

    def _adapter_task_settled(
        self,
        completed: asyncio.Task[CapturedAwaitableOutcome[dict[str, object]]],
    ) -> None:
        self._adapter_tasks.discard(completed)
        with suppress(BaseException):
            completed.result()

    def _prepare_resolved_event(
        self,
        *,
        request: CompletionResultResolutionRequest,
        receipt: CompletionDecisionApplicationReceipt,
        session_id: str,
        contract: WorkContract,
        result_kind: str,
        result_reference_id: str,
        result_digest: str,
    ) -> Event:
        identity = canonical_durable_json_bytes(
            {
                "schema_version": 1,
                "task_id": request.task_id,
                "decision_id": request.decision_id,
                "idempotency_key": request.idempotency_key,
                "application_request_sha256": receipt.request_sha256,
            },
            "completion_result_resolution_event_identity",
        )
        event = Event(
            id=f"completion-result:v1:{sha256(identity).hexdigest()}",
            type=EventType.TASK_COMPLETION_RESULT_RESOLVED,
            session_id=session_id,
            timestamp=receipt.applied_at,
            payload={
                "task_id": request.task_id,
                "decision_id": request.decision_id,
                "contract_id": contract.contract_id,
                "contract_fingerprint": contract.fingerprint,
                "resolver_id": contract.result_resolver.resolver_id,
                "resolver_version": contract.result_resolver.version,
                "resolver_configuration_fingerprint": (
                    contract.result_resolver.configuration_fingerprint
                ),
                "result_kind": result_kind,
                "result_reference_id": result_reference_id,
                "result_digest": result_digest,
                "application_request_sha256": receipt.request_sha256,
            },
        )
        event = event_with_runtime_generated_id(event)
        event = event_with_runtime_envelope_authority(event, "session_id")
        event = event_with_runtime_payload_authority(
            event,
            "task_id",
            "decision_id",
            "contract_id",
            "contract_fingerprint",
            "resolver_id",
            "resolver_version",
            "resolver_configuration_fingerprint",
            "result_kind",
            "result_reference_id",
            "result_digest",
            "application_request_sha256",
        )
        return self._event_writer.prepare(event)

    async def _publish_resolved_event(
        self,
        *,
        prepared_event: Event,
        publication_authority: _ResolvedEventPublicationAuthority,
        publication_owner: _PublicationOwner,
    ) -> None:
        prepared = prepared_event
        if await self._event_writer.is_exact_persisted(prepared):
            await self._complete_resolved_event_publication(
                publication_authority,
                publication_owner,
                require_present=False,
            )
            await self._event_writer.fan_out_persisted([prepared])
            return
        try:
            await self._session_store._publish_completion_result_event_publication(
                publication_authority.session_id,
                checkpoint_transform=lambda _session, checkpoint, store_now: (
                    _complete_completion_result_event_publication(
                        checkpoint,
                        publication_id=publication_authority.publication_id,
                        authority_sha256=publication_authority.authority_sha256,
                        owner_id=publication_owner.owner_id,
                        require_present=True,
                        now=store_now,
                    )
                ),
                events=[prepared],
            )
        except Exception as publication_error:
            try:
                exact_persisted = await self._event_writer.is_exact_persisted(prepared)
            except asyncio.CancelledError as verification_cancellation:
                verification_cancellation.__cause__ = self._detached_cleanup_failure(
                    publication_error
                )
                verification_cancellation.__suppress_context__ = True
                raise
            except Exception as verification_error:
                publication_error.add_note(
                    "Exact completion-result event verification also failed: "
                    f"{type(verification_error).__name__}."
                )
                raise publication_error from verification_error
            if not exact_persisted:
                raise
        persisted = prepared
        await self._event_writer.fan_out_persisted([persisted])


__all__ = ["CompletionResultResolverCoordinator"]
