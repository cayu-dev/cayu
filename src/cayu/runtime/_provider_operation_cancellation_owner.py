"""Durable provider-operation cancellation composed with process-local ownership."""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, ParamSpec, Protocol, TypeVar

from cayu._exception_groups import add_exception_note_safely, exception_cause, set_exception_cause
from cayu._task_wait import (
    _consume_detached_task_outcome,
    await_shielded_task_outcome,
    restore_task_cancellation_requests,
    retained_task_failure,
    unexpected_child_cancellation_error,
)
from cayu._validation import require_durable_clean_nonblank
from cayu.budgets.base import BudgetReservationRecoveryContext
from cayu.budgets.billing import BillingIdentity
from cayu.events import (
    Event,
    EventType,
    event_with_runtime_generated_id,
    event_with_runtime_payload_authority,
)
from cayu.execution_profiles import (
    event_with_execution_profile_fingerprint_authority,
)
from cayu.execution_units import ModelAttemptIdentity
from cayu.providers.operations import (
    ProviderOperationAdapter,
    ProviderOperationCancellationSupport,
    ProviderOperationMode,
    ProviderOperationSnapshot,
    ProviderOperationState,
    ProviderOperationStatus,
    copy_provider_operation_snapshot,
    copy_provider_operation_state,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.runtime._model_event_authority import _event_with_model_identity_authority
from cayu.runtime._model_execution_selection import ModelExecutionSelection
from cayu.runtime._model_failover_stage import model_failover_target_for_stored_stage
from cayu.runtime._run_limits import RunLimitController
from cayu.runtime._session_control import SessionInterruptedByRequest
from cayu.runtime.provider_operation_cancellation import (
    ProviderOperationCancellationAdmissionsSealed,
    ProviderOperationCancellationCapacityExceeded,
    ProviderOperationCancellationLifecycle,
)
from cayu.runtime.provider_operations import (
    ProviderOperationEvidenceError,
    RecoverableProviderOperation,
    load_provider_operation_cancellation_resolution,
    provider_operation_cancellation_event_id,
)
from cayu.sessions._provider_operation_cancellation_claim import (
    ProviderOperationCancellationClaim,
    checkpoint_with_provider_operation_cancellation_claim,
    checkpoint_without_provider_operation_cancellation_claim,
    provider_operation_cancellation_claim_from_checkpoint,
)
from cayu.sessions.base import ModelCompletionStage, SessionRunFenced, SessionStore
from cayu.sessions.records import Session

_PROVIDER_OPERATION_START_CLEANUP_TIMEOUT_SECONDS = 5.0

_P = ParamSpec("_P")
_R = TypeVar("_R")

_PROVIDER_OPERATION_CANCELLATION_CLAIM_LEASE = timedelta(seconds=30)

_PROVIDER_OPERATION_CANCELLATION_CLAIM_HEARTBEAT_SECONDS = 5.0


class ProviderCancellationRecoveryContext(Protocol):
    """Validated accounting and profile evidence needed by cancellation."""

    @property
    def execution_profile_fingerprint(self) -> str | None: ...

    @property
    def budget_reservations(self) -> tuple[BudgetReservationRecoveryContext, ...]: ...

    @property
    def billing_identity(self) -> BillingIdentity | None: ...


def _attach_provider_operation_cleanup_failure(
    failure: BaseException,
    cleanup_error: BaseException,
) -> None:
    prior_cause = exception_cause(failure)
    combined = BaseExceptionGroup(
        "Provider operation publication and cancellation both failed.",
        [prior_cause, cleanup_error] if prior_cause is not None else [cleanup_error],
    )
    if not set_exception_cause(failure, combined):
        add_exception_note_safely(
            failure,
            "Provider operation cancellation also failed with "
            f"{type(cleanup_error).__name__}; its exception could not be attached.",
        )


async def _cancel_provider_operation_after_definite_absence(
    *,
    lifecycle: ProviderOperationCancellationLifecycle,
    adapter: ProviderOperationAdapter,
    state: ProviderOperationState,
    failure: BaseException,
    cancellation: asyncio.CancelledError | None = None,
    ownership_lost: asyncio.Event | None = None,
    on_unobserved_failure: Callable[[BaseException], None] | None = None,
) -> tuple[asyncio.CancelledError | None, ProviderOperationSnapshot | None, int]:
    """Cancel a provider operation whose start evidence is definitely absent.

    ``on_unobserved_failure`` receives how a cancellation that outlived both of
    its bounded waits ends, if it fails; nobody else observes that outcome.
    """

    async def cancel():
        return await adapter.cancel(copy_provider_operation_state(state))

    try:
        cleanup_task = lifecycle.admit(
            adapter=adapter,
            state=state,
            cancellation=cancel,
            ownership_lost=ownership_lost,
        )
    except (
        ProviderOperationCancellationAdmissionsSealed,
        ProviderOperationCancellationCapacityExceeded,
    ) as cleanup_error:
        _attach_provider_operation_cleanup_failure(failure, cleanup_error)
        failure.add_note(
            "Provider operation cancellation could not obtain a Runtime lifecycle owner: "
            f"{type(cleanup_error).__name__}."
        )
        return cancellation, None, 0
    outcome = await await_shielded_task_outcome(
        cleanup_task,
        cancellation=cancellation,
        timeout_s=_PROVIDER_OPERATION_START_CLEANUP_TIMEOUT_SECONDS,
    )
    cancellation_requests_consumed = outcome.cancellation_requests_consumed
    if outcome.timed_out:
        cleanup_task.cancel()
        drain_outcome = await await_shielded_task_outcome(
            cleanup_task,
            cancellation=outcome.cancellation,
            timeout_s=_PROVIDER_OPERATION_START_CLEANUP_TIMEOUT_SECONDS,
        )
        cancellation_requests_consumed += drain_outcome.cancellation_requests_consumed
        if drain_outcome.timed_out:
            if on_unobserved_failure is None:
                cleanup_task.add_done_callback(_consume_detached_task_outcome)
            else:
                report = on_unobserved_failure

                def settled(completed: asyncio.Task[Any]) -> None:
                    late_failure = retained_task_failure(completed)
                    if late_failure is not None:
                        report(late_failure)

                cleanup_task.add_done_callback(settled)
            failure.add_note(
                "Provider operation cancellation remained in flight after local task "
                "cancellation; the Runtime retained lifecycle ownership with uncertain "
                "cleanup evidence."
            )
            return drain_outcome.cancellation, None, cancellation_requests_consumed
        cleanup_error = drain_outcome.error
        if isinstance(cleanup_error, asyncio.CancelledError):
            cleanup_error = None
        failure.add_note(
            "Provider operation cancellation exceeded its bounded timeout; "
            "the local cancellation task was drained before ownership was released."
        )
        if cleanup_error is not None:
            _attach_provider_operation_cleanup_failure(failure, cleanup_error)
        return drain_outcome.cancellation, None, cancellation_requests_consumed
    cleanup_error = outcome.error
    if isinstance(cleanup_error, asyncio.CancelledError) and outcome.cancellation is None:
        cleanup_error = unexpected_child_cancellation_error(
            cleanup_error,
            operation="Provider operation cancellation",
        )
    if cleanup_error is not None:
        _attach_provider_operation_cleanup_failure(failure, cleanup_error)
        failure.add_note(
            "Provider operation cleanup after start-evidence failure also failed: "
            f"{type(cleanup_error).__name__}."
        )
        return outcome.cancellation, None, cancellation_requests_consumed
    try:
        if outcome.result is None:
            raise RuntimeError("Provider operation cancellation returned no snapshot.")
        cancellation_snapshot = copy_provider_operation_snapshot(outcome.result)
        if cancellation_snapshot.state != state:
            raise RuntimeError("Provider operation cancellation returned a different identity.")
        if not cancellation_snapshot.status.terminal:
            failure.add_note("Provider operation cancellation did not reach a terminal state.")
    except BaseException as cleanup_error:
        _attach_provider_operation_cleanup_failure(failure, cleanup_error)
        failure.add_note(
            "Provider operation cleanup after start-evidence failure also failed: "
            f"{type(cleanup_error).__name__}."
        )
        return outcome.cancellation, None, cancellation_requests_consumed
    return outcome.cancellation, cancellation_snapshot, cancellation_requests_consumed


class _ProviderOperationCancellationClaimReleaseObserved(RuntimeError):
    """An intentional durable claim release won a concurrent heartbeat."""


@dataclass
class _ProviderOperationCancellationHeartbeat:
    stop: asyncio.Event
    release_intended: asyncio.Event
    claim_deadline_monotonic: float
    task: asyncio.Task[None] | None = None
    owner_task: asyncio.Task[Any] | None = None


def _stops_owned_heartbeats_on_failure(
    operation: Callable[_P, Awaitable[_R]],
) -> Callable[_P, Awaitable[_R]]:
    """Stop heartbeats this task started when the cancellation fails.

    Otherwise a heartbeat keeps renewing its claim until its owner task ends,
    which never happens when that task goes on to shut the application down.
    An expired lease is the safe outcome of a failed cancellation.
    """

    @functools.wraps(operation)
    async def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        owner_task = asyncio.current_task()
        try:
            return await operation(*args, **kwargs)
        except GeneratorExit:
            raise
        except BaseException:
            owner = args[0]
            assert isinstance(owner, ProviderOperationCancellationOwner)
            owner._stop_heartbeats_owned_by(owner_task)
            raise

    return guarded


def _provider_operation_target_model(session: Session, stage: ModelCompletionStage) -> str:
    target = model_failover_target_for_stored_stage(session=session, stage=stage)
    return session.model if target is None else target.model


class ProviderOperationCancellationOwner:
    """Own durable cancellation claims, renewal, evidence and accounting handoff.

    The supplied process-local lifecycle retains provider tasks across caller
    cancellation. The context reader validates the current stage on each use;
    this owner neither caches recovery authority nor changes backend atomicity.
    """

    def __init__(
        self,
        *,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        run_limit_controller: RunLimitController,
        lifecycle: ProviderOperationCancellationLifecycle,
        read_recovery_context: Callable[
            [ModelCompletionStage], ProviderCancellationRecoveryContext | None
        ],
    ) -> None:
        self._session_store = session_store
        self._event_writer = event_writer
        self._run_limit_controller = run_limit_controller
        self._provider_operation_cancellation_lifecycle = lifecycle
        self._read_recovery_context = read_recovery_context
        self._provider_operation_cancellation_heartbeats: dict[
            str, _ProviderOperationCancellationHeartbeat
        ] = {}
        # Claim renewal writes that outlived their lease: waited for at
        # shutdown, never cancelled, so stores stay open under them.
        self._detached_renewals: set[asyncio.Task[Any]] = set()

    def _retain_detached_renewal(self, task: asyncio.Task[Any]) -> None:
        self._detached_renewals.add(task)

        def settled(completed: asyncio.Task[Any]) -> None:
            self._detached_renewals.discard(completed)
            _consume_detached_task_outcome(completed)

        task.add_done_callback(settled)

    def _stop_heartbeats_owned_by(self, owner_task: asyncio.Task[Any] | None) -> None:
        for control in self._provider_operation_cancellation_heartbeats.values():
            if control.owner_task is owner_task:
                control.stop.set()

    def detached_renewals(self) -> set[asyncio.Future[Any]]:
        """Claim renewal writes that outlived their lease and still run."""

        return set(self._detached_renewals)

    def running(self) -> set[asyncio.Future[Any]]:
        """Claim heartbeats and renewal writes still running."""

        heartbeats = {
            control.task
            for control in self._provider_operation_cancellation_heartbeats.values()
            if control.task is not None
        }
        return {*heartbeats, *self._detached_renewals}

    def stop_all_heartbeats(self) -> None:
        """Stop every claim heartbeat; each claim's lease then expires.

        Only safe once no operation that may still hold a claim is running.
        """

        for control in self._provider_operation_cancellation_heartbeats.values():
            control.stop.set()

    @property
    def pending(self) -> bool:
        return any(not task.done() for task in self.running())

    def _start_provider_operation_cancellation_heartbeat(
        self,
        *,
        session: Session,
        claim: ProviderOperationCancellationClaim,
        ownership_lost: asyncio.Event,
        claim_deadline_monotonic: float,
    ) -> _ProviderOperationCancellationHeartbeat:
        control = _ProviderOperationCancellationHeartbeat(
            stop=asyncio.Event(),
            release_intended=asyncio.Event(),
            claim_deadline_monotonic=claim_deadline_monotonic,
        )
        task = asyncio.create_task(
            self._heartbeat_provider_operation_cancellation_claim(
                session=session,
                claim=claim,
                owner_task=asyncio.current_task(),
                ownership_lost=ownership_lost,
                stop=control.stop,
                release_intended=control.release_intended,
                control=control,
            )
        )
        control.task = task
        control.owner_task = asyncio.current_task()
        self._provider_operation_cancellation_heartbeats[claim.claim_id] = control

        def settled(completed: asyncio.Task[None]) -> None:
            if self._provider_operation_cancellation_heartbeats.get(claim.claim_id) is control:
                self._provider_operation_cancellation_heartbeats.pop(claim.claim_id, None)
            _consume_detached_task_outcome(completed)

        task.add_done_callback(settled)
        return control

    def _mark_provider_operation_cancellation_claim_release(
        self,
        claim: ProviderOperationCancellationClaim,
    ) -> None:
        control = self._provider_operation_cancellation_heartbeats.get(claim.claim_id)
        if control is not None:
            control.release_intended.set()

    async def _stop_provider_operation_cancellation_heartbeat(
        self,
        claim: ProviderOperationCancellationClaim,
    ) -> None:
        control = self._provider_operation_cancellation_heartbeats.get(claim.claim_id)
        if control is None or control.task is None:
            return
        control.stop.set()
        await control.task

    async def _acquire_provider_operation_cancellation_claim(
        self,
        *,
        session: Session,
        claim: ProviderOperationCancellationClaim,
    ) -> ProviderOperationCancellationClaim:
        persisted_claim: ProviderOperationCancellationClaim | None = None

        def acquire_claim(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            store_now: datetime,
        ) -> dict[str, Any]:
            nonlocal persisted_claim
            if current_session.run_epoch != session.run_epoch:
                raise SessionRunFenced(
                    "Provider-operation cancellation ownership changed at acquisition."
                )
            persisted_claim = claim.model_copy(
                update={"expires_at": (store_now + _PROVIDER_OPERATION_CANCELLATION_CLAIM_LEASE)}
            )
            return checkpoint_with_provider_operation_cancellation_claim(
                checkpoint,
                persisted_claim,
                now=store_now,
            )

        def commit_time_guard(commit_now: datetime) -> None:
            if persisted_claim is None or not persisted_claim.active_at(commit_now):
                raise RuntimeError(
                    "Provider-operation cancellation claim expired before acquisition."
                )

        await self._session_store.publish_checkpoint_and_events_with_store_time(
            session.id,
            idempotency_key=f"provider-operation-cancellation-lease:{claim.claim_id}",
            checkpoint_transform=acquire_claim,
            commit_time_guard=commit_time_guard,
            events=[],
            expected_run_epoch=session.run_epoch,
        )
        if persisted_claim is None:
            raise RuntimeError("Provider-operation cancellation claim was not persisted.")
        return persisted_claim

    async def _persist_provider_operation_cancellation_event(
        self,
        *,
        event_type: EventType,
        cancellation_status: str,
        session: Session,
        stage: ModelCompletionStage,
        state: ProviderOperationState,
        interaction_id: str,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
        step: int,
        attempt: int,
        max_attempts: int,
        model_attempt_identity: ModelAttemptIdentity,
        provider_status: ProviderOperationStatus | None = None,
        error_type: str | None = None,
        cancellation_claim: ProviderOperationCancellationClaim | None = None,
        release_cancellation_claim: bool = False,
    ) -> ProviderOperationCancellationClaim | None:
        payload: dict[str, Any] = {
            "provider": registered_provider.name,
            "model": _provider_operation_target_model(session, stage),
            "step": step,
            "attempt": attempt,
            "max_attempts": max_attempts,
            **model_attempt_identity.payload(),
            "source_run_epoch": stage.source_run_epoch,
            "run_epoch": session.run_epoch,
            "operation_id": state.operation_id,
            "stream_protocol": state.stream_protocol,
            "cancellation_status": cancellation_status,
        }
        if provider_status is not None:
            payload["provider_status"] = provider_status.value
        if error_type is not None:
            payload["error_type"] = require_durable_clean_nonblank(error_type, "error_type")
        event = _event_with_model_identity_authority(
            Event(
                id=provider_operation_cancellation_event_id(
                    stage_id=stage.stage_id,
                    run_epoch=session.run_epoch,
                    event_type=event_type,
                    cancellation_status=cancellation_status,
                    provider_status=provider_status,
                    error_type=error_type,
                ),
                type=event_type,
                session_id=session.id,
                interaction_id=interaction_id,
                timestamp=stage.prepared_at,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload=payload,
            ),
            model_attempt_identity,
        )
        recovery_context = self._read_recovery_context(stage)
        event = event_with_execution_profile_fingerprint_authority(
            event,
            (None if recovery_context is None else recovery_context.execution_profile_fingerprint),
        )
        event = event_with_runtime_payload_authority(
            event,
            "operation_id",
            "stream_protocol",
        )
        prepared = self._event_writer.prepare(event_with_runtime_generated_id(event))
        if cancellation_claim is None:
            if release_cancellation_claim:
                raise ValueError("Cancellation-claim release requires the exact claim.")
            persisted = await self._event_writer.persist_exact_replay(prepared)
        else:
            persisted_claim: ProviderOperationCancellationClaim | None = None

            def checkpoint_transform(
                current_session: Session,
                checkpoint: dict[str, Any] | None,
                store_now: datetime,
            ) -> dict[str, Any]:
                nonlocal persisted_claim
                if current_session.run_epoch != session.run_epoch:
                    raise SessionRunFenced(
                        "Provider-operation cancellation ownership changed at publication."
                    )
                if release_cancellation_claim:
                    existing = provider_operation_cancellation_claim_from_checkpoint(checkpoint)
                    if (
                        existing is None
                        or not existing.same_owner(cancellation_claim)
                        or not existing.active_at(store_now)
                    ):
                        raise RuntimeError(
                            "Provider-operation cancellation ownership changed before release."
                        )
                    persisted_claim = existing
                    return checkpoint_without_provider_operation_cancellation_claim(
                        checkpoint,
                        cancellation_claim,
                        now=store_now,
                    )
                persisted_claim = cancellation_claim.model_copy(
                    update={
                        "expires_at": (store_now + _PROVIDER_OPERATION_CANCELLATION_CLAIM_LEASE)
                    }
                )
                return checkpoint_with_provider_operation_cancellation_claim(
                    checkpoint,
                    persisted_claim,
                    now=store_now,
                )

            def commit_time_guard(commit_now: datetime) -> None:
                if persisted_claim is None or not persisted_claim.active_at(commit_now):
                    raise RuntimeError(
                        "Provider-operation cancellation claim expired before publication."
                    )

            if release_cancellation_claim:
                self._mark_provider_operation_cancellation_claim_release(cancellation_claim)
            await self._session_store.publish_checkpoint_and_events_with_store_time(
                session.id,
                idempotency_key=(
                    f"provider-operation-cancellation-lease:{cancellation_claim.claim_id}"
                ),
                checkpoint_transform=checkpoint_transform,
                commit_time_guard=commit_time_guard,
                events=[prepared],
                expected_run_epoch=session.run_epoch,
            )
            if release_cancellation_claim:
                await self._stop_provider_operation_cancellation_heartbeat(cancellation_claim)
            persisted = prepared
        [emitted] = await self._event_writer.fan_out_persisted([persisted])
        del emitted
        if cancellation_claim is None or release_cancellation_claim:
            return None
        assert persisted_claim is not None
        return persisted_claim

    async def release_claim(
        self,
        *,
        session: Session,
        claim: ProviderOperationCancellationClaim,
    ) -> None:
        persisted_claim: ProviderOperationCancellationClaim | None = None

        def release_claim(
            current_session: Session,
            checkpoint: dict[str, Any] | None,
            store_now: datetime,
        ) -> dict[str, Any]:
            nonlocal persisted_claim
            if current_session.run_epoch != session.run_epoch:
                raise SessionRunFenced(
                    "Provider-operation cancellation ownership changed before claim release."
                )
            existing = provider_operation_cancellation_claim_from_checkpoint(checkpoint)
            if (
                existing is None
                or not existing.same_owner(claim)
                or not existing.active_at(store_now)
            ):
                raise RuntimeError(
                    "Provider-operation cancellation ownership changed before release."
                )
            persisted_claim = existing
            return checkpoint_without_provider_operation_cancellation_claim(
                checkpoint,
                claim,
                now=store_now,
            )

        def commit_time_guard(commit_now: datetime) -> None:
            if persisted_claim is None or not persisted_claim.active_at(commit_now):
                raise RuntimeError("Provider-operation cancellation claim expired before release.")

        self._mark_provider_operation_cancellation_claim_release(claim)
        try:
            await self._session_store.publish_checkpoint_and_events_with_store_time(
                session.id,
                idempotency_key=f"provider-operation-cancellation-lease:{claim.claim_id}",
                checkpoint_transform=release_claim,
                commit_time_guard=commit_time_guard,
                events=[],
                expected_run_epoch=session.run_epoch,
            )
        except BaseException as release_error:
            # A failed release still ends renewal and its lease then expires;
            # the release error stays the one reported.
            control = self._provider_operation_cancellation_heartbeats.get(claim.claim_id)
            heartbeat = None if control is None else control.task
            if control is not None and heartbeat is not None:
                control.stop.set()
                # Wait without cancelling: a renewal in flight finishes and stays owned.
                await asyncio.wait((heartbeat,))
                if not heartbeat.cancelled() and isinstance(heartbeat.exception(), Exception):
                    add_exception_note_safely(
                        release_error,
                        "Provider-operation cancellation claim renewal also failed: "
                        f"{type(heartbeat.exception()).__name__}.",
                    )
            raise
        await self._stop_provider_operation_cancellation_heartbeat(claim)

    async def _heartbeat_provider_operation_cancellation_claim(
        self,
        *,
        session: Session,
        claim: ProviderOperationCancellationClaim,
        owner_task: asyncio.Task[Any] | None,
        ownership_lost: asyncio.Event,
        stop: asyncio.Event,
        release_intended: asyncio.Event,
        control: _ProviderOperationCancellationHeartbeat,
    ) -> None:
        """Renew one cancellation lease until its owner releases or loses it."""

        def lose_ownership(reason: str) -> None:
            ownership_lost.set()
            # An owner that already stopped this heartbeat gave the claim up;
            # cancelling it would replace the outcome it is reporting.
            if not stop.is_set() and owner_task is not None and not owner_task.done():
                owner_task.cancel(reason)

        while not stop.is_set():
            remaining = control.claim_deadline_monotonic - time.monotonic()
            if remaining <= 0:
                lose_ownership("Provider-operation cancellation ownership lease expired.")
                return
            try:
                wait_seconds = (
                    _PROVIDER_OPERATION_CANCELLATION_CLAIM_HEARTBEAT_SECONDS
                    if remaining > _PROVIDER_OPERATION_CANCELLATION_CLAIM_HEARTBEAT_SECONDS
                    else 0.0
                )
                await asyncio.wait_for(
                    stop.wait(),
                    timeout=wait_seconds,
                )
                return
            except TimeoutError:
                pass
            if owner_task is not None and owner_task.done():
                return
            renewal_started_monotonic = time.monotonic()
            renewed_claim_ref: dict[str, ProviderOperationCancellationClaim] = {}

            def renew_claim(
                current_session: Session,
                checkpoint: dict[str, Any] | None,
                store_now: datetime,
                claim_ref: dict[
                    str,
                    ProviderOperationCancellationClaim,
                ] = renewed_claim_ref,
            ) -> dict[str, Any]:
                if current_session.run_epoch != session.run_epoch:
                    raise SessionRunFenced(
                        "Provider-operation cancellation ownership changed before renewal."
                    )
                existing = provider_operation_cancellation_claim_from_checkpoint(checkpoint)
                if existing is None and release_intended.is_set():
                    raise _ProviderOperationCancellationClaimReleaseObserved
                if (
                    existing is None
                    or not existing.same_owner(claim)
                    or not existing.active_at(store_now)
                ):
                    raise RuntimeError("Provider-operation cancellation claim is no longer active.")
                renewed = existing.model_copy(
                    update={
                        "expires_at": (store_now + _PROVIDER_OPERATION_CANCELLATION_CLAIM_LEASE)
                    }
                )
                claim_ref["claim"] = renewed
                return checkpoint_with_provider_operation_cancellation_claim(
                    checkpoint,
                    renewed,
                    now=store_now,
                )

            def commit_time_guard(
                commit_now: datetime,
                claim_ref: dict[
                    str,
                    ProviderOperationCancellationClaim,
                ] = renewed_claim_ref,
            ) -> None:
                renewed = claim_ref.get("claim")
                if renewed is None or not renewed.active_at(commit_now):
                    raise RuntimeError(
                        "Provider-operation cancellation claim expired before renewal."
                    )

            renewal_task = asyncio.create_task(
                self._session_store.publish_checkpoint_and_events_with_store_time(
                    session.id,
                    idempotency_key=(f"provider-operation-cancellation-lease:{claim.claim_id}"),
                    checkpoint_transform=renew_claim,
                    commit_time_guard=commit_time_guard,
                    events=[],
                    expected_run_epoch=session.run_epoch,
                )
            )
            try:
                outcome = await await_shielded_task_outcome(
                    renewal_task,
                    timeout_s=max(
                        0.0,
                        control.claim_deadline_monotonic - time.monotonic(),
                    ),
                )
            except _ProviderOperationCancellationClaimReleaseObserved:
                return
            except asyncio.CancelledError:
                self._retain_detached_renewal(renewal_task)
                raise
            if outcome.cancellation is not None:
                self._retain_detached_renewal(renewal_task)
                raise outcome.cancellation
            if outcome.timed_out:
                self._retain_detached_renewal(renewal_task)
                lose_ownership(
                    "Provider-operation cancellation ownership renewal was not acknowledged."
                )
                return
            if outcome.error is not None:
                if isinstance(
                    outcome.error,
                    _ProviderOperationCancellationClaimReleaseObserved,
                ):
                    return
                lose_ownership("Provider-operation cancellation ownership heartbeat failed.")
                raise outcome.error
            control.claim_deadline_monotonic = (
                renewal_started_monotonic
                + _PROVIDER_OPERATION_CANCELLATION_CLAIM_LEASE.total_seconds()
            )
            if time.monotonic() >= control.claim_deadline_monotonic:
                lose_ownership("Provider-operation cancellation ownership acknowledgement expired.")
                return

    @_stops_owned_heartbeats_on_failure
    async def cancel_started_operation(
        self,
        *,
        adapter: ProviderOperationAdapter,
        state: ProviderOperationState,
        failure: SessionInterruptedByRequest | asyncio.CancelledError,
        session: Session,
        stage: ModelCompletionStage,
        interaction_id: str,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
        step: int,
        attempt: int,
        max_attempts: int,
        model_attempt_identity: ModelAttemptIdentity,
        settle_budget: bool = False,
    ) -> ProviderOperationSnapshot | None:
        heartbeat_control: _ProviderOperationCancellationHeartbeat | None = None
        cancellation_ownership_lost: asyncio.Event | None = None
        durable_resolution = await load_provider_operation_cancellation_resolution(
            self._session_store,
            stage,
            RecoverableProviderOperation(
                interaction_id=interaction_id,
                provider=registered_provider.name,
                model=_provider_operation_target_model(session, stage),
                model_attempt_identity=model_attempt_identity,
                state=state,
                status=ProviderOperationStatus.IN_PROGRESS,
                step=step,
                attempt=attempt,
                max_attempts=max_attempts,
                source_run_epoch=stage.source_run_epoch,
            ),
        )

        async def require_cancellation_owner() -> None:
            if heartbeat_control is not None and (
                cancellation_ownership_lost is None
                or cancellation_ownership_lost.is_set()
                or time.monotonic() >= heartbeat_control.claim_deadline_monotonic
            ):
                raise RuntimeError("Provider-operation cancellation claim is no longer active.")
            current = await self._session_store.load(session.id)
            if current is None:
                raise KeyError(f"Session not found: {session.id}")
            if current.run_epoch != session.run_epoch:
                raise SessionRunFenced(
                    "Provider-operation cancellation run epoch is stale: expected "
                    f"{session.run_epoch}, current {current.run_epoch}."
                )
            if heartbeat_control is not None:

                def inspect_claim(
                    current_session: Session,
                    checkpoint: dict[str, Any] | None,
                    store_now: datetime,
                ) -> None:
                    if current_session.run_epoch != session.run_epoch:
                        raise SessionRunFenced("Provider-operation cancellation run epoch changed.")
                    existing = provider_operation_cancellation_claim_from_checkpoint(checkpoint)
                    if (
                        existing is None
                        or not existing.same_owner(cancellation_claim)
                        or not existing.active_at(store_now)
                    ):
                        raise RuntimeError(
                            "Provider-operation cancellation claim is no longer active."
                        )

                await self._session_store.transform_checkpoint_with_store_time(
                    session.id,
                    inspect_claim,
                )
                if (
                    cancellation_ownership_lost is None
                    or cancellation_ownership_lost.is_set()
                    or time.monotonic() >= heartbeat_control.claim_deadline_monotonic
                ):
                    raise RuntimeError(
                        "Provider-operation cancellation claim expired during validation."
                    )

        await require_cancellation_owner()
        cancellation_claim = ProviderOperationCancellationClaim(
            claim_id=(
                f"provider-cancel:{stage.stage_id}:{session.run_epoch}:"
                f"{state.operation_id}:{state.stream_protocol}"
            ),
            stage_id=stage.stage_id,
            run_epoch=session.run_epoch,
            operation_id=state.operation_id,
            stream_protocol=state.stream_protocol,
            expires_at=session.updated_at,
        )
        claim_started_monotonic = time.monotonic()
        if durable_resolution is None:
            support = adapter.cancellation_support
            if type(support) is not ProviderOperationCancellationSupport:
                raise TypeError(
                    "ProviderOperationAdapter.cancellation_support must return "
                    "ProviderOperationCancellationSupport."
                )
            persisted_cancellation_claim = (
                await self._persist_provider_operation_cancellation_event(
                    event_type=EventType.PROVIDER_OPERATION_CANCEL_REQUESTED,
                    cancellation_status="requested",
                    session=session,
                    stage=stage,
                    state=state,
                    interaction_id=interaction_id,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    environment_name=environment_name,
                    step=step,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    model_attempt_identity=model_attempt_identity,
                    cancellation_claim=cancellation_claim,
                )
            )
        elif not durable_resolution.resolution_recorded:
            support = None
            persisted_cancellation_claim = (
                await self._persist_provider_operation_cancellation_event(
                    event_type=EventType.PROVIDER_OPERATION_CANCEL_RESOLVED,
                    cancellation_status=durable_resolution.status.value,
                    session=session,
                    stage=stage,
                    state=state,
                    interaction_id=interaction_id,
                    registered_agent=registered_agent,
                    registered_provider=registered_provider,
                    environment_name=environment_name,
                    step=step,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    model_attempt_identity=model_attempt_identity,
                    provider_status=durable_resolution.provider_status,
                    error_type=durable_resolution.error_type,
                    cancellation_claim=cancellation_claim,
                )
            )
        else:
            support = None
            persisted_cancellation_claim = (
                await self._acquire_provider_operation_cancellation_claim(
                    session=session,
                    claim=cancellation_claim,
                )
            )
        if persisted_cancellation_claim is None:
            raise RuntimeError("Provider-operation cancellation claim was not persisted.")
        cancellation_claim = persisted_cancellation_claim
        claim_deadline_monotonic = (
            claim_started_monotonic + _PROVIDER_OPERATION_CANCELLATION_CLAIM_LEASE.total_seconds()
        )
        if claim_deadline_monotonic <= time.monotonic():
            raise RuntimeError(
                "Provider-operation cancellation claim acknowledgement consumed its lease."
            )
        cancellation_ownership_lost = asyncio.Event()
        heartbeat_control = self._start_provider_operation_cancellation_heartbeat(
            session=session,
            claim=cancellation_claim,
            ownership_lost=cancellation_ownership_lost,
            claim_deadline_monotonic=claim_deadline_monotonic,
        )
        await require_cancellation_owner()
        if durable_resolution is not None and durable_resolution.provider_status is None:
            await self.release_claim(
                session=session,
                claim=cancellation_claim,
            )
            failure.__dict__["provider_operation_accounting_pending"] = True
            return None
        if support is ProviderOperationCancellationSupport.UNSUPPORTED:
            await self._persist_provider_operation_cancellation_event(
                event_type=EventType.PROVIDER_OPERATION_CANCEL_RESOLVED,
                cancellation_status="unsupported",
                session=session,
                stage=stage,
                state=state,
                interaction_id=interaction_id,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                environment_name=environment_name,
                step=step,
                attempt=attempt,
                max_attempts=max_attempts,
                model_attempt_identity=model_attempt_identity,
                cancellation_claim=cancellation_claim,
                release_cancellation_claim=True,
            )
            failure.__dict__["provider_operation_accounting_pending"] = True
            return None
        if durable_resolution is None:
            (
                cancellation,
                snapshot,
                cancellation_requests_consumed,
            ) = await _cancel_provider_operation_after_definite_absence(
                lifecycle=self._provider_operation_cancellation_lifecycle,
                adapter=adapter,
                state=state,
                failure=failure,
                cancellation=(failure if isinstance(failure, asyncio.CancelledError) else None),
                ownership_lost=cancellation_ownership_lost,
            )
            if cancellation is not None and cancellation is not failure:
                restore_task_cancellation_requests(
                    cancellation_requests_consumed,
                    cancellation=cancellation,
                )
                raise cancellation from failure
        else:
            assert durable_resolution.provider_status is not None
            snapshot = ProviderOperationSnapshot(
                state=copy_provider_operation_state(state),
                status=durable_resolution.provider_status,
            )
        if snapshot is None:
            await require_cancellation_owner()
            await self._persist_provider_operation_cancellation_event(
                event_type=EventType.PROVIDER_OPERATION_CANCEL_RESOLVED,
                cancellation_status="failed",
                session=session,
                stage=stage,
                state=state,
                interaction_id=interaction_id,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                environment_name=environment_name,
                step=step,
                attempt=attempt,
                max_attempts=max_attempts,
                model_attempt_identity=model_attempt_identity,
                error_type="CancellationUnconfirmed",
                cancellation_claim=cancellation_claim,
                release_cancellation_claim=True,
            )
            failure.__dict__["provider_operation_accounting_pending"] = True
            return None
        cancellation_status = {
            ProviderOperationStatus.CANCELLED: "cancelled",
            ProviderOperationStatus.COMPLETED: "completed",
            ProviderOperationStatus.QUEUED: "pending",
            ProviderOperationStatus.IN_PROGRESS: "pending",
            ProviderOperationStatus.UNAVAILABLE: "unavailable",
            ProviderOperationStatus.FAILED: "failed",
            ProviderOperationStatus.EXPIRED: "failed",
        }[snapshot.status]
        await require_cancellation_owner()
        retain_claim_after_resolution = snapshot.status is ProviderOperationStatus.COMPLETED or (
            snapshot.status is ProviderOperationStatus.CANCELLED and bool(stage.reservation_ids)
        )
        if durable_resolution is None:
            await self._persist_provider_operation_cancellation_event(
                event_type=EventType.PROVIDER_OPERATION_CANCEL_RESOLVED,
                cancellation_status=cancellation_status,
                session=session,
                stage=stage,
                state=state,
                interaction_id=interaction_id,
                registered_agent=registered_agent,
                registered_provider=registered_provider,
                environment_name=environment_name,
                step=step,
                attempt=attempt,
                max_attempts=max_attempts,
                model_attempt_identity=model_attempt_identity,
                provider_status=snapshot.status,
                cancellation_claim=cancellation_claim,
                release_cancellation_claim=not retain_claim_after_resolution,
            )
        elif not retain_claim_after_resolution:
            await self.release_claim(
                session=session,
                claim=cancellation_claim,
            )
        if snapshot.status is not ProviderOperationStatus.CANCELLED:
            failure.__dict__["provider_operation_accounting_pending"] = True
        elif settle_budget and stage.reservation_ids:
            await require_cancellation_owner()
            recovery_context = self._read_recovery_context(stage)
            if recovery_context is None:
                raise ProviderOperationEvidenceError(
                    "Budgeted provider-operation cancellation has no durable accounting context."
                )
            try:
                await (
                    self._run_limit_controller.reconcile_cancelled_provider_operation_reservations(
                        reservation_ids=stage.reservation_ids,
                        recovery_contexts=recovery_context.budget_reservations,
                        session=session,
                        provider_name=registered_provider.name,
                        model_attempt_identity=model_attempt_identity,
                        dispatch_id=stage.stage_id,
                        request_billing_identity=recovery_context.billing_identity,
                    )
                )
            except (KeyError, NotImplementedError, TypeError, ValueError) as accounting_error:
                raise ProviderOperationEvidenceError(
                    "Provider-operation cancellation could not reconstruct its original "
                    "budget reservation and pricing context."
                ) from accounting_error
            await self.release_claim(
                session=session,
                claim=cancellation_claim,
            )
        elif stage.reservation_ids:
            failure.__dict__["provider_operation_cancellation_claim"] = cancellation_claim
        return snapshot

    async def cancel_for_interruption(
        self,
        *,
        session: Session,
        stage: ModelCompletionStage,
        operation: RecoverableProviderOperation,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
        invocation_context: InvocationContext | None = None,
        model_execution_selection: ModelExecutionSelection | None = None,
    ) -> ProviderOperationSnapshot | None:
        """Cancel one durably identified operation after its worker disappears."""

        selected_model = _provider_operation_target_model(session, stage)
        if model_execution_selection is not None:
            model_execution_selection.require_recovery_scope(
                session=session,
                stage=stage,
                invocation_context=invocation_context,
                registered_provider=registered_provider,
            )
        elif model_failover_target_for_stored_stage(session=session, stage=stage) is not None:
            raise RuntimeError("Routed provider-operation recovery requires admitted selection.")
        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or registered_agent is not invocation_context.registered_agent
            or (
                model_execution_selection is None
                and registered_provider is not invocation_context.registered_provider
            )
            or environment_name
            != (
                None
                if invocation_context.registered_environment is None
                else invocation_context.registered_environment.spec.name
            )
        ):
            raise RuntimeError(
                "Provider-operation cancellation substituted frozen invocation authority."
            )

        provider = registered_provider.provider
        adapter = provider.provider_operations
        if (
            provider.provider_operation_mode is not ProviderOperationMode.BACKGROUND
            or not isinstance(adapter, ProviderOperationAdapter)
        ):
            raise RuntimeError(
                "The registered provider no longer supports its durable background operation."
            )
        if operation.provider != registered_provider.name:
            raise RuntimeError("Provider-operation cancellation resolved a different provider.")
        if operation.model != selected_model:
            raise RuntimeError("Provider-operation cancellation resolved a different model.")
        if operation.model_attempt_identity.model_step_id != stage.logical_step_id:
            raise RuntimeError(
                "Provider-operation cancellation belongs to a different model stage."
            )
        return await self.cancel_started_operation(
            adapter=adapter,
            state=operation.state,
            failure=SessionInterruptedByRequest(session.id),
            session=session,
            stage=stage,
            interaction_id=operation.interaction_id,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            environment_name=environment_name,
            step=operation.step,
            attempt=operation.attempt,
            max_attempts=operation.max_attempts,
            model_attempt_identity=operation.model_attempt_identity,
            settle_budget=True,
        )
