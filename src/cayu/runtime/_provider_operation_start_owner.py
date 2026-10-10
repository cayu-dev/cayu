"""Live provider-operation startup, exact identity publication and late settlement."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextvars import Context
from dataclasses import dataclass
from typing import Any, Never

from cayu._exception_groups import exception_cause, iter_exception_tree
from cayu._task_wait import (
    LateFailures,
    await_shielded_task_outcome,
    restore_task_cancellation_requests,
    retained_task_failure,
    unexpected_child_cancellation_error,
)
from cayu._validation import copy_durable_record
from cayu.events import Event, EventType, event_with_runtime_payload_authority
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
    event_with_execution_profile_authority,
)
from cayu.execution_units import ModelAttemptIdentity
from cayu.providers._credential_boundary import aclosing_provider_stream
from cayu.providers.base import (
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamDeadlineError,
    ModelStreamEvent,
)
from cayu.providers.deadlines import (
    ProviderStreamDeadlineAdmission,
    bind_provider_deadline_admission,
    reset_provider_deadline_admission,
)
from cayu.providers.operations import (
    ProviderOperationAdapter,
    ProviderOperationConnection,
    ProviderOperationStartIdempotencySupport,
    ProviderOperationStartRequest,
    ProviderOperationState,
    ProviderOperationStatus,
    copy_provider_operation_connection,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._model_completion_contracts import ModelCompletionDispatch
from cayu.runtime._model_event_authority import _event_with_model_identity_authority
from cayu.runtime._provider_operation_cancellation_owner import (
    _cancel_provider_operation_after_definite_absence,
)
from cayu.runtime._provider_stream import _close_async_iterator
from cayu.runtime._session_control import SessionInterruptedByRequest
from cayu.runtime.provider_operation_cancellation import (
    ProviderOperationCancellationAdmissionsSealed,
    ProviderOperationCancellationLifecycle,
)
from cayu.runtime.provider_operations import provider_operation_started_event_id
from cayu.sessions.base import SessionRunFenced, SessionStore
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.records import Session, SessionStatus, SessionStatusConflict

_PROVIDER_OPERATION_START_SETTLEMENT_TIMEOUT_SECONDS = 5.0


def _ambiguous_provider_operation_start_error(
    *,
    provider_name: str,
    cause: BaseException,
) -> ModelProviderError:
    return ModelProviderError(
        "Provider operation start outcome is ambiguous; automatic retry is disabled.",
        provider=provider_name,
        error_type=type(cause).__name__,
        error_code="provider_operation_start_ambiguous",
        retryable=False,
    )


def is_ambiguous_provider_operation_start_error(failure: BaseException) -> bool:
    """Return whether one failure retains start-only provider ambiguity."""

    return any(
        isinstance(candidate, ModelProviderError)
        and candidate.error_code == "provider_operation_start_ambiguous"
        for candidate in iter_exception_tree(failure)
    )


def _late_cancellation_failure(failure: BaseException) -> BaseException | None:
    """Return why cancelling a late-started operation failed, if it did.

    Sealed admission is not reported here: shutdown already reports it as an
    unowned cancellation.
    """

    cause = exception_cause(failure)
    if not isinstance(cause, BaseExceptionGroup) or not cause.exceptions:
        return None
    cleanup_error = cause.exceptions[-1]
    if isinstance(cleanup_error, ProviderOperationCancellationAdmissionsSealed):
        return None
    return cleanup_error


@dataclass
class ProviderOperationStartState:
    """Observed startup effects, retained even when iteration raises or closes.

    The start owner writes this state; the live executor takes over the stream
    and current operation identity when startup leaves its event boundary.
    Failed identity publication must remain distinguishable from definite
    absence so caller cleanup cannot cancel an operation on uncertain evidence.
    """

    adapter: ProviderOperationAdapter | None = None
    operation_state: ProviderOperationState | None = None
    interaction_id: str | None = None
    events: AsyncIterator[ModelStreamEvent] | None = None
    identity_durable: bool = False
    dispatch_invoked: bool = False


class ProviderOperationStartOwner:
    """Start exact operations and retain late acknowledgement reconciliation.

    Admission and context-exposure callbacks use the caller's existing run
    authority. This owner orders those boundaries with dispatch and durable
    identity publication, and shares the existing cancellation lifecycle.
    """

    def __init__(
        self,
        *,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        cancellation_lifecycle: ProviderOperationCancellationLifecycle,
    ) -> None:
        self._session_store = session_store
        self._event_writer = event_writer
        self._provider_operation_cancellation_lifecycle = cancellation_lifecycle
        self._reconciliation_tasks: set[asyncio.Task[None]] = set()
        self.reconciliation_failures = LateFailures("A provider-operation reconciliation")

    def running_reconciliations(self) -> set[asyncio.Future[Any]]:
        """Late-start reconciliations still running after their model step."""

        return set(self._reconciliation_tasks)

    def _record_reconciliation_failure(self, failure: BaseException) -> None:
        self.reconciliation_failures.record(type(failure).__qualname__)

    def _retain_reconciliation(self, task: asyncio.Task[None]) -> None:
        self._reconciliation_tasks.add(task)

        def settled(completed: asyncio.Task[None]) -> None:
            self._reconciliation_tasks.discard(completed)
            failure = retained_task_failure(completed)
            if failure is not None:
                self._record_reconciliation_failure(failure)

        task.add_done_callback(settled)

    async def start(
        self,
        *,
        progress: ProviderOperationStartState,
        provider: ModelProvider,
        model_request: ModelRequest,
        deadline_admission: ProviderStreamDeadlineAdmission,
        completion_dispatch: ModelCompletionDispatch | None,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_provider: runtime_records.RegisteredProvider,
        environment_name: str | None,
        step: int,
        attempt: int,
        max_attempts: int,
        model_attempt_identity: ModelAttemptIdentity,
        execution_profile: ExecutionProfileIdentity | None,
        refresh_live_model_semantics: Callable[[], Awaitable[None]],
        consume_child_session_notifications: Callable[[], Awaitable[None]],
        acknowledge_context_exposure: Callable[[str], Awaitable[None]],
    ) -> AsyncGenerator[Event, None]:
        """Yield starting/started events while retaining exact failure-state evidence."""

        provider_operation_adapter = provider.provider_operations
        progress.adapter = provider_operation_adapter
        if not isinstance(provider_operation_adapter, ProviderOperationAdapter):
            raise RuntimeError(
                "Background provider-operation mode requires a ProviderOperationAdapter."
            )
        start_idempotency_support = provider_operation_adapter.start_idempotency_support
        if type(start_idempotency_support) is not ProviderOperationStartIdempotencySupport:
            raise TypeError(
                "ProviderOperationAdapter.start_idempotency_support must return "
                "ProviderOperationStartIdempotencySupport."
            )
        start_id = f"provider-operation:{model_attempt_identity.model_attempt_id}"
        if completion_dispatch is None:
            raise RuntimeError(
                "Background provider operations require a durable model-completion stage."
            )
        staged_start = completion_dispatch.stage.intent.get("provider_operation_start")
        if (
            type(staged_start) is not dict
            or staged_start.get("schema_version") != 1
            or staged_start.get("idempotency_key") != start_id
            or staged_start.get("idempotency_support") != start_idempotency_support.value
        ):
            raise RuntimeError("Provider-operation start contract changed after durable staging.")
        starting_event = _event_with_model_identity_authority(
            Event(
                type=EventType.PROVIDER_OPERATION_STARTING,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload={
                    "provider": registered_provider.name,
                    "model": model_request.model,
                    "step": step,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    **model_attempt_identity.payload(),
                    "source_run_epoch": session.run_epoch,
                    "start_id": start_id,
                    "start_idempotency_support": start_idempotency_support.value,
                },
            ),
            model_attempt_identity,
        )
        starting_event = event_with_execution_profile_authority(
            starting_event,
            execution_profile,
        )
        starting_event = event_with_runtime_payload_authority(
            starting_event,
            "start_id",
        )
        emitted_starting_event = await self._event_writer.emit(starting_event)
        yield emitted_starting_event
        progress.interaction_id = emitted_starting_event.interaction_id
        if progress.interaction_id is None:
            raise RuntimeError("Provider-operation dispatch requires an owning interaction.")

        async def start_provider_operation() -> ProviderOperationConnection:
            # Durable staging and event publication above can yield to
            # application code. Recheck at the last pre-dispatch seam.
            await refresh_live_model_semantics()
            await consume_child_session_notifications()
            await refresh_live_model_semantics()
            start_request = ProviderOperationStartRequest(
                request=model_request,
                idempotency_key=start_id,
            )
            token = bind_provider_deadline_admission(deadline_admission)
            try:
                from cayu.resource_access import require_dispatch

                await require_dispatch()
                progress.dispatch_invoked = True
                return await provider_operation_adapter.start(start_request)
            finally:
                reset_provider_deadline_admission(token)

        start_task = asyncio.create_task(start_provider_operation())

        def operation_event_for(
            operation_state: ProviderOperationState,
            operation_status: ProviderOperationStatus,
        ) -> Event:
            event = _event_with_model_identity_authority(
                Event(
                    id=provider_operation_started_event_id(start_id),
                    type=EventType.PROVIDER_OPERATION_STARTED,
                    session_id=session.id,
                    interaction_id=emitted_starting_event.interaction_id,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    payload={
                        "provider": registered_provider.name,
                        "model": model_request.model,
                        "step": step,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        **model_attempt_identity.payload(),
                        "source_run_epoch": session.run_epoch,
                        "start_id": start_id,
                        "state_version": operation_state.version,
                        "operation_id": operation_state.operation_id,
                        "stream_protocol": operation_state.stream_protocol,
                        "status": operation_status.value,
                        "recovery_metadata": operation_state.recovery_metadata.model_dump(
                            mode="json",
                            exclude_none=True,
                        ),
                    },
                ),
                model_attempt_identity,
            )
            event = event_with_execution_profile_authority(
                event,
                execution_profile,
            )
            return event_with_runtime_payload_authority(event, "start_id")

        start_outcome = await await_shielded_task_outcome(
            start_task,
            timeout_after_cancellation_s=(_PROVIDER_OPERATION_START_SETTLEMENT_TIMEOUT_SECONDS),
        )

        def raise_start_cancellation(
            cause: BaseException | None = None,
            *,
            additional_requests_consumed: int = 0,
        ) -> Never:
            cancellation = start_outcome.cancellation
            if cancellation is None:
                raise RuntimeError("Provider start has no caller cancellation.")
            if start_outcome.cancellation_requests_consumed < 1:
                raise RuntimeError(
                    "Provider start cancellation lost its consumed request count."
                ) from cancellation
            restore_task_cancellation_requests(
                start_outcome.cancellation_requests_consumed + additional_requests_consumed,
                cancellation=cancellation,
            )
            if cause is None:
                raise cancellation
            raise cancellation from cause

        if start_outcome.timed_out:
            start_task.cancel()
            start_cancellation = start_outcome.cancellation
            if start_cancellation is None:  # pragma: no cover - armed by cancellation
                raise RuntimeError("Provider operation start settlement timed out.")

        async def reconcile_late_start() -> None:
            late_outcome = await await_shielded_task_outcome(start_task)
            if late_outcome.error is not None or late_outcome.result is None:
                return
            raw_late_operation = late_outcome.result
            try:
                late_operation = copy_provider_operation_connection(raw_late_operation)
            except BaseException:
                if type(raw_late_operation) is ProviderOperationConnection:
                    async with aclosing_provider_stream(raw_late_operation.events):
                        raise
                raise
            # The helper attaches a failed cancellation to this exception's cause.
            late_start_failure = RuntimeError(
                "Caller cancellation preceded provider start acknowledgement."
            )
            try:
                (
                    late_cleanup_cancellation,
                    cancellation_snapshot,
                    late_cleanup_cancellation_requests_consumed,
                ) = await _cancel_provider_operation_after_definite_absence(
                    lifecycle=self._provider_operation_cancellation_lifecycle,
                    adapter=provider_operation_adapter,
                    state=late_operation.state,
                    failure=late_start_failure,
                    on_unobserved_failure=self._record_reconciliation_failure,
                )
                if late_cleanup_cancellation is not None:
                    restore_task_cancellation_requests(
                        late_cleanup_cancellation_requests_consumed,
                        cancellation=late_cleanup_cancellation,
                    )
                    raise late_cleanup_cancellation
                reconciled_status = (
                    late_operation.status
                    if cancellation_snapshot is None
                    else cancellation_snapshot.status
                )
                reconciliation_event = self._event_writer.prepare(
                    operation_event_for(
                        late_operation.state,
                        reconciled_status,
                    )
                )

                def preserve_checkpoint(
                    _current: Session,
                    checkpoint: dict[str, Any] | None,
                ) -> dict[str, Any]:
                    if checkpoint is None:
                        raise RuntimeError(
                            "Late provider-operation reconciliation requires an "
                            "existing session checkpoint."
                        )
                    copied = copy_durable_record(checkpoint, "checkpoint")
                    if type(copied) is not dict:
                        raise TypeError("Session checkpoint must be an object.")
                    return copied

                for publication_attempt in range(2):
                    reconciliation_session = await self._session_store.load(session.id)
                    if reconciliation_session is None:
                        return
                    if reconciliation_session.run_epoch == session.run_epoch:
                        eligible_statuses = {
                            SessionStatus.RUNNING,
                            SessionStatus.INTERRUPTING,
                        }
                    elif reconciliation_session.run_epoch == session.run_epoch + 1:
                        eligible_statuses = {
                            SessionStatus.INTERRUPTED,
                            SessionStatus.FAILED,
                        }
                    else:
                        return
                    try:
                        await self._session_store.publish_checkpoint_and_events(
                            session.id,
                            checkpoint_transform=preserve_checkpoint,
                            events=[reconciliation_event],
                            expected_statuses=eligible_statuses,
                            expected_run_epoch=reconciliation_session.run_epoch,
                        )
                        break
                    except (SessionRunFenced, SessionStatusConflict):
                        if publication_attempt == 1:
                            raise
                persisted = await self._session_store.query_events(
                    EventQuery(
                        session_id=session.id,
                        event_id=reconciliation_event.id,
                        limit=1,
                    )
                )
                if len(persisted) != 1 or persisted[0].event != reconciliation_event:
                    raise RuntimeError(
                        "Late provider-operation reconciliation readback did not "
                        "match its durable event."
                    )
                await self._event_writer.fan_out_persisted([persisted[0].event])
            finally:
                # On every exit, before closing the stream can raise, so shutdown
                # reports a failed cancel once even when publication fails too.
                failed_cancellation = _late_cancellation_failure(late_start_failure)
                if failed_cancellation is not None:
                    self._record_reconciliation_failure(failed_cancellation)
                await _close_async_iterator(raw_late_operation.events)

        if start_outcome.timed_out:
            start_cancellation = start_outcome.cancellation
            if start_cancellation is None:  # pragma: no cover - validated above
                raise AssertionError("Timed-out provider start lost caller cancellation.")
            reconciliation_task = asyncio.create_task(
                reconcile_late_start(),
                context=Context(),
            )
            self._retain_reconciliation(reconciliation_task)
            start_cancellation.add_note(
                "Provider operation start remained in flight after bounded cancellation "
                "settlement; durable starting evidence prevents automatic retry."
            )
            raise_start_cancellation()

        start_error = start_outcome.error
        if isinstance(start_error, asyncio.CancelledError) and start_outcome.cancellation is None:
            start_error = unexpected_child_cancellation_error(
                start_error,
                operation="Provider operation start",
            )
        if start_error is not None:
            if start_outcome.cancellation is not None:
                raise_start_cancellation(start_error)
            if not isinstance(start_error, Exception):
                raise start_error
            if not progress.dispatch_invoked:
                raise start_error
            if isinstance(start_error, ModelStreamDeadlineError):
                # Preserve the typed error for the common provider-error
                # boundary to defensively copy and publish. The normal
                # deadline stage fence then owns this pre-identity
                # background outcome.
                raise start_error from None
            raise _ambiguous_provider_operation_start_error(
                provider_name=registered_provider.name,
                cause=start_error,
            ) from start_error
        try:
            if start_outcome.result is None:
                raise RuntimeError("Provider operation start returned no connection.")
            raw_provider_operation = start_outcome.result
            try:
                provider_operation = copy_provider_operation_connection(raw_provider_operation)
            except BaseException:
                if type(raw_provider_operation) is ProviderOperationConnection:
                    async with aclosing_provider_stream(raw_provider_operation.events):
                        raise
                raise
        except Exception as start_validation_error:
            if start_outcome.cancellation is not None:
                raise_start_cancellation(start_validation_error)
            raise _ambiguous_provider_operation_start_error(
                provider_name=registered_provider.name,
                cause=start_validation_error,
            ) from start_validation_error
        operation_state = provider_operation.state
        progress.operation_state = operation_state
        operation_event = operation_event_for(
            provider_operation.state,
            provider_operation.status,
        )
        progress.events = provider_operation.events
        try:
            operation_event = self._event_writer.prepare(operation_event)
        except BaseException as preparation_error:
            (
                cleanup_cancellation,
                _,
                cleanup_cancellation_requests_consumed,
            ) = await _cancel_provider_operation_after_definite_absence(
                lifecycle=self._provider_operation_cancellation_lifecycle,
                adapter=provider_operation_adapter,
                state=operation_state,
                failure=preparation_error,
                cancellation=start_outcome.cancellation,
            )
            progress.operation_state = None
            if cleanup_cancellation is not None:
                if cleanup_cancellation is start_outcome.cancellation:
                    raise_start_cancellation(
                        preparation_error,
                        additional_requests_consumed=(cleanup_cancellation_requests_consumed),
                    )
                restore_task_cancellation_requests(
                    cleanup_cancellation_requests_consumed,
                    cancellation=cleanup_cancellation,
                )
                raise cleanup_cancellation from preparation_error
            if isinstance(
                preparation_error,
                SessionInterruptedByRequest | SessionRunFenced,
            ):
                raise
            if isinstance(preparation_error, Exception):
                raise _ambiguous_provider_operation_start_error(
                    provider_name=registered_provider.name,
                    cause=preparation_error,
                ) from preparation_error
            raise
        try:
            persisted_operation_event = await self._event_writer.persist_exact_replay(
                operation_event
            )
        except BaseException as persistence_error:
            try:
                exact_identity_is_durable = await self._event_writer.is_exact_persisted(
                    operation_event
                )
            except BaseException as verification_error:
                persistence_error.add_note(
                    "Provider operation start-evidence readback also failed: "
                    f"{type(verification_error).__name__}."
                )
                if start_outcome.cancellation is not None:
                    raise_start_cancellation(
                        BaseExceptionGroup(
                            "Provider operation publication and readback both failed.",
                            [persistence_error, verification_error],
                        )
                    )
                if isinstance(
                    verification_error,
                    SessionInterruptedByRequest | SessionRunFenced,
                ) or not isinstance(verification_error, Exception):
                    raise
                if isinstance(
                    persistence_error,
                    SessionInterruptedByRequest | SessionRunFenced,
                ):
                    raise persistence_error from verification_error
                if isinstance(persistence_error, Exception):
                    raise _ambiguous_provider_operation_start_error(
                        provider_name=registered_provider.name,
                        cause=persistence_error,
                    ) from verification_error
                raise persistence_error from verification_error
            if not exact_identity_is_durable:
                (
                    cleanup_cancellation,
                    _,
                    cleanup_cancellation_requests_consumed,
                ) = await _cancel_provider_operation_after_definite_absence(
                    lifecycle=self._provider_operation_cancellation_lifecycle,
                    adapter=provider_operation_adapter,
                    state=operation_state,
                    failure=persistence_error,
                    cancellation=start_outcome.cancellation,
                )
                progress.operation_state = None
                if cleanup_cancellation is not None:
                    if cleanup_cancellation is start_outcome.cancellation:
                        raise_start_cancellation(
                            persistence_error,
                            additional_requests_consumed=(cleanup_cancellation_requests_consumed),
                        )
                    restore_task_cancellation_requests(
                        cleanup_cancellation_requests_consumed,
                        cancellation=cleanup_cancellation,
                    )
                    raise cleanup_cancellation from persistence_error
            if start_outcome.cancellation is not None:
                raise_start_cancellation(persistence_error)
            if isinstance(
                persistence_error,
                SessionInterruptedByRequest | SessionRunFenced,
            ):
                raise
            if isinstance(persistence_error, Exception):
                raise _ambiguous_provider_operation_start_error(
                    provider_name=registered_provider.name,
                    cause=persistence_error,
                ) from persistence_error
            raise
        try:
            [emitted_operation_event] = await self._event_writer.fan_out_persisted(
                [persisted_operation_event]
            )
        except BaseException as delivery_error:
            if start_outcome.cancellation is not None:
                raise_start_cancellation(delivery_error)
            if isinstance(
                delivery_error,
                SessionInterruptedByRequest | SessionRunFenced,
            ):
                raise
            if isinstance(delivery_error, Exception):
                raise _ambiguous_provider_operation_start_error(
                    provider_name=registered_provider.name,
                    cause=delivery_error,
                ) from delivery_error
            raise
        progress.identity_durable = True
        await acknowledge_context_exposure(progress.operation_state.operation_id)
        if start_outcome.cancellation is not None:
            raise_start_cancellation()
        yield emitted_operation_event
