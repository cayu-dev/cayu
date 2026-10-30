"""Optional ordinary-session assembly over external waits and native continuation."""

from __future__ import annotations

from contextlib import aclosing
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, uuid5

from cayu.external_wait_tasks import (
    CHILD_SESSION_REFUSAL,
    ended_during_delivery,
    service_under_task,
    task_ended,
)
from cayu.external_waits import (
    ExternalEventWaits,
    ExternalWaitContext,
    ExternalWaitSnapshot,
    _snapshot,
)
from cayu.runtime._external_execution_to_wait import _ExternalExecutionToWait
from cayu.runtime._external_wait_binding import load_prepared_continuation
from cayu.runtime._external_wait_observation import external_wait_entrance
from cayu.sessions.external_waits import (
    EXTERNAL_RECOVERY_INACTIVITY_S,
    ExternalWaitConflict,
    ExternalWaitRecord,
    ExternalWaitRegistration,
    ExternalWaitUnavailable,
    Identifier,
    _Value,
    external_wait_digest,
)
from cayu.sessions.requests import ResumeRequest, RunRequest, copy_resume_request, copy_run_request
from cayu.tasks.parked_continuation import declares_parked_handoffs

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cayu.applications import CayuApp
    from cayu.runtime.loop_policies import LoopPolicy


class SessionExternalWaitReceipt(_Value):
    session_id: Identifier
    wait: ExternalWaitSnapshot


class SessionExternalWaitAdapter:
    def __init__(
        self,
        app: CayuApp,
        waits: ExternalEventWaits,
        *,
        request_loop_policies: Iterable[LoopPolicy] = (),
    ) -> None:
        from cayu.runtime.loop_policies import validate_loop_policies

        if app.session_store is not waits.store:
            raise ValueError("External session adapter requires the application's exact store.")
        self.app = app
        self.waits = waits
        self._admission = app._admission
        self._request_loop_policies = validate_loop_policies(
            request_loop_policies, field_name="request_loop_policies"
        )

    def _require_task_parking(
        self, task_worker_id: str | None, registration: ExternalWaitRegistration
    ) -> None:
        """Refuse a task-owned wait that no worker could park, continue or stop."""
        hosting = self.app._external_wait_hosting
        if hosting is None or hosting.adapter is not self:
            raise ValueError(
                "External waits in a session a task owns need app.host_external_waits(...): "
                "its worker parks the task and continues it from the wait's service."
            )
        if task_worker_id is None:
            raise ValueError(
                "External waits in a session a task owns need the task worker's claim "
                "(task_worker_id): a task-backed run that no task worker owns can't hand "
                "its task off to wait."
            )
        if not declares_parked_handoffs(self.app.task_store):
            raise ValueError(
                "External waits in a session a task owns park that task; give the "
                "application a task store that declares bound and parked interrupted-task "
                "handoffs, as the built-in stores do."
            )
        try:
            self.waits._authorize(registration.correlation.request, hosting.context, "cancel")
        except PermissionError:
            raise ValueError(
                "External waits in a session a task owns are stopped, when the session is "
                "interrupted or its task ends, by cancelling them as the hosting context; "
                "the access policy must grant that context 'cancel'."
            ) from None

    @external_wait_entrance
    async def recover_to_wait(
        self,
        registration: ExternalWaitRegistration,
        *,
        context: ExternalWaitContext,
        inactive_for_seconds: int = EXTERNAL_RECOVERY_INACTIVITY_S,
    ) -> SessionExternalWaitReceipt:
        """Recover an admitted unfinished turn through the native recovery owner."""
        from cayu.sessions.recovery import IncompleteSessionRecoveryRequest

        if type(inactive_for_seconds) is not int or inactive_for_seconds < 0:
            raise ValueError("External recovery requires a nonnegative inactivity interval.")
        registration = _snapshot(registration, ExternalWaitRegistration)
        context = _snapshot(context, ExternalWaitContext)
        self.waits._authorize(registration.correlation.request, context, "service")
        retained = await self.waits.store._read_external_wait(
            registration.correlation.request.scope,
            registration.correlation.request.correlation_key,
        )
        if retained is None or retained.registration != registration or retained.execution is None:
            raise ExternalWaitUnavailable("External recovery preparation is unavailable.")
        session_id = retained.execution.intent.session_id
        recovery_request = IncompleteSessionRecoveryRequest(
            session_id=session_id, inactive_for_seconds=inactive_for_seconds
        )
        session = await self.waits.store.load(session_id)
        if session is None or session.instance_id != retained.execution.session_instance_id:
            raise ExternalWaitUnavailable("External recovery source incarnation is unavailable.")
        await self.app._require_participant_execution(session, None)
        boundary = _ExternalExecutionToWait(
            self.waits,
            registration,
            context,
            invocation_store=self.app._session_engine.session_store,
            request_loop_policies=self._request_loop_policies,
        )
        if await boundary.recovery_is_settled(retained):
            return SessionExternalWaitReceipt(
                session_id=session_id,
                wait=await self.waits.inspect(registration.correlation, context=context),
            )
        await self.app._session_recovery.recover_incomplete_session(
            recovery_request,
            execution_to_wait=boundary,
        )
        current = await self.waits.store._read_external_wait(
            registration.correlation.request.scope,
            registration.correlation.request.correlation_key,
        )
        if current is None or current.continuation is None:
            raise ExternalWaitUnavailable("External recovery has no durable continuation.")
        native = await load_prepared_continuation(self.waits.store, current.continuation)
        if native is None or (
            native.ticket.state != "WAITING"
            and not (
                native.ticket.state == "ARMING"
                and _ExternalExecutionToWait._cleanup_requested(current)
            )
        ):
            raise ExternalWaitUnavailable("External recovery did not reach its whole-turn wait.")
        return SessionExternalWaitReceipt(
            session_id=session_id,
            wait=await self.waits.inspect(registration.correlation, context=context),
        )

    @external_wait_entrance
    async def reconcile_binding(
        self, registration: ExternalWaitRegistration, *, context: ExternalWaitContext
    ) -> ExternalWaitSnapshot:
        """Repair a committed ticket's missing binding without executing the session."""
        from cayu.runtime._external_wait_receiver import ExternalWaitLatchReceiver

        receiver = ExternalWaitLatchReceiver(self.waits, registration, context)
        await receiver.reconcile_binding()
        return await self.waits.inspect(registration.correlation, context=context)

    @external_wait_entrance
    async def exclude_prepared_execution(
        self, registration: ExternalWaitRegistration, *, context: ExternalWaitContext
    ) -> ExternalWaitSnapshot:
        """Explicitly exclude unbound execution without replacing its elected outcome.

        Native admission must still be excluded or its writer positively released.
        This does not cancel a provider, tool, external job, or bound continuation.
        """
        from cayu.runtime._external_wait_execution_scope import execution_preparation_scope

        registration = _snapshot(registration, ExternalWaitRegistration)
        context = _snapshot(context, ExternalWaitContext)
        correlation = registration.correlation
        self.waits._authorize(correlation.request, context, "service")
        record = await self.waits.store._read_external_wait(
            correlation.request.scope, correlation.request.correlation_key
        )
        if record is None or record.registration != registration or record.execution is None:
            raise ExternalWaitUnavailable("External execution preparation is unavailable.")
        command = self.waits._command(
            "exclude_execution",
            correlation,
            registration=registration,
            execution_intent=record.execution.intent,
            preparation_owner_id=record.execution.preparation_owner_id,
        )
        self.waits._authorize(correlation.request, context, "service")
        with execution_preparation_scope(command):
            await self.waits._mutate(command)
        return await self.waits.inspect(correlation, context=context)

    @external_wait_entrance
    async def retire_execution(
        self,
        registration: ExternalWaitRegistration,
        *,
        operation_key: str,
        context: ExternalWaitContext,
    ) -> ExternalWaitSnapshot:
        """Request exact released-writer retirement without changing the elected outcome.

        The request is durable, not a claim that execution has stopped. Native
        retirement must still arbitrate against consumption and prove release.
        """
        from cayu.runtime._external_wait_settlement import settlement_scope

        registration = _snapshot(registration, ExternalWaitRegistration)
        context = _snapshot(context, ExternalWaitContext)
        correlation = registration.correlation
        self.waits._authorize(correlation.request, context, "service")
        retained = await self.waits.store._read_external_wait(
            correlation.request.scope, correlation.request.correlation_key
        )
        if (
            retained is None
            or retained.registration != registration
            or retained.continuation is None
        ):
            raise ExternalWaitUnavailable("External retirement has no native continuation binding.")
        command = self.waits._command(
            "prepare_retirement",
            correlation,
            registration=registration,
            continuation=retained.continuation,
            operation_key=operation_key,
        )
        self.waits._authorize(correlation.request, context, "service")
        with settlement_scope(command):
            await self.waits._mutate(command)
        await self.service_wait(registration, context=context)
        return await self.waits.inspect(correlation, context=context)

    @external_wait_entrance
    async def run_to_wait(
        self,
        request: RunRequest,
        registration: ExternalWaitRegistration,
        *,
        context: ExternalWaitContext,
    ) -> SessionExternalWaitReceipt:
        request = copy_run_request(request)
        registration = _snapshot(registration, ExternalWaitRegistration)
        context = _snapshot(context, ExternalWaitContext)
        self.waits._authorize(registration.correlation.request, context, "service")
        if request.parent_session_id is not None:
            raise ValueError(CHILD_SESSION_REFUSAL)
        if request.task_id is not None:
            self._require_task_parking(request.task_worker_id, registration)
        source_digest = self.app._work_attempt_execution.work_attempt_source_request_sha256(
            request, kind="initial", without_task_claim=True
        )
        session_id = request.session_id or str(
            uuid5(NAMESPACE_URL, "cayu-external-session:" + external_wait_digest(registration))
        )
        return await self._execute_to_wait(
            request.model_copy(update={"session_id": session_id}),
            registration,
            context=context,
            source_digest=source_digest,
        )

    @external_wait_entrance
    async def resume_to_wait(
        self,
        request: ResumeRequest,
        registration: ExternalWaitRegistration,
        *,
        context: ExternalWaitContext,
    ) -> SessionExternalWaitReceipt:
        """Run one ordinary resumed turn and park its exact completed boundary."""
        request = copy_resume_request(request)
        registration = _snapshot(registration, ExternalWaitRegistration)
        context = _snapshot(context, ExternalWaitContext)
        self.waits._authorize(registration.correlation.request, context, "service")
        session = await self.waits.store.load(request.session_id)
        if session is None:
            raise ExternalWaitUnavailable("External resume source is unavailable.")
        if session.parent_session_id is not None:
            raise ValueError(CHILD_SESSION_REFUSAL)
        linked_task, _ = await self.app._session_tasks.linked_resume_task_id(request)
        if linked_task is not None:
            self._require_task_parking(request.task_worker_id, registration)
        await self.app._require_participant_execution(session, None)
        source_digest = self.app._work_attempt_execution.work_attempt_source_request_sha256(
            request, kind="continuation", without_task_claim=True
        )
        return await self._execute_to_wait(
            request,
            registration,
            context=context,
            source_digest=source_digest,
            session_instance_id=session.instance_id,
            task_id=linked_task,
        )

    async def _execute_to_wait(
        self,
        request: RunRequest | ResumeRequest,
        registration: ExternalWaitRegistration,
        *,
        context: ExternalWaitContext,
        source_digest: str,
        session_instance_id: str | None = None,
        task_id: str | None = None,
    ) -> SessionExternalWaitReceipt:
        # Public entrances copy and validate the request before sharing orchestration.
        # The original digest precedes generated session IDs and must not be rebuilt.
        mode = "run" if isinstance(request, RunRequest) else "resume"
        session_id = request.session_id
        assert session_id is not None
        current = await self.waits.store._read_external_wait(
            registration.correlation.request.scope, registration.correlation.request.correlation_key
        )
        if current is None or current.registration != registration:
            raise ExternalWaitConflict("External session wait is not registered exactly.")
        if current.execution is not None:
            execution = current.execution
            if (
                execution.intent.mode != mode
                or execution.intent.source_request_sha256 != source_digest
                or execution.intent.session_id != session_id
                or (mode == "resume" and execution.session_instance_id != session_instance_id)
            ):
                raise ExternalWaitConflict(
                    "External execution replay changed its request or source."
                )
            native = (
                None
                if current.continuation is None
                else await load_prepared_continuation(self.waits.store, current.continuation)
            )
            if (
                native is None
                or native.preparation != current.continuation
                or native.ticket.state not in {"WAITING", "CONSUMED", "RETIRED"}
            ):
                raise ExternalWaitUnavailable(
                    "External execution requires exact native reconciliation."
                )
            self.waits._authorize(registration.correlation.request, context, "service")
        else:
            boundary = _ExternalExecutionToWait(
                self.waits,
                registration,
                context,
                invocation_store=self.app._session_engine.session_store,
                source_request_sha256=source_digest,
                request_loop_policies=self._request_loop_policies,
                task_id=task_id,
            )
            stream = (
                self.app._run_private(request, execution_to_wait=boundary)
                if isinstance(request, RunRequest)
                else self.app._resume_private(request, execution_to_wait=boundary)
            )
            async with aclosing(stream):
                async for _ in stream:
                    pass
            retained = await self.waits.store._read_external_wait(
                registration.correlation.request.scope,
                registration.correlation.request.correlation_key,
            )
            if retained is None or retained.continuation is None:
                raise ExternalWaitUnavailable(
                    "External execution did not reach its durable wait boundary."
                )
            native = await load_prepared_continuation(self.waits.store, retained.continuation)
            if native is None or native.ticket.state != "WAITING":
                raise ExternalWaitUnavailable(
                    "External execution still requires wait-boundary recovery."
                )
        return SessionExternalWaitReceipt(
            session_id=session_id,
            wait=await self.waits.inspect(registration.correlation, context=context),
        )

    @external_wait_entrance
    async def service_wait(
        self, registration: ExternalWaitRegistration, *, context: ExternalWaitContext
    ) -> SessionExternalWaitReceipt:
        """Continue one elected wait through native fenced resume, or replay its receipt."""
        from cayu.runtime._external_wait_receiver import ExternalWaitLatchReceiver
        from cayu.runtime._external_wait_settlement import settlement_scope
        from cayu.sessions._external_wait_records import elected_external_latch
        from cayu.sessions._session_continuation import ContinuationService

        registration = _snapshot(registration, ExternalWaitRegistration)
        context = _snapshot(context, ExternalWaitContext)
        await self.waits.observe(registration.correlation, context=context)
        retained = await self.waits.store._read_external_wait(
            registration.correlation.request.scope, registration.correlation.request.correlation_key
        )
        if (
            retained is None
            or retained.registration != registration
            or retained.continuation is None
        ):
            raise ExternalWaitUnavailable("External service has no native continuation binding.")
        self.waits._authorize(registration.correlation.request, context, "service")
        if retained.handoff == "settled":
            # Exact receiving evidence has already discharged this handoff. Do
            # not require the source's model material after authorized deletion.
            return SessionExternalWaitReceipt(
                session_id=retained.continuation.intent.session_id,
                wait=await self.waits.inspect(registration.correlation, context=context),
            )
        boundary = _ExternalExecutionToWait(
            self.waits,
            registration,
            context,
            invocation_store=self.app._session_engine.session_store,
            request_loop_policies=self._request_loop_policies,
        )
        receiver = boundary.owner.receiver
        assert isinstance(receiver, ExternalWaitLatchReceiver)
        session_id = retained.continuation.intent.session_id
        retire_requested = retained.execution_retirement is not None
        if retire_requested and not retained.retirement_complete:
            native = await load_prepared_continuation(self.waits.store, retained.continuation)
            if (
                native is not None
                and native.preparation == retained.continuation
                and native.consumption is not None
                and native.consumption.receipt_stage == "prepared"
            ):
                # An already-started native admission may have raced the retirement
                # handoff. Reconcile that exact service through its existing owner;
                # the retirement request is not exclusion or permission to replace it.
                if retained.service is None:
                    raise ExternalWaitUnavailable("External admission lost its exact service.")
                retire_requested = False
        if retire_requested or (
            retained.outcome is not None and retained.outcome.kind in {"cancelled", "unavailable"}
        ):
            if not retained.retirement_complete:
                native = await load_prepared_continuation(self.waits.store, retained.continuation)
                if (
                    native is not None
                    and native.preparation == retained.continuation
                    and native.ticket.state in {"ARMING", "WAITING"}
                    and native.consumption is None
                    and not native.services
                ):
                    # Reconcile writer release through ordinary native interruption
                    # recovery. Its live-owner, claim and effect cleanup gates still
                    # arbitrate takeover; terminal control cannot grant dispatch.
                    # A live execution lease fences takeover; stores without leases
                    # keep the ordinary inactivity grace as their only liveness signal.
                    await self.recover_to_wait(
                        registration,
                        context=context,
                        inactive_for_seconds=0
                        if self.waits.store.supports_session_execution
                        else EXTERNAL_RECOVERY_INACTIVITY_S,
                    )
            await receiver.retire_released(boundary.owner)
        else:
            await self.waits.project(registration, context=context)
            retained = await self.waits.store._read_external_wait(
                registration.correlation.request.scope,
                registration.correlation.request.correlation_key,
            )
            assert retained is not None and retained.continuation is not None
            if retained.service is None:
                from cayu.runtime import _model_completion_publication

                native = await load_prepared_continuation(self.waits.store, retained.continuation)
                if native is None or native.ticket.state != "WAITING":
                    raise ExternalWaitUnavailable(
                        "External service has not reached its parked boundary."
                    )
                checkpoint = await self.app._session_engine.session_store.load_checkpoint(
                    session_id
                )
                pointer = _model_completion_publication.model_step_publication_from_checkpoint(
                    checkpoint
                )
                if pointer is None:
                    raise ExternalWaitUnavailable(
                        "External service lost its completed model boundary."
                    )
                # The exact retained stage must restore its controls before service.
                await boundary.service_task_id(retained, pointer.stage_id)
                latch = elected_external_latch(retained)
                service = ContinuationService(
                    ticket=native.ticket,
                    latch=latch,
                    continuation_id="external-service:" + external_wait_digest(registration),
                    mode="inline",
                    accepted_at=latch.accepted_at,
                )
                command = self.waits._command(
                    "prepare_service",
                    registration.correlation,
                    registration=registration,
                    continuation=retained.continuation,
                    service=service,
                    service_stage_id=pointer.stage_id,
                )
                with settlement_scope(command):
                    retained = await self.waits._mutate(command)
            assert retained.service is not None and retained.projection_json is not None
            self.waits._authorize(registration.correlation.request, context, "service")
            task_id = await boundary.service_task_id(retained, retained.service_stage_id)
            if task_id is not None:
                assert retained.continuation is not None
                native = await load_prepared_continuation(self.waits.store, retained.continuation)
                if native is not None and native.ticket.state == "CONSUMED":
                    # An earlier servicing continued it under the task's claim; only
                    # its handoff is left to settle, which needs no claim.
                    await receiver.reconcile_handoff()
                    return SessionExternalWaitReceipt(
                        session_id=session_id,
                        wait=await self.waits.inspect(registration.correlation, context=context),
                    )
                if await task_ended(self.app, task_id, session_id):
                    if native is not None and native.consumption is not None:
                        raise ended_during_delivery(session_id, task_id)
                    return await self._retire_for_ended_task(registration, task_id, context)
            await boundary.owner.latch(retained.service.latch)
            if task_id is None:
                await boundary.owner.service(
                    self.app,
                    await boundary.service_request(retained, retained.service_stage_id),
                    retained.service,
                )
            else:
                await service_under_task(
                    self,
                    boundary,
                    retained,
                    session_id=session_id,
                    stage_id=retained.service_stage_id,
                )
            await receiver.reconcile_handoff()
        return SessionExternalWaitReceipt(
            session_id=session_id,
            wait=await self.waits.inspect(registration.correlation, context=context),
        )

    async def _waiting_turn_task_id(
        self, retained: ExternalWaitRecord, *, context: ExternalWaitContext
    ) -> str | None:
        """The task a parked turn ran under, from its exact completed model stage."""
        from cayu.runtime import _model_completion_publication

        if (
            retained.registration is None
            or retained.continuation is None
            or retained.execution is None
        ):
            return None
        stage_id = retained.service_stage_id
        if stage_id is None:
            native = await load_prepared_continuation(self.waits.store, retained.continuation)
            if native is None or native.ticket.state != "WAITING":
                return None
            pointer = _model_completion_publication.model_step_publication_from_checkpoint(
                await self.app._session_engine.session_store.load_checkpoint(
                    retained.continuation.intent.session_id
                )
            )
            if pointer is None:
                return None
            stage_id = pointer.stage_id
        boundary = _ExternalExecutionToWait(
            self.waits,
            retained.registration,
            context,
            invocation_store=self.app._session_engine.session_store,
        )
        return await boundary.service_task_id(retained, stage_id)

    async def _retire_for_ended_task(
        self,
        registration: ExternalWaitRegistration,
        task_id: str,
        context: ExternalWaitContext,
    ) -> SessionExternalWaitReceipt:
        """A wait never continues a session without the task it waited under."""
        await self.retire_execution(
            registration, operation_key=f"task-ended:{task_id}", context=context
        )
        retained = await self.waits.store._read_external_wait(
            registration.correlation.request.scope, registration.correlation.request.correlation_key
        )
        assert retained is not None and retained.continuation is not None
        return SessionExternalWaitReceipt(
            session_id=retained.continuation.intent.session_id,
            wait=await self.waits.inspect(registration.correlation, context=context),
        )
