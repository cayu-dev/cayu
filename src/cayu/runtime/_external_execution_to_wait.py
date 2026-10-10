"""Explicit external whole-turn boundary, called with native runtime authority."""

from uuid import uuid4

from cayu.collaboration._capabilities import CapabilityDescriptor
from cayu.collaboration._contracts import OwnerRef
from cayu.external_waits import ExternalEventWaits, ExternalWaitContext, _snapshot
from cayu.runtime._external_wait_binding import binding_scope, load_prepared_continuation
from cayu.runtime._external_wait_execution_scope import execution_preparation_scope
from cayu.runtime._external_wait_observation import external_wait_tracker
from cayu.runtime._external_wait_receiver import ExternalWaitLatchReceiver
from cayu.runtime._session_continuation_owner import LATCH_FAMILY, SessionContinuationOwner
from cayu.sessions._external_wait_records import external_continuation_intent, external_wait_owner
from cayu.sessions.external_waits import (
    ExternalWaitConflict,
    ExternalWaitExecutionIntent,
    ExternalWaitRecord,
    ExternalWaitRegistration,
    ExternalWaitUnavailable,
)


class _ExternalExecutionToWait:
    def __init__(
        self,
        waits: ExternalEventWaits,
        registration: ExternalWaitRegistration,
        context: ExternalWaitContext,
        *,
        invocation_store,
        source_request_sha256: str | None = None,
        request_loop_policies=(),
    ):
        self.waits = waits
        self.registration = _snapshot(registration, ExternalWaitRegistration)
        self.context = _snapshot(context, ExternalWaitContext)
        self.source_request_sha256 = source_request_sha256
        self.request_loop_policies = request_loop_policies
        self._invocation_store = invocation_store
        self._preparation_owner_id = str(uuid4())
        # Cache only receiving evidence within this one invocation. Authorization,
        # cancellation and execution identity are still rechecked on every call;
        # parking and every dispatch retain their native writer/claim guards.
        self._prepared_invocation = None
        self._prepared = None
        scope = registration.correlation.request.scope
        self.owner = SessionContinuationOwner(
            store=waits.store,
            owner=OwnerRef(
                application_scope=scope.application_scope,
                owner_id="external-session-runtime",
                incarnation=str(scope.generation),
            ),
            receiver=ExternalWaitLatchReceiver(waits, self.registration, self.context),
            receiver_capability=CapabilityDescriptor(
                owner=external_wait_owner(self.registration),
                mutations=(),
                readbacks=(LATCH_FAMILY,),
            ),
            redactor=waits.redactor,
            track=external_wait_tracker(),
        )

    async def prepare_initial(self, request, profile, *, request_sha256):
        self.waits._authorize(self.registration.correlation.request, self.context, "service")
        if (
            request.session_id is None
            or request.parent_session_id is not None
            or request.task_id is not None
        ):
            raise ValueError("External execution requires an ordinary root session identity.")
        command = self.waits._command(
            "prepare_execution",
            self.registration.correlation,
            registration=self.registration,
            preparation_owner_id=self._preparation_owner_id,
            execution_intent=ExternalWaitExecutionIntent(
                mode="run",
                request_sha256=request_sha256,
                source_request_sha256=self.source_request_sha256,
                profile_sha256=profile.fingerprint,
                session_id=request.session_id,
            ),
        )
        with execution_preparation_scope(command):
            retained = await self.waits._mutate(command)
        assert retained.execution is not None
        if retained.execution.preparation_owner_id != self._preparation_owner_id:
            raise ExternalWaitUnavailable(
                "External execution preparation belongs to another worker; reconcile it."
            )
        return retained.execution

    async def recovery_is_settled(self, retained: ExternalWaitRecord) -> bool:
        """Read positive completion/release evidence; this never authorizes takeover."""
        from cayu.runtime._checkpoint_store import (
            load_runtime_session_checkpoint_snapshot,
            runtime_checkpoint_session_store,
        )
        from cayu.sessions._session_continuation import (
            ContinuationConflict,
            continuation_writer_frontier,
        )
        from cayu.sessions._session_continuation_store import require_released_wait_invocation
        from cayu.sessions.base import _continuation_writer_was_released

        if retained.continuation is None:
            return False
        native = await load_prepared_continuation(self.waits.store, retained.continuation)
        if native is None or native.preparation != retained.continuation:
            return False
        if native.ticket.state in {"CONSUMED", "RETIRED"}:
            return True
        if native.ticket.state != "WAITING" and not (
            native.ticket.state == "ARMING" and self._cleanup_requested(retained)
        ):
            return False
        current, checkpoint = await load_runtime_session_checkpoint_snapshot(
            runtime_checkpoint_session_store(self.waits.store), native.ticket.session_id
        )
        writer, _ = continuation_writer_frontier(native)
        if (
            current.instance_id != native.ticket.session_instance_id
            or current.run_epoch != writer + 1
            or not _continuation_writer_was_released(current, checkpoint)
        ):
            return False
        try:
            require_released_wait_invocation(
                native.ticket,
                checkpoint,
                permit_operation=None,
                permit_commitment=None,
                record=native,
            )
        except ContinuationConflict:
            return False
        return True

    async def inspect_recovery(self, session):
        """Authenticate the retained wait before native recovery can acquire an epoch."""
        from cayu.runtime._continuation_recovery import require_resolved_native_pause
        from cayu.sessions._execution_profile_checkpoint import (
            execution_profile_from_session_metadata,
        )

        self.waits._authorize(self.registration.correlation.request, self.context, "service")
        retained = await self.waits.store._read_external_wait(
            self.registration.correlation.request.scope,
            self.registration.correlation.request.correlation_key,
        )
        profile = execution_profile_from_session_metadata(session.metadata)
        if (
            retained is None
            or retained.registration != self.registration
            or retained.execution is None
            or retained.execution.intent.session_id != session.id
            or retained.execution.session_instance_id != session.instance_id
            or profile is None
            or profile.fingerprint != retained.execution.intent.profile_sha256
            or retained.handoff != "pending"
            or retained.continuation is None
            or retained.execution_excluded
        ):
            raise ExternalWaitUnavailable("External recovery lost its exact retained execution.")
        native = await load_prepared_continuation(self.waits.store, retained.continuation)
        if (
            native is None
            or native.preparation != retained.continuation
            or native.ticket.state not in {"ARMING", "WAITING"}
            or native.consumption is not None
            or native.services
        ):
            raise ExternalWaitUnavailable("External recovery requires an unconsumed native wait.")
        cleanup_requested = self._cleanup_requested(retained)
        if not cleanup_requested:
            require_resolved_native_pause(await self._invocation_store.load_checkpoint(session.id))
        self.waits._authorize(self.registration.correlation.request, self.context, "service")
        return native, cleanup_requested

    @staticmethod
    def _cleanup_requested(retained):
        return retained.execution_retirement is not None or (
            retained.outcome is not None and retained.outcome.kind in {"cancelled", "unavailable"}
        )

    async def recovery_cleanup_requested(self, invocation):
        """Read a terminal control for this already authenticated recovered writer."""
        invocation.require_runtime_authority()
        self.waits._authorize(self.registration.correlation.request, self.context, "service")
        retained = await self.waits.store._read_external_wait(
            self.registration.correlation.request.scope,
            self.registration.correlation.request.correlation_key,
        )
        if (
            invocation.recovery_claim_id is None
            or retained is None
            or retained.registration != self.registration
            or retained.execution is None
            or retained.execution.intent.session_id != invocation.binding.session_id
            or retained.execution.session_instance_id != invocation.binding.session_instance_id
            or retained.execution.interaction_id != invocation.binding.interaction_id
            or retained.execution.intent.profile_sha256 != invocation.profile.fingerprint
        ):
            raise ExternalWaitUnavailable("External cleanup lost its recovered execution identity.")
        self.waits._authorize(self.registration.correlation.request, self.context, "service")
        return self._cleanup_requested(retained)

    async def require_recovered_result(self, invocation, boundary):
        """Bind native completion replay to this wait's genuine recovered writer."""
        from cayu.runtime._model_completion_contracts import (
            model_completion_recovery_context_from_stage,
        )

        native = await self.prepare(invocation)
        stage, pointer = boundary.completed_stage, boundary.pointer
        semantics = None if stage is None else model_completion_recovery_context_from_stage(stage)
        if (
            native.recovery_writer is None
            or stage is None
            or pointer is None
            or stage.stage_id != pointer.stage_id
            or stage.source_run_epoch > invocation.binding.run_epoch
            or stage.source_run_epoch < native.ticket.writer_generation
            or semantics is None
            or semantics.interaction_id != invocation.binding.interaction_id
            or semantics.execution_profile_fingerprint != invocation.profile.fingerprint
            or semantics.task_id is not None
            or (
                pointer.tool_round_id is None
                and boundary.transcript_cursor != pointer.transcript_end_cursor
            )
            or (
                pointer.tool_round_id is not None
                and boundary.pending_tool_round is not None
                and boundary.transcript_cursor != pointer.transcript_end_cursor
            )
            or (
                pointer.tool_round_id is not None
                and boundary.pending_tool_round is None
                and (
                    boundary.closed_tool_receipt is None
                    or boundary.transcript_cursor
                    != pointer.transcript_end_cursor
                    + (2 if pointer.assistant_message_deferred else 1)
                )
            )
        ):
            raise ExternalWaitUnavailable("External recovery has no exact completed model result.")
        return semantics

    async def service_request(self, retained, stage_id):
        """Restore controls from the exact immutable stage retained before service."""
        from cayu.messages import Message
        from cayu.runtime._model_completion_contracts import (
            model_completion_recovery_context_from_stage,
        )
        from cayu.sessions.requests import ResumeRequest

        if retained.execution is None or retained.projection_json is None or stage_id is None:
            raise ExternalWaitUnavailable("External service has no retained execution controls.")
        execution = retained.execution
        stage = await self.waits.store.load_model_completion_stage(
            execution.intent.session_id, stage_id
        )
        semantics = None if stage is None else model_completion_recovery_context_from_stage(stage)
        if (
            stage is None
            or stage.stage_id != stage_id
            or stage.state != "completed"
            or stage.purpose != "assistant-turn"
            or stage.session_id != execution.intent.session_id
            or semantics is None
            or semantics.interaction_id != execution.interaction_id
            or semantics.execution_profile_fingerprint != execution.intent.profile_sha256
            or semantics.task_id is not None
        ):
            raise ExternalWaitUnavailable("External service execution controls conflict.")
        return ResumeRequest(
            session_id=execution.intent.session_id,
            messages=[Message.text("user", retained.projection_json)],
            max_steps=semantics.max_steps,
            limits=semantics.limits,
            budget_limits=semantics.budget_limits,
            retry_policy=semantics.retry_policy,
            structured_output=semantics.structured_output,
            tool_completion=semantics.tool_completion,
            thinking=semantics.thinking,
            metadata=semantics.request_metadata,
            loop_policies=self.request_loop_policies,
        )

    async def publish_structured_event(self, invocation, writer, event):
        """Reuse exact post-model validation evidence during native recovery."""
        from cayu._validation import canonical_durable_json_bytes
        from cayu.events import EventType, copy_event
        from cayu.sessions.event_queries import EventQuery

        invocation.require_runtime_authority()
        if invocation.recovery_claim_id is None:
            return await writer.emit(event)
        self.waits._authorize(self.registration.correlation.request, self.context, "service")
        expected = writer.prepare(event)
        step_id = expected.payload.get("model_step_id")
        attempt_id = expected.payload.get("model_attempt_id")
        if (
            expected.type
            not in {
                EventType.STRUCTURED_OUTPUT_VALIDATING,
                EventType.STRUCTURED_OUTPUT_VALIDATED,
                EventType.STRUCTURED_OUTPUT_FAILED,
                EventType.STRUCTURED_OUTPUT_RETRY,
            }
            or type(step_id) is not str
            or type(attempt_id) is not str
            or expected.session_id != invocation.binding.session_id
        ):
            raise ExternalWaitUnavailable("External structured replay lost its model identity.")
        records = await self.waits.store.query_events(
            EventQuery(
                session_id=expected.session_id,
                model_step_id=step_id,
                event_type=expected.type,
                limit=2,
            )
        )
        if not records:
            return await writer.emit(event)
        actual = records[0].event
        if (
            len(records) != 1
            or actual.session_id != expected.session_id
            or actual.type != expected.type
            or actual.interaction_id != invocation.binding.interaction_id
            or actual.agent_name != expected.agent_name
            or actual.environment_name != expected.environment_name
            or canonical_durable_json_bytes(actual.payload, "structured output evidence")
            != canonical_durable_json_bytes(expected.payload, "structured output evidence")
        ):
            raise ExternalWaitUnavailable(
                "External structured replay conflicts with durable evidence."
            )
        await writer.fan_out_persisted([actual])
        return copy_event(actual)

    async def prepare(self, invocation):
        return await self._prepare(invocation, cleanup_only=False)

    async def attach_cleanup_writer(self, invocation):
        if not await self.recovery_cleanup_requested(invocation):
            raise ExternalWaitUnavailable("External recovery has no terminal cleanup request.")
        return await self._prepare(invocation, cleanup_only=True)

    async def _prepare(self, invocation, *, cleanup_only):
        invocation.require_runtime_authority()
        self.waits._authorize(self.registration.correlation.request, self.context, "service")
        retained = await self.waits.store._read_external_wait(
            self.registration.correlation.request.scope,
            self.registration.correlation.request.correlation_key,
        )
        if (
            retained is None
            or retained.registration != self.registration
            or retained.execution is None
            or retained.execution_excluded
        ):
            raise ExternalWaitUnavailable("External execution preparation is unavailable.")
        execution = retained.execution
        binding = invocation.binding
        if (binding.session_id, binding.session_instance_id, binding.interaction_id) != (
            execution.intent.session_id,
            execution.session_instance_id,
            execution.interaction_id,
        ):
            raise PermissionError(
                "External execution does not match its retained runtime identity."
            )
        if invocation.profile.fingerprint != execution.intent.profile_sha256:
            raise ExternalWaitConflict("External recovery changed the retained execution profile.")
        if self._cleanup_requested(retained):
            if (
                not cleanup_only
                or invocation.recovery_claim_id is None
                or retained.continuation is None
            ):
                raise ExternalWaitUnavailable("External cleanup cannot prepare new execution.")
            native = await load_prepared_continuation(self.waits.store, retained.continuation)
            if (
                native is None
                or native.preparation != retained.continuation
                or native.ticket.state not in {"ARMING", "WAITING"}
                or native.consumption is not None
                or native.services
            ):
                raise ExternalWaitUnavailable("External cleanup lost its native wait boundary.")
        if (
            self._prepared_invocation is invocation
            and self._prepared is not None
            and retained.continuation == self._prepared.preparation
        ):
            return self._prepared
        if invocation.recovery_claim_id is not None and retained.continuation is not None:
            prepared = await self.owner.recover_writer(
                retained.continuation.intent, invocation=invocation
            )
        else:
            prepared = await self.owner.prepare(
                external_continuation_intent(self.registration), invocation=invocation
            )
            command = self.waits._command(
                "bind",
                self.registration.correlation,
                registration=self.registration,
                continuation=prepared.preparation,
            )
            with binding_scope(command, invocation):
                await self.waits._mutate(command)
        self._prepared_invocation = invocation
        self._prepared = prepared
        return prepared

    async def admit_resume(self, command, invocation):
        from cayu.runtime._external_wait_admission import (
            external_resume_admission,
            external_resume_preparation,
        )
        from cayu.sessions._invocation_lifecycle import _invocation_lifecycle_command_sha256
        from cayu.sessions.external_waits import external_wait_digest

        invocation.require_runtime_authority()
        self.waits._authorize(self.registration.correlation.request, self.context, "service")
        digest = _invocation_lifecycle_command_sha256(command)
        mutation = self.waits._command(
            "prepare_execution",
            self.registration.correlation,
            registration=self.registration,
            preparation_owner_id=self._preparation_owner_id,
            execution_intent=ExternalWaitExecutionIntent(
                mode="resume",
                request_sha256=digest,
                source_request_sha256=self.source_request_sha256,
                profile_sha256=command.target_active_profile.profile.fingerprint,
                session_id=command.session_id,
                expected_session_instance_id=command.expected_session_instance_id,
                expected_run_epoch=command.expected_run_epoch,
                admission_sha256=digest,
            ),
        )
        with external_resume_preparation(command), execution_preparation_scope(mutation):
            retained = await self.waits._mutate(mutation)
        execution = retained.execution
        assert execution is not None
        if execution.preparation_owner_id != self._preparation_owner_id:
            raise ExternalWaitUnavailable(
                "External resume belongs to another worker; reconcile it."
            )
        self.waits._authorize(self.registration.correlation.request, self.context, "service")
        with external_resume_admission(self.registration, execution, command):
            result = await self.waits._observe_operation(
                lambda: self._invocation_store.apply_invocation_lifecycle_command(command),
                key=("external-resume", external_wait_digest(execution)),
                expectation=digest.encode(),
            )
        return self._receive_admission(result)

    async def create_invocation(self, command, execution):
        from cayu.runtime._external_wait_creation import external_creation_scope
        from cayu.sessions._invocation_lifecycle import _invocation_lifecycle_command_sha256
        from cayu.sessions.external_waits import external_wait_digest

        self.waits._authorize(self.registration.correlation.request, self.context, "service")
        with external_creation_scope(self.registration, execution):
            result = await self.waits._observe_operation(
                lambda: self._invocation_store.apply_invocation_lifecycle_command(command),
                key=("external-create", external_wait_digest(execution)),
                expectation=_invocation_lifecycle_command_sha256(command).encode(),
            )
        return self._receive_admission(result)

    @staticmethod
    def _receive_admission(result):
        from cayu.sessions._invocation_lifecycle import InvocationMutationResult
        from cayu.sessions.base import _activate_session_run_fence

        if type(result) is not InvocationMutationResult:
            raise RuntimeError("External execution received an invalid native admission.")
        # Admission runs in an owned task so cancellation cannot abandon its
        # mutation. Its context variables do not propagate back to this caller.
        # Restore the normal local fence only after positive native admission.
        _activate_session_run_fence(result.session)
        return result

    async def park(self, invocation):
        prepared = await self.prepare(invocation)
        if prepared.recovery_writer is not None:
            self._prepared = await self.owner.park_recovered(prepared.ticket, invocation=invocation)
        else:
            self._prepared = await self.owner.park(prepared.ticket, invocation=invocation)
