"""Session authority and terminal settlement for durable queued dispatch."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, Protocol

from cayu.context.structured_output import (
    StructuredOutputStrategy,
    _require_native_structured_output_support,
)
from cayu.events import Event, EventType
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
    unavailable_execution_profile_components,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._checkpoint_store import load_runtime_session_checkpoint_snapshot
from cayu.runtime._delegated_event_stream import _close_delegated_event_stream
from cayu.runtime._durable_subagents import durable_subagent_worker_incompatible
from cayu.runtime._task_store_operation_boundary import raise_task_store_operation_failure
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_is_released,
    active_invocation_execution_profile_matches_session_epoch,
    execution_profile_from_session_metadata,
)
from cayu.sessions._terminal_evidence import (
    SESSION_RUN_OPERATION_ID_PAYLOAD_KEY,
    TERMINAL_EVENT_TYPES,
)
from cayu.sessions.base import (
    ActiveModelCompletionStage,
    EventQuery,
    QueuedDispatchTerminalReceipt,
    QueuedDispatchTerminalReceiptQuery,
    RunRequest,
    Session,
    SessionRunFenced,
    SessionStatus,
    SessionStore,
    _checkpoint_after_queued_dispatch_acknowledgement,
    _queued_dispatch_session_instance_fingerprint,
    _queued_dispatch_terminal_receipts_from_checkpoint,
    _session_run_operation_from_checkpoint,
    session_fork_profile_relationship,
)
from cayu.tasks.contracts import TaskCompletionDecisionRequired
from cayu.tasks.dispatch import (
    DispatchRequest,
    DispatchStatus,
    _copy_queued_dispatch_envelope,
    _new_queued_dispatch_envelope,
    _QueuedDispatchAuthorityRejected,
    _QueuedDispatchEnvelope,
    _QueuedDispatchSettlement,
    _QueuedDispatchSettlementState,
    copy_dispatch_request,
)
from cayu.vaults.redaction import SecretRedactor


class QueuedDispatchEngine(Protocol):
    """Session profile resolution and contracted-task admission."""

    async def _verifier_aware_task_execution_outcome(
        self,
        task_id: str | None,
        *,
        session_id: str | None = None,
        admit_session: bool = True,
        allow_missing_task: bool = False,
    ) -> tuple[bool, BaseException | None]: ...

    def _queued_dispatch_required_profile(
        self,
        *,
        session: Session,
        source_profile: ExecutionProfileIdentity,
        request: DispatchRequest,
    ) -> ExecutionProfileIdentity: ...


class QueuedDispatchSubagents(Protocol):
    """Prepared-child authority retained by the durable subagent owner."""

    def prepare_queued_child_run(
        self,
        *,
        envelope: _QueuedDispatchEnvelope,
        session: Session,
        checkpoint: dict[str, Any] | None,
    ) -> RunRequest: ...

    async def require_prepared_subagent_parent_authority(
        self,
        envelope: _QueuedDispatchEnvelope,
    ) -> None: ...


class QueuedDispatchExecution(Protocol):
    """Execute an admitted request under the queue's exact frozen authority."""

    def __call__(
        self,
        request: DispatchRequest,
        *,
        store_resolved_session_id: str | None = None,
        source_execution_profile: ExecutionProfileIdentity | None = None,
        required_execution_profile: ExecutionProfileIdentity | None = None,
        required_session_instance_fingerprint: str | None = None,
        dispatch_operation_id: str | None = None,
        dispatch_terminal_event_id: str | None = None,
        queue_task_id: str | None = None,
    ) -> AsyncGenerator[Event, None]: ...


class QueuedDispatchCoordinator:
    """Own the session side of queued preparation, replay and acknowledgement.

    Dispatchers own task leases and durable queue outcomes. Stores own atomic
    checkpoint mutations. This coordinator binds those outcomes to exact session,
    invocation, profile and terminal-event evidence before releasing retention.
    Callers apply admission policy around execution; collaborators retain session
    execution, public identity projection and prepared-child authority.
    """

    def __init__(
        self,
        *,
        get_session_store: Callable[[], SessionStore],
        runtime_session_store: SessionStore,
        engine: QueuedDispatchEngine,
        subagents: QueuedDispatchSubagents,
        redactor: SecretRedactor,
        resolve_session: Callable[[str], Awaitable[tuple[str, str | None]]],
        get_agent: Callable[[str], runtime_records.RegisteredAgentState],
        get_provider: Callable[[str | None], runtime_records.RegisteredProvider],
        get_environment: Callable[[str | None], runtime_records.RegisteredEnvironment | None],
        redact_request: Callable[[DispatchRequest], DispatchRequest],
        project_event: Callable[[Event], Awaitable[Event]],
        load_model_completion: Callable[[Session], Awaitable[ActiveModelCompletionStage | None]],
        run: Callable[[RunRequest], AsyncGenerator[Event, None]],
        dispatch: QueuedDispatchExecution,
    ) -> None:
        self._get_session_store = get_session_store
        self._runtime_store = runtime_session_store
        self._engine = engine
        self._subagents = subagents
        self._redactor = redactor
        self._resolve_session = resolve_session
        self._get_agent = get_agent
        self._get_provider = get_provider
        self._get_environment = get_environment
        self._redact_request = redact_request
        self._project_event = project_event
        self._load_model_completion = load_model_completion
        self._run = run
        self._dispatch = dispatch

    @property
    def session_store(self) -> SessionStore:
        return self._get_session_store()

    async def _load_session_snapshot(
        self,
        session_id: str,
    ) -> tuple[Session, dict[str, Any] | None]:
        """Load session and checkpoint authority under one store-owned boundary."""

        return await load_runtime_session_checkpoint_snapshot(
            self._runtime_store,
            session_id,
        )

    async def _load_terminal_event(
        self,
        *,
        private_session_id: str,
        envelope: _QueuedDispatchEnvelope,
    ) -> Event | None:
        """Load and validate the exact terminal event bound to an envelope."""

        records = await self.session_store.query_events(
            EventQuery(
                session_id=private_session_id,
                event_id=envelope.terminal_event_id,
                limit=2,
            )
        )
        if not records:
            return None
        if len(records) != 1:
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch terminal evidence is duplicated."
            )
        terminal_event = records[0].event
        if (
            terminal_event.type not in TERMINAL_EVENT_TYPES
            or terminal_event.payload.get(SESSION_RUN_OPERATION_ID_PAYLOAD_KEY)
            != envelope.dispatch_operation_id
        ):
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch terminal evidence conflicts with its envelope."
            )
        return terminal_event

    async def prepare(
        self,
        request: DispatchRequest,
        *,
        queue_task_id: str,
    ) -> _QueuedDispatchEnvelope:
        """Freeze runtime-owned profile authority before a queue task is published."""

        if type(request) is not DispatchRequest:
            raise TypeError("Queued dispatch preparation requires a DispatchRequest.")
        request = copy_dispatch_request(request)
        private_session_id, _ = await self._resolve_session(request.session_id)
        (
            contract_rejected,
            admission_failure,
        ) = await self._engine._verifier_aware_task_execution_outcome(
            request.task_id,
            session_id=private_session_id,
            admit_session=False,
        )
        if admission_failure is not None:
            del private_session_id, request
            raise_task_store_operation_failure(admission_failure)
        if contract_rejected:
            del private_session_id, request
            raise TaskCompletionDecisionRequired(
                "Contracted tasks require the verifier-aware execution entrance."
            ) from None
        session, checkpoint = await self._load_session_snapshot(private_session_id)
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        pending_tool_round = pending_round_reader.pending_tool_round_from_checkpoint(
            checkpoint,
            redactor=self._redactor,
            consume_on_rejection=True,
            runtime_session=session,
        )
        pending_model_completion = await self._load_model_completion(session)
        continues_active_invocation = (
            pending_tool_round is not None or pending_model_completion is not None
        )
        if active_profile is not None:
            if not active_invocation_execution_profile_matches_session_epoch(
                active_profile,
                session_id=session.id,
                run_epoch=session.run_epoch,
            ):
                raise RuntimeError(
                    "Active invocation execution profile conflicts with the session epoch."
                )
            if (
                active_invocation_execution_profile_is_released(
                    active_profile,
                    session_id=session.id,
                    run_epoch=session.run_epoch,
                )
                and not continues_active_invocation
            ):
                source_profile = execution_profile_from_session_metadata(session.metadata)
            else:
                source_profile = active_profile.profile
        else:
            if continues_active_invocation:
                raise RuntimeError(
                    "Queued dispatch recovery has no durable active invocation execution profile."
                )
            if session.status in {SessionStatus.RUNNING, SessionStatus.INTERRUPTING}:
                raise RuntimeError(
                    "A live session has no durable active invocation execution profile."
                )
            source_profile = execution_profile_from_session_metadata(session.metadata)
        durable_request = self._redact_request(request)
        if durable_request.failover is not None or source_profile.model_failover is not None:
            self.session_store._require_model_failover_stage_protocol()
        target_changed = durable_request.target is not None and (
            durable_request.target.provider_name != session.provider_name
            or durable_request.target.model != session.model
        )
        if target_changed:
            if continues_active_invocation:
                raise RuntimeError(
                    "A queued dispatch model target cannot change while model or tool "
                    "recovery is pending."
                )
            source_profile = execution_profile_from_session_metadata(session.metadata)
        required_profile = source_profile
        if not continues_active_invocation:
            required_profile = self._engine._queued_dispatch_required_profile(
                session=session,
                source_profile=source_profile,
                request=durable_request,
            )
        unavailable = set(unavailable_execution_profile_components(source_profile))
        unavailable.update(unavailable_execution_profile_components(required_profile))
        if unavailable:
            raise RuntimeError(
                "Queued dispatch requires an execution profile with available components: "
                + ", ".join(
                    component.value
                    for component in sorted(unavailable, key=lambda item: item.value)
                )
            )
        if (
            durable_request.structured_output is not None
            and durable_request.structured_output.strategy is StructuredOutputStrategy.NATIVE
        ):
            registered_provider = self._get_provider(
                durable_request.target.provider_name
                if durable_request.target is not None
                else session.provider_name
            )
            _require_native_structured_output_support(
                durable_request.structured_output,
                provider_name=registered_provider.name,
                provider=registered_provider.provider,
            )
        (
            contract_rejected,
            admission_failure,
        ) = await self._engine._verifier_aware_task_execution_outcome(
            request.task_id,
            session_id=private_session_id,
        )
        if admission_failure is not None:
            del durable_request, private_session_id, request
            raise_task_store_operation_failure(admission_failure)
        if contract_rejected:
            del durable_request, private_session_id, request
            raise TaskCompletionDecisionRequired(
                "Contracted tasks require the verifier-aware execution entrance."
            ) from None
        fork_relationship = session_fork_profile_relationship(session)
        return _new_queued_dispatch_envelope(
            queue_task_id=queue_task_id,
            request=durable_request,
            session_instance_fingerprint=(_queued_dispatch_session_instance_fingerprint(session)),
            source_profile=source_profile,
            required_profile=required_profile,
            exact_fork_source_state_sha256=(
                None if fork_relationship is None else fork_relationship.source_state_sha256
            ),
        )

    async def requests_match(
        self,
        existing: DispatchRequest,
        candidate: DispatchRequest,
    ) -> bool:
        """Compare retries by private session authority, not rotating public aliases."""

        existing = copy_dispatch_request(existing)
        candidate = copy_dispatch_request(candidate)
        existing_session_id, _ = await self._resolve_session(existing.session_id)
        candidate_session_id, _ = await self._resolve_session(candidate.session_id)
        if existing_session_id != candidate_session_id:
            return False
        comparison_session_id = "cayu-equivalent-session-authority"
        return existing.model_copy(
            update={"session_id": comparison_session_id},
            deep=True,
        ) == candidate.model_copy(
            update={"session_id": comparison_session_id},
            deep=True,
        )

    async def acknowledge(
        self,
        envelope: _QueuedDispatchEnvelope,
        *,
        dispatch_status: DispatchStatus,
        receipt: QueuedDispatchTerminalReceipt | None = None,
    ) -> None:
        """Release exact terminal retention after the queue outcome is durable."""

        envelope = _copy_queued_dispatch_envelope(envelope)
        if type(dispatch_status) is not DispatchStatus:
            raise TypeError("Queued dispatch acknowledgement status has an invalid type.")
        settlement = await self.settlement_state(envelope)
        if settlement.state is _QueuedDispatchSettlementState.NOT_ADMITTED:
            return
        if settlement.state is not _QueuedDispatchSettlementState.TERMINAL_EVIDENCE_DURABLE:
            raise RuntimeError(
                "Queued dispatch terminal evidence is not durable enough to acknowledge."
            )
        if settlement.terminal_status is not dispatch_status:
            raise RuntimeError(
                "Queued dispatch task status conflicts with its exact terminal event."
            )
        private_session_id, _ = await self._resolve_session(envelope.request.session_id)
        if receipt is not None:
            if type(receipt) is not QueuedDispatchTerminalReceipt:
                raise TypeError("Queued dispatch acknowledgement receipt has an invalid type.")
            receipt = QueuedDispatchTerminalReceipt(
                session_id=receipt.session_id,
                queue_task_id=receipt.queue_task_id,
                operation_id=receipt.operation_id,
                terminal_event_id=receipt.terminal_event_id,
            )
            if (
                receipt.session_id != private_session_id
                or receipt.queue_task_id != envelope.queue_task_id
                or receipt.operation_id != envelope.dispatch_operation_id
                or receipt.terminal_event_id != envelope.terminal_event_id
            ):
                raise RuntimeError(
                    "Queued dispatch acknowledgement receipt conflicts with its envelope."
                )

        def acknowledge(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any] | None:
            return _checkpoint_after_queued_dispatch_acknowledgement(
                checkpoint,
                queue_task_id=envelope.queue_task_id,
                operation_id=envelope.dispatch_operation_id,
                terminal_event_id=envelope.terminal_event_id,
            )

        # A terminal session writer may finish its final checkpoint publication just
        # after the task terminalization acknowledgement starts.  Repeat this exact,
        # idempotent removal with read-after-write confirmation so that a late writer
        # cannot leave the queue-owned receipt stranded after this method returns.
        for _attempt in range(3):
            await self.session_store.transform_checkpoint(private_session_id, acknowledge)
            checkpoint = await self.session_store.load_checkpoint(private_session_id)
            receipts = _queued_dispatch_terminal_receipts_from_checkpoint(checkpoint)
            if envelope.dispatch_operation_id not in receipts:
                return
        raise RuntimeError(
            "Queued dispatch terminal receipt remained after bounded acknowledgement retry."
        )

    async def list_terminal_receipts(
        self,
        query: QueuedDispatchTerminalReceiptQuery,
    ) -> list[QueuedDispatchTerminalReceipt]:
        """Delegate bounded restart discovery to the durable session store."""

        return await self.session_store.list_queued_dispatch_terminal_receipts(query)

    @staticmethod
    def _terminal_ownership_released(
        *,
        session: Session,
        checkpoint: dict[str, Any] | None,
        envelope: _QueuedDispatchEnvelope,
    ) -> bool:
        """Validate and classify the exact invocation generation behind a terminal event."""

        try:
            run_operation = _session_run_operation_from_checkpoint(checkpoint)
            receipt = _queued_dispatch_terminal_receipts_from_checkpoint(checkpoint).get(
                envelope.dispatch_operation_id
            )
        except (TypeError, ValueError) as exc:
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch terminal ownership evidence is malformed."
            ) from exc
        if receipt is not None and (
            receipt.queue_task_id != envelope.queue_task_id
            or receipt.terminal_event_id != envelope.terminal_event_id
        ):
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch terminal receipt identity conflicts."
            )
        terminal_run_epoch = None if receipt is None else receipt.run_epoch
        if run_operation is not None and run_operation.operation_id == (
            envelope.dispatch_operation_id
        ):
            if (
                run_operation.queue_task_id != envelope.queue_task_id
                or run_operation.terminal_event_id != envelope.terminal_event_id
            ):
                raise _QueuedDispatchAuthorityRejected(
                    "Queued dispatch run operation identity conflicts."
                )
            if terminal_run_epoch not in {None, run_operation.run_epoch}:
                raise _QueuedDispatchAuthorityRejected(
                    "Queued dispatch terminal ownership epoch conflicts."
                )
            terminal_run_epoch = run_operation.run_epoch

        # The exact terminal event is supplied by the caller of this helper. If
        # neither live handoff representation remains, acknowledgement already
        # completed: terminal publication first leaves either the run marker or
        # its receipt in the checkpoint, and only an exact durable queue outcome
        # removes the last one. A later invocation's active profile must not
        # become ownership evidence for that already-settled operation.
        if terminal_run_epoch is None:
            return True

        try:
            active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        except (TypeError, ValueError) as exc:
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch terminal invocation profile is malformed."
            ) from exc
        if active_profile is None:
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch terminal evidence has no durable invocation profile."
            )
        if not active_invocation_execution_profile_matches_session_epoch(
            active_profile,
            session_id=session.id,
            run_epoch=session.run_epoch,
        ):
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch terminal ownership conflicts with the session epoch."
            )
        if session.run_epoch < terminal_run_epoch:
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch terminal ownership belongs to a future session epoch."
            )
        if active_profile.run_epoch < terminal_run_epoch:
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch terminal profile predates its ownership epoch."
            )
        if (
            active_profile.run_epoch == terminal_run_epoch
            and active_profile.profile != envelope.required_profile
        ):
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch terminal profile conflicts with its envelope."
            )
        return session.run_epoch > terminal_run_epoch

    async def settlement_state(
        self,
        envelope: _QueuedDispatchEnvelope,
    ) -> _QueuedDispatchSettlement:
        """Classify the exact event/ownership evidence for one queued operation."""

        envelope = _copy_queued_dispatch_envelope(envelope)
        if envelope.operation_kind == "prepared_subagent":
            await self._subagents.require_prepared_subagent_parent_authority(envelope)
        private_session_id, _ = await self._resolve_session(envelope.request.session_id)
        terminal_event = await self._load_terminal_event(
            private_session_id=private_session_id,
            envelope=envelope,
        )
        terminal_event_durable = terminal_event is not None

        try:
            session, checkpoint = await self._load_session_snapshot(private_session_id)
        except KeyError as exc:
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch target session no longer exists."
            ) from exc
        if (
            _queued_dispatch_session_instance_fingerprint(session)
            != envelope.session_instance_fingerprint
        ):
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch target session instance changed."
            )
        self._require_fork_protocol(session, envelope)
        try:
            run_operation = _session_run_operation_from_checkpoint(checkpoint)
            receipts = _queued_dispatch_terminal_receipts_from_checkpoint(checkpoint)
        except (TypeError, ValueError) as exc:
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch terminal ownership evidence is malformed."
            ) from exc
        receipt = receipts.get(envelope.dispatch_operation_id)
        if receipt is not None and (
            receipt.queue_task_id != envelope.queue_task_id
            or receipt.terminal_event_id != envelope.terminal_event_id
        ):
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch terminal receipt identity conflicts."
            )
        terminal_run_epoch = None if receipt is None else receipt.run_epoch
        if run_operation is not None and run_operation.operation_id == (
            envelope.dispatch_operation_id
        ):
            if (
                run_operation.queue_task_id != envelope.queue_task_id
                or run_operation.terminal_event_id != envelope.terminal_event_id
            ):
                raise _QueuedDispatchAuthorityRejected(
                    "Queued dispatch run operation identity conflicts."
                )
            if terminal_run_epoch not in {None, run_operation.run_epoch}:
                raise _QueuedDispatchAuthorityRejected(
                    "Queued dispatch terminal ownership epoch conflicts."
                )
            terminal_run_epoch = run_operation.run_epoch
            if not terminal_event_durable:
                return _QueuedDispatchSettlement(
                    _QueuedDispatchSettlementState.TERMINAL_EVIDENCE_PENDING
                )
        if receipt is not None and not terminal_event_durable:
            # Publication and receipt transfer can commit after the first event
            # query. The exact receipt pins the event, so one read after observing
            # that receipt closes the cross-store classification race.
            terminal_event = await self._load_terminal_event(
                private_session_id=private_session_id,
                envelope=envelope,
            )
            if terminal_event is None:
                raise _QueuedDispatchAuthorityRejected(
                    "Queued dispatch receipt has no exact durable terminal event."
                )
            terminal_event_durable = True
        if terminal_event_durable:
            assert terminal_event is not None
            if not self._terminal_ownership_released(
                session=session,
                checkpoint=checkpoint,
                envelope=envelope,
            ):
                return _QueuedDispatchSettlement(
                    _QueuedDispatchSettlementState.TERMINAL_EVIDENCE_PENDING
                )
            terminal_status_by_type = {
                str(EventType.SESSION_COMPLETED): DispatchStatus.COMPLETED,
                str(EventType.SESSION_FAILED): DispatchStatus.FAILED,
                str(EventType.SESSION_INTERRUPTED): DispatchStatus.INTERRUPTED,
            }
            try:
                terminal_status = terminal_status_by_type[terminal_event.type]
            except KeyError:
                raise _QueuedDispatchAuthorityRejected(
                    "Queued dispatch terminal event has no dispatch status mapping."
                ) from None
            return _QueuedDispatchSettlement(
                _QueuedDispatchSettlementState.TERMINAL_EVIDENCE_DURABLE,
                terminal_status=terminal_status,
            )
        return _QueuedDispatchSettlement(_QueuedDispatchSettlementState.NOT_ADMITTED)

    @staticmethod
    def _require_fork_protocol(
        session: Session,
        envelope: _QueuedDispatchEnvelope,
    ) -> None:
        """Bind the queue protocol to the target's immutable fork relationship."""

        try:
            relationship = session_fork_profile_relationship(session)
        except (TypeError, ValueError) as exc:
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch target fork relationship is malformed."
            ) from exc
        expected_source_state_sha256 = (
            None if relationship is None else relationship.source_state_sha256
        )
        if envelope.exact_fork_source_state_sha256 != expected_source_state_sha256:
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch protocol conflicts with its target fork relationship."
            )
        if (
            session.run_epoch == 0
            and relationship is not None
            and relationship.initial_dispatch_id is not None
            and envelope.request.dispatch_id != relationship.initial_dispatch_id
        ):
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch identity conflicts with the target fork's first invocation."
            )

    async def execute(
        self,
        envelope: _QueuedDispatchEnvelope,
    ) -> AsyncGenerator[Event, None]:
        """Run or replay one queue-owned dispatch under its frozen profile."""

        envelope = _copy_queued_dispatch_envelope(envelope)
        request = envelope.request
        (
            private_session_id,
            store_resolved_session_id,
        ) = await self._resolve_session(request.session_id)
        try:
            session, checkpoint = await self._load_session_snapshot(private_session_id)
        except KeyError as exc:
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch target session no longer exists."
            ) from exc
        if (
            _queued_dispatch_session_instance_fingerprint(session)
            != envelope.session_instance_fingerprint
        ):
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch target session instance changed."
            )
        self._require_fork_protocol(session, envelope)
        replay_event = await self._load_terminal_event(
            private_session_id=private_session_id,
            envelope=envelope,
        )
        if replay_event is not None:
            if not self._terminal_ownership_released(
                session=session,
                checkpoint=checkpoint,
                envelope=envelope,
            ):
                raise SessionRunFenced(
                    "Queued dispatch terminal hooks or trailing cleanup still own the "
                    "session run fence."
                )
            yield await self._project_event(replay_event)
            return
        try:
            self._get_agent(session.agent_name)
            self._get_provider(
                request.target.provider_name
                if request.target is not None
                else session.provider_name
            )
            self._get_environment(session.environment_name)
        except KeyError as exc:
            if envelope.operation_kind == "prepared_subagent":
                raise durable_subagent_worker_incompatible() from exc
            raise _QueuedDispatchAuthorityRejected(
                "Queued dispatch required runtime component is unavailable."
            ) from exc

        if envelope.operation_kind == "prepared_subagent":
            run_request = self._subagents.prepare_queued_child_run(
                envelope=envelope,
                session=session,
                checkpoint=checkpoint,
            )
            stream = self._run(run_request)
            async with _close_delegated_event_stream(stream) as owned_stream:
                async for event in owned_stream:
                    yield await self._project_event(event)
            return

        private_request = request.model_copy(
            update={"session_id": private_session_id},
            deep=True,
        )
        stream = self._dispatch(
            private_request,
            store_resolved_session_id=store_resolved_session_id,
            source_execution_profile=envelope.source_profile,
            required_execution_profile=envelope.required_profile,
            required_session_instance_fingerprint=(envelope.session_instance_fingerprint),
            dispatch_operation_id=envelope.dispatch_operation_id,
            dispatch_terminal_event_id=envelope.terminal_event_id,
            queue_task_id=envelope.queue_task_id,
        )
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for event in owned_stream:
                yield await self._project_event(event)
