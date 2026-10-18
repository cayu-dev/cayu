"""Linked-session task authority, lifecycle and terminal failure replay."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from cayu.budgets.base import (
    BudgetPolicy,
)
from cayu.events import (
    Event,
    EventType,
    copy_event,
    event_with_runtime_generated_id,
    event_with_runtime_nested_payload_authority,
    event_with_runtime_payload_authority,
)
from cayu.exceptions import (
    TerminalEventPublicationUncertain,
)
from cayu.execution_profiles import (
    event_with_execution_profile_authority,
)
from cayu.observability.hooks import (
    RuntimeHookPhase,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import (
    _session_execution_profile,
)
from cayu.runtime import _session_request_boundary as session_request_boundary
from cayu.runtime._continuation_task_failure import (
    ApprovalTaskFailureIdentity,
    approval_task_failure_payload,
    approval_task_failure_receipt_matches,
    approval_task_terminalization_idempotency_key,
    load_direct_runtime_task_failure_replay,
    load_direct_task_failure_replay,
    provider_operation_task_failure_payload,
    runtime_task_failure_decision_receipt_matches,
    runtime_task_failure_event_id,
    runtime_task_failure_identity_from_task,
    runtime_task_failure_receipt_matches,
    runtime_task_failure_session_payload,
    runtime_task_failure_turn_payload,
    runtime_task_terminalization_idempotency_key,
)
from cayu.runtime._diagnostics import (
    ExceptionDiagnostic,
    task_failure_payload_from_diagnostic,
)
from cayu.runtime._durable_tool_round import _environment_name as _environment_name
from cayu.runtime._event_writer import (
    RuntimeEventWriter,
)
from cayu.runtime._execution_profile_continuation import (
    ExecutionProfileContinuation,
)
from cayu.runtime._invocation_lifecycle import (
    _authenticated_invocation_context,
)
from cayu.runtime._session_finalization import (
    SessionFinalization,
    _task_event,
    _task_terminalization_idempotency_key,
)
from cayu.runtime._task_lease_authority import (
    active_task_lease_authority,
)
from cayu.runtime._terminal_event_publication import (
    TerminalEventPublication,
)
from cayu.runtime.loop_policies import (
    LoopPolicy,
)
from cayu.sessions._checkpoint_mutations import (
    SessionRuntimePublicationConflict,
)
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_is_released,
)
from cayu.sessions._invocation_lifecycle import (
    AdmittedInvocationBinding,
)
from cayu.sessions._invocation_scope import (
    _deactivate_session_interaction,
)
from cayu.sessions._invocation_terminal_decision import (
    InvocationTerminalOutcome,
    invocation_terminal_decision_from_checkpoint,
    invocation_terminal_decision_matches_active_profile,
)
from cayu.sessions._terminal_evidence import _SESSION_RUN_OPERATION_ID_PAYLOAD_KEY
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.invocation import SessionInvocationBinding
from cayu.sessions.records import (
    Session,
    SessionStatus,
)
from cayu.sessions.recovery import (
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryResult,
)
from cayu.sessions.requests import (
    ResumeRequest,
)
from cayu.sessions.store import SessionStore
from cayu.tasks._terminalization import _terminalize_claimed_task
from cayu.tasks.creation import (
    _task_invocation_for_attachment,
    _task_session_instance_for_attachment,
)
from cayu.tasks.queries import TaskQuery
from cayu.tasks.records import Task, TaskClaimLost, TaskStatus
from cayu.tasks.store import TaskStore
from cayu.tasks.terminalization import (
    TaskTerminalizationRequest,
    TaskTerminalKind,
    prepare_task_terminalization,
)
from cayu.vaults.redaction import SecretRedactor


def _provider_operation_task_terminalization_request(
    *,
    task_id: str,
    task_worker_id: str,
    task_handoff_id: str | None,
    session_id: str,
) -> TaskTerminalizationRequest:
    """Build the exact receipt-backed task failure used by first-run and replay."""

    return TaskTerminalizationRequest(
        task_id=task_id,
        worker_id=task_worker_id,
        handoff_id=task_handoff_id,
        kind=TaskTerminalKind.FAILED,
        error=provider_operation_task_failure_payload(session_id=session_id),
        idempotency_key=_task_terminalization_idempotency_key(
            task_id=task_id,
            session_id=session_id,
            kind=TaskTerminalKind.FAILED,
        ),
    )


class SessionTaskLifecycle:
    """Own attached-task admission, terminal settlement and exact failure replay."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        task_store: TaskStore | None,
        session_finalization: SessionFinalization,
        event_writer: RuntimeEventWriter,
        terminal_event_publication: TerminalEventPublication,
        execution_profile_continuation: ExecutionProfileContinuation,
        secret_redactor: SecretRedactor,
        get_budget_policy: Callable[[], BudgetPolicy | None],
        get_registered_agent: Callable[[str], runtime_records.RegisteredAgentState],
        get_registered_provider: Callable[[str | None], runtime_records.RegisteredProvider],
        get_registered_environment_for_session: Callable[
            [str | None], runtime_records.RegisteredEnvironment | None
        ],
        runtime_hooks: tuple[runtime_records.RegisteredRuntimeHook, ...],
        loop_policies: tuple[LoopPolicy, ...],
    ) -> None:
        self.session_store = session_store
        self.task_store = task_store
        self._session_finalization = session_finalization
        self._event_writer = event_writer
        self._terminal_event_publication = terminal_event_publication
        self._execution_profile_continuation = execution_profile_continuation
        self._secret_redactor = secret_redactor
        self._get_budget_policy = get_budget_policy
        self._get_registered_agent = get_registered_agent
        self._get_registered_provider = get_registered_provider
        self._get_registered_environment_for_session = get_registered_environment_for_session
        self._runtime_hooks = runtime_hooks
        self._loop_policies = loop_policies

    async def recover_committed_runtime_task_failure(
        self,
        session: Session,
        checkpoint: dict[str, Any] | None,
        previous_status: SessionStatus,
        before_mutation: Callable[[], Awaitable[None]],
    ) -> IncompleteSessionRecoveryResult | None:
        """Finish one durable FAILED winner after its task owner disappeared."""

        terminal_decision = invocation_terminal_decision_from_checkpoint(checkpoint)
        if (
            terminal_decision is None
            or terminal_decision.outcome is not InvocationTerminalOutcome.FAILED
            or terminal_decision.task_id is None
        ):
            return None
        self._session_finalization.require_terminal_interaction_publication_protocol()
        if self.task_store is None:
            raise RuntimeError("Runtime task failure recovery requires a TaskStore.")
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if (
            active_profile is None
            or session.status
            not in {SessionStatus.RUNNING, SessionStatus.INTERRUPTING, SessionStatus.FAILED}
            or not invocation_terminal_decision_matches_active_profile(
                terminal_decision,
                session_id=session.id,
                session_instance_id=session.instance_id,
                run_epoch=session.run_epoch,
                interaction_id=active_profile.interaction_id,
                execution_profile_fingerprint=active_profile.profile.fingerprint,
            )
            or terminal_decision.task_error_payload is None
            or terminal_decision.runtime_task_failure_id is None
        ):
            raise TaskClaimLost(
                "Committed runtime task failure has conflicting recovery authority."
            ) from None
        task = await self.task_store.load_task(terminal_decision.task_id)
        if (
            task is None
            or task.session_id != session.id
            or task.session_instance_id != session.instance_id
        ):
            raise TaskClaimLost(
                "Committed runtime task failure lost its exact attached task."
            ) from None
        original_worker_id = task.worker_id
        original_lease_expires_at = task.lease_expires_at
        original_handoff_id = task.interrupted_handoff_id
        proposed_failure_task = (
            task
            if task.status is TaskStatus.FAILED
            else task.model_copy(
                update={
                    "status": TaskStatus.FAILED,
                    "error": terminal_decision.task_error_payload,
                },
                deep=True,
            )
        )
        failure_identity = runtime_task_failure_identity_from_task(
            proposed_failure_task,
            session_id=session.id,
            session_instance_id=session.instance_id,
        )
        if (
            failure_identity is None
            or failure_identity.failure_id != terminal_decision.runtime_task_failure_id
            or failure_identity.run_epoch != terminal_decision.run_epoch
            or failure_identity.interaction_id != terminal_decision.interaction_id
            or failure_identity.execution_profile_fingerprint
            != terminal_decision.execution_profile_fingerprint
            or failure_identity.observed_at != terminal_decision.observed_at
            or runtime_task_failure_session_payload(proposed_failure_task)
            != terminal_decision.terminal_payload
            or runtime_task_failure_turn_payload(proposed_failure_task)
            != terminal_decision.turn_completed_payload
        ):
            raise TaskClaimLost(
                "Committed runtime task failure has conflicting task failure evidence."
            ) from None
        if task.status is TaskStatus.RUNNING:
            await before_mutation()
            if original_worker_id is None:
                task = await self.task_store.fail_task(
                    task.id,
                    terminal_decision.task_error_payload,
                    worker_id=None,
                )
            else:
                if task.lease_expires_at is None:
                    raise TaskClaimLost(
                        "Committed runtime task failure lost its worker lease generation."
                    ) from None
                if not self.task_store.supports_attached_task_recovery_terminalization:
                    raise NotImplementedError(
                        "Runtime task failure recovery requires atomic expired-owner "
                        "terminalization."
                    )
                try:
                    task = await self.task_store.recover_attached_task_failure(
                        TaskTerminalizationRequest(
                            task_id=task.id,
                            worker_id=original_worker_id,
                            lease_expires_at=task.lease_expires_at,
                            handoff_id=original_handoff_id,
                            kind=TaskTerminalKind.FAILED,
                            error=terminal_decision.task_error_payload,
                            idempotency_key=runtime_task_terminalization_idempotency_key(
                                task_id=task.id,
                                session_id=session.id,
                                kind=TaskTerminalKind.FAILED,
                            ),
                        ),
                        session_id=session.id,
                        session_instance_id=session.instance_id,
                    )
                except TaskClaimLost:
                    current = await self.task_store.load_task(task.id)
                    if (
                        current is not None
                        and current.status is TaskStatus.RUNNING
                        and current.session_id == session.id
                        and current.session_instance_id == session.instance_id
                        and current.worker_id == original_worker_id
                        and current.lease_expires_at == task.lease_expires_at
                        and current.interrupted_handoff_id == original_handoff_id
                    ):
                        return IncompleteSessionRecoveryResult(
                            session_id=session.id,
                            previous_status=previous_status,
                            status=session.status,
                            actions=(IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,),
                            events=(),
                            message=(
                                "The task owner for the committed failure is still live; "
                                "recovery skipped."
                            ),
                        )
                    task = current
                    if task is None or task.status is not TaskStatus.FAILED:
                        raise
        if task.status is not TaskStatus.FAILED:
            raise TaskClaimLost(
                "Committed runtime task failure conflicts with the task terminal outcome."
            ) from None
        await before_mutation()
        replayed, events = await self.replay_runtime_task_failure_if_needed(
            session_id=session.id,
            task_id=task.id,
            task_worker_id=original_worker_id,
            task_lease_expires_at=original_lease_expires_at,
            task_handoff_id=original_handoff_id,
        )
        if not replayed:
            raise TaskClaimLost(
                "Committed runtime task failure could not reconstruct its terminal evidence."
            ) from None
        current_session = await self.session_store.load(session.id)
        if current_session is None or current_session.instance_id != session.instance_id:
            raise TaskClaimLost(
                "Committed runtime task failure changed its session incarnation."
            ) from None
        return IncompleteSessionRecoveryResult(
            session_id=session.id,
            previous_status=previous_status,
            status=current_session.status,
            actions=(IncompleteSessionRecoveryAction.REPAIRED_TERMINAL_OWNERSHIP,),
            events=events,
            message="Recovered the committed runtime task failure without redispatching work.",
        )

    async def require_continuation_task_authority(
        self,
        *,
        session_id: str,
        session_instance_id: str | None,
        task_id: str | None,
        task_worker_id: str | None,
        task_handoff_id: str | None = None,
        allow_terminal_failure_replay: bool = False,
        approval_failure_identity: ApprovalTaskFailureIdentity | None = None,
    ) -> None:
        if task_handoff_id is not None and task_worker_id is None:
            raise RuntimeError("Task handoff authority requires a continuation worker.")
        if task_id is None and session_instance_id is None and task_worker_id is None:
            if task_handoff_id is not None:
                raise RuntimeError("Workerless continuation retained task handoff authority.")
            return
        if task_id is None or session_instance_id is None:
            raise RuntimeError("Attached-task continuation authority is incomplete.")
        await self.require_linked_continuation_task_authority(
            task_id=task_id,
            task_worker_id=task_worker_id,
            task_handoff_id=task_handoff_id,
            session_id=session_id,
            session_instance_id=session_instance_id,
            allow_terminal_failure_replay=allow_terminal_failure_replay,
            approval_failure_identity=approval_failure_identity,
        )

    async def start_task(
        self,
        *,
        task_id: str,
        session: Session,
        worker_id: str | None = None,
        lease_expires_at: datetime | None = None,
        handoff_id: str | None = None,
    ) -> Task:
        if self.task_store is None:
            raise RuntimeError("task_store is required when RunRequest.task_id is set.")
        session_invocation = SessionInvocationBinding(
            id=session.id,
            session_instance_id=session.instance_id,
            invocation=session.invocation,
        )
        existing = await self.task_store.load_task(task_id)
        if (
            worker_id is None
            and existing is not None
            and existing.status is TaskStatus.RUNNING
            and existing.session_id == session.id
            and existing.worker_id is None
        ):
            _task_invocation_for_attachment(
                existing.invocation,
                session_id=session.id,
                session_binding=session_invocation,
            )
            _task_session_instance_for_attachment(
                stored_session_instance_id=existing.session_instance_id,
                session_id=session.id,
                session_binding=session_invocation,
            )
            return existing
        if worker_id is not None:
            if lease_expires_at is None:
                raise TaskClaimLost("Worker-owned task attachment requires its exact lease.")
            lease_authority = active_task_lease_authority(
                task_id=task_id,
                worker_id=worker_id,
                handoff_id=handoff_id,
                presented_lease_expires_at=lease_expires_at,
            )
            if lease_authority is not None:
                async with lease_authority.lock:
                    attached = await self.task_store.attach_task(
                        task_id,
                        session_id=session.id,
                        session_invocation=session_invocation,
                        worker_id=worker_id,
                        lease_expires_at=lease_authority.lease_expires_at,
                    )
                    if (
                        lease_authority.deadline_monotonic is not None
                        and time.monotonic() >= lease_authority.deadline_monotonic
                    ):
                        raise TaskClaimLost(
                            "Task attachment acknowledgement consumed the current lease."
                        )
                    return attached
            return await self.task_store.attach_task(
                task_id,
                session_id=session.id,
                session_invocation=session_invocation,
                worker_id=worker_id,
                lease_expires_at=lease_expires_at,
            )
        return await self.task_store.start_task(
            task_id,
            session_id=session.id,
            session_invocation=session_invocation,
        )

    async def fail_task_for_execution_deadline(
        self,
        *,
        task_id: str | None,
        task_worker_id: str | None,
        task_lease_expires_at: datetime | None = None,
        task_handoff_id: str | None,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
    ) -> bool:
        """Settle an owned ordinary task after deadline cancellation has drained.

        Exact latest lifecycle evidence covers setup as well as dispatched work.
        The interrupted session remains available for external-effect recovery.
        """
        from cayu.sessions.event_queries import EventOrder, EventQuery
        from cayu.sessions.terminal_evidence import TERMINAL_EVIDENCE_EVENT_TYPES

        if task_id is None or self.task_store is None:
            return False
        task = await self.task_store.load_task(task_id)
        current = await self.session_store.load(session.id)
        if (
            task is None
            or current is None
            or current.instance_id != session.instance_id
            or current.status is not SessionStatus.INTERRUPTED
            or task.status not in {TaskStatus.PENDING, TaskStatus.CLAIMED, TaskStatus.RUNNING}
            or task.worker_id != task_worker_id
            or task.session_id not in {None, session.id}
            or task.session_instance_id not in {None, session.instance_id}
            or task.retry_series is not None
            or task.work_contract is not None
        ):
            return False
        records = await self.session_store.query_events(
            EventQuery(
                session_id=session.id,
                event_types=TERMINAL_EVIDENCE_EVENT_TYPES,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=1,
            )
        )
        if not records:
            return False
        terminal = records[0].event
        evidence = terminal.payload.get("failure_evidence")
        if (
            terminal.type != EventType.SESSION_INTERRUPTED
            or type(evidence) is not dict
            or evidence.get("classification") != "deadline"
            or evidence.get("session_id") != session.id
            or evidence.get("run_epoch") != session.run_epoch
        ):
            return False
        from cayu.runtime._task_lease_authority import managed_task_lease_mutation

        async with managed_task_lease_mutation(
            task_id=task.id,
            worker_id=task_worker_id,
            handoff_id=task_handoff_id,
            presented_lease_expires_at=task_lease_expires_at,
        ) as exact_lease:
            failed = await self._session_finalization.fail_task(
                task_id=task.id,
                task_worker_id=task_worker_id,
                task_lease_expires_at=exact_lease,
                task_handoff_id=task_handoff_id,
                session=session,
                error=self._secret_redactor.redact_json(
                    {
                        "error": "ExecutionDeadlineExceeded",
                        "message": "Execution deadline expired; task timed out.",
                        "failure_evidence": evidence,
                        "execution_deadline": terminal.payload.get("execution_deadline"),
                        "deadline_failure": {
                            "session_id": session.id,
                            "session_instance_id": session.instance_id,
                            "run_epoch": session.run_epoch,
                            "terminal_event_id": terminal.id,
                            "task_started": task.started_at is not None,
                            "task_handoff_id": task_handoff_id,
                        },
                    }
                ),
            )
        await self._event_writer.emit(
            _task_event(
                event_type=EventType.TASK_FAILED,
                task=failed,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            )
        )
        return True

    async def fail_task_for_run_setup_error(
        self,
        *,
        task_id: str | None,
        task_worker_id: str | None,
        task_handoff_id: str | None,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        diagnostic: ExceptionDiagnostic,
    ) -> tuple[Event | None, Exception | None]:
        """Fail only a task this run can prove it owns before `_run_session`."""
        if task_id is None:
            return None, None
        if self.task_store is None:
            return None, RuntimeError("task_store is required when RunRequest.task_id is set.")
        try:
            task = await self.task_store.load_task(task_id)
            if task is None:
                return None, KeyError(f"Task not found: {task_id}")
            worker_matches = task.worker_id == task_worker_id
            attached_to_session = task.session_id == session.id
            owned_claim = (
                task_worker_id is not None
                and task.status is TaskStatus.CLAIMED
                and task.session_id is None
                and worker_matches
            )
            unclaimed_pending = (
                task_worker_id is None
                and task.status is TaskStatus.PENDING
                and task.session_id is None
                and task.worker_id is None
            )
            if not ((attached_to_session and worker_matches) or owned_claim or unclaimed_pending):
                return None, None
            task = await self._session_finalization.fail_task(
                task_id=task_id,
                task_worker_id=task_worker_id,
                task_handoff_id=task_handoff_id,
                session=session,
                error=task_failure_payload_from_diagnostic(
                    diagnostic,
                    session_id=session.id,
                ),
            )
            return (
                await self._event_writer.emit(
                    _task_event(
                        event_type=EventType.TASK_FAILED,
                        task=task,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                    )
                ),
                None,
            )
        except Exception as task_error:
            return None, task_error

    async def replay_runtime_task_failure_if_needed(
        self,
        *,
        session_id: str,
        task_id: str | None,
        task_worker_id: str | None,
        task_lease_expires_at: datetime | None = None,
        task_handoff_id: str | None,
    ) -> tuple[bool, tuple[Event, ...]]:
        """Converge a receipt-backed generic task failure without redispatching work."""

        if task_id is None:
            return False, ()
        if self.task_store is None:  # pragma: no cover - guarded by task lookup
            raise RuntimeError("task_store is required for task failure replay.")
        session = await self.session_store.load(session_id)
        task = await self.task_store.load_task(task_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}") from None
        if task is None:
            raise KeyError(f"Task not found: {task_id}") from None
        checkpoint = await self.session_store.load_checkpoint(session.id)
        terminal_decision = invocation_terminal_decision_from_checkpoint(checkpoint)
        if terminal_decision is not None:
            decision_active_profile = active_invocation_execution_profile_from_checkpoint(
                checkpoint
            )
            if (
                decision_active_profile is None
                or decision_active_profile.session_id != session.id
                or decision_active_profile.run_epoch != session.run_epoch
                or not invocation_terminal_decision_matches_active_profile(
                    terminal_decision,
                    session_id=session.id,
                    session_instance_id=session.instance_id,
                    run_epoch=session.run_epoch,
                    interaction_id=decision_active_profile.interaction_id,
                    execution_profile_fingerprint=(decision_active_profile.profile.fingerprint),
                )
                or terminal_decision.task_id != task.id
            ):
                raise TaskClaimLost(
                    "Invocation terminal decision conflicts with its attached task."
                ) from None
            if terminal_decision.outcome is InvocationTerminalOutcome.INTERRUPTED:
                return False, ()
            if task.status is TaskStatus.RUNNING:
                await self.require_linked_continuation_task_authority(
                    task_id=task.id,
                    task_worker_id=task_worker_id,
                    task_handoff_id=task_handoff_id,
                    session_id=session.id,
                    session_instance_id=session.instance_id,
                )
                if terminal_decision.task_error_payload is None:
                    raise TaskClaimLost(
                        "Failure terminal decision lost its task error evidence."
                    ) from None
                task = await self._session_finalization.fail_task(
                    task_id=task.id,
                    task_worker_id=task_worker_id,
                    task_handoff_id=task_handoff_id,
                    session=session,
                    error=terminal_decision.task_error_payload,
                )
        identity = runtime_task_failure_identity_from_task(
            task,
            session_id=session.id,
            session_instance_id=session.instance_id,
        )
        if identity is None:
            return False, ()
        if terminal_decision is not None and (
            terminal_decision.outcome is not InvocationTerminalOutcome.FAILED
            or terminal_decision.runtime_task_failure_id != identity.failure_id
            or terminal_decision.interaction_id != identity.interaction_id
            or terminal_decision.execution_profile_fingerprint
            != identity.execution_profile_fingerprint
            or terminal_decision.observed_at != identity.observed_at
            or terminal_decision.interaction_event_id
            != runtime_task_failure_event_id(identity, "interaction_failed")
            or terminal_decision.terminal_event_id
            != runtime_task_failure_event_id(identity, "session_failed")
            or terminal_decision.task_error_payload != task.error
        ):
            raise TaskClaimLost(
                "Failure terminal decision conflicts with durable task evidence."
            ) from None
        if identity.run_epoch != session.run_epoch:
            active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
            released_failure_epoch = bool(
                session.status is SessionStatus.FAILED
                and active_profile is not None
                and active_profile.run_epoch == identity.run_epoch
                and active_profile.interaction_id == identity.interaction_id
                and active_invocation_execution_profile_is_released(
                    active_profile,
                    session_id=session.id,
                    run_epoch=session.run_epoch,
                )
            )
            if not released_failure_epoch:
                raise TaskClaimLost(
                    "Terminal task failure belongs to a stale session run epoch."
                ) from None
        await self.require_linked_continuation_task_authority(
            task_id=task.id,
            task_worker_id=task_worker_id,
            task_lease_expires_at=task_lease_expires_at,
            task_handoff_id=task_handoff_id,
            session_id=session.id,
            session_instance_id=session.instance_id,
            expected_runtime_task_terminalization_request_sha256=(
                None
                if terminal_decision is None
                else terminal_decision.task_terminalization_request_sha256
            ),
        )
        failure_payload = runtime_task_failure_session_payload(task)
        if failure_payload is None:
            raise TaskClaimLost(
                "Terminal task failure lost its durable session failure payload."
            ) from None
        if failure_payload.get("runtime_task_failure_id") != identity.failure_id:
            raise TaskClaimLost(
                "Terminal task failure has conflicting durable session evidence."
            ) from None
        turn_completed_payload = runtime_task_failure_turn_payload(task)
        if turn_completed_payload is None:
            raise TaskClaimLost(
                "Terminal task failure lost its durable turn completion payload."
            ) from None
        if terminal_decision is not None and (
            terminal_decision.terminal_payload != failure_payload
            or terminal_decision.turn_completed_payload != turn_completed_payload
        ):
            raise TaskClaimLost(
                "Failure terminal decision conflicts with its replay payloads."
            ) from None

        def terminal_event_matches(event: Event) -> bool:
            return bool(
                event.type is EventType.SESSION_FAILED
                and event.session_id == session.id
                and event.interaction_id is None
                and event.agent_name == session.agent_name
                and event.environment_name == session.environment_name
                and event.timestamp == identity.observed_at
                and event.payload.get("execution_profile_fingerprint")
                == identity.execution_profile_fingerprint
                and all(
                    event.payload.get(field_name) == field_value
                    for field_name, field_value in failure_payload.items()
                )
            )

        def turn_completed_event_matches(event: Event) -> bool:
            return bool(
                event.type is EventType.TURN_COMPLETED
                and event.session_id == session.id
                and event.interaction_id is None
                and event.agent_name == session.agent_name
                and event.environment_name == session.environment_name
                and event.timestamp == identity.observed_at
                and event.payload == turn_completed_payload
            )

        terminal_event_id = runtime_task_failure_event_id(identity, "session_failed")
        turn_completed_event_id = runtime_task_failure_event_id(identity, "turn_completed")
        durable_terminal_event: Event | None = None
        terminal_records = await self.session_store.query_events(
            EventQuery(session_id=session.id, event_id=terminal_event_id, limit=2)
        )
        if terminal_records:
            if len(terminal_records) != 1:
                raise SessionRuntimePublicationConflict(
                    "Runtime task failure has duplicate terminal session evidence."
                )
            terminal_event = terminal_records[0].event
            if (
                not terminal_event_matches(terminal_event)
                or session.status is not SessionStatus.FAILED
            ):
                raise SessionRuntimePublicationConflict(
                    "Runtime task failure terminal evidence conflicts with durable state."
                )
            turn_records = await self.session_store.query_events(
                EventQuery(
                    session_id=session.id,
                    event_id=turn_completed_event_id,
                    limit=2,
                )
            )
            if len(turn_records) != 1 or not turn_completed_event_matches(turn_records[0].event):
                raise SessionRuntimePublicationConflict(
                    "Runtime task failure terminal evidence lost its turn completion."
                )
            durable_terminal_event = copy_event(terminal_event)

        if session.status not in {
            SessionStatus.RUNNING,
            SessionStatus.INTERRUPTING,
            SessionStatus.FAILED,
        }:
            raise SessionRuntimePublicationConflict(
                "Runtime task failure conflicts with the session terminal outcome."
            )
        registered_agent = self._get_registered_agent(session.agent_name)
        registered_provider = self._get_registered_provider(session.provider_name)
        registered_environment = self._get_registered_environment_for_session(
            session.environment_name
        )
        environment_name = _environment_name(registered_environment)
        budget_policy = self._get_budget_policy()
        execution_profile_snapshot = await self._execution_profile_continuation.validate(
            session=session,
            checkpoint=checkpoint,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            budget_policy=budget_policy,
            require_open_interaction=False,
            record_rejection=False,
        )
        if (
            execution_profile_snapshot.interaction_id != identity.interaction_id
            or execution_profile_snapshot.profile.fingerprint
            != identity.execution_profile_fingerprint
        ):
            raise TaskClaimLost(
                "Terminal task failure changed its invocation execution profile."
            ) from None
        replay_active_profile = execution_profile_snapshot.model_copy(
            update={"run_epoch": session.run_epoch}
        )
        invocation_context = _authenticated_invocation_context(
            active_profile=replay_active_profile,
            binding=AdmittedInvocationBinding(
                session_id=session.id,
                session_instance_id=session.instance_id,
                interaction_id=identity.interaction_id,
                run_epoch=session.run_epoch,
                agent_name=session.agent_name,
                provider_name=session.provider_name,
                model=session.model,
                runtime_name=session.runtime_name,
                runtime_version=session.runtime_version,
                runtime_build_provenance=session.runtime_build_provenance,
                environment_name=session.environment_name,
            ),
            validated_profile=replay_active_profile.profile,
            registered_agent=registered_agent,
            registered_provider=registered_provider,
            registered_environment=registered_environment,
            runtime_hooks=self._runtime_hooks,
            loop_policies=self._loop_policies,
            request_loop_policies=(),
            budget_policy=budget_policy,
            tool_capability_ceiling=_session_execution_profile._session_tool_capability_ceiling(
                session
            ),
        )
        execution_profile = invocation_context.profile
        replay_events: list[Event] = []

        if durable_terminal_event is not None:
            try:
                async for event in self._terminal_event_publication.replay(
                    event=durable_terminal_event,
                    phase=RuntimeHookPhase.AFTER_SESSION_FAILED,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                ):
                    replay_events.append(event)
            finally:
                _deactivate_session_interaction(session.id)
            return True, tuple(replay_events)

        task_failed_template = _task_event(
            event_type=EventType.TASK_FAILED,
            task=task,
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
        )
        task_failed = event_with_runtime_generated_id(
            event_with_execution_profile_authority(
                event_with_runtime_payload_authority(
                    task_failed_template.model_copy(
                        update={
                            "id": runtime_task_failure_event_id(identity, "task_failed"),
                            "interaction_id": identity.interaction_id,
                            "timestamp": identity.observed_at,
                            "payload": {
                                **task_failed_template.payload,
                                "runtime_task_failure_id": identity.failure_id,
                            },
                        }
                    ),
                    "runtime_task_failure_id",
                ),
                execution_profile,
            )
        )
        persisted_task_failed = await self._event_writer.persist_exact_replay(task_failed)
        replay_events.extend(await self._event_writer.fan_out_persisted([persisted_task_failed]))

        if terminal_decision is not None:
            interaction_id = await self._session_finalization.activate_latest_open_interaction(
                session.id
            )
            if interaction_id != identity.interaction_id:
                raise SessionRuntimePublicationConflict(
                    "Runtime task failure lost its open durable interaction."
                )
        elif session.status is SessionStatus.FAILED:
            interaction_records = await self.session_store.query_events(
                EventQuery(
                    session_id=session.id,
                    event_id=runtime_task_failure_event_id(identity, "interaction_failed"),
                    limit=2,
                )
            )
            if (
                len(interaction_records) != 1
                or interaction_records[0].event.type is not EventType.INTERACTION_FAILED
                or interaction_records[0].event.session_id != session.id
                or interaction_records[0].event.interaction_id != identity.interaction_id
                or interaction_records[0].event.timestamp != identity.observed_at
            ):
                raise SessionRuntimePublicationConflict(
                    "Failed session lost its atomic runtime task interaction evidence."
                )
        else:
            interaction_id = await self._session_finalization.activate_latest_open_interaction(
                session.id
            )
            if interaction_id != identity.interaction_id:
                raise SessionRuntimePublicationConflict(
                    "Runtime task failure lost its open durable interaction."
                )
            interaction_event_id = runtime_task_failure_event_id(
                identity,
                "interaction_failed",
            )
            try:
                (
                    session,
                    interaction_failed,
                    _,
                ) = await self._session_finalization.publish_sibling_interaction_transition(
                    session=session,
                    invocation_context=invocation_context,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    environment_name=environment_name,
                    to_status=SessionStatus.FAILED,
                    observed_at=identity.observed_at,
                    event_id=interaction_event_id,
                    execution_profile=execution_profile,
                    # Receipt evidence proves that the runtime observed a failure, but
                    # cannot prove an already-dispatched provider had no effect.
                    model_completion_failure=RuntimeError(
                        "Recovered receipt-backed runtime task failure."
                    ),
                )
            except Exception:
                # A concurrent exact replay can calculate the same transition
                # before its peer atomically settles the active model stage. The
                # second calculated spec then differs only because that stage is
                # already settled. Adopt the winner only from the deterministic
                # event, terminal session state, and absence of active model work.
                interaction_records = await self.session_store.query_events(
                    EventQuery(
                        session_id=session.id,
                        event_id=interaction_event_id,
                        limit=2,
                    )
                )
                concurrent_session = await self.session_store.load(session.id)
                active_model_completion = (
                    await self.session_store.load_active_model_completion_stage(session.id)
                )
                if (
                    len(interaction_records) != 1
                    or interaction_records[0].event.type is not EventType.INTERACTION_FAILED
                    or interaction_records[0].event.session_id != session.id
                    or interaction_records[0].event.interaction_id != identity.interaction_id
                    or interaction_records[0].event.timestamp != identity.observed_at
                    or concurrent_session is None
                    or concurrent_session.status is not SessionStatus.FAILED
                    or active_model_completion is not None
                ):
                    raise
                session = concurrent_session
                interaction_failed = copy_event(interaction_records[0].event)
            if interaction_failed is not None:
                replay_events.append(interaction_failed)

        turn_completed = event_with_runtime_generated_id(
            event_with_runtime_nested_payload_authority(
                Event(
                    id=turn_completed_event_id,
                    type=EventType.TURN_COMPLETED,
                    session_id=session.id,
                    interaction_id=None,
                    agent_name=registered_agent.spec.name,
                    environment_name=environment_name,
                    timestamp=identity.observed_at,
                    payload=turn_completed_payload,
                ),
                ("interaction_ids", "*"),
            )
        )
        persisted_turn_completed = await self._event_writer.persist_exact_replay(turn_completed)
        replay_events.extend(await self._event_writer.fan_out_persisted([persisted_turn_completed]))

        session_failed = event_with_runtime_generated_id(
            event_with_execution_profile_authority(
                event_with_runtime_payload_authority(
                    Event(
                        id=terminal_event_id,
                        type=EventType.SESSION_FAILED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        timestamp=identity.observed_at,
                        payload=failure_payload,
                    ),
                    "runtime_task_failure_id",
                ),
                execution_profile,
            )
        )
        if terminal_decision is not None:
            (
                finalize_result,
                prepared_terminal_event,
            ) = await self._session_finalization.prepare_terminal_event_for_atomic_transition(
                event=session_failed,
                session=session,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
            )
            try:
                try:
                    (
                        session,
                        interaction_failed,
                        _,
                    ) = await self._session_finalization.publish_sibling_interaction_transition(
                        session=session,
                        invocation_context=invocation_context,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        environment_name=environment_name,
                        to_status=SessionStatus.FAILED,
                        from_statuses={SessionStatus.RUNNING},
                        observed_at=identity.observed_at,
                        event_id=terminal_decision.interaction_event_id,
                        execution_profile=execution_profile,
                        model_completion_failure=RuntimeError(
                            "Recovered receipt-backed runtime task failure."
                        ),
                        terminal_event=prepared_terminal_event,
                        terminal_decision=terminal_decision,
                    )
                except Exception:
                    interaction_records, terminal_records = await asyncio.gather(
                        self.session_store.query_events(
                            EventQuery(
                                session_id=session.id,
                                event_id=terminal_decision.interaction_event_id,
                                limit=2,
                            )
                        ),
                        self.session_store.query_events(
                            EventQuery(
                                session_id=session.id,
                                event_id=terminal_decision.terminal_event_id,
                                limit=2,
                            )
                        ),
                    )
                    concurrent_session = await self.session_store.load(session.id)
                    active_model_completion = (
                        await self.session_store.load_active_model_completion_stage(session.id)
                    )
                    if (
                        len(interaction_records) != 1
                        or interaction_records[0].event.type is not EventType.INTERACTION_FAILED
                        or interaction_records[0].event.interaction_id != identity.interaction_id
                        or interaction_records[0].event.timestamp != identity.observed_at
                        or len(terminal_records) != 1
                        or not terminal_event_matches(terminal_records[0].event)
                        or concurrent_session is None
                        or concurrent_session.status is not SessionStatus.FAILED
                        or active_model_completion is not None
                    ):
                        raise
                    session = concurrent_session
                    interaction_failed = copy_event(interaction_records[0].event)
                    prepared_terminal_event = copy_event(terminal_records[0].event)
                    finalize_result = None
                if interaction_failed is not None:
                    replay_events.append(interaction_failed)
                if finalize_result is None:
                    async for event in self._terminal_event_publication.replay(
                        event=prepared_terminal_event,
                        phase=RuntimeHookPhase.AFTER_SESSION_FAILED,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                    ):
                        replay_events.append(event)
                else:
                    async for event in self._session_finalization.emit_atomically_persisted_terminal_event_with_hooks(
                        finalize_result=finalize_result,
                        event=prepared_terminal_event,
                        phase=RuntimeHookPhase.AFTER_SESSION_FAILED,
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        execution_profile=execution_profile,
                        invocation_context=invocation_context,
                    ):
                        replay_events.append(event)
            finally:
                _deactivate_session_interaction(session.id)
            return True, tuple(replay_events)
        try:
            try:
                async for event in self._terminal_event_publication.emit(
                    event=session_failed,
                    phase=RuntimeHookPhase.AFTER_SESSION_FAILED,
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    execution_profile=execution_profile,
                    invocation_context=invocation_context,
                    expected_run_operation_epoch=identity.run_epoch,
                ):
                    replay_events.append(event)
            except TerminalEventPublicationUncertain:
                # A concurrent exact replay can clear the run-operation marker or
                # finish environment cleanup before this caller prepares the same
                # deterministic terminal ID. Accept only the already-durable
                # receipt-backed failure outcome; all other conflicts stay uncertain.
                concurrent_records = await self.session_store.query_events(
                    EventQuery(
                        session_id=session.id,
                        event_id=terminal_event_id,
                        limit=2,
                    )
                )
                concurrent_session = await self.session_store.load(session.id)
                if (
                    len(concurrent_records) != 1
                    or not terminal_event_matches(concurrent_records[0].event)
                    or concurrent_session is None
                    or concurrent_session.status is not SessionStatus.FAILED
                ):
                    raise
                concurrent_terminal = concurrent_records[0].event
                terminal_run_operation_id = concurrent_terminal.payload.get(
                    _SESSION_RUN_OPERATION_ID_PAYLOAD_KEY
                )
                if terminal_run_operation_id is not None:
                    await self._terminal_event_publication.clear_run_operation_after_terminal_event(
                        session=session,
                        terminal_event=concurrent_terminal,
                    )
                replay_events.append(copy_event(concurrent_terminal))
        finally:
            _deactivate_session_interaction(session.id)
        return True, tuple(replay_events)

    async def linked_resume_task_id(
        self,
        request: ResumeRequest,
    ) -> tuple[str | None, str | None]:
        return await self.linked_continuation_task_id(
            session_id=request.session_id,
            task_worker_id=request.task_worker_id,
            task_handoff_id=request.task_handoff_id,
        )

    async def linked_continuation_task_id(
        self,
        *,
        session_id: str,
        task_worker_id: str | None,
        task_handoff_id: str | None,
        allow_terminal_failure_replay: bool = False,
        approval_failure_identity: ApprovalTaskFailureIdentity | None = None,
    ) -> tuple[str | None, str | None]:
        """Find the running task that ordinary resume must terminalize.

        Contract authority is checked independently from the session identity at
        the shared resume boundary. This lookup retains the historical behavior of
        re-associating the one running task for ordinary task terminalization. A
        worker-owned task is admitted only after a store-time lease proof, before
        session admission can dispatch provider or tool work.
        """
        session_request_boundary.require_secret_free_session_authority(
            task_worker_id,
            field_name="task_worker_id",
            redactor=self._secret_redactor,
            authority_kind="task continuation authority",
        )
        session_request_boundary.require_secret_free_session_authority(
            task_handoff_id,
            field_name="task_handoff_id",
            redactor=self._secret_redactor,
            authority_kind="task continuation authority",
        )
        if task_handoff_id is not None and task_worker_id is None:
            raise TaskClaimLost("Task handoff authority requires a continuation worker.")
        if self.task_store is None:
            if task_worker_id is not None:
                raise RuntimeError("task_store is required when task_worker_id is set.")
            return None, None
        live_session = await self.session_store.load(session_id)
        if live_session is None:
            raise KeyError(f"Session not found: {session_id}")
        tasks = await self.task_store.list_tasks(
            TaskQuery(
                status=TaskStatus.RUNNING,
                session_id=session_id,
                limit=2,
            )
        )
        if len(tasks) > 1:
            raise RuntimeError(f"Session has multiple running tasks attached: {session_id}")
        if not tasks:
            terminal_tasks = await self.task_store.list_tasks(
                TaskQuery(
                    status=TaskStatus.FAILED,
                    session_id=session_id,
                    limit=2,
                )
            )
            if len(terminal_tasks) > 1:
                raise RuntimeError(f"Session has multiple failed tasks attached: {session_id}")
            if terminal_tasks:
                task = terminal_tasks[0]
                if task.session_instance_id is None:
                    raise TaskClaimLost(
                        "Attached task failure replay lost its session incarnation."
                    )
                await self.require_linked_continuation_task_authority(
                    task_id=task.id,
                    task_worker_id=task_worker_id,
                    task_handoff_id=task_handoff_id,
                    session_id=session_id,
                    session_instance_id=live_session.instance_id,
                    allow_terminal_failure_replay=allow_terminal_failure_replay,
                    approval_failure_identity=approval_failure_identity,
                )
                return task.id, live_session.instance_id
            if task_worker_id is not None:
                raise TaskClaimLost(
                    "Continuation worker authority has no running task attached to the session."
                )
            # Recheck after observing no live attachment: restart atomically
            # detaches the task and retires its previous session. An admission
            # read made before that transaction must not turn the old run into
            # independent work when this attachment lookup happens afterward.
            await self.task_store.require_unretired_task_session(session_id)
            return None, None
        task = tasks[0]
        if task.session_instance_id is None:
            raise TaskClaimLost("Attached task continuation lost its session incarnation.")
        if task.worker_id is None:
            if task_worker_id is not None:
                raise TaskClaimLost(
                    "Continuation worker authority conflicts with a workerless attached task."
                )
            await self.require_linked_continuation_task_authority(
                task_id=task.id,
                task_worker_id=None,
                task_handoff_id=None,
                session_id=session_id,
                session_instance_id=live_session.instance_id,
            )
            return task.id, live_session.instance_id
        if task_worker_id is None:
            raise TaskClaimLost(
                "A worker-owned attached task requires exact continuation worker authority."
            )
        await self.require_linked_continuation_task_authority(
            task_id=task.id,
            task_worker_id=task_worker_id,
            task_handoff_id=task_handoff_id,
            session_id=session_id,
            session_instance_id=live_session.instance_id,
        )
        return task.id, live_session.instance_id

    async def require_linked_continuation_task_authority(
        self,
        *,
        task_id: str,
        task_worker_id: str | None,
        task_lease_expires_at: datetime | None = None,
        task_handoff_id: str | None,
        session_id: str,
        session_instance_id: str,
        allow_terminal_failure_replay: bool = False,
        approval_failure_identity: ApprovalTaskFailureIdentity | None = None,
        expected_runtime_task_terminalization_request_sha256: str | None = None,
    ) -> None:
        if self.task_store is None:
            raise RuntimeError("task_store is required for attached-task continuation authority.")
        self._session_finalization.require_terminal_interaction_publication_protocol()
        session_request_boundary.require_secret_free_session_authority(
            task_worker_id,
            field_name="task_worker_id",
            redactor=self._secret_redactor,
            authority_kind="task continuation authority",
        )
        session_request_boundary.require_secret_free_session_authority(
            task_handoff_id,
            field_name="task_handoff_id",
            redactor=self._secret_redactor,
            authority_kind="task continuation authority",
        )
        if allow_terminal_failure_replay:
            from cayu.runtime._deadline_task_failure import load_deadline_task_failure

            deadline_failure = await load_deadline_task_failure(
                self.task_store,
                self.session_store,
                task_id=task_id,
                session_id=session_id,
                session_instance_id=session_instance_id,
                task_worker_id=task_worker_id,
                task_handoff_id=task_handoff_id,
            )
            if deadline_failure is not None:
                return
        if task_worker_id is None:
            if expected_runtime_task_terminalization_request_sha256 is not None:
                if await self._runtime_task_failure_decision_receipt_is_durable(
                    task_id=task_id,
                    expected_request_sha256=(expected_runtime_task_terminalization_request_sha256),
                    session_id=session_id,
                    session_instance_id=session_instance_id,
                ):
                    return
                raise TaskClaimLost(
                    "Terminal runtime task failure does not match its elected receipt."
                ) from None
            direct_runtime_failure = await load_direct_runtime_task_failure_replay(
                self.task_store,
                task_id=task_id,
                session_id=session_id,
                session_instance_id=session_instance_id,
            )
            if direct_runtime_failure is not None:
                return
            if allow_terminal_failure_replay or approval_failure_identity is not None:
                if approval_failure_identity is not None:
                    expected_error = approval_task_failure_payload(
                        session_id=session_id,
                        identity=approval_failure_identity,
                    )
                    idempotency_key = approval_task_terminalization_idempotency_key(
                        task_id=task_id,
                        session_id=session_id,
                        identity=approval_failure_identity,
                    )
                else:
                    expected_error = provider_operation_task_failure_payload(session_id=session_id)
                    idempotency_key = _task_terminalization_idempotency_key(
                        task_id=task_id,
                        session_id=session_id,
                        kind=TaskTerminalKind.FAILED,
                    )
                direct_failure = await load_direct_task_failure_replay(
                    self.task_store,
                    task_id=task_id,
                    session_id=session_id,
                    session_instance_id=session_instance_id,
                    expected_error=expected_error,
                    claimed_terminalization_idempotency_key=idempotency_key,
                )
                if direct_failure is not None:
                    return
            if task_handoff_id is not None:
                raise TaskClaimLost("Workerless continuation cannot own a handoff generation.")
            await self.task_store.load_direct_attached_task_resume(
                task_id,
                session_id=session_id,
                session_instance_id=session_instance_id,
            )
            return
        if (
            type(self.task_store).load_active_attached_task_worker
            is TaskStore.load_active_attached_task_worker
        ):
            raise NotImplementedError(
                "Worker-owned continuation requires a store-authoritative active lease read."
            )
        try:
            active_task = await self.task_store.load_active_attached_task_worker(
                task_id,
                task_worker_id,
                session_id=session_id,
                session_instance_id=session_instance_id,
            )
            if active_task.interrupted_handoff_id != task_handoff_id:
                raise TaskClaimLost("Attached-task continuation handoff generation changed.")
        except (KeyError, TaskClaimLost):
            if await self._runtime_task_failure_receipt_is_durable(
                task_id=task_id,
                task_worker_id=task_worker_id,
                task_lease_expires_at=task_lease_expires_at,
                task_handoff_id=task_handoff_id,
                session_id=session_id,
                session_instance_id=session_instance_id,
            ):
                return
            if approval_failure_identity is not None:
                await self._require_approval_task_failure_replay(
                    task_id=task_id,
                    task_worker_id=task_worker_id,
                    task_handoff_id=task_handoff_id,
                    session_id=session_id,
                    session_instance_id=session_instance_id,
                    identity=approval_failure_identity,
                )
                return
            if allow_terminal_failure_replay:
                await self._require_provider_operation_task_failure_replay(
                    task_id=task_id,
                    task_worker_id=task_worker_id,
                    task_handoff_id=task_handoff_id,
                    session_id=session_id,
                    session_instance_id=session_instance_id,
                )
                return
            raise

    async def _runtime_task_failure_receipt_is_durable(
        self,
        *,
        task_id: str,
        task_worker_id: str,
        task_lease_expires_at: datetime | None,
        task_handoff_id: str | None,
        session_id: str,
        session_instance_id: str,
    ) -> bool:
        """Return whether one generic failure receipt proves exact worker authority."""

        if self.task_store is None or not self.task_store.supports_idempotent_terminalization:
            return False
        receipt = await self.task_store.load_task_terminalization_receipt(
            task_id,
            _task_terminalization_idempotency_key(
                task_id=task_id,
                session_id=session_id,
                kind=TaskTerminalKind.FAILED,
            ),
        )
        task = await self.task_store.load_task(task_id)
        return runtime_task_failure_receipt_matches(
            receipt=receipt,
            task=task,
            task_worker_id=task_worker_id,
            task_lease_expires_at=task_lease_expires_at,
            task_handoff_id=task_handoff_id,
            session_id=session_id,
            session_instance_id=session_instance_id,
        )

    async def _runtime_task_failure_decision_receipt_is_durable(
        self,
        *,
        task_id: str,
        expected_request_sha256: str,
        session_id: str,
        session_instance_id: str,
    ) -> bool:
        """Authenticate a claimed failure after terminal state cleared live authority."""

        if self.task_store is None or not self.task_store.supports_idempotent_terminalization:
            return False
        receipt = await self.task_store.load_task_terminalization_receipt(
            task_id,
            _task_terminalization_idempotency_key(
                task_id=task_id,
                session_id=session_id,
                kind=TaskTerminalKind.FAILED,
            ),
        )
        task = await self.task_store.load_task(task_id)
        return runtime_task_failure_decision_receipt_matches(
            receipt=receipt,
            task=task,
            expected_request_sha256=expected_request_sha256,
            session_id=session_id,
            session_instance_id=session_instance_id,
        )

    async def _require_approval_task_failure_replay(
        self,
        *,
        task_id: str,
        task_worker_id: str,
        task_handoff_id: str | None,
        session_id: str,
        session_instance_id: str,
        identity: ApprovalTaskFailureIdentity,
    ) -> None:
        """Authenticate one closed-approval retry from its exact task receipt."""

        if self.task_store is None:  # pragma: no cover - guarded by the caller
            raise RuntimeError("task_store is required for task failure replay authority.")
        if not self.task_store.supports_idempotent_terminalization:
            raise TaskClaimLost(
                "Terminal task failure replay requires receipt-backed terminalization."
            )
        receipt = await self.task_store.load_task_terminalization_receipt(
            task_id,
            approval_task_terminalization_idempotency_key(
                task_id=task_id,
                session_id=session_id,
                identity=identity,
            ),
        )
        task = await self.task_store.load_task(task_id)
        if not approval_task_failure_receipt_matches(
            receipt=receipt,
            task=task,
            task_id=task_id,
            task_worker_id=task_worker_id,
            task_handoff_id=task_handoff_id,
            session_id=session_id,
            session_instance_id=session_instance_id,
            identity=identity,
        ):
            raise TaskClaimLost(
                "Terminal approval failure does not prove exact continuation replay authority."
            )

    async def _require_provider_operation_task_failure_replay(
        self,
        *,
        task_id: str,
        task_worker_id: str,
        task_handoff_id: str | None,
        session_id: str,
        session_instance_id: str,
    ) -> None:
        """Authenticate a terminal retry from exact immutable task-failure evidence."""

        if self.task_store is None:  # pragma: no cover - guarded by the caller
            raise RuntimeError("task_store is required for task failure replay authority.")
        if not self.task_store.supports_idempotent_terminalization:
            raise TaskClaimLost(
                "Terminal task failure replay requires receipt-backed terminalization."
            )
        request, request_sha256 = prepare_task_terminalization(
            _provider_operation_task_terminalization_request(
                task_id=task_id,
                task_worker_id=task_worker_id,
                task_handoff_id=task_handoff_id,
                session_id=session_id,
            )
        )
        receipt = await self.task_store.load_task_terminalization_receipt(
            task_id,
            request.idempotency_key,
        )
        task = await self.task_store.load_task(task_id)
        if (
            receipt is None
            or task is None
            or receipt.task_id != task_id
            or receipt.worker_id != task_worker_id
            or receipt.kind is not TaskTerminalKind.FAILED
            or receipt.request_sha256 != request_sha256
            or receipt.task != task
            or task.status is not TaskStatus.FAILED
            or task.session_id != session_id
            or task.session_instance_id != session_instance_id
        ):
            raise TaskClaimLost(
                "Terminal task failure does not prove exact continuation replay authority."
            )

    async def complete_task(
        self,
        *,
        task_id: str,
        task_worker_id: str | None,
        task_handoff_id: str | None,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
    ) -> Task:
        if self.task_store is None:
            raise RuntimeError("task_store is required when RunRequest.task_id is set.")
        result = {
            "session_id": session.id,
            "agent_name": registered_agent.spec.name,
            "environment_name": _environment_name(registered_environment),
        }
        if task_worker_id is not None:
            return await _terminalize_claimed_task(
                self.task_store,
                TaskTerminalizationRequest(
                    task_id=task_id,
                    worker_id=task_worker_id,
                    handoff_id=task_handoff_id,
                    kind=TaskTerminalKind.COMPLETED,
                    result=result,
                    idempotency_key=_task_terminalization_idempotency_key(
                        task_id=task_id,
                        session_id=session.id,
                        kind=TaskTerminalKind.COMPLETED,
                    ),
                ),
            )
        return await self.task_store.complete_task(
            task_id,
            result,
            worker_id=task_worker_id,
        )
