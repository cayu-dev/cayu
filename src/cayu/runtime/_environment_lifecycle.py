"""Session-scoped environment provisioning, binding, and finalization.

The lifecycle owns concrete environment resources and their durable reconnect
state. Session orchestration, task ownership, hook execution, and terminal
status decisions belong to the session engine behind :class:`CayuApp`.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from hashlib import sha256
from math import isfinite
from typing import Any

from cayu._coding_product_authority import (
    CODING_PRODUCT_FINAL_GIT_RECEIPT_SCHEMA,
    is_final_git_result_envelope,
)
from cayu._exception_groups import (
    exception_group_children,
    exception_tree_contains,
    iter_exception_tree,
)
from cayu._task_wait import (
    CapturedAwaitableOutcome,
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    unexpected_child_cancellation_error,
)
from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    copy_json_value,
    copy_label_map,
    require_clean_nonblank,
)
from cayu._workspace_mutation import (
    WorkspaceMutationSettlementError,
    workspace_mutation_task_settlement_probe,
)
from cayu.core.events import Event, EventType, copy_event, event_with_runtime_payload_authority
from cayu.environments import (
    BoundWorkspace,
    DockerCodingWorkspaceBinding,
    Environment,
    EnvironmentAllocationScope,
    EnvironmentAllocationState,
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    ExecutionAdmissionCandidate,
    ExecutionAdmissionError,
    ExecutionRequirements,
    WorkspaceBinding,
    WorkspaceInstructions,
    WorkspaceSnapshot,
    copy_environment,
    copy_workspace_snapshot,
    evaluate_execution_admission,
    load_workspace_instructions,
)
from cayu.environments.bindings import (
    SyncBinding,
    _EnvironmentLifecycleBindAttempt,
    _runtime_owned_workspace_observer_name,
)
from cayu.environments.factory import (
    attach_environment_factory_cleanup_settlement_task,
    combine_environment_factory_cleanup_settlement_tasks,
    environment_factory_cleanup_retry_available,
    environment_factory_cleanup_settlement_task,
    environment_factory_cleanup_settlement_tasks,
    register_environment_factory_cleanup_retry,
    retry_environment_factory_cleanup_settlement_task,
)
from cayu.runners import Runner
from cayu.runtime import _environment_operation_boundary as environment_operation_boundary
from cayu.runtime import _invocation_secrets as invocation_secrets
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._binding_cleanup import (
    append_binding_finalize_cancellation,
    attach_binding_cleanup_status,
    attach_binding_finalize_safe_payload,
    binding_cleanup_payload,
    binding_cleanup_status,
    binding_finalize_cancellation,
    binding_finalize_error_details,
    binding_finalize_explicit_cancellation,
    binding_finalize_failure_payload,
    binding_finalize_fatal_signal,
    binding_finalize_safe_payload,
)
from cayu.runtime._diagnostics import (
    ExceptionDiagnostic,
    _attach_runtime_exception_payload,
    _runtime_exception_payload,
    bound_diagnostic_text,
    exception_diagnostic,
)
from cayu.runtime._environment_allocation import (
    ENVIRONMENT_FACTORY_ALLOCATION_INTENTS_CHECKPOINT_KEY,
    ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY,
    ENVIRONMENT_FACTORY_ALLOCATION_RECEIPTS_CHECKPOINT_KEY,
    ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY,
    DurableEnvironmentAllocationContext,
    EnvironmentAllocationCoordinator,
    EnvironmentAllocationReceipt,
    environment_allocation_parent_session_id,
    environment_allocation_source_owner_session_id,
)
from cayu.runtime._environment_allocation import (
    environment_factory_checkpoint_may_be_committed as _environment_factory_checkpoint_may_be_committed,
)
from cayu.runtime._environment_allocation import (
    mark_environment_factory_checkpoint_may_be_committed as _mark_environment_factory_checkpoint_may_be_committed,
)
from cayu.runtime._environment_allocation import (
    require_bounded_reconnect_metadata as _require_bounded_reconnect_metadata,
)
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._invocation_lifecycle import (
    InvocationContext,
    ReleaseInvocationCommand,
    _release_invocation_command_with_cleanup_authority,
    invocation_lifecycle_receipt_history_present,
)
from cayu.runtime._terminal_evidence import (
    TERMINAL_EVIDENCE_EVENT_TYPES,
    TERMINAL_EVIDENCE_QUERY_LIMIT,
    classify_current_terminal_evidence,
)
from cayu.runtime.egress_authority_transitions import (
    _EGRESS_AUTHORITY_PARKED_OUTCOME,
    EgressAuthorityAdoptionHandler,
    _complete_egress_authority_allocation_parking,
    _discard_egress_authority_allocation_parking_reservation,
    _reserve_egress_authority_allocation_parking,
)
from cayu.runtime.execution_profiles import (
    ExecutionProfileIdentity,
    active_invocation_execution_profile_from_checkpoint,
    event_with_execution_profile_authority,
)
from cayu.runtime.public_authority import PublicAuthorityAliasCodec
from cayu.runtime.sessions import (
    CheckpointTransform,
    EventOrder,
    EventQuery,
    Session,
    SessionRunFenced,
    SessionStatus,
    SessionStore,
    _current_session_invocation_settlement_transition,
    _current_session_invocation_terminal_event,
    _current_session_run_epoch,
    _deactivate_session_run_fence,
    _session_run_operation_from_checkpoint,
    session_user_metadata,
)
from cayu.runtime.workspace_observation_recovery import (
    _WORKSPACE_OBSERVATION_OBSERVER_ALIAS_FIELD,
    _WORKSPACE_OBSERVATION_WORKSPACE_ALIAS_FIELD,
    _project_workspace_observation_authority,
    raise_workspace_observation_concurrent_control,
    restore_workspace_observation_cancellation_requests,
    retain_workspace_observation_pending_cancellation_requests,
)
from cayu.tools._operation_boundary import BoundedInvocationOperationRegistry
from cayu.vaults import SecretRedactor
from cayu.workspaces import (
    WorkspaceIdentity,
    WorkspaceMutationAttributionConfidence,
    WorkspacePathRevision,
    WorkspaceRevisionDeltaStatus,
    WorkspaceRevisionObservation,
    WorkspaceRevisionObservationLimits,
    WorkspaceRevisionObservationStatus,
    compare_workspace_revisions,
)
from cayu.workspaces.revisions import (
    WorkspaceRevisionObservationLimitExceeded,
    copy_bounded_workspace_revision_observation,
)

FAILURE_DIAGNOSTIC_TEXT_MAX_BYTES = 4096
_ENVIRONMENT_FACTORY_RELEASE_ERROR_ATTRIBUTE = "_cayu_environment_factory_release"
_MAX_LAZY_ENVIRONMENT_CLEANUP_SETTLEMENTS = 16
_LAZY_ENVIRONMENT_CLEANUP_ADMISSION_BUDGET_SECONDS = 0.01
DEFAULT_MAX_ENVIRONMENT_LIFECYCLE_OWNERS = 256
_FINAL_WORKSPACE_OBSERVATION_TIMEOUT_SECONDS = 30.0
_MAX_RETAINED_FINAL_WORKSPACE_OBSERVATIONS = 64

_RunFenceReleaseKey = tuple[str, int]

logger = logging.getLogger(__name__)

CheckpointTransformFactory = Callable[[dict[str, Any]], CheckpointTransform]


def _live_allocation_fingerprint(
    allocation: DurableEnvironmentAllocationContext | None,
    receipt: EnvironmentAllocationReceipt | None,
) -> str | None:
    """Return content-free continuity authority for one acknowledged allocation."""

    if allocation is None and receipt is None:
        return None
    reconnect_metadata = (
        receipt.reconnect_metadata
        if allocation is None and receipt is not None
        else allocation.acknowledged_reconnect_metadata
        if allocation is not None
        else None
    )
    if reconnect_metadata is None:
        return None
    if allocation is not None:
        intent = allocation.intent
    elif receipt is not None:
        intent = receipt.intent
    else:  # pragma: no cover - guarded above
        return None
    return sha256(
        canonical_durable_json_bytes(
            {
                "record_type": "cayu.live-environment-allocation",
                "schema_version": 1,
                "allocation_id": intent.allocation_id,
                "provider": intent.provider,
                "adapter_generation": intent.adapter_generation,
                "session_id": intent.session_id,
                "environment_name": intent.environment_name,
                "reconnect_metadata": reconnect_metadata,
            },
            "live_environment_allocation",
        )
    ).hexdigest()


def _reconnect_allocation_fingerprint(reconnect_metadata: dict[str, Any]) -> str | None:
    """Read one factory-published backend identity without exposing provider data."""

    value = reconnect_metadata.get("allocation_fingerprint")
    if value is None:
        return None
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            "Environment factory allocation fingerprint must be a lowercase SHA-256 digest."
        )
    return value


class EnvironmentCapacityError(RuntimeError):
    """Raised before provisioning when process-local lifecycle capacity is full."""


@dataclass(frozen=True)
class EnvironmentBindingResult:
    registered_environment: runtime_records.RegisteredEnvironment | None
    events: list[Event]
    error: Exception | None = None


@dataclass(frozen=True)
class EnvironmentFactoryResolutionResult:
    registered_environment: runtime_records.RegisteredEnvironment | None
    events: list[Event]
    error: Exception | None = None


@dataclass(frozen=True)
class EnvironmentBindingFinalizeResult:
    event: Event
    events: list[Event]
    cancellation: asyncio.CancelledError | None = None
    cancellation_requests_consumed: int = 0


@dataclass(frozen=True)
class _EnvironmentCleanupSettlementOutcome:
    error: BaseException | None = None
    task_cancelled: bool = False


@dataclass
class _ActiveEnvironmentSetup:
    registered_environment: runtime_records.RegisteredEnvironment
    execution_profile: ExecutionProfileIdentity | None = None
    invocation_context: InvocationContext | None = field(default=None, repr=False)
    cleanup_started: bool = False
    cleanup_finished: bool = False
    prebind_release_tombstone: bool = False
    cleanup_error: BaseException | None = None
    cleanup_release_safe: bool = False
    pending_finalize_failure_event: Event | None = None
    cleanup_settlement_started: bool = False
    cleanup_settlement_deferred: bool = False
    cleanup_requires_finalize_retry: bool = False
    cleanup_retry_outcome: str | None = None
    cleanup_retry_metadata: dict[str, Any] | None = None
    cleanup_settlement_task: asyncio.Task[_EnvironmentCleanupSettlementOutcome] | None = None
    release_failed_binding_reservations: Callable[[], None] | None = field(
        default=None,
        repr=False,
    )


def _retain_cleanup_execution_profile(
    owner: _ActiveEnvironmentSetup,
    execution_profile: ExecutionProfileIdentity | None,
) -> None:
    if execution_profile is None:
        return
    if owner.execution_profile is None:
        owner.execution_profile = execution_profile
        return
    if owner.execution_profile != execution_profile:
        raise RuntimeError("Environment cleanup owner execution profile changed.")


def _retain_cleanup_invocation_context(
    owner: _ActiveEnvironmentSetup,
    invocation_context: InvocationContext | None,
) -> None:
    if invocation_context is None:
        return
    if type(invocation_context) is not InvocationContext:
        raise TypeError("invocation_context must be an InvocationContext.")
    current = owner.invocation_context
    if current is invocation_context:
        return
    if current is not None and (
        current.active_profile != invocation_context.active_profile
        or current.binding != invocation_context.binding
        or current.profile is not invocation_context.profile
        or current.registered_agent is not invocation_context.registered_agent
        or current.registered_provider is not invocation_context.registered_provider
        or current.runtime_hooks is not invocation_context.runtime_hooks
        or current.loop_policies is not invocation_context.loop_policies
        or current.request_loop_policies is not invocation_context.request_loop_policies
        or current.budget_policy is not invocation_context.budget_policy
        or current.tool_capability_ceiling != invocation_context.tool_capability_ceiling
        or current.targeted_tool_grants != invocation_context.targeted_tool_grants
    ):
        raise RuntimeError("Environment cleanup owner invocation context changed.")
    if current is not None:
        if current.registered_environment is not owner.registered_environment:
            raise RuntimeError("Environment cleanup owner lost its retained environment context.")
        if invocation_context.registered_environment is owner.registered_environment:
            owner.invocation_context = invocation_context
        return
    if invocation_context.registered_environment is not owner.registered_environment:
        raise RuntimeError("Environment cleanup context does not own the retained environment.")
    owner.invocation_context = invocation_context


def _advance_cleanup_environment(
    owner: _ActiveEnvironmentSetup,
    registered_environment: runtime_records.RegisteredEnvironment,
) -> None:
    context = owner.invocation_context
    if context is not None:
        context = context.with_registered_environment(
            registered_environment,
            validated_profile=context.profile,
        )
    owner.registered_environment = registered_environment
    owner.invocation_context = context


class EnvironmentLifecycle:
    """Own environment factory, workspace binding, and reconnect state."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        checkpoint_transform: CheckpointTransformFactory,
        secret_redactor: SecretRedactor | None = None,
        max_environment_lifecycle_owners: int = DEFAULT_MAX_ENVIRONMENT_LIFECYCLE_OWNERS,
        egress_authority_adoption_handler: EgressAuthorityAdoptionHandler | None = None,
    ) -> None:
        self._session_store = session_store
        self._event_writer = event_writer
        self._checkpoint_transform = checkpoint_transform
        self._secret_redactor = secret_redactor or SecretRedactor()
        self._allocation_coordinator = EnvironmentAllocationCoordinator(
            session_store=session_store,
            checkpoint_transform=checkpoint_transform,
            secret_redactor=self._secret_redactor,
        )
        if (
            type(max_environment_lifecycle_owners) is not int
            or max_environment_lifecycle_owners <= 0
        ):
            raise ValueError("max_environment_lifecycle_owners must be a positive integer.")
        self._max_environment_lifecycle_owners = max_environment_lifecycle_owners
        self._egress_authority_adoption_handler = egress_authority_adoption_handler
        # Factory results and bound workspaces contain process-local handles
        # that cannot be reconstructed from durable session state. Retain the
        # authoritative owner across async-generator yield boundaries until the
        # setup is adopted or finalized. The run fence permits one owner per
        # session.
        self._active_environment_setups: dict[str, _ActiveEnvironmentSetup] = {}
        self._pending_environment_owner_admissions: set[str] = set()
        self._deferred_factory_cleanup_tasks: dict[str, asyncio.Task[None]] = {}
        self._deferred_factory_cleanup_profiles: dict[str, ExecutionProfileIdentity] = {}
        self._deferred_run_fence_release_events: dict[_RunFenceReleaseKey, asyncio.Event] = {}
        self._deferred_run_fence_release_tasks: dict[_RunFenceReleaseKey, asyncio.Task[None]] = {}
        self._final_workspace_observation_operations = BoundedInvocationOperationRegistry(
            max_operations=_MAX_RETAINED_FINAL_WORKSPACE_OBSERVATIONS
        )

    def _retain_deferred_factory_cleanup_execution_profile(
        self,
        session_id: str,
        execution_profile: ExecutionProfileIdentity | None,
    ) -> None:
        if execution_profile is None:
            return
        profiles = getattr(self, "_deferred_factory_cleanup_profiles", None)
        if profiles is None:
            profiles = {}
            self._deferred_factory_cleanup_profiles = profiles
        current = profiles.get(session_id)
        if current is None:
            profiles[session_id] = execution_profile
            return
        if current != execution_profile:
            raise RuntimeError("Environment cleanup owner execution profile changed.")

    def _has_retained_environment_cleanup(self, session_id: str) -> bool:
        return (
            session_id in self._active_environment_setups
            or session_id in self._deferred_factory_cleanup_tasks
        )

    def _signal_environment_cleanup_state_changed(self, session_id: str) -> None:
        events = getattr(self, "_deferred_run_fence_release_events", None)
        if events is None:
            return
        for (owned_session_id, _run_epoch), event in tuple(events.items()):
            if owned_session_id == session_id:
                event.set()

    def _deferred_factory_cleanup_completed(
        self,
        session_id: str,
        _task: asyncio.Task[None],
    ) -> None:
        self._harvest_deferred_factory_cleanups()
        self._signal_environment_cleanup_state_changed(session_id)

    def _harvest_deferred_run_fence_release(
        self,
        key: _RunFenceReleaseKey,
        task: asyncio.Task[None],
    ) -> None:
        tasks = getattr(self, "_deferred_run_fence_release_tasks", None)
        if tasks is None or tasks.get(key) is not task:
            return
        session_id, run_epoch = key
        try:
            task.result()
        except BaseException as error:
            # Failure or loop-shutdown cancellation leaves the durable epoch
            # fenced for explicit worker-startup recovery.
            diagnostic = exception_diagnostic(
                error,
                empty_message="run fence release failed",
                nonportable_message="Run fence release failed with a non-portable diagnostic.",
                redactor=self._secret_redactor,
            )
            logger.warning(
                "Deferred environment run-fence release failed: "
                "session_id=%s run_epoch=%s error_type=%s error=%s",
                session_id,
                run_epoch,
                diagnostic.error_type,
                diagnostic.message,
            )
            return
        del tasks[key]
        events = getattr(self, "_deferred_run_fence_release_events", None)
        if events is not None:
            events.pop(key, None)

    def retire_repaired_run_fence_releases(
        self,
        *,
        session_id: str,
        repaired_run_epoch: int,
    ) -> None:
        """Forget failed older release tasks after durable recovery supersedes them."""

        tasks = getattr(self, "_deferred_run_fence_release_tasks", None)
        if tasks is None:
            return
        events = getattr(self, "_deferred_run_fence_release_events", None)
        for key, task in tuple(tasks.items()):
            owned_session_id, owned_run_epoch = key
            if (
                owned_session_id == session_id
                and owned_run_epoch < repaired_run_epoch
                and task.done()
            ):
                del tasks[key]
                if events is not None:
                    events.pop(key, None)

    async def release_run_fence_after_environment_cleanup(
        self,
        *,
        session_id: str,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
    ) -> None:
        """Release one invocation epoch only after retained cleanup is quiescent."""

        if invocation_context is not None:
            if type(invocation_context) is not InvocationContext:
                raise TypeError("invocation_context must be an InvocationContext.")
            if invocation_context.binding.session_id != session_id:
                raise ValueError("Cleanup context belongs to another session.")
            if (
                execution_profile is not None
                and execution_profile is not invocation_context.profile
            ):
                raise ValueError("Cleanup execution profile conflicts with its context.")
            execution_profile = invocation_context.profile

        run_epoch = _current_session_run_epoch(session_id)
        if invocation_context is not None:
            context_run_epoch = invocation_context.binding.run_epoch
            # Recovery and retained cleanup can execute while a successor epoch
            # is task-locally active. The frozen context, not that ambient
            # token, is the exact authority for the cleanup being settled.
            # _release_quiescent_invocation_fence separately refuses to mutate
            # a positively newer durable epoch.
            run_epoch = context_run_epoch
        if run_epoch is None:
            raise RuntimeError("Environment cleanup has no active session run-fence epoch.")
        terminal_event = _current_session_invocation_terminal_event(session_id)
        key = (session_id, run_epoch)
        tasks = getattr(self, "_deferred_run_fence_release_tasks", None)
        if tasks is None:
            tasks = {}
            self._deferred_run_fence_release_tasks = tasks
        events = getattr(self, "_deferred_run_fence_release_events", None)
        if events is None:
            events = {}
            self._deferred_run_fence_release_events = events
        self.retire_repaired_run_fence_releases(
            session_id=session_id,
            repaired_run_epoch=run_epoch,
        )

        setup_owner = self._active_environment_setups.get(session_id)
        if setup_owner is not None:
            _retain_cleanup_execution_profile(setup_owner, execution_profile)
            _retain_cleanup_invocation_context(setup_owner, invocation_context)
        if session_id in self._deferred_factory_cleanup_tasks:
            self._retain_deferred_factory_cleanup_execution_profile(
                session_id,
                execution_profile,
            )
        existing = tasks.get(key)
        if existing is not None:
            if not self._has_retained_environment_cleanup(session_id):
                self._signal_environment_cleanup_state_changed(session_id)
            _deactivate_session_run_fence(session_id)
            return
        if not self._has_retained_environment_cleanup(session_id):
            await self._release_quiescent_invocation_fence(
                session_id,
                invocation_context=invocation_context,
                terminal_event=terminal_event,
            )
            return
        state_changed = asyncio.Event()
        events[key] = state_changed

        async def release_when_quiescent() -> None:
            while self._has_retained_environment_cleanup(session_id):
                state_changed.clear()
                if not self._has_retained_environment_cleanup(session_id):
                    break
                await state_changed.wait()
            await self._release_quiescent_invocation_fence(
                session_id,
                invocation_context=invocation_context,
                terminal_event=terminal_event,
            )

        task = asyncio.create_task(
            release_when_quiescent(),
            name=f"cayu-environment-run-fence-release-{session_id}",
        )
        tasks[key] = task
        # ``asyncio.create_task`` copied the caller's exact run-fence context.
        # The retained cleanup owner now owns that token; the caller must not
        # continue to present itself as a second in-process owner.
        _deactivate_session_run_fence(session_id)
        task.add_done_callback(
            lambda completed, owned_key=key: self._harvest_deferred_run_fence_release(
                owned_key,
                completed,
            )
        )

    async def _release_quiescent_invocation_fence(
        self,
        session_id: str,
        *,
        invocation_context: InvocationContext | None,
        terminal_event: Event | None,
    ) -> None:
        """Release active invocation authority only after exact terminal settlement."""

        session = await self._session_store.load(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        checkpoint = await self._session_store.load_checkpoint(session_id)
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        if active_profile is None:
            if invocation_lifecycle_receipt_history_present(checkpoint):
                raise RuntimeError("Environment cleanup lost durable invocation profile authority.")
            await self._session_store.release_run_fence(session_id)
            return
        stale_invocation = False
        if invocation_context is not None:
            if invocation_context.binding.session_instance_id != session.instance_id:
                raise RuntimeError("Cleanup context lost exact session-incarnation authority.")
            if session.run_epoch < invocation_context.binding.run_epoch:
                raise RuntimeError("Cleanup context is ahead of durable invocation authority.")
            stale_invocation = session.run_epoch > invocation_context.binding.run_epoch
            if not stale_invocation and invocation_context.active_profile != active_profile:
                raise RuntimeError("Cleanup context lost exact active invocation authority.")
            if stale_invocation:
                # A positively newer session epoch owns every subsequent
                # cleanup decision. Even if an invalid/custom transition left
                # the older profile projection in place, the stale finalizer
                # must not inspect settlement receipts or release authority on
                # the successor's behalf.
                return
        transition = _current_session_invocation_settlement_transition(session_id)
        run_operation = _session_run_operation_from_checkpoint(checkpoint)
        # Prefer exact terminal evidence for the current durable status. An
        # older invocation's settlement receipt can legitimately remain after
        # a later run reaches its own terminal state; consulting that stale
        # receipt first would misclassify positive current evidence as an
        # authority conflict.
        if transition is None and terminal_event is None and run_operation is not None:
            if run_operation.run_epoch != active_profile.run_epoch:
                raise RuntimeError(
                    "Invocation cleanup run operation conflicts with active authority."
                )
            expected_terminal_type = {
                SessionStatus.COMPLETED: EventType.SESSION_COMPLETED,
                SessionStatus.FAILED: EventType.SESSION_FAILED,
                SessionStatus.INTERRUPTED: EventType.SESSION_INTERRUPTED,
            }.get(session.status)
            if expected_terminal_type is not None:
                evidence_records = await self._session_store.query_events(
                    EventQuery(
                        session_id=session_id,
                        event_types=TERMINAL_EVIDENCE_EVENT_TYPES,
                        order_by=EventOrder.SEQUENCE_DESC,
                        limit=TERMINAL_EVIDENCE_QUERY_LIMIT,
                    )
                )
                durable_evidence = classify_current_terminal_evidence(
                    evidence_events=tuple(record.event for record in evidence_records),
                    expected_event_type=expected_terminal_type,
                    run_operation_id=run_operation.operation_id,
                    interruption_request_id=None,
                )
                if len(durable_evidence.events) > 1:
                    raise RuntimeError(
                        "Invocation cleanup found conflicting durable terminal evidence."
                    )
                if durable_evidence.events:
                    terminal_event = durable_evidence.events[0]
        if transition is None and terminal_event is None:
            try:
                transition = await self._session_store.load_invocation_settlement_transition(
                    session_id,
                    expected_session_instance_id=session.instance_id,
                    expected_active_invocation_profile=active_profile,
                )
            except SessionRunFenced:
                # A receipt from an older invocation is not settlement proof
                # for this run. Retain the current fence and preserve the
                # process-control or primary failure that initiated cleanup.
                return
        use_terminal_event = terminal_event is not None and (
            transition is None or transition.to_status is not session.status
        )
        if transition is None and not use_terminal_event:
            if stale_invocation:
                # A newer epoch won before this owner durably settled. Preserve
                # the original run-fence failure and leave successor cleanup to
                # that epoch's exact owner.
                return
            # A process-loss signal can unwind this in-process owner without a
            # terminal publication even though a recoverable provider/tool
            # operation remains durable. Absence of settlement is authority to
            # retain the fence, not authority to replace the primary signal
            # with a cleanup error. Recovery must reconstruct and rebind the
            # invocation before it can release this epoch.
            return
        command_fields: dict[str, Any]
        if use_terminal_event:
            assert terminal_event is not None
            command_fields = {"terminal_session_event": terminal_event}
        else:
            assert transition is not None
            command_fields = {"settlement_transition": transition}
        command = _release_invocation_command_with_cleanup_authority(
            ReleaseInvocationCommand(
                session_id=session.id,
                expected_session_instance_id=session.instance_id,
                expected_run_epoch=active_profile.run_epoch,
                expected_active_profile=active_profile,
                **command_fields,
            )
        )
        try:
            await self._session_store.apply_invocation_lifecycle_command(command)
        except SessionRunFenced:
            if stale_invocation:
                return
            raise

    def _reserve_environment_owner_admission(self, session_id: str) -> None:
        if (
            session_id in self._pending_environment_owner_admissions
            or session_id in self._active_environment_setups
        ):
            return
        owner_count = len(
            self._active_environment_setups.keys() | self._pending_environment_owner_admissions
        )
        if owner_count >= self._max_environment_lifecycle_owners:
            raise EnvironmentCapacityError(
                "Environment lifecycle owner capacity is exhausted "
                f"({owner_count}/{self._max_environment_lifecycle_owners}); "
                "retry after retained cleanup settles or increase "
                "max_environment_lifecycle_owners."
            )
        self._pending_environment_owner_admissions.add(session_id)

    def _release_pending_environment_owner_admission(self, session_id: str) -> None:
        self._pending_environment_owner_admissions.discard(session_id)

    def _promote_environment_owner_admission(
        self,
        session_id: str,
        setup_owner: _ActiveEnvironmentSetup,
    ) -> None:
        """Replace one pending capacity owner with its active setup."""

        if session_id in self._active_environment_setups:
            raise RuntimeError(f"Session {session_id!r} already owns an active environment setup.")
        self._active_environment_setups[session_id] = setup_owner
        # Promotion is synchronous: pending and active are mutually exclusive
        # representations of one lifecycle-capacity owner.
        self._release_pending_environment_owner_admission(session_id)

    async def load_workspace_instructions(
        self,
        registered_environment: runtime_records.RegisteredEnvironment | None,
    ) -> WorkspaceInstructions | None:
        if registered_environment is None:
            return None
        await registered_environment.workspace_mutation_fence.wait_until_available()
        return await load_workspace_instructions(registered_environment.environment)

    async def _settle_retained_environment_cleanups(self) -> None:
        """Start and poll a bounded batch of process-local cleanup owners.

        Normal runtime activity is the delivery mechanism for cleanup retained
        after an ambiguous durable write or incomplete managed teardown. A
        cleanup task remains owned by its exact setup until it reaches a
        terminal outcome; admission never cancels or waits indefinitely for
        dispatched mutation-capable work.
        """

        self._harvest_deferred_factory_cleanups()
        eligible = tuple(
            (session_id, setup_owner)
            for session_id, setup_owner in self._active_environment_setups.items()
            if setup_owner.cleanup_started and setup_owner.cleanup_finished
        )

        async def settle_one(
            session_id: str,
            expected_owner: _ActiveEnvironmentSetup,
        ) -> _EnvironmentCleanupSettlementOutcome:
            try:
                if self._active_environment_setups.get(session_id) is not expected_owner:
                    return _EnvironmentCleanupSettlementOutcome()
                await self.abort_environment_setup(
                    session_id=session_id,
                    original_error=None,
                    allow_deferred_settlement=True,
                    execution_profile=expected_owner.execution_profile,
                    invocation_context=expected_owner.invocation_context,
                )
            except asyncio.CancelledError as error:
                task = asyncio.current_task()
                return _EnvironmentCleanupSettlementOutcome(
                    error=error,
                    task_cancelled=task is not None and task.cancelling() > 0,
                )
            except BaseException as error:
                return _EnvironmentCleanupSettlementOutcome(error=error)
            finally:
                self._signal_environment_cleanup_state_changed(session_id)
            return _EnvironmentCleanupSettlementOutcome()

        def harvest_completed(
            setup_owner: _ActiveEnvironmentSetup,
            *,
            propagate_control_signal: bool,
        ) -> None:
            task = setup_owner.cleanup_settlement_task
            if task is None or not task.done():
                return
            try:
                outcome = task.result()
            except asyncio.CancelledError as error:
                # Cancellation before the coroutine first ran has no structured
                # outcome. It still belongs to the internal settlement task,
                # not to a later environment admission.
                outcome = _EnvironmentCleanupSettlementOutcome(
                    error=error,
                    task_cancelled=True,
                )
            if setup_owner.cleanup_settlement_task is task:
                setup_owner.cleanup_settlement_task = None
            cleanup_error = outcome.error
            if cleanup_error is None or outcome.task_cancelled:
                return
            fatal_signal = binding_finalize_fatal_signal(cleanup_error)
            if (
                fatal_signal is None
                and binding_finalize_explicit_cancellation(cleanup_error) is not None
            ):
                # The cancellation completed inside this retained owner's
                # private settlement task. It is retry state for that owner,
                # not a request to cancel whichever unrelated admission
                # happened to poll it. Direct cancellation of this admission
                # still propagates from the asyncio.wait above.
                return
            if not propagate_control_signal:
                # A control signal completed outside this admission's polling
                # window. Retain the exact owner and retry it; do not replay a
                # historical signal into an unrelated caller.
                return
            if fatal_signal is not None:
                raise cleanup_error

        # Harvest results from an earlier admission before allocating slots.
        # In particular, asyncio.run() loop shutdown may have cancelled the
        # private task after the prior admission returned.
        for _session_id, setup_owner in eligible:
            harvest_completed(
                setup_owner,
                propagate_control_signal=False,
            )

        pending_count = sum(
            setup_owner.cleanup_settlement_task is not None
            and not setup_owner.cleanup_settlement_task.done()
            for setup_owner in self._active_environment_setups.values()
        )
        available_slots = max(
            0,
            _MAX_LAZY_ENVIRONMENT_CLEANUP_SETTLEMENTS - pending_count,
        )
        polled: list[tuple[str, _ActiveEnvironmentSetup]] = []
        for session_id, setup_owner in eligible:
            task = setup_owner.cleanup_settlement_task
            if task is not None:
                polled.append((session_id, setup_owner))
                continue
            if available_slots == 0:
                continue
            setup_owner.cleanup_settlement_task = asyncio.create_task(
                settle_one(session_id, setup_owner),
                name=f"cayu-environment-cleanup-{session_id}",
            )
            available_slots -= 1
            polled.append((session_id, setup_owner))

        # Poll the owned tasks as a group for one small admission budget. The
        # timeout never cancels them: a quick settlement preserves the previous
        # eager cleanup behavior, while one permanently unresolved provider or
        # store call adds only bounded latency to unrelated environment setup.
        tasks = tuple(
            setup_owner.cleanup_settlement_task
            for _session_id, setup_owner in polled
            if setup_owner.cleanup_settlement_task is not None
            and not setup_owner.cleanup_settlement_task.done()
        )
        tasks = (
            *tasks,
            *(task for task in self._deferred_factory_cleanup_tasks.values() if not task.done()),
            *(task for task in self._deferred_run_fence_release_tasks.values() if not task.done()),
        )
        if tasks:
            await asyncio.wait(
                tasks,
                timeout=_LAZY_ENVIRONMENT_CLEANUP_ADMISSION_BUDGET_SECONDS,
            )
        self._harvest_deferred_factory_cleanups()

        for session_id, setup_owner in polled:
            current_owner = self._active_environment_setups.get(session_id)
            if current_owner is not setup_owner:
                continue
            harvest_completed(
                setup_owner,
                propagate_control_signal=True,
            )
            # A pending or failing prefix must not starve later owners. Move
            # only the exact owner observed by this sweep; concurrent
            # replacement or successful retirement wins.
            if self._active_environment_setups.get(session_id) is setup_owner:
                del self._active_environment_setups[session_id]
                self._active_environment_setups[session_id] = setup_owner

    def _require_no_retained_cleanup_for_session(self, session_id: str) -> None:
        setup_owner = self._active_environment_setups.get(session_id)
        if setup_owner is not None and setup_owner.cleanup_started:
            raise RuntimeError(f"Session {session_id!r} still owns incomplete environment cleanup.")
        task = self._deferred_factory_cleanup_tasks.get(session_id)
        if task is not None:
            raise RuntimeError(
                f"Session {session_id!r} still owns incomplete environment factory cleanup."
            )

    def _harvest_deferred_factory_cleanups(self) -> None:
        """Release capacity only after an exact factory cleanup task succeeds."""

        for session_id, task in tuple(self._deferred_factory_cleanup_tasks.items()):
            if not task.done():
                continue
            try:
                task.result()
            except BaseException:
                # A failed settlement has not proved the external mutation or
                # ownership claim quiescent. Retain this admission and reject
                # retries in this process rather than allowing a new owner to
                # obscure an unrecovered resource.
                continue
            if self._deferred_factory_cleanup_tasks.get(session_id) is task:
                del self._deferred_factory_cleanup_tasks[session_id]
                profiles = getattr(self, "_deferred_factory_cleanup_profiles", None)
                if profiles is not None:
                    profiles.pop(session_id, None)
                self._release_pending_environment_owner_admission(session_id)
                self._signal_environment_cleanup_state_changed(session_id)

    def _retry_failed_deferred_factory_cleanups(
        self,
        *,
        attempted_sessions: set[str],
    ) -> None:
        """Dispatch at most one explicit recovery attempt per retained owner."""

        for session_id, task in tuple(self._deferred_factory_cleanup_tasks.items()):
            if session_id in attempted_sessions or not task.done():
                continue
            try:
                task.result()
            except BaseException:
                replacement = retry_environment_factory_cleanup_settlement_task(task)
                if replacement is not task:
                    self._deferred_factory_cleanup_tasks[session_id] = replacement
                    replacement.add_done_callback(
                        lambda completed, owned_session_id=session_id: (
                            self._deferred_factory_cleanup_completed(
                                owned_session_id,
                                completed,
                            )
                        )
                    )
                    attempted_sessions.add(session_id)

    def _transfer_deferred_factory_cleanup(
        self,
        *,
        session_id: str,
        error: BaseException,
    ) -> None:
        """Transfer a timed-out factory release out of an active setup owner."""

        setup_owner = self._active_environment_setups.get(session_id)
        task = self._adopt_deferred_factory_cleanup(
            session_id=session_id,
            error=error,
            execution_profile=(None if setup_owner is None else setup_owner.execution_profile),
        )
        if task is None:
            return
        # This is an ownership transfer, not a new admission: replace the
        # process-local setup owner with a pending admission so the same unit
        # of capacity remains consumed until the exact task succeeds.
        self._pending_environment_owner_admissions.add(session_id)
        self._active_environment_setups.pop(session_id, None)

    def _adopt_deferred_factory_cleanup(
        self,
        *,
        session_id: str,
        error: BaseException,
        execution_profile: ExecutionProfileIdentity | None = None,
    ) -> asyncio.Task[None] | None:
        """Retain every authenticated cleanup owner carried by one failure tree."""

        current = self._deferred_factory_cleanup_tasks.get(session_id)
        task = combine_environment_factory_cleanup_settlement_tasks(
            (
                *((current,) if current is not None else ()),
                *environment_factory_cleanup_settlement_tasks(error),
            ),
            task_name=f"cayu-environment-factory-cleanup-{session_id}",
            failure_message="Environment factory cleanup settlement tasks failed.",
        )
        if task is not None:
            self._deferred_factory_cleanup_tasks[session_id] = task
            self._retain_deferred_factory_cleanup_execution_profile(
                session_id,
                execution_profile,
            )
            task.add_done_callback(
                lambda completed, owned_session_id=session_id: (
                    self._deferred_factory_cleanup_completed(
                        owned_session_id,
                        completed,
                    )
                )
            )
        return task

    async def emit_factory_started(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
    ) -> Event | None:
        """Persist the factory acceptance boundary before provisioning begins."""

        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or registered_agent is not invocation_context.registered_agent
            or registered_environment is not invocation_context.registered_environment
            or execution_profile is not invocation_context.profile
        ):
            raise RuntimeError("Environment factory start lost frozen invocation authority.")
        if registered_environment is not None:
            await registered_environment.workspace_mutation_fence.wait_until_available()
        await self._settle_retained_environment_cleanups()
        self._require_no_retained_cleanup_for_session(session.id)
        if registered_environment is None or registered_environment.factory is None:
            return None
        self._reserve_environment_owner_admission(session.id)
        environment_name = registered_environment.spec.name
        try:
            return await self._event_writer.emit(
                event_with_execution_profile_authority(
                    Event(
                        type=EventType.ENVIRONMENT_FACTORY_STARTED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        payload=_environment_factory_base_payload(
                            session=session,
                            registered_environment=registered_environment,
                        ),
                    ),
                    execution_profile,
                )
            )
        except BaseException:
            self._release_pending_environment_owner_admission(session.id)
            raise

    async def resolve_factory(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        started_event: Event | None,
        operation: EnvironmentFactoryOperation,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
        adopted_factory_result: EnvironmentFactoryResult | None = None,
    ) -> EnvironmentFactoryResolutionResult:
        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or registered_agent is not invocation_context.registered_agent
            or registered_environment is not invocation_context.registered_environment
            or execution_profile is not invocation_context.profile
        ):
            raise RuntimeError("Environment factory resolution lost frozen invocation authority.")
        if registered_environment is None or registered_environment.factory is None:
            if started_event is not None:
                raise AssertionError("Factory start event exists without a registered factory.")
            return EnvironmentFactoryResolutionResult(
                registered_environment=registered_environment,
                events=[],
            )
        if started_event is None:
            raise AssertionError("Registered environment factory was not started.")
        if (
            adopted_factory_result is not None
            and type(adopted_factory_result) is not EnvironmentFactoryResult
        ):
            raise TypeError("adopted_factory_result must be an EnvironmentFactoryResult.")

        factory = registered_environment.factory
        environment_name = registered_environment.spec.name
        base_payload = _environment_factory_base_payload(
            session=session,
            registered_environment=registered_environment,
        )
        events: list[Event] = []
        result: EnvironmentFactoryResult | None = adopted_factory_result
        environment: Environment | None = None
        allocation_context: DurableEnvironmentAllocationContext | None = None
        allocation_checkpointed = False
        allocation_checkpoint_may_be_committed = False
        effective_operation = operation
        try:
            parent_session_id = environment_allocation_parent_session_id(session)
            source_allocation_owner_session_id = environment_allocation_source_owner_session_id(
                session,
                environment_name=environment_name,
            )
            checkpoint = await self._session_store.load_checkpoint(session.id)
            reconnect_metadata, allocation_owner = _factory_reconnect_state_from_checkpoint(
                checkpoint,
                environment_name=environment_name,
            )
            allocation_record = self._allocation_coordinator.record_from_checkpoint(
                checkpoint,
                environment_name=environment_name,
            )
            allocation_receipt = self._allocation_coordinator.receipt_from_checkpoint(
                checkpoint,
                environment_name=environment_name,
            )
            if allocation_record is not None and allocation_receipt is not None:
                raise ValueError(
                    "Environment allocation checkpoint contains both a pending intent "
                    "and a published receipt."
                )
            if allocation_receipt is not None and (
                allocation_receipt.intent.environment_name != environment_name
                or allocation_receipt.intent.session_id != allocation_owner
                or allocation_receipt.reconnect_metadata != reconnect_metadata
            ):
                raise ValueError(
                    "Environment allocation receipt conflicts with durable reconnect state."
                )
            if (
                operation is EnvironmentFactoryOperation.RECONNECT
                and allocation_owner != session.id
            ):
                # A failed setup has no session-owned allocation to reconnect,
                # and a fork inherits its parent's checkpoint only as context.
                # Allocation provenance is authoritative; Cayu does not scan
                # historical events to infer ownership.
                effective_operation = EnvironmentFactoryOperation.CREATE
            request = EnvironmentFactoryRequest(
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                execution_profile_fingerprint=(
                    None if execution_profile is None else execution_profile.fingerprint
                ),
                operation=effective_operation,
                parent_session_id=parent_session_id,
                causal_budget_id=session.causal_budget_id,
                labels=session.labels,
                metadata=session_user_metadata(session.metadata),
                reconnect_metadata=reconnect_metadata,
                execution_requirements=registered_agent.execution_requirements,
            )
            admission_candidate = factory.execution_admission_candidate(request)
            if admission_candidate is not None and not isinstance(
                admission_candidate,
                ExecutionAdmissionCandidate,
            ):
                raise TypeError(
                    "EnvironmentFactory.execution_admission_candidate must return "
                    "ExecutionAdmissionCandidate or None."
                )
            evaluate_execution_admission(
                candidate=(
                    environment_name
                    if admission_candidate is None
                    else admission_candidate.candidate
                ),
                requirements=request.execution_requirements,
                evidence=None if admission_candidate is None else admission_candidate.evidence,
                stage="pre_create",
            ).require_admitted()
            if adopted_factory_result is not None:
                if operation is not EnvironmentFactoryOperation.RECONNECT:
                    raise ValueError(
                        "An adopted egress environment can only continue a resumed session."
                    )
                if allocation_record is not None:
                    raise RuntimeError(
                        "An adopted egress environment conflicts with an incomplete "
                        "remote allocation intent."
                    )
                if allocation_owner != session.id:
                    raise RuntimeError(
                        "An adopted egress environment has no session-owned allocation checkpoint."
                    )
                effective_operation = EnvironmentFactoryOperation.RECONNECT
            elif effective_operation is EnvironmentFactoryOperation.CREATE:
                allocation_scope = factory.allocation_scope(request)
                if allocation_scope is not None and type(allocation_scope) is not (
                    EnvironmentAllocationScope
                ):
                    raise TypeError(
                        "EnvironmentFactory.allocation_scope must return "
                        "EnvironmentAllocationScope or None."
                    )
                if allocation_scope is None:
                    if allocation_record is not None:
                        raise RuntimeError(
                            "Environment factory has an incomplete remote allocation "
                            "but no longer declares its durable allocation scope."
                        )
                    if allocation_receipt is not None:
                        raise RuntimeError(
                            "Environment factory has a published remote allocation receipt "
                            "but no longer declares its durable allocation scope."
                        )
                    result = await environment_operation_boundary.await_environment_operation(
                        lambda: factory.create(request),
                        operation_name="Environment factory creation",
                        redactor=self._secret_redactor,
                    )
                else:
                    allocation_context = self._allocation_coordinator.context(
                        session_id=session.id,
                        inherited_owner_session_id=source_allocation_owner_session_id,
                        environment_name=environment_name,
                        scope=allocation_scope,
                        existing=allocation_record,
                    )
                    result = await environment_operation_boundary.await_environment_operation(
                        lambda: factory.create_recoverable(request, allocation_context),
                        operation_name="Recoverable environment factory creation",
                        redactor=self._secret_redactor,
                    )
            else:
                if allocation_record is not None:
                    raise RuntimeError(
                        "Environment factory reconnect state conflicts with an incomplete "
                        "remote allocation intent."
                    )
                result = await environment_operation_boundary.await_environment_operation(
                    lambda: factory.create(request),
                    operation_name="Environment factory reconnect",
                    redactor=self._secret_redactor,
                )
            if type(result) is not EnvironmentFactoryResult:
                raise TypeError(
                    "Environment factory resolution must return EnvironmentFactoryResult."
                )
            environment = copy_environment(result.environment)
            if environment.spec.name != environment_name:
                raise ValueError(
                    "Environment factory returned a different environment name: "
                    f"{environment.spec.name!r} != {environment_name!r}"
                )
            if environment.runner is not None or environment.binding is None:
                self._require_runner_admitted(
                    execution_candidate=(
                        None if admission_candidate is None else admission_candidate.candidate
                    ),
                    fallback_candidate=environment_name,
                    requirements=request.execution_requirements,
                    runner=environment.runner,
                )
            reconnect_metadata = copy_json_value(
                result.reconnect_metadata,
                "reconnect_metadata",
            )
            self._secret_redactor.require_no_secret_keys(
                reconnect_metadata,
                field_name="EnvironmentFactoryResult.reconnect_metadata",
                preserve_keys={"allocation_fingerprint"},
                match_short_substrings=True,
            )
            if self._secret_redactor.redact_json_values(reconnect_metadata) != (reconnect_metadata):
                raise ValueError(
                    "EnvironmentFactoryResult.reconnect_metadata contains a workload "
                    "secret and cannot be checkpointed without changing reconnect semantics."
                )
            _require_bounded_reconnect_metadata(reconnect_metadata)
            reconnect_allocation_fingerprint = _reconnect_allocation_fingerprint(reconnect_metadata)
            if (
                effective_operation is EnvironmentFactoryOperation.RECONNECT
                and allocation_receipt is not None
                and reconnect_metadata != allocation_receipt.reconnect_metadata
            ):
                raise RuntimeError(
                    "Recoverable environment factory changed its immutable reconnect "
                    "identity during reconnect."
                )
            try:
                if allocation_context is None:
                    await self._checkpoint_factory_reconnect_metadata(
                        session_id=session.id,
                        environment_name=environment_name,
                        reconnect_metadata=reconnect_metadata,
                    )
                else:
                    if allocation_context.state is not EnvironmentAllocationState.ACKNOWLEDGED:
                        raise RuntimeError(
                            "Recoverable environment factory returned before provider "
                            "acknowledgement was durable."
                        )
                    if allocation_context.acknowledged_reconnect_metadata != reconnect_metadata:
                        raise RuntimeError(
                            "Recoverable environment factory result changed the acknowledged "
                            "reconnect identity."
                        )
                    await allocation_context.publish()
            except BaseException as exc:
                # The checkpoint helper reconciles any failure after the
                # transactional write begins. Only a durable read proving the
                # expected owner and metadata absent permits the allocation to
                # be discarded.
                allocation_checkpoint_may_be_committed = (
                    _environment_factory_checkpoint_may_be_committed(exc)
                )
                raise
            allocation_checkpointed = True
            completed_event = Event(
                type=EventType.ENVIRONMENT_FACTORY_COMPLETED,
                session_id=session.id,
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                payload={
                    **base_payload,
                    "environment_name": environment.spec.name,
                    "result_metadata": copy_json_value(
                        result.metadata,
                        "result_metadata",
                    ),
                    "reconnect_metadata": reconnect_metadata,
                    **(
                        {}
                        if allocation_context is None
                        else {"allocation_id": (allocation_context.intent.allocation_id)}
                    ),
                },
            )
            if allocation_context is not None:
                completed_event = event_with_runtime_payload_authority(
                    completed_event,
                    "allocation_id",
                )
            completed_event = event_with_execution_profile_authority(
                completed_event,
                execution_profile,
            )
            events.append(await self._event_writer.emit(completed_event))
            if result is None:
                raise RuntimeError("Environment factory did not return an owned result.")
            if environment is None:
                raise RuntimeError("Environment factory did not produce an environment.")
            resolved_environment = runtime_records.RegisteredEnvironment(
                spec=registered_environment.spec,
                environment=environment,
                factory_backed=True,
                factory_execution_profile_identity=(
                    registered_environment.factory_execution_profile_identity
                ),
                execution_candidate=(
                    None if admission_candidate is None else admission_candidate.candidate
                ),
                unclaimed_factory_result=result,
                live_allocation_fingerprint=(
                    reconnect_allocation_fingerprint
                    or _live_allocation_fingerprint(
                        allocation_context,
                        allocation_receipt,
                    )
                ),
                registration_source=registered_environment.registration_source,
                registration_symbol=registered_environment.registration_symbol,
                workspace_mutation_fence=(
                    registered_environment.workspace_mutation_fence.child_fence()
                ),
            )
            self._promote_environment_owner_admission(
                session.id,
                _ActiveEnvironmentSetup(
                    registered_environment=resolved_environment,
                    execution_profile=execution_profile,
                    invocation_context=(
                        None
                        if invocation_context is None
                        else invocation_context.with_registered_environment(
                            resolved_environment,
                            validated_profile=invocation_context.profile,
                        )
                    ),
                ),
            )
        except BaseException as exc:
            self._adopt_deferred_factory_cleanup(
                session_id=session.id,
                error=exc,
                execution_profile=execution_profile,
            )
            if result is not None:
                release_action = (
                    EnvironmentFactoryReleaseAction.PRESERVE
                    if allocation_checkpointed
                    or allocation_checkpoint_may_be_committed
                    or effective_operation is EnvironmentFactoryOperation.RECONNECT
                    else EnvironmentFactoryReleaseAction.DISCARD
                )
                discard_fence_acquired: bool | None = None
                discard_fence_error: BaseException | None = None
                if (
                    release_action is EnvironmentFactoryReleaseAction.DISCARD
                    and allocation_context is not None
                ):
                    try:
                        discard_fence_acquired = await allocation_context.mark_reaping()
                    except BaseException as fence_error:
                        # Without a durable cleanup fence, provider deletion is
                        # never safe. Preserve the exact allocation and report
                        # the fence failure alongside the original setup error.
                        discard_fence_error = fence_error
                        release_action = EnvironmentFactoryReleaseAction.PRESERVE
                    else:
                        if not discard_fence_acquired:
                            # Another worker atomically published the same
                            # acknowledged allocation before this worker could
                            # claim cleanup. Detach local handles without
                            # deleting the now-durable provider resource.
                            release_action = EnvironmentFactoryReleaseAction.PRESERVE
                try:
                    release_payload = await _release_unclaimed_factory_result(
                        result,
                        action=release_action,
                        original_error=exc,
                        redactor=self._secret_redactor,
                    )
                finally:
                    self._adopt_deferred_factory_cleanup(
                        session_id=session.id,
                        error=exc,
                    )
                if discard_fence_acquired is not None:
                    release_payload["discard_fence_acquired"] = discard_fence_acquired
                if discard_fence_error is not None:
                    diagnostic = exception_diagnostic(
                        discard_fence_error,
                        empty_message="environment allocation cleanup fence failed",
                        nonportable_message=(
                            "Environment allocation cleanup fence failed with a "
                            "non-portable diagnostic."
                        ),
                        redactor=self._secret_redactor,
                    )
                    release_payload.update(
                        {
                            "discard_fence_acquired": False,
                            "discard_fence_error": diagnostic.message,
                            "discard_fence_error_type": diagnostic.error_type,
                        }
                    )
                    _add_exception_note_safely(
                        exc,
                        "Environment allocation cleanup was preserved because its durable "
                        f"fence failed: {diagnostic.error_type}: {diagnostic.message}.",
                    )
                _attach_environment_factory_release_payload(exc, release_payload)
                if discard_fence_error is not None:
                    fatal_signal = binding_finalize_fatal_signal(discard_fence_error)
                    if fatal_signal is not None:
                        raise fatal_signal from exc
                    if binding_finalize_explicit_cancellation(discard_fence_error) is not None:
                        raise discard_fence_error from exc
            ordinary_failure = isinstance(exc, Exception) or exception_tree_contains(exc, Exception)
            fatal_signal = binding_finalize_fatal_signal(exc)
            if fatal_signal is not None and not ordinary_failure:
                raise
            if ordinary_failure:
                try:
                    failed_event = Event(
                        type=EventType.ENVIRONMENT_FACTORY_FAILED,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                        payload={
                            **base_payload,
                            **(
                                {}
                                if allocation_context is None
                                else {"allocation_id": (allocation_context.intent.allocation_id)}
                            ),
                            **exception_failure_payload(
                                exc,
                                redactor=self._secret_redactor,
                            ),
                        },
                    )
                    if allocation_context is not None:
                        failed_event = event_with_runtime_payload_authority(
                            failed_event,
                            "allocation_id",
                        )
                    failed_event = event_with_execution_profile_authority(
                        failed_event,
                        execution_profile,
                    )
                    events.append(await self._event_writer.emit(failed_event))
                except BaseException as publication_error:
                    raise BaseExceptionGroup(
                        "Environment factory failure publication also failed.",
                        [exc, publication_error],
                    ) from publication_error
            if fatal_signal is not None or not isinstance(exc, Exception):
                raise
            return EnvironmentFactoryResolutionResult(
                registered_environment=registered_environment,
                events=events,
                error=exc,
            )
        finally:
            self._harvest_deferred_factory_cleanups()
            if session.id not in self._deferred_factory_cleanup_tasks:
                self._release_pending_environment_owner_admission(session.id)

        return EnvironmentFactoryResolutionResult(
            registered_environment=resolved_environment,
            events=events,
        )

    async def checkpoint_preserving_runtime_state(
        self,
        session_id: str,
        checkpoint: dict[str, Any],
    ) -> None:
        await self._session_store.transform_checkpoint(
            session_id,
            self.checkpoint_transform_preserving_runtime_state(checkpoint),
        )

    def checkpoint_transform_preserving_runtime_state(
        self,
        checkpoint: dict[str, Any],
    ) -> CheckpointTransform:
        """Build one atomic transform retaining environment-owned checkpoint state."""

        copied_checkpoint = copy_json_value(checkpoint, "checkpoint")
        runtime_keys = (
            ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY,
            ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY,
            ENVIRONMENT_FACTORY_ALLOCATION_INTENTS_CHECKPOINT_KEY,
            ENVIRONMENT_FACTORY_ALLOCATION_RECEIPTS_CHECKPOINT_KEY,
        )

        def transform(session: Session, current: dict[str, Any] | None) -> dict[str, Any]:
            replacement = copy_json_value(copied_checkpoint, "checkpoint")
            if current is not None:
                for key in runtime_keys:
                    if key in replacement:
                        continue
                    state = current.get(key)
                    if state is not None:
                        if type(state) is not dict:
                            raise ValueError(f"{key} checkpoint state must be an object.")
                        replacement[key] = copy_json_value(state, key)
            transformed = self._checkpoint_transform(replacement)(session, current)
            if transformed is None:
                raise RuntimeError(
                    "Checkpoint preservation transform unexpectedly deleted the checkpoint."
                )
            return transformed

        return transform

    async def emit_binding_started(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
    ) -> Event | None:
        """Persist the binding acceptance boundary before workspace setup begins."""

        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or registered_agent is not invocation_context.registered_agent
            or registered_environment is not invocation_context.registered_environment
            or execution_profile is not invocation_context.profile
        ):
            raise RuntimeError("Environment binding start lost frozen invocation authority.")
        await self._settle_retained_environment_cleanups()
        self._require_no_retained_cleanup_for_session(session.id)
        if (
            registered_environment is None
            or registered_environment.bound_workspace is not None
            or registered_environment.environment.binding is None
        ):
            return None
        self._reserve_environment_owner_admission(session.id)
        environment_name = _environment_name(registered_environment)
        try:
            return await self._event_writer.emit(
                _event_with_binding_generation_authority(
                    event_with_execution_profile_authority(
                        Event(
                            type=EventType.ENVIRONMENT_BINDING_STARTED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            payload=_binding_base_payload(
                                registered_environment,
                                session_id=session.id,
                                public_authority_alias_codec=(
                                    self._session_store.public_authority_alias_codec
                                ),
                                redactor=self._secret_redactor,
                            ),
                        ),
                        execution_profile,
                    )
                )
            )
        except BaseException:
            self._release_pending_environment_owner_admission(session.id)
            raise

    def _require_runner_admitted(
        self,
        *,
        execution_candidate: str | None,
        fallback_candidate: str,
        requirements: ExecutionRequirements,
        runner: Runner | None,
    ) -> None:
        if execution_candidate is None and not requirements.required_capabilities():
            return
        admission_candidate = None if runner is None else runner.execution_admission_candidate()
        if admission_candidate is not None and not isinstance(
            admission_candidate,
            ExecutionAdmissionCandidate,
        ):
            raise TypeError(
                "Runner.execution_admission_candidate must return "
                "ExecutionAdmissionCandidate or None."
            )
        if execution_candidate is not None and admission_candidate is None:
            missing_evidence = evaluate_execution_admission(
                candidate=execution_candidate,
                requirements=requirements,
                evidence=None,
                stage="pre_exposure",
            )
            if missing_evidence.status == "refused":
                missing_evidence.require_admitted()
            raise RuntimeError(
                f"Execution candidate {execution_candidate!r} supplied pre-create evidence, "
                "but the final runner supplied no execution admission evidence."
            )
        candidate = execution_candidate
        if candidate is None:
            candidate = (
                fallback_candidate if admission_candidate is None else admission_candidate.candidate
            )
        evaluate_execution_admission(
            candidate=candidate,
            requirements=requirements,
            evidence=None if admission_candidate is None else admission_candidate.evidence,
            stage="pre_exposure",
        ).require_admitted()

    def _require_registered_environment_admitted(
        self,
        *,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment,
    ) -> None:
        self._require_runner_admitted(
            execution_candidate=registered_environment.execution_candidate,
            fallback_candidate=registered_environment.spec.name,
            requirements=registered_agent.execution_requirements,
            runner=registered_environment.environment.runner,
        )

    async def _release_unexposed_factory_environment(
        self,
        registered_environment: runtime_records.RegisteredEnvironment,
        *,
        error: BaseException,
        release_failed_binding_reservations: Callable[[], None] | None = None,
    ) -> tuple[runtime_records.RegisteredEnvironment, dict[str, Any] | None]:
        result = registered_environment.unclaimed_factory_result
        if result is None:
            return registered_environment, None
        # Resolution checkpoints every factory result before returning it to
        # binding. Once committed, release may detach live handles but must not
        # destroy the durable allocation that a later resume will reconnect.
        release_payload = await _release_unclaimed_factory_result(
            result,
            action=EnvironmentFactoryReleaseAction.PRESERVE,
            original_error=error,
            redactor=self._secret_redactor,
            on_quiescent=(release_failed_binding_reservations),
        )
        _attach_environment_factory_release_payload(error, release_payload)
        return (
            replace(
                registered_environment,
                unclaimed_factory_result=None,
            ),
            release_payload,
        )

    async def bind(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        started_event: Event | None,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
    ) -> EnvironmentBindingResult:
        if invocation_context is not None and (
            invocation_context.binding.session_id != session.id
            or registered_agent is not invocation_context.registered_agent
            or registered_environment is not invocation_context.registered_environment
            or execution_profile is not invocation_context.profile
        ):
            raise RuntimeError("Environment binding lost frozen invocation authority.")
        if registered_environment is None:
            if started_event is not None:
                raise AssertionError("Binding start event exists without an environment.")
            return EnvironmentBindingResult(registered_environment=None, events=[])
        try:
            await registered_environment.workspace_mutation_fence.wait_until_available()
        except Exception as exc:
            return EnvironmentBindingResult(
                registered_environment=registered_environment,
                events=[],
                error=exc,
            )
        if registered_environment.bound_workspace is not None:
            if started_event is not None:
                raise AssertionError("Binding start event exists for an already-bound workspace.")
            try:
                self._require_registered_environment_admitted(
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                )
            except Exception as exc:
                return EnvironmentBindingResult(
                    registered_environment=registered_environment,
                    events=[],
                    error=exc,
                )
            adopted_environment = (
                registered_environment
                if registered_environment.unclaimed_factory_result is None
                else replace(
                    registered_environment,
                    unclaimed_factory_result=None,
                )
            )
            setup_owner = self._active_environment_setups.get(session.id)
            if setup_owner is not None:
                _advance_cleanup_environment(setup_owner, adopted_environment)
            return EnvironmentBindingResult(
                registered_environment=adopted_environment,
                events=[],
            )
        binding = registered_environment.environment.binding
        if binding is None:
            if started_event is not None:
                raise AssertionError("Binding start event exists without a workspace binding.")
            try:
                self._require_registered_environment_admitted(
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                )
            except Exception as exc:
                try:
                    (
                        registered_environment,
                        _release_payload,
                    ) = await self._release_unexposed_factory_environment(
                        registered_environment,
                        error=exc,
                    )
                finally:
                    self._transfer_deferred_factory_cleanup(
                        session_id=session.id,
                        error=exc,
                    )
                self._active_environment_setups.pop(session.id, None)
                return EnvironmentBindingResult(
                    registered_environment=registered_environment,
                    events=[],
                    error=exc,
                )
            adopted_environment = (
                registered_environment
                if registered_environment.unclaimed_factory_result is None
                else replace(
                    registered_environment,
                    unclaimed_factory_result=None,
                )
            )
            self._active_environment_setups.pop(session.id, None)
            return EnvironmentBindingResult(
                registered_environment=adopted_environment,
                events=[],
            )
        if started_event is None:
            raise AssertionError("Registered workspace binding was not started.")

        environment_name = _environment_name(registered_environment)
        events: list[Event] = []
        base_payload = _binding_base_payload(
            registered_environment,
            session_id=session.id,
            public_authority_alias_codec=(self._session_store.public_authority_alias_codec),
            redactor=self._secret_redactor,
        )
        setup_owner = self._active_environment_setups.get(session.id)
        if setup_owner is None:
            setup_owner = _ActiveEnvironmentSetup(
                registered_environment=registered_environment,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
            )
            self._active_environment_setups[session.id] = setup_owner
        else:
            _retain_cleanup_execution_profile(setup_owner, execution_profile)
            _retain_cleanup_invocation_context(setup_owner, invocation_context)
        self._release_pending_environment_owner_admission(session.id)
        release_failed_binding_reservations: Callable[[], None] | None = None
        try:
            if registered_environment.unclaimed_factory_result is not None:
                attempt = _EnvironmentLifecycleBindAttempt()
                try:
                    bound = await environment_operation_boundary.await_environment_operation(
                        lambda: binding._bind_for_environment_lifecycle(
                            registered_environment.environment.workspace,
                            registered_environment.environment.runner,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            _attempt=attempt,
                        ),
                        operation_name="Environment workspace binding",
                        redactor=self._secret_redactor,
                    )
                finally:
                    release_failed_binding_reservations = attempt.release_failed_reservations
                    setup_owner.release_failed_binding_reservations = (
                        release_failed_binding_reservations
                    )
            else:
                bound = await environment_operation_boundary.await_environment_operation(
                    lambda: binding.bind(
                        registered_environment.environment.workspace,
                        registered_environment.environment.runner,
                        session_id=session.id,
                        agent_name=registered_agent.spec.name,
                        environment_name=environment_name,
                    ),
                    operation_name="Environment workspace binding",
                    redactor=self._secret_redactor,
                )
        except BaseException as exc:
            cleanup_status = binding_cleanup_status(exc)
            retry_error: BaseException | None = None
            if cleanup_status is not None:
                cleanup_status.retry_attempted = True
                retry_operation = cleanup_status.retry
                try:
                    await environment_operation_boundary.await_environment_operation(
                        retry_operation,
                        operation_name="Environment binding cleanup retry",
                        redactor=self._secret_redactor,
                    )
                except asyncio.CancelledError as cleanup_exc:
                    cleanup_status.retry_error = cleanup_exc
                    retry_error = cleanup_exc
                except BaseException as cleanup_exc:
                    cleanup_status.retry_error = cleanup_exc
                    retry_error = cleanup_exc
                finally:
                    # The callback is binding-owned authority needed only for this
                    # retry. Do not leave it reachable from an exception that may
                    # cross the lifecycle boundary.
                    cleanup_status.retry = (
                        environment_operation_boundary.completed_environment_operation
                    )
                    del retry_operation
            ordinary_failure = isinstance(exc, Exception) or exception_tree_contains(exc, Exception)
            fatal_signal = binding_finalize_fatal_signal(exc)
            if fatal_signal is not None and not ordinary_failure:
                raise
            propagated_error: BaseException = exc
            if retry_error is not None:
                propagated_error = BaseExceptionGroup(
                    "Binding and binding-owned cleanup both failed.",
                    [exc, retry_error],
                )
                if cleanup_status is not None:
                    attach_binding_cleanup_status(propagated_error, cleanup_status)
            try:
                (
                    registered_environment,
                    _release_payload,
                ) = await self._release_unexposed_factory_environment(
                    registered_environment,
                    error=exc,
                    release_failed_binding_reservations=(release_failed_binding_reservations),
                )
            finally:
                self._transfer_deferred_factory_cleanup(
                    session_id=session.id,
                    error=exc,
                )
                if session.id in self._deferred_factory_cleanup_tasks and setup_owner is not None:
                    # Main retains this tombstone so terminalization cannot
                    # reuse its stale pre-bind factory result. The deferred
                    # task remains the mutation owner, while both records share
                    # one session identity for admission accounting.
                    _advance_cleanup_environment(
                        setup_owner,
                        replace(
                            registered_environment,
                            unclaimed_factory_result=None,
                        ),
                    )
                    setup_owner.cleanup_started = True
                    setup_owner.prebind_release_tombstone = True
                    self._active_environment_setups[session.id] = setup_owner
            # Retain a cleanup tombstone until the run finalizer executes. A
            # caller cancellation is terminalized after this method unwinds,
            # and that terminal path still holds its pre-bind environment
            # snapshot. Removing the owner here would make that stale snapshot
            # look authoritative and release the same factory result twice.
            setup_owner = self._active_environment_setups.get(session.id)
            if setup_owner is not None:
                _advance_cleanup_environment(setup_owner, registered_environment)
                setup_owner.cleanup_started = True
                setup_owner.prebind_release_tombstone = True
            if ordinary_failure:
                failure_payload = {
                    **base_payload,
                    **exception_failure_payload(
                        exc,
                        redactor=self._secret_redactor,
                    ),
                }
                try:
                    events.append(
                        await self._event_writer.emit(
                            _event_with_binding_generation_authority(
                                event_with_execution_profile_authority(
                                    Event(
                                        type=EventType.ENVIRONMENT_BINDING_FAILED,
                                        session_id=session.id,
                                        agent_name=registered_agent.spec.name,
                                        environment_name=environment_name,
                                        payload=failure_payload,
                                    ),
                                    execution_profile,
                                ),
                            )
                        )
                    )
                except BaseException as publication_error:
                    publication_failure = BaseExceptionGroup(
                        "Binding failure publication also failed.",
                        [propagated_error, publication_error],
                    )
                    if cleanup_status is not None:
                        attach_binding_cleanup_status(publication_failure, cleanup_status)
                    raise publication_failure from (
                        fatal_signal or binding_finalize_cancellation(exc) or publication_error
                    )
            if fatal_signal is not None or not isinstance(exc, Exception):
                if retry_error is not None:
                    raise propagated_error from (fatal_signal or binding_finalize_cancellation(exc))
                raise
            if (
                retry_error is not None
                and binding_finalize_explicit_cancellation(retry_error) is not None
            ):
                raise propagated_error from retry_error
            return EnvironmentBindingResult(
                registered_environment=registered_environment,
                events=events,
                error=exc,
            )

        bound_environment = copy_environment(registered_environment.environment)
        bound_environment.workspace = bound.workspace
        bound_environment.runner = bound.runner
        bound_registered_environment = runtime_records.RegisteredEnvironment(
            spec=registered_environment.spec,
            environment=bound_environment,
            factory_backed=registered_environment.factory_backed,
            runner_execution_profile_identity=(
                registered_environment.runner_execution_profile_identity
            ),
            factory_execution_profile_identity=(
                registered_environment.factory_execution_profile_identity
            ),
            bound_workspace=bound,
            binding_payload=copy_json_value(base_payload, "binding_payload"),
            execution_candidate=registered_environment.execution_candidate,
            retained_factory_result=(registered_environment.unclaimed_factory_result),
            preserve_factory_allocation=(
                registered_environment.unclaimed_factory_result is not None
            ),
            live_allocation_fingerprint=(registered_environment.live_allocation_fingerprint),
            registration_source=registered_environment.registration_source,
            registration_symbol=registered_environment.registration_symbol,
            binding_generation_id=registered_environment.binding_generation_id,
            workspace_mutation_fence=(registered_environment.workspace_mutation_fence),
        )
        # Binding owns the live handles from this point. Record that transfer
        # before publishing it so cancellation or an event-store failure cannot
        # leave cleanup using the stale pre-bound value.
        setup_owner.release_failed_binding_reservations = None
        _advance_cleanup_environment(setup_owner, bound_registered_environment)
        events.append(
            await self._event_writer.emit(
                _event_with_binding_generation_authority(
                    event_with_execution_profile_authority(
                        Event(
                            type=EventType.ENVIRONMENT_BINDING_COMPLETED,
                            session_id=session.id,
                            agent_name=registered_agent.spec.name,
                            environment_name=environment_name,
                            payload={
                                **base_payload,
                                **_bound_workspace_payload(
                                    bound,
                                    registered_environment=registered_environment,
                                    session_id=session.id,
                                    public_authority_alias_codec=(
                                        self._session_store.public_authority_alias_codec
                                    ),
                                ),
                            },
                        ),
                        execution_profile,
                    )
                )
            )
        )
        try:
            self._require_registered_environment_admitted(
                registered_agent=registered_agent,
                registered_environment=bound_registered_environment,
            )
        except Exception as exc:
            return EnvironmentBindingResult(
                registered_environment=bound_registered_environment,
                events=events,
                error=exc,
            )
        adopted_environment = replace(
            bound_registered_environment,
            preserve_factory_allocation=False,
        )
        _advance_cleanup_environment(setup_owner, adopted_environment)
        return EnvironmentBindingResult(
            registered_environment=adopted_environment,
            events=events,
        )

    async def drain_retained_cleanups(self, *, timeout_s: float = 10.0) -> bool:
        """Settle retained cleanup owners without cancelling dispatched work."""

        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not isfinite(timeout_s)
            or timeout_s <= 0
        ):
            raise ValueError("timeout_s must be a finite positive number.")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + float(timeout_s)
        # An explicit drain is the operator recovery boundary for a cleanup
        # that stopped automatic retries after a permanent or ambiguous
        # provider failure. Retry each authenticated owner once; a later drain
        # call may request another attempt without creating a busy loop.
        attempted_factory_recoveries: set[str] = set()
        while True:
            self._harvest_deferred_factory_cleanups()
            self._retry_failed_deferred_factory_cleanups(
                attempted_sessions=attempted_factory_recoveries,
            )
            retained = tuple(
                owner
                for owner in self._active_environment_setups.values()
                if owner.cleanup_started and owner.cleanup_finished
            )
            if (
                not retained
                and not self._deferred_factory_cleanup_tasks
                and not self._deferred_run_fence_release_tasks
            ):
                return True
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            # A retained mutation fence is part of the cleanup owner's
            # authority. Explicit operator drain is therefore also a valid
            # positive-settlement entrance; polling guarded finalization alone
            # cannot start the runner-owned probe.
            seen_fences: set[int] = set()
            fence_unavailable = False
            for owner in retained:
                fence = owner.registered_environment.workspace_mutation_fence
                fence_identity = id(fence)
                if fence_identity in seen_fences:
                    continue
                seen_fences.add(fence_identity)
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return False
                try:
                    async with asyncio.timeout(remaining):
                        await fence.wait_until_available()
                except WorkspaceMutationSettlementError:
                    fence_unavailable = True
                except TimeoutError:
                    return False
            try:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return False
                async with asyncio.timeout(remaining):
                    await self._settle_retained_environment_cleanups()
            except TimeoutError:
                return False
            if fence_unavailable:
                return False
            if (
                not any(
                    owner.cleanup_started and owner.cleanup_finished
                    for owner in self._active_environment_setups.values()
                )
                and not self._deferred_factory_cleanup_tasks
                and not self._deferred_run_fence_release_tasks
            ):
                return True
            if (
                not any(
                    owner.cleanup_started and owner.cleanup_finished
                    for owner in self._active_environment_setups.values()
                )
                and self._deferred_factory_cleanup_tasks
                and all(task.done() for task in self._deferred_factory_cleanup_tasks.values())
            ):
                # Every explicit recovery attempt in this drain call reached a
                # terminal failure. Retain ownership and return control so an
                # operator can correct provider state before retrying.
                return False
            if (
                not any(
                    owner.cleanup_started and owner.cleanup_finished
                    for owner in self._active_environment_setups.values()
                )
                and not self._deferred_factory_cleanup_tasks
                and self._deferred_run_fence_release_tasks
                and all(task.done() for task in self._deferred_run_fence_release_tasks.values())
                and any(
                    task.cancelled() or task.exception() is not None
                    for task in self._deferred_run_fence_release_tasks.values()
                )
            ):
                # Cleanup is quiescent, but the durable fence could not be
                # advanced. Worker-startup recovery remains the safe retry
                # boundary; an explicit drain must not report success.
                return False
            # Retry unavailable cleanup without turning the explicit drain path
            # into a busy loop. In-flight mutation tasks remain singly owned.
            await asyncio.sleep(min(0.05, max(0.0, deadline - loop.time())))

    async def finalize_terminal_event(
        self,
        *,
        event: Event,
        session: Session,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
    ) -> EnvironmentBindingFinalizeResult:
        if invocation_context is not None:
            if invocation_context.binding.session_id != session.id:
                raise ValueError("Terminal cleanup context belongs to another session.")
            if registered_environment is not invocation_context.registered_environment:
                raise RuntimeError(
                    "Terminal cleanup substituted the frozen registered environment."
                )
            if (
                execution_profile is not None
                and execution_profile is not invocation_context.profile
            ):
                raise RuntimeError("Terminal cleanup substituted its execution profile.")
            execution_profile = invocation_context.profile
        setup_owner = self._active_environment_setups.get(session.id)
        if setup_owner is not None:
            _retain_cleanup_execution_profile(setup_owner, execution_profile)
            _retain_cleanup_invocation_context(setup_owner, invocation_context)
        if session.id in self._deferred_factory_cleanup_tasks:
            self._retain_deferred_factory_cleanup_execution_profile(
                session.id,
                execution_profile,
            )
        owns_cleanup = setup_owner is not None and not setup_owner.cleanup_started
        owns_prebind_tombstone = setup_owner is not None and setup_owner.prebind_release_tombstone
        try:
            return await self._finalize_terminal_event_once(
                event=event,
                session=session,
                registered_environment=registered_environment,
                execution_profile=execution_profile,
            )
        except BaseException as exc:
            if owns_cleanup and setup_owner is not None and setup_owner.cleanup_started:
                setup_owner.cleanup_error = exc
                if (
                    not setup_owner.cleanup_release_safe
                    and setup_owner.pending_finalize_failure_event is None
                    and setup_owner.cleanup_retry_outcome is not None
                ):
                    # Bare fatal control-flow signals bypass ordinary failure
                    # publication. Keep the exact handle and let a later
                    # bounded sweep retry finalization to a positive terminal
                    # boundary instead of retaining it forever.
                    setup_owner.cleanup_requires_finalize_retry = True
            raise
        finally:
            # `cleanup_started` is a claim, not proof of quiescence. Only the
            # call that observed and claimed the unstarted owner may publish
            # completion; a concurrent duplicate terminalizer must not make an
            # abort release binding ownership while the first call is awaiting.
            if owns_cleanup and setup_owner is not None:
                setup_owner.cleanup_finished = True
            if (
                owns_prebind_tombstone
                and setup_owner is not None
                and self._active_environment_setups.get(session.id) is setup_owner
            ):
                del self._active_environment_setups[session.id]
            self._signal_environment_cleanup_state_changed(session.id)

    async def _finalize_terminal_event_once(
        self,
        *,
        event: Event,
        session: Session,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        execution_profile: ExecutionProfileIdentity | None,
    ) -> EnvironmentBindingFinalizeResult:
        setup_owner = self._active_environment_setups.get(session.id)
        if execution_profile is None and setup_owner is not None:
            execution_profile = setup_owner.execution_profile
        if setup_owner is not None:
            if setup_owner.cleanup_started:
                if setup_owner.cleanup_error is not None:
                    raise setup_owner.cleanup_error
                return EnvironmentBindingFinalizeResult(event=event, events=[])
            setup_owner.cleanup_started = True
            registered_environment = setup_owner.registered_environment
        if (
            registered_environment is not None
            and registered_environment.unclaimed_factory_result is not None
        ):
            setup_error = RuntimeError(
                "Environment setup ended before the factory result was adopted."
            )
            try:
                (
                    registered_environment,
                    release_payload,
                ) = await self._release_unexposed_factory_environment(
                    registered_environment,
                    error=setup_error,
                    release_failed_binding_reservations=(
                        None
                        if setup_owner is None
                        else setup_owner.release_failed_binding_reservations
                    ),
                )
            finally:
                self._transfer_deferred_factory_cleanup(
                    session_id=session.id,
                    error=setup_error,
                )
            if release_payload is not None:
                terminal_payload = copy_json_value(event.payload, "payload")
                terminal_payload["environment_factory_release"] = release_payload
                event = _copy_event_with_payload(event, terminal_payload)
        if registered_environment is None or registered_environment.bound_workspace is None:
            return EnvironmentBindingFinalizeResult(event=event, events=[])
        binding = registered_environment.environment.binding
        if binding is None:
            return EnvironmentBindingFinalizeResult(event=event, events=[])
        bound_workspace = registered_environment.bound_workspace

        terminal_outcome = _binding_outcome_for_terminal_event(event.type)
        preserve_factory_allocation = registered_environment.preserve_factory_allocation
        parked_factory_result = registered_environment.retained_factory_result
        park_for_egress_adoption = (
            self._egress_authority_adoption_handler is not None
            and parked_factory_result is not None
            and registered_environment.live_allocation_fingerprint is not None
            and execution_profile is not None
            and execution_profile.egress_authority is not None
        )
        parking_reservation = None
        parking_managed_runner: Runner | None = None
        parking_handler: EgressAuthorityAdoptionHandler | None = None
        parking_environment_name: str | None = None
        parking_fingerprint: str | None = None
        take_parking_cancellation: (
            Callable[[], tuple[asyncio.CancelledError | None, int]] | None
        ) = None
        if park_for_egress_adoption:
            if parked_factory_result is None:
                raise AssertionError("Egress parking lost its exact factory result.")
            fingerprint = registered_environment.live_allocation_fingerprint
            if fingerprint is None:
                raise AssertionError("Egress parking lost its allocation fingerprint.")
            environment_name = _environment_name(registered_environment)
            if environment_name is None:
                raise AssertionError("Egress parking lost its environment name.")
            handler = self._egress_authority_adoption_handler
            if handler is None:
                raise AssertionError("Egress parking lost its adoption handler.")
            parking_handler = handler
            parking_environment_name = environment_name
            parking_fingerprint = fingerprint
            parking_managed_runner = parked_factory_result.environment.runner
            if not isinstance(parking_managed_runner, Runner):
                raise TypeError("Parked egress allocation lost its managed runner.")
            raw_take_parking_cancellation = getattr(
                parking_managed_runner,
                "_take_authority_parking_cancellation",
                None,
            )
            if not callable(raw_take_parking_cancellation):
                raise TypeError("Parked egress allocation lost its cancellation owner.")
            take_parking_cancellation = raw_take_parking_cancellation
        outcome = (
            _EGRESS_AUTHORITY_PARKED_OUTCOME
            if park_for_egress_adoption
            else "interrupted"
            if preserve_factory_allocation
            else terminal_outcome
        )
        environment_name = _environment_name(registered_environment)
        finalize_metadata = {
            "event_type": str(event.type),
            "session_id": session.id,
        }
        if setup_owner is not None:
            setup_owner.cleanup_retry_outcome = outcome
            setup_owner.cleanup_retry_metadata = copy_json_value(
                finalize_metadata,
                "binding finalize metadata",
            )
        base_payload = {
            **_binding_base_payload(
                registered_environment,
                session_id=session.id,
                public_authority_alias_codec=(self._session_store.public_authority_alias_codec),
                redactor=self._secret_redactor,
            ),
            **_bound_workspace_payload(
                bound_workspace,
                registered_environment=registered_environment,
                session_id=session.id,
                public_authority_alias_codec=(self._session_store.public_authority_alias_codec),
            ),
            # The private parking outcome steers binding teardown, but the
            # durable event continues to describe the invocation's real
            # terminal result.  Parking is represented separately as the
            # allocation action.
            "outcome": terminal_outcome,
        }
        if park_for_egress_adoption:
            base_payload["terminal_outcome"] = terminal_outcome
            base_payload["factory_allocation_action"] = "park"
        elif preserve_factory_allocation:
            base_payload["terminal_outcome"] = terminal_outcome
            base_payload["factory_allocation_action"] = "preserve"
        events: list[Event] = []
        start_publication_error: BaseException | None = None
        try:
            events.append(
                await self._event_writer.emit(
                    _event_with_binding_generation_authority(
                        event_with_execution_profile_authority(
                            Event(
                                type=EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED,
                                session_id=session.id,
                                agent_name=event.agent_name,
                                environment_name=environment_name,
                                payload=base_payload,
                            ),
                            execution_profile,
                        ),
                    )
                )
            )
        except BaseException as exc:
            start_publication_error = exc

        final_revision: WorkspaceRevisionObservation | None = None
        finalization_delta: dict[str, Any] | None = None
        try:
            final_revision = await environment_operation_boundary.await_environment_operation(
                lambda: _observe_final_workspace_revision(
                    registered_environment,
                    binding,
                    bound_workspace,
                    operation_registry=self._final_workspace_observation_operations,
                ),
                operation_name="Final workspace revision observation",
                redactor=self._secret_redactor,
            )
            finalization_delta = await _final_workspace_delta_payload(
                session_store=self._session_store,
                session_id=session.id,
                binding_generation_id=registered_environment.binding_generation_id,
                final_observation=final_revision,
            )
            if park_for_egress_adoption:
                if (
                    parking_handler is None
                    or parking_environment_name is None
                    or parking_fingerprint is None
                    or parked_factory_result is None
                ):
                    raise AssertionError("Egress parking lost its preflight authority.")
                parking_reservation = _reserve_egress_authority_allocation_parking(
                    parking_handler,
                    session_id=session.id,
                    environment_name=parking_environment_name,
                    fingerprint=parking_fingerprint,
                    factory_result=parked_factory_result,
                    max_parked_allocations=self._max_environment_lifecycle_owners,
                )

            async def finalize_and_transfer_parked_allocation() -> tuple[
                WorkspaceSnapshot | None,
                asyncio.CancelledError | None,
                int,
            ]:
                final_snapshot_value = await _finalize_binding_after_mutation_quiescence(
                    registered_environment,
                    binding,
                    bound_workspace,
                    outcome=outcome,
                    metadata=finalize_metadata,
                )
                final_snapshot = copy_workspace_snapshot(final_snapshot_value)
                if parking_reservation is None:
                    return final_snapshot, None, 0
                handler = self._egress_authority_adoption_handler
                if handler is None or take_parking_cancellation is None:
                    raise AssertionError("Egress parking lost its reserved handoff owner.")
                _complete_egress_authority_allocation_parking(
                    handler,
                    reservation=parking_reservation,
                )
                parking_cancellation, parking_cancellation_requests = take_parking_cancellation()
                return (
                    final_snapshot,
                    parking_cancellation,
                    parking_cancellation_requests,
                )

            (
                final_snapshot,
                parking_cancellation,
                parking_cancellation_requests,
            ) = await environment_operation_boundary.await_environment_operation(
                finalize_and_transfer_parked_allocation,
                operation_name="Environment binding finalization",
                redactor=self._secret_redactor,
            )
            if setup_owner is not None:
                # The binding reached its own terminal boundary. Any retained
                # exact-owner retry state is now safe to discard even if
                # validating or publishing the resulting snapshot later fails.
                setup_owner.cleanup_release_safe = True
        except (BaseExceptionGroup, Exception, asyncio.CancelledError) as exc:
            if (
                parking_reservation is not None
                and not parking_reservation.ready
                and (
                    parking_managed_runner is None
                    or getattr(
                        parking_managed_runner,
                        "is_parked_for_authority_adoption",
                        False,
                    )
                    is not True
                )
            ):
                handler = self._egress_authority_adoption_handler
                if handler is not None:
                    _discard_egress_authority_allocation_parking_reservation(
                        handler,
                        reservation=parking_reservation,
                    )
            if start_publication_error is not None:
                exc = BaseExceptionGroup(
                    "Binding finalization and start-event publication failed.",
                    [start_publication_error, exc],
                )
            if setup_owner is not None:
                setup_owner.cleanup_error = exc
            finalize_error_payload = _binding_finalize_error_payload(
                exc,
                outcome=terminal_outcome,
                redactor=self._secret_redactor,
            )
            final_revision_payload = (
                None
                if final_revision is None
                else _final_workspace_revision_payload(
                    final_revision,
                    registered_environment=registered_environment,
                    session_id=session.id,
                    redactor=self._secret_redactor,
                    public_authority_alias_codec=(self._session_store.public_authority_alias_codec),
                    finalization_delta=finalization_delta,
                )
            )
            error_payload = {
                **base_payload,
                **finalize_error_payload,
            }
            if final_revision_payload is not None:
                error_payload["final_revision"] = final_revision_payload
            pending_failure_event = _event_with_binding_generation_authority(
                event_with_execution_profile_authority(
                    Event(
                        type=EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED,
                        session_id=session.id,
                        agent_name=event.agent_name,
                        environment_name=environment_name,
                        payload=error_payload,
                    ),
                    execution_profile,
                ),
            )
            if setup_owner is not None:
                # Retain the stable event identity until persistence or
                # reconciliation positively proves the failure durable. A
                # retry must not create a second diagnostic for the same
                # failed finalization attempt.
                setup_owner.pending_finalize_failure_event = pending_failure_event
            try:
                persistence = await _persist_binding_finalize_failure_event(
                    self._event_writer,
                    pending_failure_event,
                )
                failure_event = persistence.event
                persist_cancellation = persistence.cancellation
            except BaseException as diagnostic_error:
                attach_binding_finalize_safe_payload(exc, finalize_error_payload)
                diagnostic = exception_diagnostic(
                    diagnostic_error,
                    empty_message="binding finalization failure publication failed",
                    nonportable_message=(
                        "Binding finalization failure publication failed with a "
                        "non-portable diagnostic."
                    ),
                    redactor=self._secret_redactor,
                )
                _add_exception_note_safely(
                    exc,
                    "Binding finalization durable failure publication also failed: "
                    f"{diagnostic.error_type}: {diagnostic.message}.",
                )
                fatal_signal = binding_finalize_fatal_signal(diagnostic_error)
                if fatal_signal is not None:
                    raise fatal_signal from diagnostic_error
                cancellation = (
                    diagnostic_error
                    if isinstance(diagnostic_error, asyncio.CancelledError)
                    else binding_finalize_explicit_cancellation(diagnostic_error)
                )
                if cancellation is not None:
                    aggregate = append_binding_finalize_cancellation(exc, cancellation)
                    aggregate.add_note(
                        "Binding finalization durable failure publication also failed."
                    )
                    raise aggregate from diagnostic_error
                raise exc from diagnostic_error
            if setup_owner is not None:
                setup_owner.cleanup_release_safe = True
                setup_owner.pending_finalize_failure_event = None
            if persist_cancellation is not None:
                aggregate = append_binding_finalize_cancellation(
                    exc,
                    persist_cancellation,
                )
                if persistence.cancellation_requests_consumed:
                    retain_workspace_observation_pending_cancellation_requests(
                        aggregate,
                        persistence.cancellation_requests_consumed,
                    )
                restore_workspace_observation_cancellation_requests(
                    persistence.cancellation_requests_consumed
                )
                raise aggregate from persist_cancellation
            try:
                fanout_task = asyncio.create_task(
                    self._event_writer.fan_out_persisted([failure_event])
                )
                failure_event = (await asyncio.shield(fanout_task))[0]
            except asyncio.CancelledError as cancellation:
                attach_binding_finalize_safe_payload(exc, finalize_error_payload)
                raise append_binding_finalize_cancellation(exc, cancellation) from cancellation
            except BaseException as diagnostic_error:
                attach_binding_finalize_safe_payload(exc, finalize_error_payload)
                diagnostic = exception_diagnostic(
                    diagnostic_error,
                    empty_message="binding finalization diagnostic fan-out failed",
                    nonportable_message=(
                        "Binding finalization diagnostic fan-out failed with a "
                        "non-portable diagnostic."
                    ),
                    redactor=self._secret_redactor,
                )
                _add_exception_note_safely(
                    exc,
                    "Binding finalization diagnostic fan-out failed: "
                    f"{diagnostic.error_type}: {diagnostic.message}.",
                )
                fatal_signal = binding_finalize_fatal_signal(diagnostic_error)
                if fatal_signal is not None:
                    raise fatal_signal from diagnostic_error
                cancellation = binding_finalize_explicit_cancellation(diagnostic_error)
                if cancellation is not None:
                    aggregate = append_binding_finalize_cancellation(exc, cancellation)
                    aggregate.add_note(
                        "Binding finalization durable failure publication also failed."
                    )
                    raise aggregate from diagnostic_error
                raise exc from diagnostic_error
            events.append(failure_event)
            if not isinstance(exc, Exception):
                raise
            terminal_payload = copy_json_value(event.payload, "payload")
            terminal_payload["binding_finalize_error"] = finalize_error_payload
            if final_revision_payload is not None:
                terminal_payload["final_revision"] = final_revision_payload
            return EnvironmentBindingFinalizeResult(
                event=copy_event(event).model_copy(
                    update={"payload": terminal_payload},
                    deep=True,
                ),
                events=events,
            )

        completion_publication_error: BaseException | None = None
        try:
            events.append(
                await self._event_writer.emit(
                    _event_with_binding_generation_authority(
                        event_with_execution_profile_authority(
                            Event(
                                type=EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
                                session_id=session.id,
                                agent_name=event.agent_name,
                                environment_name=environment_name,
                                payload={
                                    **base_payload,
                                    "final_snapshot": _final_workspace_snapshot_payload(
                                        final_snapshot,
                                        registered_environment=registered_environment,
                                    ),
                                    "source_publication_receipt": (
                                        _source_publication_receipt_payload(
                                            final_snapshot,
                                            binding=binding,
                                            destination_workspace_id=base_payload.get(
                                                "source_workspace_id"
                                            ),
                                            workload_workspace_id=base_payload.get(
                                                "bound_workspace_id"
                                            ),
                                            expected_destination_workspace_id=(
                                                _workspace_object_id(
                                                    bound_workspace.source_workspace
                                                )
                                            ),
                                            expected_workload_workspace_id=(
                                                _workspace_object_id(bound_workspace.workspace)
                                            ),
                                        )
                                    ),
                                    "final_git_receipt": _final_git_receipt_payload(
                                        final_snapshot,
                                        binding=binding,
                                        destination_workspace_id=base_payload.get(
                                            "source_workspace_id"
                                        ),
                                        workload_workspace_id=base_payload.get(
                                            "bound_workspace_id"
                                        ),
                                        expected_destination_workspace_id=(
                                            _workspace_object_id(bound_workspace.source_workspace)
                                        ),
                                        expected_workload_workspace_id=(
                                            _workspace_object_id(bound_workspace.workspace)
                                        ),
                                    ),
                                    "final_revision": _final_workspace_revision_payload(
                                        final_revision,
                                        registered_environment=registered_environment,
                                        session_id=session.id,
                                        redactor=self._secret_redactor,
                                        public_authority_alias_codec=(
                                            self._session_store.public_authority_alias_codec
                                        ),
                                        finalization_delta=finalization_delta,
                                    ),
                                },
                            ),
                            execution_profile,
                        )
                    )
                )
            )
        except BaseException as exc:
            completion_publication_error = exc
        publication_failures = tuple(
            (phase, error)
            for phase, error in (
                ("finalize_started_event", start_publication_error),
                ("finalize_completed_event", completion_publication_error),
            )
            if error is not None
        )
        if publication_failures:
            terminal_payload = copy_json_value(event.payload, "payload")
            terminal_payload["binding_finalize_publication_error"] = (
                _binding_finalize_publication_failure_payload(
                    publication_failures,
                    outcome=terminal_outcome,
                    redactor=self._secret_redactor,
                )
            )
            terminal_payload["final_revision"] = _final_workspace_revision_payload(
                final_revision,
                registered_environment=registered_environment,
                session_id=session.id,
                redactor=self._secret_redactor,
                public_authority_alias_codec=(self._session_store.public_authority_alias_codec),
                finalization_delta=finalization_delta,
            )
            event = _copy_event_with_payload(event, terminal_payload)
        return EnvironmentBindingFinalizeResult(
            event=event,
            events=events,
            cancellation=parking_cancellation,
            cancellation_requests_consumed=parking_cancellation_requests,
        )

    async def abort_environment_setup(
        self,
        *,
        session_id: str,
        original_error: BaseException | None,
        allow_deferred_settlement: bool = False,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
    ) -> None:
        """Release a live setup when no terminal event can own its cleanup."""

        setup_owner = self._active_environment_setups.get(session_id)
        if invocation_context is None and setup_owner is not None:
            invocation_context = setup_owner.invocation_context
        if invocation_context is not None:
            if invocation_context.binding.session_id != session_id:
                raise ValueError("Environment abort context belongs to another session.")
            if (
                execution_profile is not None
                and execution_profile is not invocation_context.profile
            ):
                raise RuntimeError("Environment abort substituted its execution profile.")
            execution_profile = invocation_context.profile
        try:
            await self._abort_environment_setup_once(
                session_id=session_id,
                original_error=original_error,
                allow_deferred_settlement=allow_deferred_settlement,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
            )
        finally:
            self._signal_environment_cleanup_state_changed(session_id)

    async def _abort_environment_setup_once(
        self,
        *,
        session_id: str,
        original_error: BaseException | None,
        allow_deferred_settlement: bool = False,
        execution_profile: ExecutionProfileIdentity | None = None,
        invocation_context: InvocationContext | None = None,
    ) -> None:
        """Perform one exact-owner setup cleanup attempt."""

        setup_owner = self._active_environment_setups.get(session_id)
        if setup_owner is None:
            if session_id in self._deferred_factory_cleanup_tasks:
                self._retain_deferred_factory_cleanup_execution_profile(
                    session_id,
                    execution_profile,
                )
            # A stream may be abandoned after the durable started event but
            # before factory creation or binding mutation begins.
            if session_id not in self._deferred_factory_cleanup_tasks:
                self._release_pending_environment_owner_admission(session_id)
            return
        _retain_cleanup_execution_profile(setup_owner, execution_profile)
        _retain_cleanup_invocation_context(setup_owner, invocation_context)
        if setup_owner.cleanup_started and not setup_owner.cleanup_finished:
            return
        if setup_owner.cleanup_settlement_started:
            return
        if self._active_environment_setups.get(session_id) is not setup_owner:
            return
        setup_owner.cleanup_settlement_started = True
        registered_environment = setup_owner.registered_environment
        binding = registered_environment.environment.binding
        bound_workspace = registered_environment.bound_workspace
        if setup_owner.cleanup_started:
            if binding is None or bound_workspace is None:
                self._active_environment_setups.pop(session_id, None)
                return
            if (
                setup_owner.cleanup_error is not None
                and not setup_owner.cleanup_settlement_deferred
            ):
                if setup_owner.cleanup_release_safe:
                    try:
                        released = _abandon_binding_after_mutation_quiescence(
                            registered_environment,
                            binding,
                            bound_workspace,
                        )
                        if released is _BINDING_ABANDON_BLOCKED_BY_MUTATION_FENCE:
                            setup_owner.cleanup_settlement_started = False
                            return
                    except BaseException as abandon_error:
                        setup_owner.cleanup_settlement_started = False
                        if original_error is None or abandon_error is original_error:
                            raise
                        raise BaseExceptionGroup(
                            "Environment binding abandonment failed after terminal cleanup.",
                            [original_error, abandon_error],
                        ) from abandon_error
                    if released is not False:
                        if self._active_environment_setups.get(session_id) is setup_owner:
                            del self._active_environment_setups[session_id]
                        return
                    setup_owner.cleanup_requires_finalize_retry = True
                # Preserve the authoritative terminal exception during its
                # unwind. The next ordinary lifecycle entry delivers one
                # bounded retry through `_settle_retained_environment_cleanups`.
                setup_owner.cleanup_settlement_deferred = True
                setup_owner.cleanup_settlement_started = False
                return
            if setup_owner.cleanup_settlement_deferred and not allow_deferred_settlement:
                setup_owner.cleanup_settlement_started = False
                return
            settlement_error: BaseException | None = None

            async def retry_binding_finalize() -> BaseException | None:
                try:
                    await environment_operation_boundary.await_environment_operation(
                        lambda: _finalize_binding_after_mutation_quiescence(
                            registered_environment,
                            binding,
                            bound_workspace,
                            outcome=setup_owner.cleanup_retry_outcome,
                            metadata=setup_owner.cleanup_retry_metadata,
                        ),
                        operation_name="Environment binding cleanup retry",
                        redactor=self._secret_redactor,
                    )
                except BaseException as retry_error:
                    setup_owner.cleanup_error = retry_error
                    return retry_error
                setup_owner.cleanup_requires_finalize_retry = False
                return None

            pending_failure_event = setup_owner.pending_finalize_failure_event
            retry_error: BaseException | None = None
            if (
                not setup_owner.cleanup_release_safe
                and pending_failure_event is None
                and setup_owner.cleanup_requires_finalize_retry
            ):
                retry_error = await retry_binding_finalize()
                if retry_error is None:
                    setup_owner.cleanup_release_safe = True
                else:
                    setup_owner.cleanup_settlement_started = False
                    if original_error is None or retry_error is original_error:
                        raise retry_error
                    raise BaseExceptionGroup(
                        "Environment binding fatal cleanup retry failed.",
                        [original_error, retry_error],
                    ) from retry_error
            if not setup_owner.cleanup_release_safe and pending_failure_event is not None:
                try:
                    persistence = await _persist_binding_finalize_failure_event(
                        self._event_writer,
                        pending_failure_event,
                    )
                    failure_event = persistence.event
                    persist_cancellation = persistence.cancellation
                    setup_owner.cleanup_release_safe = True
                    setup_owner.pending_finalize_failure_event = None
                    fanout_outcome = await await_shielded_task_outcome(
                        asyncio.create_task(self._event_writer.fan_out_persisted([failure_event])),
                        cancellation=persist_cancellation,
                    )
                    if fanout_outcome.cancellation is not None:
                        restore_workspace_observation_cancellation_requests(
                            persistence.cancellation_requests_consumed
                            + fanout_outcome.cancellation_requests_consumed
                        )
                    settlement_error = _environment_cleanup_settlement_error(
                        fanout_error=fanout_outcome.error,
                        cancellation=fanout_outcome.cancellation,
                    )
                except BaseException as exc:
                    settlement_error = exc
            if not setup_owner.cleanup_release_safe:
                setup_owner.cleanup_settlement_started = False
                if settlement_error is None:
                    # No positive terminal or durable-failure evidence exists
                    # from which this lifecycle owner can safely retire.
                    return
                if original_error is None or settlement_error is original_error:
                    raise settlement_error
                raise BaseExceptionGroup(
                    "Environment binding failure evidence remains non-durable.",
                    [original_error, settlement_error],
                ) from settlement_error

            if setup_owner.cleanup_requires_finalize_retry:
                retry_error = await retry_binding_finalize()
            if retry_error is None:
                try:
                    # `False` is an explicit refusal: a composite binding still
                    # owns an executable resource and must retain this handle.
                    released = _abandon_binding_after_mutation_quiescence(
                        registered_environment,
                        binding,
                        bound_workspace,
                    )
                    if released is _BINDING_ABANDON_BLOCKED_BY_MUTATION_FENCE:
                        setup_owner.cleanup_settlement_started = False
                        return
                except BaseException as abandon_error:
                    setup_owner.cleanup_settlement_started = False
                    if original_error is None or abandon_error is original_error:
                        raise
                    raise BaseExceptionGroup(
                        "Environment binding abandonment failed after terminal cleanup.",
                        [original_error, abandon_error],
                    ) from abandon_error
                if released is False:
                    setup_owner.cleanup_requires_finalize_retry = True
                    retry_error = await retry_binding_finalize()
                    if retry_error is None:
                        try:
                            released = _abandon_binding_after_mutation_quiescence(
                                registered_environment,
                                binding,
                                bound_workspace,
                            )
                            if released is _BINDING_ABANDON_BLOCKED_BY_MUTATION_FENCE:
                                setup_owner.cleanup_settlement_started = False
                                return
                        except BaseException as abandon_error:
                            setup_owner.cleanup_settlement_started = False
                            if original_error is None or abandon_error is original_error:
                                raise
                            raise BaseExceptionGroup(
                                "Environment binding abandonment failed after cleanup retry.",
                                [original_error, abandon_error],
                            ) from abandon_error
                        if released is False:
                            retry_error = RuntimeError(
                                "Environment binding refused abandonment after successful "
                                "cleanup retry."
                            )
                            setup_owner.cleanup_requires_finalize_retry = True
                            setup_owner.cleanup_error = retry_error

            if retry_error is not None:
                setup_owner.cleanup_settlement_started = False
                if settlement_error is not None and settlement_error is not retry_error:
                    retry_error = BaseExceptionGroup(
                        "Environment binding durability settlement and cleanup retry failed.",
                        [settlement_error, retry_error],
                    )
                if original_error is None or retry_error is original_error:
                    raise retry_error
                raise BaseExceptionGroup(
                    "Environment binding cleanup remains incomplete.",
                    [original_error, retry_error],
                ) from retry_error
            if self._active_environment_setups.get(session_id) is setup_owner:
                del self._active_environment_setups[session_id]
            if settlement_error is not None:
                if original_error is None or settlement_error is original_error:
                    raise settlement_error
                raise BaseExceptionGroup(
                    "Environment binding failure evidence committed after a control signal.",
                    [original_error, settlement_error],
                ) from settlement_error
            return
        setup_owner.cleanup_started = True
        setup_owner.cleanup_finished = False
        if original_error is None:
            original_error = RuntimeError("Environment setup ended without terminal cleanup.")
        if registered_environment.unclaimed_factory_result is not None:
            try:
                try:
                    await self._release_unexposed_factory_environment(
                        registered_environment,
                        error=original_error,
                        release_failed_binding_reservations=(
                            setup_owner.release_failed_binding_reservations
                        ),
                    )
                finally:
                    self._transfer_deferred_factory_cleanup(
                        session_id=session_id,
                        error=original_error,
                    )
            except BaseException as cleanup_error:
                if cleanup_error is original_error:
                    raise
                raise BaseExceptionGroup(
                    "Environment factory cleanup failed while aborting setup.",
                    [original_error, cleanup_error],
                ) from cleanup_error
            finally:
                if self._active_environment_setups.get(session_id) is setup_owner:
                    del self._active_environment_setups[session_id]
            return
        if binding is None or bound_workspace is None:
            self._active_environment_setups.pop(session_id, None)
            return
        setup_owner.cleanup_retry_outcome = "interrupted"
        setup_owner.cleanup_retry_metadata = {
            "event_type": "environment_setup_aborted",
            "session_id": session_id,
        }
        cleanup_error: BaseException | None = None
        try:
            await environment_operation_boundary.await_environment_operation(
                lambda: _finalize_binding_after_mutation_quiescence(
                    registered_environment,
                    binding,
                    bound_workspace,
                    outcome="interrupted",
                    metadata=setup_owner.cleanup_retry_metadata,
                ),
                operation_name="Environment binding cleanup finalization",
                redactor=self._secret_redactor,
            )
        except BaseException as exc:
            cleanup_error = exc
            setup_owner.cleanup_error = exc
            diagnostic = exception_diagnostic(
                cleanup_error,
                empty_message="environment binding cleanup failed",
                nonportable_message=(
                    "Environment binding cleanup failed with a non-portable diagnostic."
                ),
                redactor=self._secret_redactor,
            )
            _add_exception_note_safely(
                original_error,
                "Environment binding cleanup failed while aborting setup: "
                f"{diagnostic.error_type}: {diagnostic.message}.",
            )
        # This non-terminal abort has no surviving public retry handle. Once
        # the finalize attempt is quiescent, exact-owner abandonment is the
        # authoritative retirement path even when that attempt failed.
        setup_owner.cleanup_release_safe = True
        setup_owner.cleanup_finished = True
        abandon_error: BaseException | None = None
        released = True
        try:
            # No lifecycle owner survives this abort path. Finalize success
            # makes this a no-op; failure leaves retry state that must now be
            # released by the exact bound generation.
            release_outcome = _abandon_binding_after_mutation_quiescence(
                registered_environment,
                binding,
                bound_workspace,
            )
            if release_outcome is _BINDING_ABANDON_BLOCKED_BY_MUTATION_FENCE:
                setup_owner.cleanup_settlement_started = False
                return
            released = release_outcome is not False
        except BaseException as exc:
            setup_owner.cleanup_settlement_started = False
            abandon_error = exc
            diagnostic = exception_diagnostic(
                abandon_error,
                empty_message="environment binding abandonment failed",
                nonportable_message=(
                    "Environment binding abandonment failed with a non-portable diagnostic."
                ),
            )
            _add_exception_note_safely(
                original_error,
                "Environment binding abandonment failed while aborting setup: "
                f"{diagnostic.error_type}: {diagnostic.message}.",
            )
        if (
            abandon_error is None
            and released
            and self._active_environment_setups.get(session_id) is setup_owner
        ):
            del self._active_environment_setups[session_id]
        elif abandon_error is not None or not released:
            setup_owner.cleanup_settlement_started = False
            setup_owner.cleanup_finished = True
            # This abort already performed the first cleanup and abandonment
            # attempt. The next ordinary lifecycle entry should execute the
            # retained retry, not spend one request merely arming it.
            setup_owner.cleanup_settlement_deferred = True
            setup_owner.cleanup_requires_finalize_retry = not released
            if cleanup_error is None and not released:
                cleanup_error = RuntimeError(
                    "Environment binding retained ownership after aborted setup cleanup."
                )
                setup_owner.cleanup_error = cleanup_error
        failures = [
            error for error in (original_error, cleanup_error, abandon_error) if error is not None
        ]
        unique_failures: list[BaseException] = []
        for error in failures:
            if all(error is not existing for existing in unique_failures):
                unique_failures.append(error)
        if cleanup_error is None and abandon_error is None:
            return
        if len(unique_failures) == 1:
            raise unique_failures[0]
        raise BaseExceptionGroup(
            "Environment binding cleanup failed while aborting setup.",
            unique_failures,
        ) from (abandon_error or cleanup_error)

    async def _load_factory_reconnect_state(
        self,
        *,
        session_id: str,
        environment_name: str,
    ) -> tuple[dict[str, Any], str | None]:
        checkpoint = await self._session_store.load_checkpoint(session_id)
        return _factory_reconnect_state_from_checkpoint(
            checkpoint,
            environment_name=environment_name,
        )

    async def require_live_allocation_fingerprint(
        self,
        *,
        session_id: str,
        environment_name: str,
    ) -> str:
        """Load the session-owned pre-cutover backend identity from durable state."""

        reconnect_metadata, allocation_owner = await self._load_factory_reconnect_state(
            session_id=session_id,
            environment_name=environment_name,
        )
        if allocation_owner != session_id:
            raise RuntimeError(
                "Egress authority adoption has no session-owned environment allocation."
            )
        fingerprint = _reconnect_allocation_fingerprint(reconnect_metadata)
        if fingerprint is None:
            raise RuntimeError(
                "Egress authority adoption has no durable pre-cutover allocation identity."
            )
        return fingerprint

    async def durable_live_allocation_fingerprint(
        self,
        *,
        session_id: str,
        environment_name: str,
    ) -> str | None:
        """Reconstruct the session-owned allocation identity without reconnecting it."""

        checkpoint = await self._session_store.load_checkpoint(session_id)
        reconnect_metadata, allocation_owner = _factory_reconnect_state_from_checkpoint(
            checkpoint,
            environment_name=environment_name,
        )
        if allocation_owner != session_id:
            raise RuntimeError(
                "Environment recovery has no session-owned durable allocation identity."
            )
        allocation_record = self._allocation_coordinator.record_from_checkpoint(
            checkpoint,
            environment_name=environment_name,
        )
        allocation_receipt = self._allocation_coordinator.receipt_from_checkpoint(
            checkpoint,
            environment_name=environment_name,
        )
        if allocation_record is not None:
            raise RuntimeError(
                "Environment recovery conflicts with an incomplete allocation intent."
            )
        if allocation_receipt is not None and (
            allocation_receipt.intent.environment_name != environment_name
            or allocation_receipt.intent.session_id != session_id
            or allocation_receipt.reconnect_metadata != reconnect_metadata
        ):
            raise RuntimeError(
                "Environment allocation receipt conflicts with durable reconnect state."
            )
        return _reconnect_allocation_fingerprint(
            reconnect_metadata
        ) or _live_allocation_fingerprint(None, allocation_receipt)

    async def _checkpoint_factory_reconnect_metadata(
        self,
        *,
        session_id: str,
        environment_name: str,
        reconnect_metadata: dict[str, Any],
    ) -> None:
        checkpoint = await self._session_store.load_checkpoint(session_id)
        copied_checkpoint = {} if checkpoint is None else copy_json_value(checkpoint, "checkpoint")
        state = copied_checkpoint.get(ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY)
        if state is None:
            state = {}
        elif type(state) is not dict:
            raise ValueError("Environment factory reconnect checkpoint must be an object.")
        else:
            state = copy_json_value(state, "environment_factory_reconnect")
        state[environment_name] = copy_json_value(reconnect_metadata, "reconnect_metadata")
        copied_checkpoint[ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY] = state
        owners = copied_checkpoint.get(ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY)
        if owners is None:
            owners = {}
        elif type(owners) is not dict:
            raise ValueError("Environment factory allocation owners must be an object.")
        else:
            owners = copy_json_value(owners, "environment_factory_allocation_owner")
        owners[environment_name] = session_id
        copied_checkpoint[ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY] = owners
        try:
            await self._session_store.transform_checkpoint(
                session_id,
                self._checkpoint_transform(copied_checkpoint),
            )
        except BaseException as exc:
            current_task = asyncio.current_task()
            caller_cancellation = (
                exc
                if isinstance(exc, asyncio.CancelledError)
                and current_task is not None
                and current_task.cancelling() > 0
                else None
            )
            outcome = await await_shielded_task_outcome(
                asyncio.create_task(
                    self._factory_checkpoint_matches(
                        session_id=session_id,
                        environment_name=environment_name,
                        reconnect_metadata=reconnect_metadata,
                    )
                ),
                cancellation=caller_cancellation,
            )
            checkpoint_may_be_committed = outcome.error is not None or bool(outcome.result)
            if outcome.error is not None:
                fatal_signal = binding_finalize_fatal_signal(outcome.error)
                if fatal_signal is not None:
                    _mark_environment_factory_checkpoint_may_be_committed(fatal_signal)
                    _add_exception_note_safely(
                        fatal_signal,
                        "The environment factory checkpoint write also failed; "
                        "its commit outcome could not be reconciled.",
                    )
                    raise fatal_signal from exc
                _add_exception_note_safely(
                    exc,
                    "Could not reconcile whether the environment factory checkpoint "
                    "committed; the allocation will be preserved.",
                )
            propagated_error: BaseException = exc
            if outcome.cancellation is not None and outcome.cancellation is not exc:
                propagated_error = BaseExceptionGroup(
                    "Environment factory checkpoint write failed during caller cancellation.",
                    [exc, outcome.cancellation],
                )
            if checkpoint_may_be_committed:
                _mark_environment_factory_checkpoint_may_be_committed(propagated_error)
            if outcome.error is not None:
                raise propagated_error from outcome.error
            if propagated_error is exc:
                raise
            raise propagated_error from exc

    async def _factory_checkpoint_matches(
        self,
        *,
        session_id: str,
        environment_name: str,
        reconnect_metadata: dict[str, Any],
    ) -> bool:
        persisted_metadata, allocation_owner = await self._load_factory_reconnect_state(
            session_id=session_id,
            environment_name=environment_name,
        )
        return allocation_owner == session_id and persisted_metadata == reconnect_metadata


def render_initial_system_prompt(
    *,
    agent_system_prompt: str | None,
    workspace_instructions: WorkspaceInstructions | None,
) -> str | None:
    rendered, _ = render_initial_system_prompt_with_contributions(
        agent_system_prompt=agent_system_prompt,
        workspace_instructions=workspace_instructions,
    )
    return rendered


def render_initial_system_prompt_with_contributions(
    *,
    agent_system_prompt: str | None,
    workspace_instructions: WorkspaceInstructions | None,
) -> tuple[str | None, dict[str, tuple[str, ...]]]:
    """Render the prompt and retain exact in-memory fragments for safe measurement."""

    agent_prompt = agent_system_prompt.strip() if agent_system_prompt else ""
    if workspace_instructions is None:
        return (
            agent_prompt or None,
            {"agent_instructions": (agent_prompt,)} if agent_prompt else {},
        )

    workspace_content = workspace_instructions.content.strip()
    source_list = ", ".join(workspace_instructions.sources)
    workspace_framing = (
        "[Workspace instructions]\n"
        f"Source: {source_list}\n"
        "These instructions apply only to the active workspace. If they conflict "
        "with agent, tool, approval, sandbox, or secret policy, follow the "
        "higher-priority runtime policy.\n\n"
    )
    if not agent_prompt:
        return (
            f"{workspace_framing}{workspace_content}",
            {
                "cayu_framing": (workspace_framing,),
                "workspace_instructions": (workspace_content,),
            },
        )
    agent_framing = "[Agent instructions]\n"
    separator = "\n\n"
    return (
        f"{agent_framing}{agent_prompt}{separator}{workspace_framing}{workspace_content}",
        {
            "agent_instructions": (agent_prompt,),
            "cayu_framing": (agent_framing, separator, workspace_framing),
            "workspace_instructions": (workspace_content,),
        },
    )


def exception_failure_payload(
    error: BaseException,
    *,
    diagnostic: ExceptionDiagnostic | None = None,
    redactor: SecretRedactor | None = None,
) -> dict[str, Any]:
    """Return portable, workload-secret-safe terminal evidence."""

    resolved_redactor = redactor or SecretRedactor()
    if diagnostic is None:
        diagnostic = exception_diagnostic(error, redactor=resolved_redactor)
    elif type(diagnostic) is not ExceptionDiagnostic:
        raise TypeError("diagnostic must be an ExceptionDiagnostic.")
    fallback = _redact_and_bound_failure_payload(
        diagnostic.payload_fields(),
        redactor=resolved_redactor,
    )

    safe_payload = binding_finalize_safe_payload(error)
    if safe_payload is None and isinstance(error, BaseExceptionGroup):
        safe_payload = next(
            (
                payload
                for child in iter_exception_tree(error)
                if child is not error
                and (payload := binding_finalize_safe_payload(child)) is not None
            ),
            None,
        )
    if safe_payload is not None:
        return _redact_and_bound_failure_payload(
            safe_payload,
            redactor=resolved_redactor,
        )

    payload = dict(fallback)
    try:
        if isinstance(error, ExecutionAdmissionError):
            payload["execution_admission"] = error.decision.model_dump(mode="json")
    except BaseException:
        return fallback
    try:
        cleanup_payload = binding_cleanup_payload(
            error,
            redactor=resolved_redactor,
        )
    except BaseException:
        cleanup_payload = None
    if cleanup_payload is not None:
        payload["binding_cleanup"] = cleanup_payload
    factory_release = _environment_factory_release_payload(error)
    if factory_release is not None:
        payload["environment_factory_release"] = factory_release
    try:
        portable = copy_durable_json_object(payload, "failure_payload")
    except BaseException:
        return fallback
    return _redact_and_bound_failure_payload(
        portable,
        redactor=resolved_redactor,
    )


def _attach_environment_factory_release_payload(
    error: BaseException,
    payload: dict[str, Any],
) -> None:
    _attach_runtime_exception_payload(
        error,
        attribute_name=_ENVIRONMENT_FACTORY_RELEASE_ERROR_ATTRIBUTE,
        payload=payload,
    )


def _environment_factory_release_payload(
    error: BaseException,
) -> dict[str, Any] | None:
    return _runtime_exception_payload(
        error,
        attribute_name=_ENVIRONMENT_FACTORY_RELEASE_ERROR_ATTRIBUTE,
    )


def _redact_and_bound_failure_payload(
    value: dict[str, Any],
    *,
    redactor: SecretRedactor,
) -> dict[str, Any]:
    redacted = redactor.redact_json_values(
        value,
        preserve_string_fields={"outcome", "phase"},
    )
    if type(redacted) is not dict:
        raise AssertionError("Failure payload redaction returned a non-object.")

    def bound(item: Any) -> Any:
        if type(item) is str:
            return redactor.redact_text_bounded(
                item,
                max_bytes=FAILURE_DIAGNOSTIC_TEXT_MAX_BYTES,
            )
        if item is None or type(item) in {bool, int, float}:
            return item
        if type(item) is list:
            return [bound(child) for child in item]
        if type(item) is dict:
            return {key: bound(child) for key, child in item.items()}
        raise AssertionError("Failure payload contains non-JSON-compatible data.")

    bounded = bound(redacted)
    if type(bounded) is not dict:
        raise AssertionError("Failure payload bounding returned a non-object.")
    return bounded


def _copy_event_with_payload(event: Event, payload: dict[str, Any]) -> Event:
    return copy_event(event).model_copy(update={"payload": payload}, deep=True)


async def _reconcile_binding_finalize_failure_event(
    writer: RuntimeEventWriter,
    event: Event,
    *,
    persistence_error: BaseException,
    cancellation: asyncio.CancelledError | None,
) -> tuple[bool, asyncio.CancelledError | None, int]:
    outcome = await await_shielded_task_outcome(
        asyncio.create_task(writer.is_persisted(event)),
        cancellation=cancellation,
    )
    cancellation = outcome.cancellation
    if outcome.error is None:
        return (
            bool(outcome.result),
            cancellation,
            outcome.cancellation_requests_consumed,
        )
    fatal_signal = binding_finalize_fatal_signal(outcome.error)
    if fatal_signal is not None:
        raise fatal_signal
    if cancellation is not None:
        _add_exception_note_safely(
            persistence_error,
            "Could not reconcile whether the binding finalization failure event committed.",
        )
        raise BaseExceptionGroup(
            "Binding finalization failure reconciliation failed after caller cancellation.",
            [persistence_error, cancellation],
        ) from outcome.error
    raise persistence_error from outcome.error


@dataclass(frozen=True, slots=True)
class _BindingFinalizeFailurePersistence:
    """Durable failure event plus cancellation consumed while proving it."""

    event: Event
    cancellation: asyncio.CancelledError | None
    cancellation_requests_consumed: int

    def __iter__(self) -> Iterator[Event | asyncio.CancelledError | None]:
        # Preserve the established private two-value unpacking contract while
        # runtime callers consume the exact cancellation count by name.
        yield self.event
        yield self.cancellation


async def _persist_binding_finalize_failure_event(
    writer: RuntimeEventWriter,
    event: Event,
) -> _BindingFinalizeFailurePersistence:
    outcome = await await_shielded_task_outcome(asyncio.create_task(writer.persist(event)))
    persistence_error = outcome.error
    cancellation = outcome.cancellation
    if persistence_error is None:
        return _BindingFinalizeFailurePersistence(
            event=event,
            cancellation=cancellation,
            cancellation_requests_consumed=outcome.cancellation_requests_consumed,
        )
    fatal_signal = binding_finalize_fatal_signal(persistence_error)
    if fatal_signal is not None:
        raise fatal_signal
    (
        persisted,
        cancellation,
        reconciliation_cancellation_requests_consumed,
    ) = await _reconcile_binding_finalize_failure_event(
        writer,
        event,
        persistence_error=persistence_error,
        cancellation=cancellation,
    )
    if persisted:
        return _BindingFinalizeFailurePersistence(
            event=event,
            cancellation=cancellation,
            cancellation_requests_consumed=(
                outcome.cancellation_requests_consumed
                + reconciliation_cancellation_requests_consumed
            ),
        )
    if cancellation is not None:
        raise BaseExceptionGroup(
            "Binding finalization failure publication failed after caller cancellation.",
            [persistence_error, cancellation],
        ) from persistence_error
    raise persistence_error


def _environment_cleanup_settlement_error(
    *,
    fanout_error: BaseException | None,
    cancellation: asyncio.CancelledError | None,
) -> BaseException | None:
    """Preserve every signal observed while delivering settled failure evidence."""

    if fanout_error is None:
        return cancellation
    if cancellation is None:
        return fanout_error
    return BaseExceptionGroup(
        "Environment binding failure evidence fan-out failed after caller cancellation.",
        [fanout_error, cancellation],
    )


async def _finalize_binding_after_mutation_quiescence(
    registered_environment: runtime_records.RegisteredEnvironment,
    binding: WorkspaceBinding,
    bound_workspace: BoundWorkspace,
    *,
    outcome: str | None,
    metadata: dict[str, Any] | None,
) -> WorkspaceSnapshot | None:
    registered_environment.workspace_mutation_fence.require_available_nowait()
    return await binding.finalize(
        bound_workspace,
        outcome=outcome,
        metadata=metadata,
    )


async def _observe_final_workspace_revision(
    registered_environment: runtime_records.RegisteredEnvironment,
    binding: WorkspaceBinding,
    bound_workspace: BoundWorkspace,
    *,
    operation_registry: BoundedInvocationOperationRegistry,
) -> WorkspaceRevisionObservation:
    # A terminal revision is authoritative only after every earlier workspace
    # mutation has positively settled.  Check before dispatching the observer;
    # the binding-finalization boundary repeats this guard before its own
    # potentially mutating synchronization step.
    registered_environment.workspace_mutation_fence.require_available_nowait()
    workspace = bound_workspace.workspace or bound_workspace.source_workspace
    workspace_id = (
        "workspace-unavailable"
        if workspace is None
        else require_clean_nonblank(workspace.id, "workspace.id")
    )
    expected_identity = WorkspaceIdentity(
        workspace_id=workspace_id,
        observer=type(binding).__name__,
    )

    def failed(detail_code: str) -> WorkspaceRevisionObservation:
        return WorkspaceRevisionObservation(
            identity=expected_identity,
            status=WorkspaceRevisionObservationStatus.FAILED,
            detail_code=detail_code,
        )

    if not operation_registry.reserve():
        return failed("final_revision_observer_capacity_exhausted")
    try:
        observation_task = asyncio.create_task(
            capture_awaitable_outcome(lambda: binding.observe_revision(bound_workspace))
        )
    except BaseException:
        operation_registry.release_reservation()
        raise
    operation_registry.track(observation_task)
    try:
        observation_outcome = await await_shielded_task_outcome(
            observation_task,
            timeout_s=_FINAL_WORKSPACE_OBSERVATION_TIMEOUT_SECONDS,
            timeout_after_cancellation_s=0.0,
        )
    except BaseException:
        if not observation_task.done():
            registered_environment.workspace_mutation_fence.fail_closed(
                workspace_mutation_task_settlement_probe(observation_task)
            )
        raise
    if observation_task.done():
        operation_registry.release(observation_task)
    observed: object = None
    observer_error = observation_outcome.error
    if observer_error is None:
        captured = observation_outcome.result
        if type(captured) is not CapturedAwaitableOutcome:
            observer_error = RuntimeError(
                "Final workspace revision observer returned an invalid owned outcome."
            )
        else:
            observed = captured.result
            observer_error = captured.error
    observer_settlement_unproven = observer_error is not None and (
        exception_tree_contains(
            observer_error,
            asyncio.CancelledError,
        )
    )
    if observer_settlement_unproven:
        # A terminal observer coroutine may still have cancellation-opaque
        # work in a thread, executor, subprocess, SDK, or remote service. A
        # cancelled task is therefore retained as fail-closed evidence even
        # when caller cancellation arrived in the same scheduling turn.
        registered_environment.workspace_mutation_fence.fail_closed(
            workspace_mutation_task_settlement_probe(observation_task)
        )
    if observation_outcome.cancellation is not None and not observation_task.done():
        registered_environment.workspace_mutation_fence.fail_closed(
            workspace_mutation_task_settlement_probe(observation_task)
        )
    if observation_outcome.cancellation is not None:
        restore_workspace_observation_cancellation_requests(
            observation_outcome.cancellation_requests_consumed
        )
    raise_workspace_observation_concurrent_control(
        cancellation=observation_outcome.cancellation,
        error=observer_error,
        operation="Final workspace revision observation",
        cancellation_requests_pending=(observation_outcome.cancellation_requests_consumed),
    )
    if observation_outcome.timed_out:
        if observation_task.done():
            operation_registry.release(observation_task)
            return failed("final_revision_observer_timeout")
        # Do not cancel an opaque observer: cancellation of its coroutine does
        # not stop a thread, executor job, subprocess, or remote request. Keep
        # the exact task behind the environment fence until it really settles.
        registered_environment.workspace_mutation_fence.fail_closed(
            workspace_mutation_task_settlement_probe(observation_task)
        )
        raise WorkspaceMutationSettlementError(
            "Final workspace revision observation did not settle after its deadline."
        ) from None
    if observer_error is not None:
        if exception_tree_contains(
            observer_error,
            (GeneratorExit, KeyboardInterrupt, SystemExit),
        ):
            raise observer_error
        if observer_settlement_unproven:
            raise WorkspaceMutationSettlementError(
                "Final workspace revision observation did not prove mutation quiescence."
            ) from None
        return failed("final_revision_observer_failed")
    try:
        return copy_bounded_workspace_revision_observation(
            observed,
            expected_identity=expected_identity,
            limits=WorkspaceRevisionObservationLimits(),
        )
    except WorkspaceRevisionObservationLimitExceeded:
        return WorkspaceRevisionObservation(
            identity=expected_identity,
            status=WorkspaceRevisionObservationStatus.TRUNCATED,
            detail_code="final_revision_observer_limit_exceeded",
        )
    except Exception:
        return failed("final_revision_observer_failed")


async def _final_workspace_delta_payload(
    *,
    session_store: SessionStore,
    session_id: str,
    binding_generation_id: str,
    final_observation: WorkspaceRevisionObservation,
) -> dict[str, Any]:
    """Compare final state with the latest durable tool-window endpoint."""

    baseline_payload: dict[str, Any] | None = None
    try:
        records = await session_store.query_events(
            EventQuery(
                session_id=session_id,
                event_type=EventType.WORKSPACE_REVISION_OBSERVED,
                order_by=EventOrder.SEQUENCE_DESC,
                limit=16,
            )
        )
        for record in records:
            candidate = record.event.payload
            if (
                candidate.get("phase") == "after"
                and candidate.get("binding_generation_id") == binding_generation_id
                and candidate.get("workspace_id") == final_observation.identity.workspace_id
                and candidate.get("observer") == final_observation.identity.observer
            ):
                baseline_payload = candidate
                break
    except Exception:
        baseline_payload = None

    attribution = WorkspaceMutationAttributionConfidence.UNATTRIBUTED_FINALIZATION_CHANGE.value
    if baseline_payload is None:
        return {
            "attribution_confidence": attribution,
            "status": WorkspaceRevisionDeltaStatus.INCOMPLETE.value,
            "before_revision": None,
            "after_revision": final_observation.revision,
            "paths": [],
            "retained_paths": 0,
            "total_paths": 0,
            "truncated": False,
            "head_changed": False,
            "branch_changed": False,
            "detail_code": "finalization_baseline_unavailable",
        }

    before_revision = baseline_payload.get("revision")
    if type(before_revision) is not str or not before_revision.strip():
        before_revision = None
    try:
        raw_paths = baseline_payload.get("paths")
        total_paths = baseline_payload.get("total_paths")
        if (
            baseline_payload.get("status") != WorkspaceRevisionObservationStatus.SUPPORTED.value
            or type(raw_paths) is not list
            or type(total_paths) is not int
            or total_paths != len(raw_paths)
        ):
            raise ValueError("Finalization baseline is incomplete.")
        baseline = WorkspaceRevisionObservation(
            identity=final_observation.identity,
            status=WorkspaceRevisionObservationStatus.SUPPORTED,
            revision=before_revision,
            head_revision=baseline_payload.get("head_revision"),
            branch=baseline_payload.get("branch"),
            path_scope=baseline_payload.get("path_scope", "complete"),
            paths=tuple(WorkspacePathRevision.model_validate(path) for path in raw_paths),
            total_paths=total_paths,
        )
        delta = compare_workspace_revisions(baseline, final_observation)
    except Exception:
        changed = (
            before_revision is not None
            and final_observation.revision is not None
            and before_revision != final_observation.revision
        )
        return {
            "attribution_confidence": attribution,
            "status": (
                WorkspaceRevisionDeltaStatus.TRUNCATED.value
                if changed
                else WorkspaceRevisionDeltaStatus.INCOMPLETE.value
            ),
            "before_revision": before_revision,
            "after_revision": final_observation.revision,
            "paths": [],
            "retained_paths": 0,
            "total_paths": 0,
            "truncated": changed,
            "head_changed": False,
            "branch_changed": False,
            "detail_code": "finalization_baseline_evidence_incomplete",
        }

    retained_paths = delta.paths[:_MAX_RETAINED_FINAL_WORKSPACE_OBSERVATIONS]
    return {
        "attribution_confidence": attribution,
        "status": delta.status.value,
        "before_revision": delta.before_revision,
        "after_revision": delta.after_revision,
        "paths": [
            {
                "path_sha256": hashlib.sha256(f"{session_id}\0{path.path}".encode()).hexdigest(),
                "change": path.change,
            }
            for path in retained_paths
        ],
        "retained_paths": len(retained_paths),
        "total_paths": delta.total_paths,
        "truncated": len(retained_paths) < delta.total_paths,
        "head_changed": delta.head_changed,
        "branch_changed": delta.branch_changed,
        "detail_code": delta.detail_code,
    }


def _final_workspace_revision_payload(
    observation: WorkspaceRevisionObservation,
    *,
    registered_environment: runtime_records.RegisteredEnvironment | None,
    session_id: str,
    redactor: SecretRedactor,
    public_authority_alias_codec: PublicAuthorityAliasCodec | None,
    finalization_delta: dict[str, Any] | None,
) -> dict[str, Any]:
    if (
        invocation_secrets.registered_environment_secret_resolution_scope(registered_environment)
        != "static"
    ):
        # Invocation-resolved secret registries deliberately remain in-process
        # and invocation-scoped. Terminal finalization can run after that scope
        # has been discarded or in a fresh recovery process, so arbitrary
        # observer text cannot be proven safe for a dynamic environment. Keep a
        # fixed, content-free view instead of allowing branch/revision fields to
        # bypass the tool-round publication quarantine.
        binding = (
            None if registered_environment is None else registered_environment.environment.binding
        )
        runtime_owned_observer = (
            None if binding is None else _runtime_owned_workspace_observer_name(binding)
        )
        try:
            projected_authority = _project_workspace_observation_authority(
                session_id=session_id,
                configured_workspace_id=observation.identity.workspace_id,
                configured_observer=observation.identity.observer,
                configured_artifact_store_id=None,
                observer_is_runtime_owned=(runtime_owned_observer == observation.identity.observer),
                secret_resolution_scope="dynamic",
                redactor=redactor,
                public_authority_alias_codec=public_authority_alias_codec,
            )
            workspace_id = projected_authority.workspace_id
            observer = projected_authority.observer
        except (RuntimeError, ValueError):
            # A store without a durable alias keyring cannot authenticate raw
            # dynamic identities after the invocation secret scope is gone.
            # Finalization remains available, but publishes fixed evidence.
            workspace_id = "workspace-authority-unavailable"
            observer = (
                observation.identity.observer
                if runtime_owned_observer == observation.identity.observer
                else "workspace-observer-unavailable"
            )
        payload: dict[str, Any] = {
            "workspace_id": workspace_id,
            "observer": observer,
            "status": WorkspaceRevisionObservationStatus.TRUNCATED.value,
            "revision": None,
            "head_revision": None,
            "branch": None,
            "path_scope": observation.path_scope,
            "total_paths": observation.total_paths,
            "detail_code": "final_revision_secret_scope_unavailable",
        }
        if finalization_delta is not None:
            payload["finalization_delta"] = {
                "attribution_confidence": (
                    WorkspaceMutationAttributionConfidence.UNATTRIBUTED_FINALIZATION_CHANGE.value
                ),
                "status": WorkspaceRevisionDeltaStatus.TRUNCATED.value,
                "before_revision": None,
                "after_revision": None,
                "paths": [],
                "retained_paths": 0,
                "total_paths": 0,
                "truncated": True,
                "head_changed": False,
                "branch_changed": False,
                "detail_code": "finalization_delta_secret_scope_unavailable",
            }
        return payload
    payload = {
        "workspace_id": observation.identity.workspace_id,
        "observer": observation.identity.observer,
        "status": observation.status.value,
        "revision": observation.revision,
        "head_revision": observation.head_revision,
        "branch": observation.branch,
        "path_scope": observation.path_scope,
        "total_paths": observation.total_paths,
        "detail_code": observation.detail_code,
    }
    if finalization_delta is not None:
        payload["finalization_delta"] = copy_json_value(
            finalization_delta,
            "finalization_delta",
        )
    return payload


def _final_workspace_snapshot_payload(
    snapshot: WorkspaceSnapshot | None,
    *,
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> dict[str, Any] | None:
    if (
        invocation_secrets.registered_environment_secret_resolution_scope(registered_environment)
        != "static"
    ):
        # Binding snapshots may repeat observer-derived branch, revision, path,
        # or extension metadata. They share the terminal workspace-evidence
        # boundary and therefore require the same positive static-scope proof as
        # the final revision projection.
        return None
    return _workspace_snapshot_payload(snapshot)


def _source_publication_receipt_payload(
    snapshot: WorkspaceSnapshot | None,
    *,
    binding: WorkspaceBinding,
    destination_workspace_id: object,
    workload_workspace_id: object,
    expected_destination_workspace_id: str | None,
    expected_workload_workspace_id: str | None,
) -> dict[str, object] | None:
    """Publish the fixed, secret-free settlement fields owned by ``SyncBinding``."""

    if (
        not isinstance(binding, SyncBinding)
        or snapshot is None
        or snapshot.source != "sync"
        or expected_destination_workspace_id is None
        or expected_workload_workspace_id is None
        or snapshot.workspace_id != expected_destination_workspace_id
        or snapshot.metadata.get("target_workspace_id") != expected_workload_workspace_id
    ):
        return None
    metadata = snapshot.metadata
    copied_files = metadata.get("copied_files")
    copied_bytes = metadata.get("copied_bytes")
    deleted_files = metadata.get("deleted_files")
    outcome = metadata.get("outcome")
    source_conflict_policy = metadata.get("source_conflict_policy")
    sync_back = metadata.get("sync_back")
    delete_missing = metadata.get("delete_missing")
    if (
        type(destination_workspace_id) is not str
        or not destination_workspace_id
        or type(workload_workspace_id) is not str
        or not workload_workspace_id
        or type(copied_files) is not int
        or copied_files < 0
        or type(copied_bytes) is not int
        or copied_bytes < 0
        or type(deleted_files) is not int
        or deleted_files < 0
        or type(outcome) is not str
        or type(source_conflict_policy) is not str
        or type(sync_back) is not str
        or type(delete_missing) is not bool
    ):
        return None
    snapshot_payload = {
        "schema": "cayu.source_publication_snapshot.v1",
        "destination_workspace_id": destination_workspace_id,
        "workload_workspace_id": workload_workspace_id,
        "source": "sync",
        "outcome": outcome,
        "source_conflict_policy": source_conflict_policy,
        "sync_back": sync_back,
        "delete_missing": delete_missing,
        "copied_files": copied_files,
        "copied_bytes": copied_bytes,
        "deleted_files": deleted_files,
    }
    snapshot_sha256 = (
        "sha256:"
        + sha256(
            canonical_durable_json_bytes(snapshot_payload, "source_publication_snapshot")
        ).hexdigest()
    )
    receipt = {
        "schema": "cayu.source_publication_receipt.v1",
        "snapshot_sha256": snapshot_sha256,
        "destination_workspace_id": destination_workspace_id,
        "workload_workspace_id": workload_workspace_id,
        "outcome": outcome,
        "source_conflict_policy": source_conflict_policy,
        "sync_back": sync_back,
        "delete_missing": delete_missing,
        "copied_files": copied_files,
        "copied_bytes": copied_bytes,
        "deleted_files": deleted_files,
    }
    return {
        **receipt,
        "receipt_sha256": "sha256:"
        + sha256(canonical_durable_json_bytes(receipt, "source_publication_receipt")).hexdigest(),
    }


def _final_git_receipt_payload(
    snapshot: WorkspaceSnapshot | None,
    *,
    binding: WorkspaceBinding,
    destination_workspace_id: object,
    workload_workspace_id: object,
    expected_destination_workspace_id: str | None,
    expected_workload_workspace_id: str | None,
) -> dict[str, object] | None:
    """Publish digest-bound final Git evidence from the Docker sync boundary."""

    if (
        type(binding) is not DockerCodingWorkspaceBinding
        or snapshot is None
        or snapshot.source != "sync"
        or expected_destination_workspace_id is None
        or expected_workload_workspace_id is None
        or snapshot.workspace_id != expected_destination_workspace_id
        or snapshot.metadata.get("target_workspace_id") != expected_workload_workspace_id
    ):
        return None
    evidence = snapshot.metadata.get("final_git_evidence")
    if type(evidence) is not dict or set(evidence) != {
        "request_fingerprint",
        "source_workspace_id",
        "baseline_revision",
        "workspace_revision",
        "status",
        "summary",
        "diff",
    }:
        return None
    request_fingerprint = evidence.get("request_fingerprint")
    source_workspace_id = evidence.get("source_workspace_id")
    baseline_revision = evidence.get("baseline_revision")
    workspace_revision = evidence.get("workspace_revision")
    status = evidence.get("status")
    summary = evidence.get("summary")
    diff = evidence.get("diff")
    if (
        type(destination_workspace_id) is not str
        or not destination_workspace_id
        or type(workload_workspace_id) is not str
        or not workload_workspace_id
        or type(request_fingerprint) is not str
        or not request_fingerprint
        or source_workspace_id != expected_destination_workspace_id
        or type(baseline_revision) is not str
        or not baseline_revision
        or type(workspace_revision) is not str
        or not workspace_revision
        or type(status) is not dict
        or set(status) != {"structured"}
        or type(status.get("structured")) is not dict
        or not is_final_git_result_envelope(status.get("structured"), mode="status")
        or type(summary) is not dict
        or set(summary) != {"structured"}
        or type(summary.get("structured")) is not dict
        or not is_final_git_result_envelope(summary.get("structured"), mode="summary")
        or type(diff) is not dict
        or set(diff) != {"content", "structured"}
        or type(diff.get("content")) is not str
        or type(diff.get("structured")) is not dict
        or not is_final_git_result_envelope(diff.get("structured"), mode="diff")
    ):
        return None
    receipt = {
        "schema": CODING_PRODUCT_FINAL_GIT_RECEIPT_SCHEMA,
        "request_fingerprint": request_fingerprint,
        "destination_workspace_id": destination_workspace_id,
        "workload_workspace_id": workload_workspace_id,
        "baseline_revision": baseline_revision,
        "workspace_revision": workspace_revision,
        "status": copy_json_value(status, "final_git_status"),
        "summary": copy_json_value(summary, "final_git_summary"),
        "diff": copy_json_value(diff, "final_git_diff"),
    }
    return {
        **receipt,
        "receipt_sha256": "sha256:"
        + sha256(canonical_durable_json_bytes(receipt, "final_git_receipt")).hexdigest(),
    }


class _BindingAbandonBlockedByMutationFence:
    """Private result distinct from every valid binding abandonment result."""


_BINDING_ABANDON_BLOCKED_BY_MUTATION_FENCE = _BindingAbandonBlockedByMutationFence()


def _abandon_binding_after_mutation_quiescence(
    registered_environment: runtime_records.RegisteredEnvironment,
    binding: WorkspaceBinding,
    bound_workspace: BoundWorkspace,
) -> bool | None | _BindingAbandonBlockedByMutationFence:
    try:
        registered_environment.workspace_mutation_fence.require_available_nowait()
    except WorkspaceMutationSettlementError:
        # Keep the runtime-owned guard state structurally separate from an
        # extension binding that happens to raise the same public exception.
        return _BINDING_ABANDON_BLOCKED_BY_MUTATION_FENCE
    return binding.abandon(bound_workspace)


def _binding_finalize_error_payload(
    error: BaseException,
    *,
    outcome: str,
    redactor: Any,
) -> dict[str, Any]:
    details = binding_finalize_error_details(error, redactor=redactor)
    failures = binding_finalize_failure_payload(error, redactor=redactor)
    if failures is None:
        failures = [{"phase": "workspace_finalize", **details}]
    return {**details, "outcome": outcome, "failures": failures}


def _binding_finalize_publication_failure_payload(
    failures: tuple[tuple[str, BaseException], ...],
    *,
    outcome: str,
    redactor: Any,
) -> dict[str, Any]:
    errors = [error for _phase, error in failures]
    combined_error: BaseException = errors[0]
    if len(errors) > 1:
        combined_error = BaseExceptionGroup(
            "Binding finalization lifecycle publication failed.",
            errors,
        )
    fatal_signal = binding_finalize_fatal_signal(combined_error)
    if fatal_signal is not None:
        raise fatal_signal
    cancellation = binding_finalize_explicit_cancellation(combined_error)
    if cancellation is not None:
        if combined_error is cancellation:
            raise cancellation
        raise combined_error from cancellation
    return {
        "outcome": outcome,
        "failures": [
            {
                "phase": phase,
                **binding_finalize_error_details(error, redactor=redactor),
            }
            for phase, error in failures
        ],
    }


def _environment_factory_base_payload(
    *,
    session: Session,
    registered_environment: runtime_records.RegisteredEnvironment,
) -> dict[str, Any]:
    factory = registered_environment.factory
    if factory is None:
        raise AssertionError("Environment factory payload requires a registered factory.")
    environment_name = registered_environment.spec.name
    return {
        "factory_type": type(factory).__name__,
        "requested_environment_name": environment_name,
        "parent_session_id": environment_allocation_parent_session_id(session),
        "causal_budget_id": session.causal_budget_id,
        "labels": copy_label_map(session.labels, "labels"),
    }


def _binding_base_payload(
    registered_environment: runtime_records.RegisteredEnvironment,
    *,
    session_id: str,
    public_authority_alias_codec: PublicAuthorityAliasCodec | None,
    redactor: SecretRedactor,
) -> dict[str, Any]:
    if registered_environment.binding_payload is not None:
        payload = copy_json_value(registered_environment.binding_payload, "binding_payload")
        payload["binding_generation_id"] = registered_environment.binding_generation_id
        return payload
    binding = registered_environment.environment.binding
    return {
        "binding_type": _durable_environment_binding_type(
            binding,
            registered_environment=registered_environment,
            session_id=session_id,
            public_authority_alias_codec=public_authority_alias_codec,
            redactor=redactor,
        ),
        "binding_generation_id": registered_environment.binding_generation_id,
        "configured_workspace_id": _durable_environment_workspace_id(
            _workspace_object_id(registered_environment.environment.workspace),
            registered_environment=registered_environment,
            session_id=session_id,
            public_authority_alias_codec=public_authority_alias_codec,
        ),
        "has_configured_runner": registered_environment.environment.runner is not None,
    }


def _event_with_binding_generation_authority(event: Event) -> Event:
    """Mark one generated binding owner as structural runtime authority."""

    if "binding_generation_id" not in event.payload:
        raise ValueError("Binding lifecycle event is missing its generation authority.")
    return event_with_runtime_payload_authority(event, "binding_generation_id")


def _bound_workspace_payload(
    bound: BoundWorkspace,
    *,
    registered_environment: runtime_records.RegisteredEnvironment,
    session_id: str,
    public_authority_alias_codec: PublicAuthorityAliasCodec | None,
) -> dict[str, Any]:
    snapshot = bound.snapshot
    bound_snapshot = _workspace_snapshot_payload(snapshot)
    if bound_snapshot is not None:
        assert snapshot is not None
        bound_snapshot["workspace_id"] = _durable_environment_workspace_id(
            snapshot.workspace_id,
            registered_environment=registered_environment,
            session_id=session_id,
            public_authority_alias_codec=public_authority_alias_codec,
        )
    return {
        "source_workspace_id": _durable_environment_workspace_id(
            _workspace_object_id(bound.source_workspace),
            registered_environment=registered_environment,
            session_id=session_id,
            public_authority_alias_codec=public_authority_alias_codec,
        ),
        "bound_workspace_id": _durable_environment_workspace_id(
            _workspace_object_id(bound.workspace),
            registered_environment=registered_environment,
            session_id=session_id,
            public_authority_alias_codec=public_authority_alias_codec,
        ),
        "bound_path": bound.path,
        "bound_metadata": copy_json_value(bound.metadata, "bound_metadata"),
        "bound_snapshot": bound_snapshot,
        "has_bound_runner": bound.runner is not None,
    }


def _workspace_snapshot_payload(snapshot: WorkspaceSnapshot | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return {
        "snapshot_id": snapshot.snapshot_id,
        "workspace_id": snapshot.workspace_id,
        "version": snapshot.version,
        "source": snapshot.source,
        "metadata": copy_json_value(snapshot.metadata, "metadata"),
    }


def _workspace_object_id(workspace: Any) -> str | None:
    if workspace is None:
        return None
    workspace_id = getattr(workspace, "id", None)
    return workspace_id if isinstance(workspace_id, str) else None


def _durable_environment_workspace_id(
    workspace_id: str | None,
    *,
    registered_environment: runtime_records.RegisteredEnvironment,
    session_id: str,
    public_authority_alias_codec: PublicAuthorityAliasCodec | None,
) -> str | None:
    if workspace_id is None:
        return None
    workspace_id = require_clean_nonblank(workspace_id, "workspace_id")
    if (
        invocation_secrets.registered_environment_secret_resolution_scope(registered_environment)
        == "static"
    ):
        return workspace_id
    if not isinstance(public_authority_alias_codec, PublicAuthorityAliasCodec):
        return "workspace-authority-unavailable"
    return public_authority_alias_codec.encode(
        workspace_id,
        field_name=_WORKSPACE_OBSERVATION_WORKSPACE_ALIAS_FIELD,
        session_id=session_id,
    )


def _durable_environment_binding_type(
    binding: WorkspaceBinding | None,
    *,
    registered_environment: runtime_records.RegisteredEnvironment,
    session_id: str,
    public_authority_alias_codec: PublicAuthorityAliasCodec | None,
    redactor: SecretRedactor,
) -> str | None:
    if binding is None:
        return None
    binding_type = type(binding).__name__
    if _runtime_owned_workspace_observer_name(binding) == binding_type:
        return binding_type
    dynamic = (
        invocation_secrets.registered_environment_secret_resolution_scope(registered_environment)
        != "static"
    )
    if not dynamic and redactor.redact_text(binding_type) == binding_type:
        return binding_type
    if not isinstance(public_authority_alias_codec, PublicAuthorityAliasCodec):
        return "workspace-observer-unavailable"
    return public_authority_alias_codec.encode(
        binding_type,
        field_name=_WORKSPACE_OBSERVATION_OBSERVER_ALIAS_FIELD,
        session_id=session_id,
    )


def _binding_outcome_for_terminal_event(event_type: EventType | str) -> str:
    if event_type == EventType.SESSION_COMPLETED:
        return "completed"
    if event_type == EventType.SESSION_FAILED:
        return "failed"
    if event_type == EventType.SESSION_INTERRUPTED:
        return "interrupted"
    return str(event_type)


def _factory_reconnect_state_from_checkpoint(
    checkpoint: dict[str, Any] | None,
    *,
    environment_name: str,
) -> tuple[dict[str, Any], str | None]:
    if checkpoint is None:
        return {}, None
    state = checkpoint.get(ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY)
    if state is None:
        metadata: dict[str, Any] = {}
    else:
        if type(state) is not dict:
            raise ValueError("Environment factory reconnect checkpoint must be an object.")
        candidate_metadata = state.get(environment_name)
        if candidate_metadata is None:
            metadata = {}
        elif type(candidate_metadata) is not dict:
            raise ValueError("Environment factory reconnect metadata must be an object.")
        else:
            metadata = copy_json_value(candidate_metadata, "reconnect_metadata")
    owners = checkpoint.get(ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY)
    if owners is None:
        return metadata, None
    if type(owners) is not dict:
        raise ValueError("Environment factory allocation owners must be an object.")
    owner = owners.get(environment_name)
    if owner is None:
        return metadata, None
    if not isinstance(owner, str) or not owner:
        raise ValueError("Environment factory allocation owner must be a nonblank string.")
    return metadata, owner


def _environment_name(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> str | None:
    if registered_environment is None:
        return None
    return registered_environment.spec.name


async def _release_unclaimed_factory_result(
    result: EnvironmentFactoryResult,
    *,
    action: EnvironmentFactoryReleaseAction,
    original_error: BaseException,
    redactor: SecretRedactor | None = None,
    on_quiescent: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if redactor is not None and not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    resolved_redactor = redactor or SecretRedactor()
    if on_quiescent is not None and not callable(on_quiescent):
        raise TypeError("on_quiescent must be callable or None.")
    payload: dict[str, Any] = {
        "action": action.value,
        "callback_provided": result.release is not None,
    }
    if result.release is not None:
        release = result.release

        async def run_release() -> None:
            await environment_operation_boundary.await_environment_operation(
                lambda: release(action),
                operation_name="Environment factory release",
                redactor=resolved_redactor,
            )

        release_attempt = asyncio.create_task(run_release())
        factory_cleanup_quiescent = False

        def retire_after_factory_quiescence() -> None:
            nonlocal factory_cleanup_quiescent
            factory_cleanup_quiescent = True
            if on_quiescent is not None:
                on_quiescent()

        def retry_release_settlement() -> asyncio.Task[None]:
            for task in _environment_factory_release_retryable_handoffs(release_attempt):
                retry_environment_factory_cleanup_settlement_task(task)
            return start_release_settlement()

        def retain_successor_retry(settlement: asyncio.Task[None]) -> None:
            try:
                settlement.result()
            except BaseException:
                try:
                    release_attempt.result()
                except BaseException:
                    release_succeeded = False
                else:
                    release_succeeded = True
                if (
                    factory_cleanup_quiescent
                    or release_succeeded
                    or _environment_factory_release_retryable_handoffs(release_attempt)
                ):
                    register_environment_factory_cleanup_retry(
                        settlement,
                        retry_release_settlement,
                    )

        def start_release_settlement() -> asyncio.Task[None]:
            async def settle_release() -> None:
                await _settle_environment_factory_release(
                    release_attempt,
                    on_quiescent=retire_after_factory_quiescence,
                )

            settlement = asyncio.create_task(settle_release())
            settlement.add_done_callback(retain_successor_retry)
            return settlement

        release_task = release_attempt if on_quiescent is None else start_release_settlement()
        try:
            cancelled = await _await_bounded_environment_factory_release(
                release_task,
                timeout_s=result.release_timeout_s,
                timeout_handoff_task=(release_task if on_quiescent is not None else None),
            )
        except BaseException as cleanup_error:
            if on_quiescent is not None:
                # The complete release settlement remains the reservation owner whether it
                # failed immediately or is still running after a timeout.
                attach_environment_factory_cleanup_settlement_task(
                    original_error,
                    environment_factory_cleanup_settlement_task(cleanup_error) or release_task,
                )
            elif (
                settlement_task := environment_factory_cleanup_settlement_task(cleanup_error)
            ) is not None:
                attach_environment_factory_cleanup_settlement_task(
                    original_error,
                    settlement_task,
                )
            diagnostic = exception_diagnostic(
                cleanup_error,
                empty_message="environment factory release failed",
                nonportable_message=(
                    "Environment factory release failed with a non-portable diagnostic."
                ),
                redactor=resolved_redactor,
            )
            payload.update(
                {
                    "completed": False,
                    **diagnostic.payload_fields(),
                    "timeout_s": result.release_timeout_s,
                }
            )
            _add_exception_note_safely(
                original_error,
                "Environment factory result release failed after "
                f"{action.value}: {diagnostic.error_type}: {diagnostic.message}.",
            )
            fatal_signal = binding_finalize_fatal_signal(cleanup_error)
            if fatal_signal is not None:
                if fatal_signal is cleanup_error:
                    raise
                raise fatal_signal from cleanup_error
            if binding_finalize_explicit_cancellation(cleanup_error) is not None:
                raise cleanup_error
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                cancellation = asyncio.CancelledError()
                cancellation.add_note(
                    "Environment factory result release failed while cancellation was pending: "
                    f"{diagnostic.error_type}: {diagnostic.message}."
                )
                raise BaseExceptionGroup(
                    "Environment factory result release failed after caller cancellation.",
                    [cancellation, cleanup_error],
                ) from cleanup_error
        else:
            payload["completed"] = True
            if cancelled:
                raise asyncio.CancelledError()
        return payload
    if action is EnvironmentFactoryReleaseAction.PRESERVE:
        payload.update(
            {
                "completed": False,
                "error": "Durable factory result has no release callback.",
                "error_type": "MissingEnvironmentFactoryRelease",
            }
        )
        _add_exception_note_safely(
            original_error,
            "Environment factory result has durable reconnect state but no release callback; "
            "the runtime left the live allocation untouched rather than closing it terminally.",
        )
        if on_quiescent is not None:

            async def missing_release() -> None:
                raise RuntimeError(
                    "Environment factory result cannot settle failed binding ownership "
                    "without a release callback."
                )

            missing_release_task = asyncio.create_task(missing_release())
            attach_environment_factory_cleanup_settlement_task(
                original_error,
                missing_release_task,
            )
        return payload

    cleanup_errors: list[tuple[str, Exception]] = []

    async def run_fallback_release() -> None:
        runner = result.environment.runner
        if runner is not None:
            try:
                await environment_operation_boundary.await_environment_operation(
                    runner.close,
                    operation_name="Environment runner fallback release",
                    redactor=resolved_redactor,
                )
            except Exception as cleanup_error:
                cleanup_errors.append(("runner", cleanup_error))

        binding = result.environment.binding
        close = getattr(binding, "close", None)
        if callable(close):
            try:

                async def close_binding() -> None:
                    close_result = close()
                    if inspect.isawaitable(close_result):
                        await close_result

                await environment_operation_boundary.await_environment_operation(
                    close_binding,
                    operation_name="Environment binding fallback release",
                    redactor=resolved_redactor,
                )
            except Exception as cleanup_error:
                cleanup_errors.append(("binding", cleanup_error))
        if not cleanup_errors and on_quiescent is not None:
            on_quiescent()

    fallback_task = asyncio.create_task(run_fallback_release())
    try:
        cancelled = await _await_bounded_environment_factory_release(
            fallback_task,
            timeout_s=result.release_timeout_s,
        )
    except BaseException as cleanup_error:
        if (
            settlement_task := environment_factory_cleanup_settlement_task(cleanup_error)
        ) is not None:
            attach_environment_factory_cleanup_settlement_task(
                original_error,
                settlement_task,
            )
        diagnostic = exception_diagnostic(
            cleanup_error,
            empty_message="environment factory fallback release failed",
            nonportable_message=(
                "Environment factory fallback release failed with a non-portable diagnostic."
            ),
            redactor=resolved_redactor,
        )
        payload.update(
            {
                "completed": False,
                **diagnostic.payload_fields(),
                "timeout_s": result.release_timeout_s,
            }
        )
        current_task = asyncio.current_task()
        if current_task is not None and current_task.cancelling():
            cancellation = asyncio.CancelledError()
            cancellation.add_note(
                "Environment factory fallback release failed while cancellation was pending: "
                f"{diagnostic.error_type}: {diagnostic.message}."
            )
            raise cancellation from cleanup_error
        return payload
    if cancelled:
        raise asyncio.CancelledError()
    payload["completed"] = not cleanup_errors
    if cleanup_errors:
        diagnostics = [
            (phase, exception_diagnostic(error, redactor=resolved_redactor))
            for phase, error in cleanup_errors
        ]
        details = bound_diagnostic_text(
            "; ".join(
                f"{phase}: {diagnostic.error_type}: {diagnostic.message}"
                for phase, diagnostic in diagnostics
            )
        )
        payload["error"] = details
        payload["error_type"] = diagnostics[0][1].error_type
        _add_exception_note_safely(
            original_error,
            f"Environment factory fallback release incomplete after {action.value}: {details}.",
        )
    return payload


def _add_exception_note_safely(error: BaseException, note: str) -> None:
    """Attach runtime-owned portable context without invoking subclass accessors."""

    try:
        BaseException.add_note(error, bound_diagnostic_text(note))
    except BaseException:
        return


async def _await_bounded_environment_factory_release(
    task: asyncio.Task[None],
    *,
    timeout_s: float,
    timeout_handoff_task: asyncio.Task[None] | None = None,
) -> bool:
    """Finish a factory release despite cancellation, within its declared bound."""

    cancelled = False
    deadline = asyncio.get_running_loop().time() + timeout_s
    while not task.done():
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            error = TimeoutError(
                f"Environment factory result release did not complete within {timeout_s:g} seconds."
            )
            attach_environment_factory_cleanup_settlement_task(
                error,
                timeout_handoff_task or _defer_timed_out_environment_factory_release(task),
            )
            raise error
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
        except asyncio.CancelledError:
            if task.done():
                task.result()
            cancelled = True
        except TimeoutError as exc:
            if task.done():
                task.result()
                break
            error = TimeoutError(
                f"Environment factory result release did not complete within {timeout_s:g} seconds."
            )
            attach_environment_factory_cleanup_settlement_task(
                error,
                timeout_handoff_task or _defer_timed_out_environment_factory_release(task),
            )
            raise error from exc
    task.result()
    return cancelled


def _environment_factory_cleanup_handoffs(
    error: BaseException,
) -> tuple[tuple[asyncio.Task[None], ...], tuple[BaseException, ...]]:
    """Split grouped cleanup failures into owned successors and unresolved leaves."""

    pending = [error]
    tasks: list[asyncio.Task[None]] = []
    failures: list[BaseException] = []
    while pending:
        candidate = pending.pop()
        task = environment_factory_cleanup_settlement_task(candidate)
        if task is not None:
            tasks.append(task)
            continue
        if isinstance(candidate, BaseExceptionGroup):
            children = exception_group_children(candidate)
            if children is not None:
                pending.extend(reversed(children))
                continue
        failures.append(candidate)
    return tuple(dict.fromkeys(tasks)), tuple(failures)


def _environment_factory_release_retryable_handoffs(
    release_task: asyncio.Task[None],
) -> tuple[asyncio.Task[None], ...]:
    """Find retryable cleanup owners below one completed public release call."""

    pending = [release_task]
    seen: set[asyncio.Task[None]] = set()
    retryable: list[asyncio.Task[None]] = []
    while pending:
        task = pending.pop()
        if task in seen or not task.done():
            continue
        seen.add(task)
        try:
            task.result()
        except BaseException as error:
            nested, _ = _environment_factory_cleanup_handoffs(error)
        else:
            continue
        for nested_task in nested:
            if environment_factory_cleanup_retry_available(nested_task):
                retryable.append(nested_task)
            else:
                pending.append(nested_task)
    return tuple(dict.fromkeys(retryable))


def _defer_timed_out_environment_factory_release(
    release_task: asyncio.Task[None],
) -> asyncio.Task[None]:
    """Follow a timed-out release through any later cleanup-owner handoff."""

    return asyncio.create_task(
        _settle_environment_factory_release(release_task),
        name="cayu-timed-out-environment-factory-release-settlement",
    )


async def _settle_environment_factory_release(
    release_task: asyncio.Task[None],
    *,
    on_quiescent: Callable[[], None] | None = None,
) -> None:
    """Follow one release through authenticated successor cleanup owners.

    This task is the lifecycle's cleanup owner, not an interruptible waiter.
    Cancellation stops neither opaque factory work nor its successor owners, so
    the task completes from their authoritative outcome. If cancellation races
    a failure, preserve both without turning the owner into a cancelled task;
    once every owner and ``on_quiescent`` succeed, the cleanup is settled.
    """

    pending = [release_task]
    seen: set[asyncio.Task[None]] = set()
    failures: list[BaseException] = []
    cancellation: asyncio.CancelledError | None = None
    while pending:
        task = pending.pop()
        if task in seen:
            continue
        seen.add(task)
        outcome = await await_shielded_task_outcome(task)
        error = outcome.error
        cancellation = cancellation or outcome.cancellation
        if error is None:
            continue
        nested, unresolved = _environment_factory_cleanup_handoffs(error)
        unseen = tuple(task for task in nested if task not in seen)
        if len(unseen) != len(nested):
            # A terminal task that delegates back to itself or an earlier
            # owner supplies no new proof of quiescence. Preserve its
            # original failure so lifecycle capacity remains fenced.
            failures.append(_environment_factory_cleanup_failure(error))
        pending.extend(unseen)
        failures.extend(_environment_factory_cleanup_failure(failure) for failure in unresolved)
    if failures:
        failure: BaseException = (
            failures[0]
            if len(failures) == 1
            else BaseExceptionGroup(
                "Environment factory cleanup settlement chain failed.",
                failures,
            )
        )
        if cancellation is not None:
            raise BaseExceptionGroup(
                "Environment factory cleanup settlement failed after cancellation.",
                [cancellation, failure],
            ) from failure
        raise failure
    if on_quiescent is not None:
        try:
            on_quiescent()
        except BaseException as failure:
            if cancellation is not None:
                raise BaseExceptionGroup(
                    "Environment factory ownership release failed after cancellation.",
                    [cancellation, failure],
                ) from failure
            raise


def _environment_factory_cleanup_failure(error: BaseException) -> BaseException:
    """Classify child-only cancellation without claiming caller cancellation."""

    if isinstance(error, asyncio.CancelledError):
        return unexpected_child_cancellation_error(
            error,
            operation="Environment factory cleanup settlement",
        )
    return error
