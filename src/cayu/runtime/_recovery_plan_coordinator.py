"""Read-only recovery planning and plan-bound operator execution."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import suppress
from datetime import UTC, datetime
from hashlib import sha256
from time import monotonic
from typing import cast
from uuid import uuid4

from cayu._validation import canonical_durable_json_bytes, copy_durable_json_object
from cayu.core.events import (
    Event,
    EventType,
    event_with_runtime_envelope_authority,
    event_with_runtime_generated_id,
)
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime._durable_operation_ownership import (
    DurableOperationOwnership,
    DurableOperationOwnershipAction,
    DurableOperationOwnershipDisposition,
    DurableOperationOwnershipTransition,
    transition_durable_operation_ownership,
)
from cayu.runtime._environment_allocation import (
    ENVIRONMENT_FACTORY_ALLOCATION_INTENTS_CHECKPOINT_KEY,
    EnvironmentAllocationRecord,
    checkpoint_object_map,
)
from cayu.runtime._environment_lifecycle import (
    pending_completion_finalization_from_checkpoint,
)
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._model_step_executor import model_completion_recovery_context_from_stage
from cayu.runtime._recovery_coordinator import (
    ModelCompletionManualRecoveryRequired,
    RecoveryCoordinator,
)
from cayu.runtime._task_store_operation_boundary import (
    task_store_exact_interrupted_handoff_capability_is_complete,
)
from cayu.runtime.approvals import ToolApprovalRecoveryOutcome
from cayu.runtime.execution_profiles import execution_profile_from_session_metadata
from cayu.runtime.pending_actions import (
    checkpoint_has_pending_action_candidate,
    pending_action_from_records,
    pending_action_source_is_invalid,
    project_pending_action_checkpoint,
    project_pending_action_event_record,
)
from cayu.runtime.provider_operations import RecoverableProviderOperation
from cayu.runtime.recovery_plans import (
    RECOVERY_PLAN_MAX_CURSOR_BYTES,
    RecoveryBlockerCode,
    RecoveryClaimEvidence,
    RecoveryDecision,
    RecoveryEnvironmentEvidence,
    RecoveryExecutionRequest,
    RecoveryInterruptionCascadeEvidence,
    RecoveryItemExecutionStatus,
    RecoveryItemReceipt,
    RecoveryModelStageEvidence,
    RecoveryPendingActionEvidence,
    RecoveryPlan,
    RecoveryPlanAction,
    RecoveryPlanBlocker,
    RecoveryPlanExecutionEvidence,
    RecoveryPlanExecutionFenced,
    RecoveryPlanItem,
    RecoveryPlanRequest,
    RecoveryReceipt,
    RecoveryRegistrationEvidence,
    RecoveryRegistrationStatus,
    RecoveryTaskClaimEvidence,
    StaleRecoveryPlanError,
)
from cayu.runtime.sessions import (
    PENDING_ACTION_EVENT_TYPE_VALUES,
    EventOrder,
    EventQuery,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    IncompleteSessionRecoveryResult,
    ModelCompletionManualRecoveryRequest,
    ModelCompletionManualRecoveryResult,
    PendingActionKind,
    PendingActionQuery,
    PendingActionRecord,
    PendingActionSession,
    Session,
    SessionOrder,
    SessionQuery,
    SessionStatus,
    SessionStore,
    _incomplete_recovery_claim_from_checkpoint,
    _invocation_lifecycle_authority_read_scope,
    _session_run_operation_from_checkpoint,
)
from cayu.runtime.tasks import (
    Task,
    TaskClaimLost,
    TaskInterruptedHandoffReceipt,
    TaskQuery,
    TaskStatus,
    TaskStore,
    TaskTerminalizationRequest,
    TaskTerminalKind,
    interrupted_task_handoff_request,
)
from cayu.runtime.tool_rounds import ToolRoundRecoveryRequest

RECOVERY_PLAN_EXECUTION_CHECKPOINT_KEY = "recovery_plan_execution"
_RECOVERY_PLAN_EXECUTION_LEASE_SECONDS = 900
_RECOVERY_PLAN_EXECUTION_HEARTBEAT_SECONDS = 60.0
_RECOVERY_PLAN_TASK_LEASE_SECONDS = 900
_RECOVERY_PLAN_TASK_HEARTBEAT_SECONDS = 60.0
_RECOVERY_PLAN_CURSOR_VERSION = 1
_RECOVERY_PLAN_STATUS_ORDER = (
    SessionStatus.PENDING,
    SessionStatus.RUNNING,
    SessionStatus.INTERRUPTING,
    SessionStatus.INTERRUPTED,
    SessionStatus.FAILED,
    SessionStatus.COMPLETED,
)
_ACTIVE_TASK_STATUSES = frozenset(
    {
        TaskStatus.CLAIMED,
        TaskStatus.RUNNING,
        TaskStatus.PAUSED,
        TaskStatus.BLOCKED,
        TaskStatus.NEEDS_ATTENTION,
    }
)


class _RecoveryPlanSnapshotChanged(RuntimeError):
    """A session could not be projected from one stable read snapshot."""


ResolveRegisteredAgent = Callable[[str], runtime_records.RegisteredAgentState]
ResolveRegisteredProvider = Callable[[str | None], runtime_records.RegisteredProvider]
ResolveRegisteredEnvironment = Callable[[str | None], runtime_records.RegisteredEnvironment | None]
RecoverIncompleteSession = Callable[
    [IncompleteSessionRecoveryRequest], Awaitable[IncompleteSessionRecoveryResult]
]
RecoverModelCompletion = Callable[
    [ModelCompletionManualRecoveryRequest], Awaitable[ModelCompletionManualRecoveryResult]
]
RecoverToolRound = Callable[[ToolRoundRecoveryRequest], AsyncIterator[Event]]
RecoverInterruptionCascade = Callable[[str, int | None], Awaitable[bool]]
ProjectSessionId = Callable[[str], str]
ResolveSessionId = Callable[[str], Awaitable[str]]


def _hash_material(label: str, material: object) -> str:
    return sha256(canonical_durable_json_bytes(material, label)).hexdigest()


def _safe_ref(kind: str, value: str) -> str:
    return f"{kind}:sha256:{sha256(value.encode('utf-8')).hexdigest()}"


def _checkpoint_without_plan_execution(
    checkpoint: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if checkpoint is None:
        return None
    copied = copy_durable_json_object(dict(checkpoint), "checkpoint")
    copied.pop(RECOVERY_PLAN_EXECUTION_CHECKPOINT_KEY, None)
    return copied or None


def _session_state_fingerprint(
    session: Session,
    checkpoint: Mapping[str, object] | None,
) -> str:
    """Bind every session/checkpoint field that may authorize recovery.

    The complete session record is included so progress or operator annotation
    after planning makes the item stale. Installing the plan-execution claim
    advances store activity only after this fingerprint has matched atomically;
    retries prove that started execution through its exact checkpoint marker.
    """

    session_material = session.model_dump(mode="json", warnings=False)
    return _hash_material(
        "recovery_plan_session_state",
        {
            "session": session_material,
            "checkpoint": _checkpoint_without_plan_execution(checkpoint),
        },
    )


def _session_authority_fingerprint(
    session: Session,
    checkpoint: Mapping[str, object] | None,
) -> str:
    """Bind durable recovery authority while excluding lease-write timestamps."""

    session_material = session.model_dump(mode="json", warnings=False)
    session_material.pop("updated_at", None)
    session_material.pop("last_activity_at", None)
    return _hash_material(
        "recovery_plan_session_authority",
        {
            "session": session_material,
            "checkpoint": _checkpoint_without_plan_execution(checkpoint),
        },
    )


def _item_authority_projection(item: RecoveryPlanItem) -> dict[str, object]:
    projection = item.model_dump(mode="json", warnings=False)
    projection.pop("item_id", None)
    projection.pop("state_fingerprint", None)
    projection.pop("plan_execution", None)
    return projection


def _action_identity(action: PendingActionRecord) -> dict[str, object]:
    return {
        "kind": str(action.kind),
        "event_sequence": action.event.sequence,
        "approval_id": action.approval_id,
        "input_id": action.input_id,
        "round_id": action.round_id,
        "tool_call_id": action.tool_call_id,
    }


def _action_ref(action: PendingActionRecord) -> str:
    return "action:sha256:" + _hash_material("recovery_pending_action", _action_identity(action))


def _model_stage_ref(stage_id: str, marker_digest: str) -> str:
    return "stage:sha256:" + _hash_material(
        "recovery_model_stage",
        {"stage_id": stage_id, "marker_digest": marker_digest},
    )


def _plan_id(
    *,
    created_at: datetime,
    request: RecoveryPlanRequest,
    items: tuple[RecoveryPlanItem, ...],
    inspected_session_count: int,
    next_cursor: str | None,
) -> str:
    digest = _hash_material(
        "recovery_plan",
        {
            "record_type": "cayu.recovery-plan",
            "schema_version": 1,
            "created_at": created_at.isoformat(),
            "request": {
                "selection": {
                    "session_ids": list(request.selection.session_ids),
                    "statuses": sorted(status.value for status in request.selection.statuses),
                    "inactive_for_seconds": request.selection.inactive_for_seconds,
                    "cursor": request.selection.cursor,
                },
                "bounds": request.bounds.model_dump(mode="json", warnings=False),
            },
            "items": [item.model_dump(mode="json", warnings=False) for item in items],
            "inspected_session_count": inspected_session_count,
            "next_cursor": next_cursor,
        },
    )
    return f"recovery-plan:sha256:{digest}"


def _item_id(*, public_session_id: str, state_fingerprint: str) -> str:
    return "recovery-item:sha256:" + _hash_material(
        "recovery_plan_item",
        {"session_id": public_session_id, "state_fingerprint": state_fingerprint},
    )


def _decision_fingerprint(decision: RecoveryDecision) -> str:
    return _hash_material(
        "recovery_plan_decision",
        decision.model_dump(mode="json", warnings=False),
    )


def _selection_binding(request: RecoveryPlanRequest) -> str:
    return _hash_material(
        "recovery_plan_selection",
        {
            "statuses": sorted(status.value for status in request.selection.statuses),
            "inactive_for_seconds": request.selection.inactive_for_seconds,
        },
    )


def _encode_cursor(
    *,
    status: SessionStatus,
    session_cursor: str | None,
    request: RecoveryPlanRequest,
) -> str:
    payload = canonical_durable_json_bytes(
        {
            "version": _RECOVERY_PLAN_CURSOR_VERSION,
            "status": status.value,
            "session_cursor": session_cursor,
            "selection": _selection_binding(request),
        },
        "recovery_plan_cursor",
    )
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    if len(encoded.encode("utf-8")) > RECOVERY_PLAN_MAX_CURSOR_BYTES:
        raise ValueError("Recovery-plan cursor exceeds its byte limit.")
    return encoded


def _decode_cursor(
    cursor: str, *, request: RecoveryPlanRequest
) -> tuple[SessionStatus, str | None]:
    try:
        padding = "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode((cursor + padding).encode("ascii"))
        payload = json.loads(raw)
    except (UnicodeEncodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Recovery-plan cursor is malformed.") from exc
    if (
        type(payload) is not dict
        or payload.get("version") != _RECOVERY_PLAN_CURSOR_VERSION
        or payload.get("selection") != _selection_binding(request)
    ):
        raise ValueError("Recovery-plan cursor does not match the selection.")
    try:
        status = SessionStatus(payload.get("status"))
    except ValueError as exc:
        raise ValueError("Recovery-plan cursor has an invalid status.") from exc
    session_cursor = payload.get("session_cursor")
    if session_cursor is not None and type(session_cursor) is not str:
        raise ValueError("Recovery-plan cursor has an invalid store cursor.")
    return status, session_cursor


def _parse_execution_marker(checkpoint: Mapping[str, object] | None) -> dict[str, object] | None:
    if checkpoint is None:
        return None
    raw_value = checkpoint.get(RECOVERY_PLAN_EXECUTION_CHECKPOINT_KEY)
    if raw_value is None:
        return None
    if type(raw_value) is not dict:
        raise ValueError("Recovery-plan execution checkpoint is invalid.")
    raw = cast("dict[str, object]", raw_value)
    if raw.get("version") != 1:
        raise ValueError("Recovery-plan execution checkpoint is invalid.")
    required = {
        "version",
        "plan_id",
        "item_id",
        "execution_id",
        "state_fingerprint",
        "decision_fingerprint",
        "ownership",
    }
    if set(raw) != required:
        raise ValueError("Recovery-plan execution checkpoint has invalid fields.")
    for field_name in (
        "plan_id",
        "item_id",
        "execution_id",
        "state_fingerprint",
        "decision_fingerprint",
    ):
        value = raw.get(field_name)
        if type(value) is not str or not value.strip():
            raise ValueError("Recovery-plan execution checkpoint has invalid identity.")
    decision_fingerprint = cast("str", raw["decision_fingerprint"])
    if len(decision_fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in decision_fingerprint
    ):
        raise ValueError("Recovery-plan execution decision fingerprint is invalid.")
    ownership = DurableOperationOwnership.model_validate(raw["ownership"])
    if ownership.state.value != "active" or ownership.lease_expires_at is None:
        raise ValueError("Recovery-plan execution checkpoint has inactive ownership.")
    return {
        "version": 1,
        "plan_id": raw["plan_id"],
        "item_id": raw["item_id"],
        "execution_id": raw["execution_id"],
        "state_fingerprint": raw["state_fingerprint"],
        "decision_fingerprint": decision_fingerprint,
        "ownership": ownership,
    }


class RecoveryPlanCoordinator:
    """Compose a registered app's existing recovery authorities into one surface."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        task_store: TaskStore | None,
        event_writer: RuntimeEventWriter,
        recovery_coordinator: RecoveryCoordinator,
        resolve_registered_agent: ResolveRegisteredAgent,
        resolve_registered_provider: ResolveRegisteredProvider,
        resolve_registered_environment: ResolveRegisteredEnvironment,
        recover_incomplete_session: RecoverIncompleteSession,
        recover_model_completion: RecoverModelCompletion,
        recover_tool_round: RecoverToolRound,
        recover_interruption_cascade: RecoverInterruptionCascade,
        project_session_id: ProjectSessionId,
        resolve_session_id: ResolveSessionId,
        clock: Callable[[], datetime],
    ) -> None:
        self._session_store = session_store
        self._task_store = task_store
        self._event_writer = event_writer
        self._recovery_coordinator = recovery_coordinator
        self._resolve_registered_agent = resolve_registered_agent
        self._resolve_registered_provider = resolve_registered_provider
        self._resolve_registered_environment = resolve_registered_environment
        self._recover_incomplete_session = recover_incomplete_session
        self._recover_model_completion = recover_model_completion
        self._recover_tool_round = recover_tool_round
        self._recover_interruption_cascade = recover_interruption_cascade
        self._project_session_id = project_session_id
        self._resolve_session_id = resolve_session_id
        self._clock = clock

    async def plan_recovery(self, request: RecoveryPlanRequest) -> RecoveryPlan:
        if type(request) is not RecoveryPlanRequest:
            raise TypeError("Recovery planning requires a RecoveryPlanRequest.")
        request = request.model_copy(deep=True)
        private_ids, inspected, next_cursor = await self._select_sessions(request)
        items: list[RecoveryPlanItem] = []
        for private_session_id in private_ids:
            item = await self._plan_item(private_session_id, request=request)
            if item is not None:
                items.append(item)
        created_at = self._clock().astimezone(UTC)
        frozen_items = tuple(items)
        return RecoveryPlan(
            plan_id=_plan_id(
                created_at=created_at,
                request=request,
                items=frozen_items,
                inspected_session_count=inspected,
                next_cursor=next_cursor,
            ),
            created_at=created_at,
            request=request,
            items=frozen_items,
            inspected_session_count=inspected,
            next_cursor=next_cursor,
        )

    async def _select_sessions(
        self,
        request: RecoveryPlanRequest,
    ) -> tuple[tuple[str, ...], int, str | None]:
        selection = request.selection
        bounds = request.bounds
        if selection.session_ids:
            if len(selection.session_ids) > bounds.item_limit:
                raise ValueError("Explicit session selection exceeds item_limit.")
            resolved: list[str] = []
            for public_id in selection.session_ids:
                resolved.append(await self._resolve_session_id(public_id))
            if len(set(resolved)) != len(resolved):
                raise ValueError("Explicit recovery selection resolves to duplicate sessions.")
            return tuple(resolved), len(resolved), None

        statuses = tuple(
            status for status in _RECOVERY_PLAN_STATUS_ORDER if status in selection.statuses
        )
        start_index = 0
        initial_cursor: str | None = None
        if selection.cursor is not None:
            cursor_status, initial_cursor = _decode_cursor(selection.cursor, request=request)
            if cursor_status not in statuses:
                raise ValueError("Recovery-plan cursor status is outside the selection.")
            start_index = statuses.index(cursor_status)

        selected: list[str] = []
        selected_set: set[str] = set()
        inspected = 0

        def next_status_cursor(index: int) -> str | None:
            if index + 1 >= len(statuses):
                return None
            return _encode_cursor(
                status=statuses[index + 1],
                session_cursor=None,
                request=request,
            )

        for status_index in range(start_index, len(statuses)):
            status = statuses[status_index]
            cursor = initial_cursor if status_index == start_index else None
            seen_cursors = set() if cursor is None else {cursor}
            while len(selected) < bounds.item_limit and inspected < bounds.inspection_limit:
                remaining = min(
                    1000,
                    bounds.item_limit - len(selected),
                    bounds.inspection_limit - inspected,
                )
                page = await self._session_store.list_sessions(
                    SessionQuery(
                        status=status,
                        inactive_for_seconds=selection.inactive_for_seconds,
                        limit=remaining,
                        cursor=cursor,
                        order_by=SessionOrder.UPDATED_AT_DESC,
                    )
                )
                if not page.sessions:
                    if page.next_cursor is not None:
                        raise RuntimeError("Session store returned an empty page with a cursor.")
                    break
                if len(page.sessions) > remaining:
                    raise RuntimeError(
                        "Session store returned more recovery candidates than requested."
                    )
                page_ids = tuple(session.id for session in page.sessions)
                if selected_set.intersection(page_ids) or len(set(page_ids)) != len(page_ids):
                    raise RuntimeError("Session store repeated a recovery candidate.")
                inspected += len(page.sessions)
                selected.extend(page_ids)
                selected_set.update(page_ids)
                if len(selected) >= bounds.item_limit or inspected >= bounds.inspection_limit:
                    next_cursor = (
                        _encode_cursor(
                            status=status,
                            session_cursor=page.next_cursor,
                            request=request,
                        )
                        if page.next_cursor is not None
                        else next_status_cursor(status_index)
                    )
                    return tuple(selected), inspected, next_cursor
                if page.next_cursor is None:
                    break
                if page.next_cursor in seen_cursors:
                    raise RuntimeError("Session store repeated a recovery-plan cursor.")
                seen_cursors.add(page.next_cursor)
                cursor = page.next_cursor
            initial_cursor = None
        return tuple(selected), inspected, None

    async def _plan_item(
        self,
        private_session_id: str,
        *,
        request: RecoveryPlanRequest,
    ) -> RecoveryPlanItem | None:
        for _attempt in range(3):
            session = await self._session_store.load(private_session_id)
            if session is None:
                raise KeyError(f"Session not found: {private_session_id}")
            checkpoint = await self._session_store.load_checkpoint(private_session_id)
            fingerprint = _session_state_fingerprint(session, checkpoint)
            item = await self._project_item_state(
                session,
                checkpoint,
                fingerprint=fingerprint,
                request=request,
            )
            confirmed_session = await self._session_store.load(private_session_id)
            confirmed_checkpoint = await self._session_store.load_checkpoint(private_session_id)
            if (
                confirmed_session is not None
                and _session_state_fingerprint(confirmed_session, confirmed_checkpoint)
                == fingerprint
            ):
                return item
        raise _RecoveryPlanSnapshotChanged("Session changed repeatedly during recovery planning.")

    async def _project_item_state(
        self,
        session: Session,
        checkpoint: dict[str, object] | None,
        *,
        fingerprint: str,
        request: RecoveryPlanRequest,
    ) -> RecoveryPlanItem:
        public_session_id = self._project_session_id(session.id)
        blockers: list[RecoveryPlanBlocker] = []
        allowed_actions: list[RecoveryPlanAction] = [RecoveryPlanAction.LEAVE_INTACT]

        expected_profile_fingerprint: str | None = None
        with suppress(TypeError, ValueError):
            expected_profile_fingerprint = execution_profile_from_session_metadata(
                session.metadata
            ).fingerprint

        registration_status = RecoveryRegistrationStatus.READY
        registration_reason: str | None = None
        registered_provider: runtime_records.RegisteredProvider | None = None
        try:
            self._resolve_registered_agent(session.agent_name)
        except KeyError:
            registration_status = RecoveryRegistrationStatus.MISSING_AGENT
        if registration_status is RecoveryRegistrationStatus.READY:
            try:
                registered_provider = self._resolve_registered_provider(session.provider_name)
            except KeyError:
                registration_status = RecoveryRegistrationStatus.MISSING_PROVIDER
        if registration_status is RecoveryRegistrationStatus.READY:
            try:
                self._resolve_registered_environment(session.environment_name)
            except KeyError:
                registration_status = RecoveryRegistrationStatus.MISSING_ENVIRONMENT

        preflight: IncompleteSessionRecoveryResult | None = None
        if registration_status is RecoveryRegistrationStatus.READY:
            try:
                preflight = await self._recovery_coordinator.preflight_incomplete_session(
                    session=session,
                    inactive_for_seconds=request.selection.inactive_for_seconds,
                )
            except ModelCompletionManualRecoveryRequired:
                registration_reason = "model_completion_manual_recovery_required"
            except Exception as exc:
                registration_status = RecoveryRegistrationStatus.INCOMPATIBLE
                registration_reason = type(exc).__name__

        if registration_status is not RecoveryRegistrationStatus.READY:
            blockers.append(
                RecoveryPlanBlocker(
                    code=(
                        RecoveryBlockerCode.REGISTRATION_INCOMPATIBLE
                        if registration_status is RecoveryRegistrationStatus.INCOMPATIBLE
                        else RecoveryBlockerCode.REGISTRATION_UNAVAILABLE
                    )
                )
            )

        claim_evidence: RecoveryClaimEvidence | None = None
        try:
            claim = _incomplete_recovery_claim_from_checkpoint(checkpoint)
        except (TypeError, ValueError):
            claim = None
            blockers.append(RecoveryPlanBlocker(code=RecoveryBlockerCode.INVALID_DURABLE_STATE))
        if claim is not None:
            claim_evidence = RecoveryClaimEvidence(
                claim_ref=_safe_ref("recovery-claim", claim[0]),
                lease_expires_at=claim[1],
            )
            if claim[1] > self._clock():
                blockers.append(RecoveryPlanBlocker(code=RecoveryBlockerCode.ACTIVE_RECOVERY_CLAIM))

        try:
            run_operation = _session_run_operation_from_checkpoint(checkpoint)
        except (TypeError, ValueError):
            run_operation = None
            blockers.append(RecoveryPlanBlocker(code=RecoveryBlockerCode.INVALID_DURABLE_STATE))
        run_operation_ref = (
            None
            if run_operation is None
            else _safe_ref("run-operation", run_operation.operation_id)
        )

        pending_records, pending_source_invalid = await self._pending_actions(session, checkpoint)
        pending_actions = tuple(
            RecoveryPendingActionEvidence(
                action_ref=_action_ref(action),
                kind=action.kind,
                tool_name=action.tool_name,
                round_ref=(
                    None if action.round_id is None else _safe_ref("tool-round", action.round_id)
                ),
                tool_call_ref=(
                    None
                    if action.tool_call_id is None
                    else _safe_ref("tool-call", action.tool_call_id)
                ),
            )
            for action in pending_records
        )
        for action, evidence in zip(pending_records, pending_actions, strict=True):
            if action.kind is PendingActionKind.MANUAL_RECOVERY:
                blockers.append(
                    RecoveryPlanBlocker(
                        code=RecoveryBlockerCode.TOOL_EFFECT_OUTCOME_UNKNOWN,
                        action_ref=evidence.action_ref,
                    )
                )
                allowed_actions.extend(
                    (
                        RecoveryPlanAction.TOOL_MARK_COMPLETED,
                        RecoveryPlanAction.TOOL_MARK_FAILED,
                    )
                )
            elif action.kind is PendingActionKind.TOOL_APPROVAL:
                blockers.append(
                    RecoveryPlanBlocker(
                        code=RecoveryBlockerCode.TOOL_APPROVAL_REQUIRED,
                        action_ref=evidence.action_ref,
                    )
                )
            elif action.kind is PendingActionKind.USER_INPUT:
                blockers.append(
                    RecoveryPlanBlocker(
                        code=RecoveryBlockerCode.USER_INPUT_REQUIRED,
                        action_ref=evidence.action_ref,
                    )
                )
        if pending_source_invalid:
            blockers.append(RecoveryPlanBlocker(code=RecoveryBlockerCode.INVALID_DURABLE_STATE))

        active = await self._session_store.load_active_model_completion_stage(session.id)
        model_evidence: RecoveryModelStageEvidence | None = None
        if active is not None:
            dispatch = await self._session_store.load_model_completion_stage_dispatch(
                session.id,
                active.stage.stage_id,
            )
            provider_reattachment = False
            provider_operation_ref: str | None = None
            if registered_provider is not None:
                try:
                    recoverable_operation = (
                        await self._recovery_coordinator._recoverable_provider_operation(
                            active.stage, registered_provider=registered_provider
                        )
                    )
                    provider_reattachment = recoverable_operation is not None
                    if recoverable_operation is not None:
                        operation = recoverable_operation[0]
                        provider_operation_ref = _safe_ref(
                            "provider-operation",
                            (
                                operation.state.operation_id
                                if isinstance(operation, RecoverableProviderOperation)
                                else operation.start_id
                            ),
                        )
                except Exception:
                    provider_reattachment = False
                    provider_operation_ref = None
            context = model_completion_recovery_context_from_stage(active.stage)
            model_evidence = RecoveryModelStageEvidence(
                stage_ref=_model_stage_ref(active.stage.stage_id, active.marker_digest),
                state=active.stage.state,
                dispatched=dispatch is not None,
                provider_reattachment_supported=provider_reattachment,
                provider_operation_ref=provider_operation_ref,
                reservation_count=(0 if context is None else len(context.budget_reservations)),
            )
            if (
                active.stage.state == "in_flight"
                and dispatch is not None
                and not provider_reattachment
            ):
                blockers.append(
                    RecoveryPlanBlocker(
                        code=RecoveryBlockerCode.MODEL_EFFECT_OUTCOME_UNKNOWN,
                        action_ref=model_evidence.stage_ref,
                    )
                )
                allowed_actions.extend(
                    (
                        RecoveryPlanAction.MODEL_MARK_FAILED,
                        RecoveryPlanAction.MODEL_MARK_INTERRUPTED,
                    )
                )

        try:
            cascade = self._interruption_cascade_evidence(checkpoint)
        except (TypeError, ValueError):
            cascade = None
            blockers.append(RecoveryPlanBlocker(code=RecoveryBlockerCode.INVALID_DURABLE_STATE))

        plan_execution: RecoveryPlanExecutionEvidence | None = None
        try:
            execution_marker = _parse_execution_marker(checkpoint)
        except (TypeError, ValueError):
            execution_marker = None
            blockers.append(RecoveryPlanBlocker(code=RecoveryBlockerCode.INVALID_DURABLE_STATE))
        if execution_marker is not None:
            execution_ownership = execution_marker["ownership"]
            assert isinstance(execution_ownership, DurableOperationOwnership)
            assert execution_ownership.lease_expires_at is not None
            plan_execution = RecoveryPlanExecutionEvidence(
                plan_ref=_safe_ref("recovery-plan", str(execution_marker["plan_id"])),
                item_ref=_safe_ref("recovery-item", str(execution_marker["item_id"])),
                execution_ref=_safe_ref(
                    "recovery-execution",
                    str(execution_marker["execution_id"]),
                ),
                decision_ref=(
                    "recovery-decision:sha256:" + str(execution_marker["decision_fingerprint"])
                ),
                lease_expires_at=execution_ownership.lease_expires_at,
            )
            if execution_ownership.lease_expires_at > self._clock():
                blockers.append(RecoveryPlanBlocker(code=RecoveryBlockerCode.ACTIVE_RECOVERY_CLAIM))

        try:
            environment_recovery = self._environment_recovery_evidence(checkpoint)
        except (TypeError, ValueError):
            environment_recovery = RecoveryEnvironmentEvidence()
            blockers.append(RecoveryPlanBlocker(code=RecoveryBlockerCode.INVALID_DURABLE_STATE))

        task_claims = await self._task_claims(session)
        if any(claim.ownership_status == "active" for claim in task_claims):
            blockers.append(RecoveryPlanBlocker(code=RecoveryBlockerCode.ACTIVE_TASK_CLAIM))
        recoverable_task_claims = tuple(
            claim for claim in task_claims if claim.ownership_status in {"expired", "unowned"}
        )
        if (
            any(claim.ownership_status == "invalid" for claim in task_claims)
            or len(recoverable_task_claims) > 1
        ):
            blockers.append(RecoveryPlanBlocker(code=RecoveryBlockerCode.INVALID_DURABLE_STATE))
        if preflight is not None:
            if preflight.actions == (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,):
                code = (
                    RecoveryBlockerCode.ACTIVE_RECOVERY_CLAIM
                    if claim_evidence is not None
                    else RecoveryBlockerCode.ACTIVE_TASK_CLAIM
                )
                blockers.append(RecoveryPlanBlocker(code=code))
            elif preflight.actions == (IncompleteSessionRecoveryAction.FAILED,):
                blockers.append(RecoveryPlanBlocker(code=RecoveryBlockerCode.INVALID_DURABLE_STATE))

        hard_blockers = {
            RecoveryBlockerCode.REGISTRATION_UNAVAILABLE,
            RecoveryBlockerCode.REGISTRATION_INCOMPATIBLE,
            RecoveryBlockerCode.ACTIVE_RECOVERY_CLAIM,
            RecoveryBlockerCode.ACTIVE_TASK_CLAIM,
            RecoveryBlockerCode.TOOL_APPROVAL_REQUIRED,
            RecoveryBlockerCode.USER_INPUT_REQUIRED,
            RecoveryBlockerCode.INVALID_DURABLE_STATE,
        }
        if any(blocker.code in hard_blockers for blocker in blockers):
            allowed_actions = [RecoveryPlanAction.LEAVE_INTACT]
        else:
            decision_blockers = {
                RecoveryBlockerCode.MODEL_EFFECT_OUTCOME_UNKNOWN,
                RecoveryBlockerCode.TOOL_EFFECT_OUTCOME_UNKNOWN,
            }
            if (preflight is None or cascade is not None) and not any(
                blocker.code in decision_blockers for blocker in blockers
            ):
                allowed_actions.append(RecoveryPlanAction.AUTOMATIC_REPAIR)
            if (
                model_evidence is not None
                and model_evidence.provider_reattachment_supported
                and registration_status is RecoveryRegistrationStatus.READY
                and not any(blocker.code in decision_blockers for blocker in blockers)
            ):
                allowed_actions.append(RecoveryPlanAction.AUTOMATIC_REPAIR)

        return RecoveryPlanItem(
            item_id=_item_id(
                public_session_id=public_session_id,
                state_fingerprint=fingerprint,
            ),
            state_fingerprint=fingerprint,
            authority_fingerprint=_session_authority_fingerprint(session, checkpoint),
            session_id=public_session_id,
            agent_name=session.agent_name,
            provider_name=session.provider_name,
            environment_name=session.environment_name,
            status=session.status,
            run_epoch=session.run_epoch,
            execution_profile_fingerprint=expected_profile_fingerprint,
            registration=RecoveryRegistrationEvidence(
                status=registration_status,
                expected_execution_profile_fingerprint=expected_profile_fingerprint,
                validated_execution_profile_fingerprint=(
                    expected_profile_fingerprint
                    if registration_status is RecoveryRegistrationStatus.READY
                    and registration_reason is None
                    else None
                ),
                reason_code=registration_reason,
            ),
            recovery_claim=claim_evidence,
            plan_execution=plan_execution,
            run_fence_ref=_safe_ref(
                "session-run-fence",
                f"{session.instance_id}:{session.run_epoch}",
            ),
            run_operation_ref=run_operation_ref,
            task_claims=task_claims,
            pending_actions=pending_actions,
            active_model_stage=model_evidence,
            environment_recovery=environment_recovery,
            interruption_cascade=cascade,
            blockers=tuple(dict.fromkeys(blockers)),
            allowed_actions=tuple(dict.fromkeys(allowed_actions)),
        )

    async def _pending_actions(
        self,
        session: Session,
        checkpoint: dict[str, object] | None,
    ) -> tuple[tuple[PendingActionRecord, ...], bool]:
        """Project terminal-indexed or still-running checkpoint actions safely."""

        result = await self._session_store.query_pending_actions(
            PendingActionQuery(
                session_id=session.id,
                statuses=frozenset({session.status}),
                limit=200,
            )
        )
        actions = list(result.actions)
        invalid = bool(result.issues)
        if actions or not checkpoint_has_pending_action_candidate(checkpoint):
            return tuple(actions), invalid

        # Store-native pending-action indexes intentionally expose terminal
        # sessions. A recovery planner must also classify a checkpoint left by
        # a process that died while still marked running. Read only the bounded
        # action-event vocabulary and reuse the same secret-free projection and
        # validation logic as the store query.
        raw_records = await self._session_store.query_events(
            EventQuery(
                session_id=session.id,
                event_types=tuple(sorted(PENDING_ACTION_EVENT_TYPE_VALUES)),
                order_by=EventOrder.SEQUENCE_DESC,
                limit=5000,
            )
        )
        try:
            projected_checkpoint = project_pending_action_checkpoint(checkpoint)
            projected_records = [
                project_pending_action_event_record(record) for record in raw_records
            ]
            projected_session = PendingActionSession.from_session(session)
            action = pending_action_from_records(
                projected_session,
                projected_records,
                projected_checkpoint,
            )
            invalid = invalid or pending_action_source_is_invalid(
                projected_session,
                projected_checkpoint,
                action,
                projected_records,
            )
        except (TypeError, ValueError):
            return (), True
        if action is not None and not invalid:
            actions.append(action)
        return tuple(actions), invalid

    async def _task_claims(self, session: Session) -> tuple[RecoveryTaskClaimEvidence, ...]:
        if self._task_store is None:
            return ()
        tasks = await self._task_store.list_tasks(TaskQuery(session_id=session.id, limit=1000))
        exact_handoffs = task_store_exact_interrupted_handoff_capability_is_complete(
            self._task_store
        )
        evidence: list[RecoveryTaskClaimEvidence] = []
        for task in tasks:
            if task.status not in _ACTIVE_TASK_STATUSES:
                continue
            ownership_status = "invalid"
            worker_and_lease_present = (
                task.worker_id is not None and task.lease_expires_at is not None
            )
            if (task.worker_id is None) != (task.lease_expires_at is None):
                ownership_status = "invalid"
            elif worker_and_lease_present:
                if (
                    task.status is not TaskStatus.RUNNING
                    or task.session_instance_id != session.instance_id
                ):
                    ownership_status = "invalid"
                elif exact_handoffs:
                    expired = await (
                        self._task_store.load_expired_interrupted_task_handoff_candidate(task.id)
                    )
                    if expired == task:
                        ownership_status = "expired"
                    elif task.lease_expires_at <= self._clock():
                        ownership_status = "invalid"
                    else:
                        ownership_status = "active"
                else:
                    ownership_status = "active"
            elif (
                task.status is TaskStatus.RUNNING
                and task.session_instance_id == session.instance_id
                and task.interrupted_handoff_id is not None
                and session.status is SessionStatus.INTERRUPTED
                and exact_handoffs
            ):
                receipt = await self._task_store.load_interrupted_task_handoff_receipt(
                    task.id,
                    task.interrupted_handoff_id,
                )
                if type(receipt) is TaskInterruptedHandoffReceipt and receipt.task == task:
                    ownership_status = "unowned"
            evidence.append(
                RecoveryTaskClaimEvidence(
                    task_ref=_safe_ref("task", task.id),
                    status=task.status,
                    ownership_status=ownership_status,
                    worker_ref=(
                        None if task.worker_id is None else _safe_ref("task-worker", task.worker_id)
                    ),
                    lease_expires_at=task.lease_expires_at,
                )
            )
        return tuple(sorted(evidence, key=lambda item: item.task_ref))

    async def _recoverable_task_for_item(
        self,
        *,
        private_session_id: str,
        item: RecoveryPlanItem,
    ) -> Task | None:
        expected = tuple(
            claim for claim in item.task_claims if claim.ownership_status in {"expired", "unowned"}
        )
        if not expected:
            return None
        if len(expected) != 1 or self._task_store is None:
            raise StaleRecoveryPlanError("Recovery task ownership is no longer unambiguous.")
        if not task_store_exact_interrupted_handoff_capability_is_complete(self._task_store):
            raise StaleRecoveryPlanError("Task store no longer supports exact owner recovery.")
        matches = [
            task
            for task in await self._task_store.list_tasks(
                TaskQuery(session_id=private_session_id, limit=1000)
            )
            if _safe_ref("task", task.id) == expected[0].task_ref
        ]
        if len(matches) != 1:
            raise StaleRecoveryPlanError("Recoverable task ownership changed after planning.")
        task = matches[0]
        ownership_status = expected[0].ownership_status
        if ownership_status == "expired":
            current = await self._task_store.load_expired_interrupted_task_handoff_candidate(
                task.id
            )
        else:
            receipt = (
                None
                if task.interrupted_handoff_id is None
                else await self._task_store.load_interrupted_task_handoff_receipt(
                    task.id,
                    task.interrupted_handoff_id,
                )
            )
            current = (
                task
                if task.status is TaskStatus.RUNNING
                and task.session_id == private_session_id
                and task.worker_id is None
                and task.lease_expires_at is None
                and type(receipt) is TaskInterruptedHandoffReceipt
                and receipt.task == task
                else None
            )
        if current != task or (
            RecoveryTaskClaimEvidence(
                task_ref=_safe_ref("task", task.id),
                status=task.status,
                ownership_status=ownership_status,
                worker_ref=(
                    None if task.worker_id is None else _safe_ref("task-worker", task.worker_id)
                ),
                lease_expires_at=task.lease_expires_at,
            )
            != expected[0]
        ):
            raise StaleRecoveryPlanError("Recoverable task ownership changed after planning.")
        return task

    async def _prepare_recoverable_task_handoff(
        self,
        *,
        task: Task,
        session: Session,
    ) -> Task:
        if self._task_store is None:
            raise RuntimeError("Recoverable task handoff requires a task store.")
        if (
            session.status is not SessionStatus.INTERRUPTED
            or task.session_id != session.id
            or task.session_instance_id != session.instance_id
        ):
            raise StaleRecoveryPlanError(
                "Task handoff requires its exact interrupted session incarnation."
            )
        handed_off = task.worker_id is None and task.lease_expires_at is None
        if handed_off:
            if task.interrupted_handoff_id is None:
                raise StaleRecoveryPlanError("Recoverable task handoff identity is missing.")
            receipt = await self._task_store.load_interrupted_task_handoff_receipt(
                task.id,
                task.interrupted_handoff_id,
            )
            if type(receipt) is not TaskInterruptedHandoffReceipt or receipt.task != task:
                raise StaleRecoveryPlanError("Recoverable task handoff evidence changed.")
            return task
        current = await self._task_store.load_expired_interrupted_task_handoff_candidate(task.id)
        if current != task:
            raise StaleRecoveryPlanError("Expired task owner changed before handoff.")
        request = interrupted_task_handoff_request(
            task,
            session_run_epoch=session.run_epoch,
        )
        try:
            receipt = await self._task_store.recover_interrupted_task_worker(request)
        except Exception:
            receipt = await self._task_store.load_interrupted_task_handoff_receipt(
                request.task_id,
                request.handoff_id,
            )
            if receipt is None:
                raise
        if type(receipt) is not TaskInterruptedHandoffReceipt or receipt.request != request:
            raise RuntimeError("Task handoff returned conflicting durable receipt evidence.")
        return receipt.task

    async def _fail_recoverable_task_after_model_disposition(
        self,
        *,
        task: Task,
        session: Session,
        item: RecoveryPlanItem,
        execution_id: str,
    ) -> None:
        if self._task_store is None:
            raise RuntimeError("Recoverable task failure requires a task store.")
        if (
            not self._task_store.supports_attached_task_recovery_terminalization
            or session.status is not SessionStatus.FAILED
            or task.session_id != session.id
            or task.session_instance_id != session.instance_id
        ):
            raise RuntimeError("Task store cannot settle the failed recovery-plan session.")
        handed_off = task.worker_id is None and task.lease_expires_at is None
        if handed_off:
            if not self._task_store.supports_idempotent_terminalization:
                raise RuntimeError("Task store cannot settle a handed-off recovery task.")
            task = await self._claim_plan_task_continuation(
                task=task,
                item=item,
                execution_id=execution_id,
            )
        if task.worker_id is None or task.lease_expires_at is None:
            raise RuntimeError("Recoverable task failure requires complete task authority.")
        request = TaskTerminalizationRequest(
            task_id=task.id,
            worker_id=task.worker_id,
            lease_expires_at=task.lease_expires_at,
            handoff_id=task.interrupted_handoff_id,
            kind=TaskTerminalKind.FAILED,
            error={
                "error_type": "RecoveryPlanModelDisposition",
                "message": "Operator marked the attached model operation failed.",
            },
            idempotency_key=(
                "recovery-plan-model-failure:v1:"
                + _hash_material(
                    "recovery_plan_model_task_failure",
                    {"item_id": item.item_id, "task_ref": _safe_ref("task", task.id)},
                )
            ),
        )
        if handed_off:
            await self._task_store.terminalize_task(request)
        else:
            await self._task_store.recover_attached_task_failure(
                request,
                session_id=session.id,
                session_instance_id=session.instance_id,
            )

    async def _claim_plan_task_continuation(
        self,
        *,
        task: Task,
        item: RecoveryPlanItem,
        execution_id: str,
    ) -> Task:
        if self._task_store is None:
            raise RuntimeError("Task continuation requires a task store.")
        identity = _hash_material(
            "recovery_plan_task_continuation",
            {
                "item_id": item.item_id,
                "execution_id": execution_id,
                "task_ref": _safe_ref("task", task.id),
                "prior_handoff_id": task.interrupted_handoff_id,
            },
        )
        worker_id = f"recovery-plan-worker:v1:{identity}"
        handoff_id = f"recovery-plan-handoff:v1:{identity}"
        page = await self._task_store.claim_interrupted_task_continuation(
            worker_id,
            handoff_id=handoff_id,
            task_id=task.id,
            lease_seconds=_RECOVERY_PLAN_TASK_LEASE_SECONDS,
            scan_limit=1,
        )
        claimed = page.task
        if (
            claimed is None
            or claimed.id != task.id
            or claimed.session_id != task.session_id
            or claimed.session_instance_id != task.session_instance_id
            or claimed.worker_id != worker_id
            or claimed.interrupted_handoff_id != handoff_id
            or claimed.lease_expires_at is None
        ):
            raise TaskClaimLost("Exact recovery-plan task continuation was not acquired.")
        return claimed

    async def _heartbeat_plan_task_continuation(
        self,
        *,
        task: Task,
        stop: asyncio.Event,
        lost: asyncio.Event,
    ) -> None:
        if self._task_store is None or task.worker_id is None or task.lease_expires_at is None:
            lost.set()
            return
        lease_expires_at = task.lease_expires_at
        local_deadline = monotonic() + _RECOVERY_PLAN_TASK_LEASE_SECONDS
        while not stop.is_set():
            remaining = local_deadline - monotonic()
            if remaining <= 0:
                lost.set()
                return
            try:
                await asyncio.wait_for(
                    stop.wait(),
                    timeout=min(_RECOVERY_PLAN_TASK_HEARTBEAT_SECONDS, remaining / 2),
                )
                return
            except TimeoutError:
                pass
            try:
                renewed = await self._task_store.heartbeat(
                    task.id,
                    task.worker_id,
                    lease_expires_at=lease_expires_at,
                    handoff_id=task.interrupted_handoff_id,
                    extend_seconds=_RECOVERY_PLAN_TASK_LEASE_SECONDS,
                )
            except TaskClaimLost:
                current = await self._task_store.load_task(task.id)
                if current is not None and current.status in {
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                }:
                    return
                lost.set()
                return
            except Exception:
                continue
            if renewed.lease_expires_at is None:
                lost.set()
                return
            lease_expires_at = renewed.lease_expires_at
            local_deadline = monotonic() + _RECOVERY_PLAN_TASK_LEASE_SECONDS

    @staticmethod
    def _environment_recovery_evidence(
        checkpoint: Mapping[str, object] | None,
    ) -> RecoveryEnvironmentEvidence:
        if checkpoint is None:
            return RecoveryEnvironmentEvidence()
        records = checkpoint_object_map(
            checkpoint,
            ENVIRONMENT_FACTORY_ALLOCATION_INTENTS_CHECKPOINT_KEY,
        )
        allocation_states = []
        for payload in records.values():
            if type(payload) is not dict:
                raise ValueError("Environment allocation intent is invalid.")
            allocation_states.append(EnvironmentAllocationRecord.from_payload(payload).state)
        return RecoveryEnvironmentEvidence(
            allocation_states=tuple(sorted(allocation_states, key=lambda state: state.value)),
            completion_finalization_pending=(
                pending_completion_finalization_from_checkpoint(dict(checkpoint)) is not None
            ),
        )

    @staticmethod
    def _interruption_cascade_evidence(
        checkpoint: Mapping[str, object] | None,
    ) -> RecoveryInterruptionCascadeEvidence | None:
        if checkpoint is None or checkpoint.get("pending_interruption_cascade") is None:
            return None
        raw_value = checkpoint["pending_interruption_cascade"]
        if type(raw_value) is not dict:
            raise ValueError("Pending interruption cascade checkpoint must be an object.")
        raw = cast("dict[str, object]", raw_value)
        attempt_id = raw.get("attempt_id")
        generation = raw.get("generation", 0)
        if (
            type(attempt_id) is not str
            or not attempt_id.strip()
            or type(generation) is not int
            or generation < 0
        ):
            raise ValueError("Pending interruption cascade identity is invalid.")
        if type(raw.get("interrupt_payload")) is not dict:
            raise ValueError("Pending interruption cascade payload is invalid.")
        claim_id = raw.get("claim_id")
        raw_expiry = raw.get("claim_expires_at")
        if raw_expiry is not None and type(raw_expiry) is not str:
            raise ValueError("Pending interruption cascade expiry is invalid.")
        try:
            expiry = None if raw_expiry is None else datetime.fromisoformat(raw_expiry)
        except (TypeError, ValueError):
            raise ValueError("Pending interruption cascade expiry is invalid.") from None
        if (claim_id is None) != (expiry is None) or (
            claim_id is not None and (type(claim_id) is not str or not claim_id.strip())
        ):
            raise ValueError("Pending interruption cascade claim is invalid.")
        return RecoveryInterruptionCascadeEvidence(
            attempt_ref=_safe_ref("interruption-cascade", attempt_id),
            generation=generation,
            claim_ref=(
                _safe_ref("interruption-cascade-claim", claim_id)
                if type(claim_id) is str and claim_id.strip()
                else None
            ),
            lease_expires_at=expiry,
            failure_recorded=raw.get("failure_recorded") is True,
        )

    async def execute_recovery(self, request: RecoveryExecutionRequest) -> RecoveryReceipt:
        if type(request) is not RecoveryExecutionRequest:
            raise TypeError("Recovery execution requires a RecoveryExecutionRequest.")
        request = request.model_copy(deep=True)
        expected_plan_id = _plan_id(
            created_at=request.plan.created_at,
            request=request.plan.request,
            items=request.plan.items,
            inspected_session_count=request.plan.inspected_session_count,
            next_cursor=request.plan.next_cursor,
        )
        if request.plan.plan_id != expected_plan_id:
            raise ValueError("Recovery plan identity does not match its contents.")

        decisions = {decision.item_id: decision for decision in request.decisions}
        semaphore = asyncio.Semaphore(request.max_concurrency)

        async def execute(item: RecoveryPlanItem) -> RecoveryItemReceipt:
            decision = decisions.get(item.item_id)
            if decision is None:
                action = (
                    RecoveryPlanAction.AUTOMATIC_REPAIR
                    if RecoveryPlanAction.AUTOMATIC_REPAIR in item.allowed_actions
                    else RecoveryPlanAction.LEAVE_INTACT
                )
                decision = RecoveryDecision(item_id=item.item_id, action=action)
            if decision.action not in item.allowed_actions:
                return RecoveryItemReceipt(
                    plan_id=request.plan.plan_id,
                    item_id=item.item_id,
                    execution_id=request.execution_id,
                    session_id=item.session_id,
                    action=decision.action,
                    status=RecoveryItemExecutionStatus.BLOCKED,
                    final_session_status=item.status,
                    final_run_epoch=item.run_epoch,
                    error_code="action_not_allowed",
                )
            async with semaphore:
                try:
                    if decision.action is RecoveryPlanAction.LEAVE_INTACT:
                        return await self._leave_item_intact(request, item, decision)
                    return await self._execute_item(request, item, decision)
                except Exception as exc:
                    failure_status = (
                        RecoveryItemExecutionStatus.BLOCKED
                        if isinstance(
                            exc,
                            (RecoveryPlanExecutionFenced, StaleRecoveryPlanError),
                        )
                        else RecoveryItemExecutionStatus.FAILED
                    )
                    with suppress(Exception):
                        private_session_id = await self._resolve_session_id(item.session_id)
                        current = await self._session_store.load(private_session_id)
                        if current is not None:
                            return RecoveryItemReceipt(
                                plan_id=request.plan.plan_id,
                                item_id=item.item_id,
                                execution_id=request.execution_id,
                                session_id=item.session_id,
                                action=decision.action,
                                status=failure_status,
                                final_session_status=current.status,
                                final_run_epoch=current.run_epoch,
                                error_code=type(exc).__name__,
                            )
                    return RecoveryItemReceipt(
                        plan_id=request.plan.plan_id,
                        item_id=item.item_id,
                        execution_id=request.execution_id,
                        session_id=item.session_id,
                        action=decision.action,
                        status=failure_status,
                        final_session_status=item.status,
                        final_run_epoch=item.run_epoch,
                        error_code=type(exc).__name__,
                    )

        results = await asyncio.gather(*(execute(item) for item in request.plan.items))
        return RecoveryReceipt(
            plan_id=request.plan.plan_id,
            execution_id=request.execution_id,
            items=tuple(results),
        )

    async def _leave_item_intact(
        self,
        request: RecoveryExecutionRequest,
        item: RecoveryPlanItem,
        decision: RecoveryDecision,
    ) -> RecoveryItemReceipt:
        """Return current state without masking an execution-id decision conflict."""

        private_session_id = await self._resolve_session_id(item.session_id)
        receipt_event_id = self._receipt_event_id(request, item)
        replay = await self._load_receipt(
            private_session_id=private_session_id,
            receipt_event_id=receipt_event_id,
            request=request,
            item=item,
            decision=decision,
        )
        if replay is not None:
            return replay.model_copy(update={"replayed": True})

        session = await self._session_store.load(private_session_id)
        if session is None:
            raise KeyError(f"Session not found: {item.session_id}")
        marker = _parse_execution_marker(
            await self._session_store.load_checkpoint(private_session_id)
        )
        if marker is not None and (
            marker["plan_id"] == request.plan.plan_id
            and marker["item_id"] == item.item_id
            and marker["execution_id"] == request.execution_id
        ):
            raise RecoveryPlanExecutionFenced(
                "Recovery execution is already bound to a mutating decision."
            )
        return RecoveryItemReceipt(
            plan_id=request.plan.plan_id,
            item_id=item.item_id,
            execution_id=request.execution_id,
            session_id=item.session_id,
            action=decision.action,
            status=RecoveryItemExecutionStatus.LEFT_INTACT,
            final_session_status=session.status,
            final_run_epoch=session.run_epoch,
        )

    async def _execute_item(
        self,
        request: RecoveryExecutionRequest,
        item: RecoveryPlanItem,
        decision: RecoveryDecision,
    ) -> RecoveryItemReceipt:
        private_session_id = await self._resolve_session_id(item.session_id)
        receipt_event_id = self._receipt_event_id(request, item)
        replay = await self._load_receipt(
            private_session_id=private_session_id,
            receipt_event_id=receipt_event_id,
            request=request,
            item=item,
            decision=decision,
        )
        if replay is not None:
            return replay.model_copy(update={"replayed": True})

        session = await self._session_store.load(private_session_id)
        if session is None:
            raise KeyError(f"Session not found: {item.session_id}")
        checkpoint = await self._session_store.load_checkpoint(private_session_id)
        state_matches = _session_state_fingerprint(session, checkpoint) == item.state_fingerprint
        marker = _parse_execution_marker(checkpoint)
        decision_fingerprint = _decision_fingerprint(decision)
        resumed_started_execution = bool(
            marker is not None
            and marker["plan_id"] == request.plan.plan_id
            and marker["item_id"] == item.item_id
            and marker["execution_id"] == request.execution_id
            and marker["state_fingerprint"] == item.state_fingerprint
            and marker["decision_fingerprint"] == decision_fingerprint
        )
        exact_item_matches = False
        if resumed_started_execution:
            comparison_checkpoint = _checkpoint_without_plan_execution(checkpoint)
            authority_matches = (
                _session_authority_fingerprint(session, comparison_checkpoint)
                == item.authority_fingerprint
            )
            try:
                current_item = await self._project_item_state(
                    session,
                    comparison_checkpoint,
                    fingerprint=_session_state_fingerprint(session, comparison_checkpoint),
                    request=request.plan.request,
                )
            except (KeyError, TypeError, ValueError, _RecoveryPlanSnapshotChanged):
                current_item = None
            exact_item_matches = bool(
                authority_matches
                and current_item is not None
                and _item_authority_projection(current_item) == _item_authority_projection(item)
            )
        else:
            try:
                current_item = await self._plan_item(
                    private_session_id,
                    request=request.plan.request,
                )
            except (KeyError, TypeError, ValueError, _RecoveryPlanSnapshotChanged):
                current_item = None
            exact_item_matches = current_item == item
        if (not state_matches or not exact_item_matches) and not resumed_started_execution:
            return RecoveryItemReceipt(
                plan_id=request.plan.plan_id,
                item_id=item.item_id,
                execution_id=request.execution_id,
                session_id=item.session_id,
                action=decision.action,
                status=RecoveryItemExecutionStatus.BLOCKED,
                final_session_status=session.status,
                final_run_epoch=session.run_epoch,
                error_code=StaleRecoveryPlanError.__name__,
            )

        recoverable_task = (
            None
            if resumed_started_execution and not exact_item_matches
            else await self._recoverable_task_for_item(
                private_session_id=private_session_id,
                item=item,
            )
        )

        claim_id = str(uuid4())
        owner_id = str(uuid4())
        try:
            ownership = await self._claim_execution(
                session=session,
                item=item,
                request=request,
                claim_id=claim_id,
                owner_id=owner_id,
                decision_fingerprint=decision_fingerprint,
                allow_started_state=resumed_started_execution,
            )
        except (RecoveryPlanExecutionFenced, StaleRecoveryPlanError) as exc:
            current = await self._session_store.load(private_session_id)
            if current is None:
                raise KeyError(f"Session not found: {item.session_id}") from exc
            return RecoveryItemReceipt(
                plan_id=request.plan.plan_id,
                item_id=item.item_id,
                execution_id=request.execution_id,
                session_id=item.session_id,
                action=decision.action,
                status=RecoveryItemExecutionStatus.BLOCKED,
                final_session_status=current.status,
                final_run_epoch=current.run_epoch,
                error_code=type(exc).__name__,
            )

        recovery_actions: tuple[IncompleteSessionRecoveryAction, ...] = ()
        event_ids: tuple[str, ...] = ()
        error_code: str | None = None
        item_status = RecoveryItemExecutionStatus.EXECUTED
        heartbeat_stop = asyncio.Event()
        heartbeat_lost = asyncio.Event()
        heartbeat = asyncio.create_task(
            self._heartbeat_execution(
                private_session_id=private_session_id,
                item=item,
                request=request,
                decision_fingerprint=decision_fingerprint,
                ownership=ownership,
                stop=heartbeat_stop,
                lost=heartbeat_lost,
            )
        )
        try:
            if resumed_started_execution and not exact_item_matches:
                item_status = RecoveryItemExecutionStatus.FAILED
                error_code = "recovery_plan_outcome_unknown"
            else:
                try:
                    recovery_actions, event_ids = await self._apply_decision(
                        request=request,
                        private_session_id=private_session_id,
                        item=item,
                        decision=decision,
                        inactive_for_seconds=request.plan.request.selection.inactive_for_seconds,
                        recoverable_task=recoverable_task,
                    )
                    if IncompleteSessionRecoveryAction.SKIPPED_ACTIVE in recovery_actions:
                        item_status = RecoveryItemExecutionStatus.BLOCKED
                        error_code = RecoveryPlanExecutionFenced.__name__
                    elif IncompleteSessionRecoveryAction.FAILED in recovery_actions:
                        item_status = RecoveryItemExecutionStatus.FAILED
                        error_code = "incomplete_session_recovery_failed"
                except Exception as exc:
                    item_status = RecoveryItemExecutionStatus.FAILED
                    error_code = type(exc).__name__
        finally:
            heartbeat_stop.set()
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
        if heartbeat_lost.is_set():
            item_status = RecoveryItemExecutionStatus.FAILED
            error_code = RecoveryPlanExecutionFenced.__name__

        current = await self._session_store.load(private_session_id)
        if current is None:
            raise KeyError(f"Session not found: {item.session_id}")
        receipt = RecoveryItemReceipt(
            plan_id=request.plan.plan_id,
            item_id=item.item_id,
            execution_id=request.execution_id,
            session_id=item.session_id,
            action=decision.action,
            status=item_status,
            final_session_status=current.status,
            final_run_epoch=current.run_epoch,
            recovery_actions=recovery_actions,
            event_ids=event_ids,
            receipt_event_id=receipt_event_id,
            error_code=error_code,
        )
        try:
            await self._commit_receipt(
                private_session_id=private_session_id,
                private_agent_name=current.agent_name,
                receipt=receipt,
                ownership=ownership,
            )
        except Exception:
            # Publication can commit before its acknowledgement or event-sink
            # fan-out fails. Exact event identity makes readback authoritative.
            committed = await self._load_receipt(
                private_session_id=private_session_id,
                receipt_event_id=receipt_event_id,
                request=request,
                item=item,
                decision=decision,
            )
            if committed is not None:
                return committed
            raise
        return receipt

    async def _claim_execution(
        self,
        *,
        session: Session,
        item: RecoveryPlanItem,
        request: RecoveryExecutionRequest,
        claim_id: str,
        owner_id: str,
        decision_fingerprint: str,
        allow_started_state: bool,
    ) -> DurableOperationOwnership:
        operation_id = _safe_ref("recovery-plan-session", session.instance_id)
        requested = DurableOperationOwnershipTransition(
            operation_id=operation_id,
            claim_id=claim_id,
            owner_id=owner_id,
            action=DurableOperationOwnershipAction.CLAIM,
            lease_seconds=_RECOVERY_PLAN_EXECUTION_LEASE_SECONDS,
        )

        def claim(
            current_session: Session,
            checkpoint: dict[str, object] | None,
            store_now: datetime,
        ) -> dict[str, object]:
            current_marker = _parse_execution_marker(checkpoint)
            if current_session.instance_id != session.instance_id or (
                not allow_started_state
                and _session_state_fingerprint(current_session, checkpoint)
                != item.state_fingerprint
            ):
                raise StaleRecoveryPlanError("Recovery plan item no longer matches durable state.")
            current_ownership = None if current_marker is None else current_marker["ownership"]
            same_execution = bool(
                current_marker is not None
                and current_marker["plan_id"] == request.plan.plan_id
                and current_marker["item_id"] == item.item_id
                and current_marker["execution_id"] == request.execution_id
                and current_marker["state_fingerprint"] == item.state_fingerprint
            )
            if (
                same_execution
                and current_marker is not None
                and current_marker["decision_fingerprint"] != decision_fingerprint
            ):
                raise RecoveryPlanExecutionFenced(
                    "Recovery execution is already bound to another decision."
                )
            if current_marker is not None and not (
                same_execution and current_marker["decision_fingerprint"] == decision_fingerprint
            ):
                ownership = current_ownership
                assert isinstance(ownership, DurableOperationOwnership)
                if (
                    ownership.state.value == "active"
                    and ownership.lease_expires_at is not None
                    and ownership.lease_expires_at > store_now
                ):
                    raise RecoveryPlanExecutionFenced("Another recovery plan owns this session.")
            result = transition_durable_operation_ownership(
                current_ownership
                if isinstance(current_ownership, DurableOperationOwnership)
                else None,
                requested,
                store_now=store_now,
                operation_active=True,
            )
            if (
                result.disposition
                not in {
                    DurableOperationOwnershipDisposition.ACQUIRED,
                    DurableOperationOwnershipDisposition.EXPIRED_TAKEN_OVER,
                }
                or result.ownership is None
            ):
                raise RecoveryPlanExecutionFenced("Recovery plan execution is already owned.")
            updated = copy_durable_json_object(checkpoint or {}, "checkpoint")
            updated[RECOVERY_PLAN_EXECUTION_CHECKPOINT_KEY] = {
                "version": 1,
                "plan_id": request.plan.plan_id,
                "item_id": item.item_id,
                "execution_id": request.execution_id,
                "state_fingerprint": item.state_fingerprint,
                "decision_fingerprint": decision_fingerprint,
                "ownership": result.ownership.model_dump(mode="json"),
            }
            return updated

        # The state fingerprint covers private invocation-lifecycle authority.
        # This runtime-owned read scope exposes those roots to the atomic CAS
        # callback while the generic writer still preserves them unchanged.
        with _invocation_lifecycle_authority_read_scope():
            await self._session_store.transform_checkpoint_with_store_time(session.id, claim)
        checkpoint = await self._session_store.load_checkpoint(session.id)
        marker = _parse_execution_marker(checkpoint)
        if marker is None:
            raise RuntimeError("Recovery plan claim acknowledgement is unavailable.")
        ownership = marker["ownership"]
        if not isinstance(ownership, DurableOperationOwnership) or (
            ownership.claim_id != claim_id or ownership.owner_id != owner_id
        ):
            raise RecoveryPlanExecutionFenced("Recovery plan claim was superseded.")
        return ownership

    async def _heartbeat_execution(
        self,
        *,
        private_session_id: str,
        item: RecoveryPlanItem,
        request: RecoveryExecutionRequest,
        decision_fingerprint: str,
        ownership: DurableOperationOwnership,
        stop: asyncio.Event,
        lost: asyncio.Event,
    ) -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(
                    stop.wait(),
                    timeout=_RECOVERY_PLAN_EXECUTION_HEARTBEAT_SECONDS,
                )
                return
            except TimeoutError:
                pass

            def renew(
                _session: Session,
                checkpoint: dict[str, object] | None,
                store_now: datetime,
            ) -> dict[str, object]:
                marker = _parse_execution_marker(checkpoint)
                if marker is None or not (
                    marker["plan_id"] == request.plan.plan_id
                    and marker["item_id"] == item.item_id
                    and marker["execution_id"] == request.execution_id
                    and marker["state_fingerprint"] == item.state_fingerprint
                    and marker["decision_fingerprint"] == decision_fingerprint
                ):
                    raise RecoveryPlanExecutionFenced("Recovery plan execution claim was replaced.")
                current_ownership = marker["ownership"]
                if not isinstance(current_ownership, DurableOperationOwnership):
                    raise RuntimeError("Recovery plan execution ownership is invalid.")
                result = transition_durable_operation_ownership(
                    current_ownership,
                    DurableOperationOwnershipTransition(
                        operation_id=ownership.operation_id,
                        claim_id=ownership.claim_id,
                        owner_id=ownership.owner_id,
                        generation=ownership.generation,
                        action=DurableOperationOwnershipAction.RENEW,
                        lease_seconds=_RECOVERY_PLAN_EXECUTION_LEASE_SECONDS,
                    ),
                    store_now=store_now,
                    operation_active=True,
                )
                if (
                    result.disposition is not DurableOperationOwnershipDisposition.RENEWED
                    or result.ownership is None
                ):
                    raise RecoveryPlanExecutionFenced("Recovery plan execution lease was lost.")
                updated = copy_durable_json_object(checkpoint or {}, "checkpoint")
                updated_marker = copy_durable_json_object(
                    updated[RECOVERY_PLAN_EXECUTION_CHECKPOINT_KEY],
                    "recovery_plan_execution",
                )
                updated_marker["ownership"] = result.ownership.model_dump(mode="json")
                updated[RECOVERY_PLAN_EXECUTION_CHECKPOINT_KEY] = updated_marker
                return updated

            try:
                await self._session_store.transform_checkpoint_with_store_time(
                    private_session_id,
                    renew,
                )
            except RecoveryPlanExecutionFenced:
                lost.set()
                return
            except Exception:
                # A transient or acknowledgement-ambiguous renewal failure is
                # not proof of lease loss. The next heartbeat retries, and the
                # receipt transaction still requires the exact live owner.
                continue

    async def _apply_decision(
        self,
        *,
        request: RecoveryExecutionRequest,
        private_session_id: str,
        item: RecoveryPlanItem,
        decision: RecoveryDecision,
        inactive_for_seconds: int | None,
        recoverable_task: Task | None,
    ) -> tuple[tuple[IncompleteSessionRecoveryAction, ...], tuple[str, ...]]:
        if decision.action is RecoveryPlanAction.AUTOMATIC_REPAIR:
            result = await self._recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=private_session_id,
                    # This plan's exact checkpoint claim refreshed activity
                    # only after atomically matching the planned snapshot. A
                    # second inactivity check would compare against our own
                    # lease write instead of the operator-inspected state.
                    inactive_for_seconds=None,
                    reason="operator_executed_recovery_plan",
                    metadata={"plan_item_id": item.item_id},
                )
            )
            if item.interruption_cascade is not None:
                await self._recover_interruption_cascade(
                    private_session_id,
                    inactive_for_seconds,
                )
            if recoverable_task is not None:
                current = await self._session_store.load(private_session_id)
                if current is None:
                    raise KeyError(f"Session not found: {item.session_id}")
                await self._prepare_recoverable_task_handoff(
                    task=recoverable_task,
                    session=current,
                )
            return result.actions, tuple(_safe_ref("event", event.id) for event in result.events)

        if decision.action in {
            RecoveryPlanAction.MODEL_MARK_FAILED,
            RecoveryPlanAction.MODEL_MARK_INTERRUPTED,
        }:
            active = await self._session_store.load_active_model_completion_stage(
                private_session_id
            )
            if (
                active is None
                or item.active_model_stage is None
                or (
                    _model_stage_ref(active.stage.stage_id, active.marker_digest)
                    != item.active_model_stage.stage_ref
                )
            ):
                raise StaleRecoveryPlanError("Active model stage changed after planning.")
            result = await self._recover_model_completion(
                ModelCompletionManualRecoveryRequest(
                    session_id=private_session_id,
                    stage_id=active.stage.stage_id,
                    expected_run_epoch=item.run_epoch,
                    terminal_status=(
                        SessionStatus.FAILED
                        if decision.action is RecoveryPlanAction.MODEL_MARK_FAILED
                        else SessionStatus.INTERRUPTED
                    ),
                )
            )
            settlement = await self._recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=private_session_id,
                    inactive_for_seconds=None,
                    reason="operator_executed_model_recovery_plan",
                    metadata={"plan_item_id": item.item_id},
                )
            )
            event_ids = tuple(
                _safe_ref("event", event.id)
                for event in (*result.budget_events, *settlement.events)
            )
            if recoverable_task is not None:
                current = await self._session_store.load(private_session_id)
                if current is None:
                    raise KeyError(f"Session not found: {item.session_id}")
                if current.status is SessionStatus.INTERRUPTED:
                    await self._prepare_recoverable_task_handoff(
                        task=recoverable_task,
                        session=current,
                    )
                elif current.status is SessionStatus.FAILED:
                    await self._fail_recoverable_task_after_model_disposition(
                        task=recoverable_task,
                        session=current,
                        item=item,
                        execution_id=request.execution_id,
                    )
                else:
                    raise RuntimeError(
                        "Model disposition did not reach an attached-task terminal boundary."
                    )
            return settlement.actions, event_ids

        if decision.action in {
            RecoveryPlanAction.TOOL_MARK_COMPLETED,
            RecoveryPlanAction.TOOL_MARK_FAILED,
        }:
            current_session = await self._session_store.load(private_session_id)
            if current_session is None:
                raise KeyError(f"Session not found: {item.session_id}")
            pending, pending_invalid = await self._pending_actions(
                current_session,
                await self._session_store.load_checkpoint(private_session_id),
            )
            planned_refs = {
                evidence.action_ref
                for evidence in item.pending_actions
                if evidence.kind is PendingActionKind.MANUAL_RECOVERY
            }
            matches = [
                action
                for action in pending
                if action.kind is PendingActionKind.MANUAL_RECOVERY
                and _action_ref(action) in planned_refs
            ]
            if pending_invalid or len(matches) != 1:
                raise StaleRecoveryPlanError("Pending tool effect changed after planning.")
            action = matches[0]
            if action.round_id is None or action.tool_call_id is None:
                raise RuntimeError("Pending tool recovery lacks exact round identity.")
            events: list[Event] = []
            task_worker_id: str | None = None
            task_handoff_id: str | None = None
            task_heartbeat_stop: asyncio.Event | None = None
            task_heartbeat_lost: asyncio.Event | None = None
            task_heartbeat: asyncio.Task[None] | None = None
            if recoverable_task is not None:
                preparation = await self._recovery_coordinator.interrupt_incomplete_session_for_manual_tool_recovery(
                    IncompleteSessionRecoveryRequest(
                        session_id=private_session_id,
                        inactive_for_seconds=None,
                        reason="operator_prepared_task_backed_tool_recovery",
                        metadata={"plan_item_id": item.item_id},
                    )
                )
                events.extend(preparation.events)
                interrupted = await self._session_store.load(private_session_id)
                if interrupted is None:
                    raise KeyError(f"Session not found: {item.session_id}")
                released = await self._prepare_recoverable_task_handoff(
                    task=recoverable_task,
                    session=interrupted,
                )
                claimed = await self._claim_plan_task_continuation(
                    task=released,
                    item=item,
                    execution_id=request.execution_id,
                )
                task_worker_id = claimed.worker_id
                task_handoff_id = claimed.interrupted_handoff_id
                task_heartbeat_stop = asyncio.Event()
                task_heartbeat_lost = asyncio.Event()
                task_heartbeat = asyncio.create_task(
                    self._heartbeat_plan_task_continuation(
                        task=claimed,
                        stop=task_heartbeat_stop,
                        lost=task_heartbeat_lost,
                    )
                )
            try:
                async for event in self._recover_tool_round(
                    ToolRoundRecoveryRequest(
                        session_id=private_session_id,
                        task_worker_id=task_worker_id,
                        task_handoff_id=task_handoff_id,
                        round_id=action.round_id,
                        tool_call_id=action.tool_call_id,
                        outcome=(
                            ToolApprovalRecoveryOutcome.COMPLETED
                            if decision.action is RecoveryPlanAction.TOOL_MARK_COMPLETED
                            else ToolApprovalRecoveryOutcome.FAILED
                        ),
                        message=decision.message or "Operator supplied recovery disposition.",
                        reason="operator_executed_recovery_plan",
                        metadata={"plan_item_id": item.item_id},
                    )
                ):
                    events.append(event)
            finally:
                if task_heartbeat_stop is not None:
                    task_heartbeat_stop.set()
                if task_heartbeat is not None:
                    task_heartbeat.cancel()
                    await asyncio.gather(task_heartbeat, return_exceptions=True)
            if task_heartbeat_lost is not None and task_heartbeat_lost.is_set():
                raise TaskClaimLost("Recovery-plan task continuation lease was lost.")
            return (), tuple(_safe_ref("event", event.id) for event in events)

        raise RuntimeError(f"Unsupported recovery plan action: {decision.action}")

    @staticmethod
    def _receipt_event_id(
        request: RecoveryExecutionRequest,
        item: RecoveryPlanItem,
    ) -> str:
        digest = _hash_material(
            "recovery_plan_receipt_event",
            {
                "plan_id": request.plan.plan_id,
                "execution_id": request.execution_id,
                "item_id": item.item_id,
            },
        )
        return f"evt_recovery_plan_item_{digest}"

    async def _load_receipt(
        self,
        *,
        private_session_id: str,
        receipt_event_id: str,
        request: RecoveryExecutionRequest,
        item: RecoveryPlanItem,
        decision: RecoveryDecision,
    ) -> RecoveryItemReceipt | None:
        records = await self._session_store.query_events(
            EventQuery(
                session_id=private_session_id,
                event_id=receipt_event_id,
                limit=2,
            )
        )
        if not records:
            return None
        if len(records) != 1 or records[0].event.type != EventType.RECOVERY_PLAN_ITEM_EXECUTED:
            raise RuntimeError("Recovery receipt event identity conflicts with durable evidence.")
        payload = records[0].event.payload
        if set(payload) != {"receipt"}:
            raise RuntimeError("Recovery receipt event payload is malformed.")
        receipt = RecoveryItemReceipt.model_validate(payload["receipt"])
        if receipt.action != decision.action:
            raise RecoveryPlanExecutionFenced(
                "Recovery execution is already bound to another decision."
            )
        if (
            receipt.plan_id != request.plan.plan_id
            or receipt.execution_id != request.execution_id
            or receipt.item_id != item.item_id
            or receipt.session_id != item.session_id
            or receipt.receipt_event_id != receipt_event_id
        ):
            raise RuntimeError("Recovery receipt conflicts with the requested execution.")
        return receipt

    async def _commit_receipt(
        self,
        *,
        private_session_id: str,
        private_agent_name: str,
        receipt: RecoveryItemReceipt,
        ownership: DurableOperationOwnership,
    ) -> None:
        assert receipt.receipt_event_id is not None
        event = event_with_runtime_envelope_authority(
            event_with_runtime_generated_id(
                Event(
                    id=receipt.receipt_event_id,
                    type=EventType.RECOVERY_PLAN_ITEM_EXECUTED,
                    session_id=private_session_id,
                    agent_name=private_agent_name,
                    payload={"receipt": receipt.model_dump(mode="json", warnings=False)},
                )
            ),
            "session_id",
        )
        event = self._event_writer.prepare(event)
        current = await self._session_store.load(private_session_id)
        if current is None:
            raise KeyError(f"Session not found: {receipt.session_id}")

        def settle(
            current_session: Session,
            checkpoint: dict[str, object] | None,
            store_now: datetime,
        ) -> dict[str, object]:
            marker = _parse_execution_marker(checkpoint)
            if marker is None:
                raise RecoveryPlanExecutionFenced("Recovery plan execution claim disappeared.")
            current_ownership = marker["ownership"]
            if not isinstance(current_ownership, DurableOperationOwnership):
                raise RuntimeError("Recovery plan execution ownership is invalid.")
            result = transition_durable_operation_ownership(
                current_ownership,
                DurableOperationOwnershipTransition(
                    operation_id=ownership.operation_id,
                    claim_id=ownership.claim_id,
                    owner_id=ownership.owner_id,
                    generation=ownership.generation,
                    action=DurableOperationOwnershipAction.RELEASE,
                ),
                store_now=store_now,
                operation_active=True,
            )
            if result.disposition is not DurableOperationOwnershipDisposition.RELEASED:
                raise RecoveryPlanExecutionFenced("Recovery plan execution lease was lost.")
            updated = copy_durable_json_object(checkpoint or {}, "checkpoint")
            updated.pop(RECOVERY_PLAN_EXECUTION_CHECKPOINT_KEY, None)
            return updated

        await self._session_store.publish_checkpoint_and_events_with_store_time(
            private_session_id,
            idempotency_key=f"recovery-plan-receipt:{receipt.receipt_event_id}",
            checkpoint_transform=settle,
            commit_time_guard=lambda _store_now: None,
            events=[event],
            expected_statuses={current.status},
            expected_run_epoch=current.run_epoch,
        )
        await self._event_writer.fan_out_persisted([event])


__all__ = [
    "RECOVERY_PLAN_EXECUTION_CHECKPOINT_KEY",
    "RecoveryPlanCoordinator",
]
