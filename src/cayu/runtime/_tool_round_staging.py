"""Private durable terminal staging and projection for tool-round owners.

Execution, recovered publication and the paused-round phase share this implementation.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Mapping
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from cayu._task_wait import await_shielded_task_outcome, restore_task_cancellation_requests
from cayu._validation import (
    JsonUtf8SizeCounter,
    canonical_durable_json_bytes,
    copy_durable_json_object,
    copy_durable_json_value,
    copy_json_value,
    require_clean_nonblank,
    require_nonblank,
)
from cayu.events import (
    Event,
    EventType,
    copy_event,
    event_nested_payload_authority_is_runtime_generated,
    event_payload_authority_is_runtime_generated,
    event_with_runtime_envelope_authority,
    event_with_runtime_generated_id,
    event_with_runtime_nested_payload_authority,
    event_with_runtime_payload_authority,
)
from cayu.failure_evidence import FailureEvidence
from cayu.mcp.tools import McpToolAdapter
from cayu.observability.hooks import RuntimeHookPhase, _runtime_hook_supports_phase
from cayu.runtime import _invocation_secrets as invocation_secrets
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _shared_artifact_results as shared_artifact_results
from cayu.runtime import _tool_argument_publication as tool_argument_publication
from cayu.runtime import _tool_results as tool_results
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime import _web_access_results as web_access_results
from cayu.runtime._assistant_tool_round_publication import validate_tool_exposure_terminal_event
from cayu.runtime._event_writer import prepare_runtime_event
from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.runtime._phase_timing import current_builder, timed_phase
from cayu.runtime._tool_effect_state import (
    ToolEffectReconciliationRequired,
    ToolEffectStateOwner,
    ToolEffectTerminal,
    is_command_policy_refusal_terminal,
)
from cayu.runtime.execution_profiles import (
    EXECUTION_PROFILE_FINGERPRINT_FIELD,
    ExecutionProfileIdentity,
    event_with_execution_profile_fingerprint_authority,
)
from cayu.runtime.execution_units import ToolRoundIdentity, copy_tool_round_identity
from cayu.sessions.base import Session, SessionStore, runtime_publication_checkpoint_mutation
from cayu.tools.base import ToolResult, _bound_policy_denial_result, _bound_policy_denial_text
from cayu.tools.catalogue import ToolExecutionContract
from cayu.tools.exposure import ResolvedToolExposureAuthority, copy_resolved_tool_exposure_authority
from cayu.tools.result_projection import _TOOL_RESULT_PROJECTION_PROVENANCE_PATH
from cayu.tools.terminal_publication import (
    TOOL_TERMINAL_PUBLICATION_SLICE_BYTES,
    ToolTerminalPublicationGovernor,
)
from cayu.vaults.redaction import SecretRedactor


class ToolTerminalPublisher(Protocol):
    """Execute existing result hooks using the round owner's durable callbacks."""

    def __call__(
        self,
        *,
        event: Event,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        tool_call: runtime_records.ToolCallRequest,
        result: ToolResult,
        task_id: str | None,
        execution_profile: ExecutionProfileIdentity | None,
        invocation_context: InvocationContext | None,
        redactor: SecretRedactor | None = None,
        output_redactor: SecretRedactor | None = None,
        argument_projection: tool_argument_publication.ToolArgumentProjection,
        hook_argument_projection: tool_argument_publication.ToolArgumentProjection,
        allow_modification: bool,
        publish_before_hooks: bool,
        deferred_terminal_projection_recorder: Callable[[Event], Awaitable[Event]] | None,
        deferred_terminal_finalizer: Callable[[Event], Awaitable[Event]] | None,
        terminal_event_emitter: Callable[[Event], Awaitable[Event]] | None,
        hooks_already_completed: bool,
    ) -> AsyncGenerator[tuple[Event, runtime_records.ToolCallOutcome | None], None]: ...


CheckpointTransform = Callable[
    [Session, dict[str, Any] | None],
    dict[str, Any],
]

_POLICY_DENIAL_CONTROL_PAYLOAD_FIELDS = frozenset(
    {
        "approval_id",
        "blocked_by",
        "decision",
        "denied_by",
        "idempotency_key",
        "input_id",
        "model_attempt_id",
        "model_step_id",
        "tool_call_id",
        "tool_name",
        "tool_round_id",
    }
)

_POLICY_DENIAL_CONTROL_RESULT_FIELDS = frozenset({"decision", "error"})


def _event_with_tool_round_authority(
    event: Event,
    identity: ToolRoundIdentity,
    *additional_fields: str,
) -> Event:
    """Attest only runtime-owned linkage carried by a typed tool-round identity."""

    identity = copy_tool_round_identity(identity)
    fields = [
        field_name
        for field_name, value in identity.payload().items()
        if event.payload.get(field_name) == value
    ]
    for field_name in additional_fields:
        if field_name in event.payload:
            fields.append(field_name)
    event = event_with_runtime_envelope_authority(event, "session_id")
    return event_with_runtime_payload_authority(event, *fields) if fields else event


_TOOL_EFFECT_COMPLETED_AT_FIELD = "tool_effect_completed_at"

_TOOL_TERMINAL_STAGED_AT_FIELD = "tool_terminal_staged_at"

_TOOL_TERMINAL_PUBLICATION_STARTED_AT_FIELD = "tool_terminal_publication_started_at"

_TOOL_TERMINAL_TIMING_FIELDS = (
    _TOOL_EFFECT_COMPLETED_AT_FIELD,
    _TOOL_TERMINAL_STAGED_AT_FIELD,
    _TOOL_TERMINAL_PUBLICATION_STARTED_AT_FIELD,
)

_TOOL_TERMINAL_RUNTIME_PAYLOAD_HEADROOM_BYTES = 64 * 1024


class _ToolRoundPublicationCoordinator:
    """Serialize actual secret discovery with private terminal staging."""

    def __init__(
        self,
        *,
        session_id: str,
        session_instance_id: str,
        run_epoch: int,
        tool_round_identity: ToolRoundIdentity,
        session_store: SessionStore,
        redactor: SecretRedactor,
        execution_profile: ExecutionProfileIdentity | None,
        tool_exposure: ResolvedToolExposureAuthority | None = None,
        publication_governor: ToolTerminalPublicationGovernor | None = None,
        clock: Callable[[], datetime] | None = None,
        terminal_payload_limits: Mapping[str, int | None] | None = None,
    ) -> None:
        self._session_id = require_clean_nonblank(session_id, "session_id")
        self._session_instance_id = require_clean_nonblank(
            session_instance_id, "session_instance_id"
        )
        if type(run_epoch) is not int or run_epoch < 0:
            raise ValueError("Terminal staging requires an exact run epoch.")
        self._run_epoch = run_epoch
        self._tool_round_identity = copy_tool_round_identity(tool_round_identity)
        self._session_store = session_store
        self._redactor = redactor
        if (
            execution_profile is not None
            and type(execution_profile) is not ExecutionProfileIdentity
        ):
            raise TypeError("execution_profile must be an ExecutionProfileIdentity or None.")
        self._execution_profile_fingerprint = (
            None if execution_profile is None else execution_profile.fingerprint
        )
        self._tool_exposure = (
            None if tool_exposure is None else copy_resolved_tool_exposure_authority(tool_exposure)
        )
        self._unsafe_tool_call_ids: set[str] = set()
        self._lock = asyncio.Lock()
        self._publication_governor = publication_governor or ToolTerminalPublicationGovernor()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._terminal_payload_limits = (
            {} if terminal_payload_limits is None else dict(terminal_payload_limits)
        )
        if any(
            type(tool_call_id) is not str
            or not tool_call_id
            or (limit is not None and (type(limit) is not int or limit <= 0))
            for tool_call_id, limit in self._terminal_payload_limits.items()
        ):
            raise ValueError("Terminal payload limits must map call IDs to positive bytes or None.")
        limits = tuple(self._terminal_payload_limits.values())
        self._capacity_maximum_bytes = (
            sum(
                limit + _TOOL_TERMINAL_RUNTIME_PAYLOAD_HEADROOM_BYTES
                for limit in limits
                if limit is not None
            )
            if limits and all(limit is not None for limit in limits)
            else None
        )
        self._capacity_reserved = False
        self._capacity_sealed = False
        # Register the write attempt before the atomic checkpoint transform.
        # If its acknowledgement is lost, retain the lease until a retry or
        # recovery owner reconciles and publishes the durable stage.
        self._stage_attempted_event_ids: set[str] = set()
        self._staged_event_ids: set[str] = set()
        self._published_event_ids: set[str] = set()

    @property
    def redactor(self) -> SecretRedactor:
        return self._redactor

    @property
    def tool_round_identity(self) -> ToolRoundIdentity:
        return copy_tool_round_identity(self._tool_round_identity)

    @property
    def argument_scope_finalized(self) -> bool:
        """Return whether every sealed call contributed complete secret evidence."""

        return not self._unsafe_tool_call_ids

    async def reserve_capacity(self) -> None:
        """Acquire this complete round's byte lease before tool dispatch."""

        session = await self._session_store.load(self._session_id)
        if (
            session is None
            or session.instance_id != self._session_instance_id
            or session.run_epoch != self._run_epoch
        ):
            raise RuntimeError("Tool-round reservation lost its session authority.")
        if session.invocation is None:
            # Missing provenance cannot join another round's capacity domain.
            await self._publication_governor.reserve_round(
                session_id=self._session_id,
                tool_round_id=self._tool_round_identity.tool_round_id,
                maximum_bytes=self._capacity_maximum_bytes,
            )
        else:
            await self._publication_governor._reserve_invocation_round(
                session_id=self._session_id,
                tool_round_id=self._tool_round_identity.tool_round_id,
                maximum_bytes=self._capacity_maximum_bytes,
                invocation=session.invocation,
            )
        self._capacity_reserved = True

    async def restore_staged_capacity(
        self,
        staged_terminals: Iterable[tool_round_recovery.StagedToolCallTerminal],
    ) -> None:
        """Attach durable stages to a recovered owner of the round lease."""

        if not self._capacity_reserved:
            raise RuntimeError("Staged terminal recovery requires a round reservation.")
        for staged in staged_terminals:
            if staged.event.session_id != self._session_id:
                raise RuntimeError("Staged terminal capacity belongs to a different session.")
            payload_bytes = staged.payload_bytes
            if payload_bytes is None:
                payload_bytes = await self._publication_governor.run_cpu(
                    _terminal_publication_work_estimate(staged.event),
                    lambda staged=staged: _durable_payload_utf8_size(staged.event.payload),
                )
            self.validate_staged_payload(staged.tool_call_id, payload_bytes)
            effect_completed_at = _normalized_event_timestamp(
                staged.effect_completed_at or staged.event.timestamp
            )
            self._publication_governor.reconcile_stage(
                session_id=self._session_id,
                event_id=staged.event.id,
                payload_bytes=payload_bytes,
                effect_completed_at=effect_completed_at,
                tool_round_id=self._tool_round_identity.tool_round_id,
            )
            self._stage_attempted_event_ids.discard(staged.event.id)
            self._staged_event_ids.add(staged.event.id)

    def terminal_payload_limit(self, tool_call_id: str) -> int | None:
        return self._terminal_payload_limits.get(tool_call_id)

    def validate_staged_payload(self, tool_call_id: str, payload_bytes: int) -> None:
        declared = self.terminal_payload_limit(tool_call_id)
        if (
            declared is not None
            and payload_bytes > declared + _TOOL_TERMINAL_RUNTIME_PAYLOAD_HEADROOM_BYTES
        ):
            raise RuntimeError(
                "Bounded tool terminal exceeded its reserved runtime payload envelope."
            )

    def seal_capacity(self) -> None:
        """Declare that no additional terminal can enter this round."""

        self._capacity_sealed = True
        self._release_capacity_if_drained()

    def terminal_published(self, event_id: str) -> None:
        self._published_event_ids.add(event_id)
        self._release_capacity_if_drained()

    def _release_capacity_if_drained(self) -> None:
        if (
            not self._capacity_reserved
            or not self._capacity_sealed
            or self._stage_attempted_event_ids
            or not self._staged_event_ids.issubset(self._published_event_ids)
        ):
            return
        self._publication_governor.release_round(
            session_id=self._session_id,
            tool_round_id=self._tool_round_identity.tool_round_id,
        )
        self._capacity_reserved = False

    def restore_staged_event_authority(self, event: Event) -> Event:
        restored = restore_staged_terminal_authority(
            event,
            session_id=self._session_id,
            tool_round_identity=self._tool_round_identity,
            tool_exposure=self._tool_exposure,
        )
        restored = web_access_results.restore_persisted_web_access_result_authority(restored)
        restored = shared_artifact_results.restore_persisted_shared_artifact_result_authority(
            restored
        )
        observed_fingerprint = restored.payload.get(EXECUTION_PROFILE_FINGERPRINT_FIELD)
        if (
            observed_fingerprint is not None
            and observed_fingerprint != self._execution_profile_fingerprint
        ):
            raise RuntimeError("Staged terminal conflicts with its execution profile owner.")
        return event_with_execution_profile_fingerprint_authority(
            restored,
            self._execution_profile_fingerprint,
        )

    @timed_phase("publication")
    async def start_publication(
        self,
        staged: tool_round_recovery.StagedToolCallTerminal,
    ) -> tool_round_recovery.StagedToolCallTerminal:
        """Durably pin public timing before the first append attempt."""

        builder = current_builder()
        if builder is not None:
            builder.mark_publication_started(staged.tool_call_id)

        effect_completed_at = _normalized_event_timestamp(
            staged.effect_completed_at or staged.event.timestamp
        )
        staged_at = staged.staged_at or effect_completed_at
        payload_bytes = staged.payload_bytes
        if payload_bytes is None:
            payload_bytes = await self._publication_governor.run_cpu(
                _terminal_publication_work_estimate(staged.event),
                lambda: _durable_payload_utf8_size(staged.event.payload),
            )
        self._publication_governor.reconcile_stage(
            session_id=self._session_id,
            event_id=staged.event.id,
            payload_bytes=payload_bytes,
            effect_completed_at=effect_completed_at,
            tool_round_id=(
                self._tool_round_identity.tool_round_id if self._capacity_reserved else None
            ),
        )
        self._staged_event_ids.add(staged.event.id)
        if staged.publication_started_at is not None:
            return staged
        publication_started_at = max(self._clock(), staged_at)
        payload = dict(staged.event.payload)
        payload.update(
            {
                _TOOL_EFFECT_COMPLETED_AT_FIELD: effect_completed_at.isoformat(),
                _TOOL_TERMINAL_STAGED_AT_FIELD: staged_at.isoformat(),
                _TOOL_TERMINAL_PUBLICATION_STARTED_AT_FIELD: (publication_started_at.isoformat()),
            }
        )
        public_event = staged.event.model_copy(
            update={"timestamp": publication_started_at, "payload": payload}
        )
        public_payload_bytes = await self._publication_governor.run_cpu(
            _terminal_publication_work_estimate(public_event),
            lambda: _durable_payload_utf8_size(public_event.payload),
        )
        self.validate_staged_payload(staged.tool_call_id, public_payload_bytes)
        await self._session_store.transform_checkpoint(
            self._session_id,
            tool_round_recovery.started_staged_terminal_publication_transform(
                tool_round_identity=self._tool_round_identity,
                tool_call_id=staged.tool_call_id,
                event=public_event,
                payload_bytes=public_payload_bytes,
                effect_completed_at=effect_completed_at,
                staged_at=staged_at,
                publication_started_at=publication_started_at,
            ),
        )
        checkpoint = await self._session_store.load_checkpoint(self._session_id)
        stored = next(
            (
                item
                for item in tool_round_recovery.checkpoint_staged_terminals(
                    checkpoint,
                    tool_round_identity=self._tool_round_identity,
                )
                if item.tool_call_id == staged.tool_call_id
            ),
            None,
        )
        if (
            stored is None
            or stored.publication_started_at != publication_started_at
            or stored.payload_bytes != public_payload_bytes
        ):
            raise RuntimeError("Staged terminal publication timing was not acknowledged.")
        self._record_durable_stage(stored)
        return stored

    def restore_started_publication_authority(
        self,
        staged: tool_round_recovery.StagedToolCallTerminal,
    ) -> Event:
        """Restore typed timing authority from the durable staged record."""

        restored = self.restore_staged_event_authority(staged.event)
        if (
            staged.effect_completed_at is None
            or staged.staged_at is None
            or staged.publication_started_at is None
        ):
            return restored
        expected = {
            _TOOL_EFFECT_COMPLETED_AT_FIELD: staged.effect_completed_at.isoformat(),
            _TOOL_TERMINAL_STAGED_AT_FIELD: staged.staged_at.isoformat(),
            _TOOL_TERMINAL_PUBLICATION_STARTED_AT_FIELD: (
                staged.publication_started_at.isoformat()
            ),
        }
        if restored.timestamp != staged.publication_started_at or any(
            restored.payload.get(field_name) != value for field_name, value in expected.items()
        ):
            raise RuntimeError("Staged terminal event conflicts with its publication timing.")
        return event_with_runtime_payload_authority(restored, *expected)

    @timed_phase("staging")
    async def register_redactor(
        self,
        *,
        tool_call_id: str,
        redactor: SecretRedactor,
    ) -> None:
        """Persist one real invocation redactor before its secret is returned."""

        async with self._lock:
            self._redactor = self._redactor.merged_with(redactor)
            await self._session_store.transform_checkpoint(
                self._session_id,
                self._checkpoint_transform(
                    tool_call_id=tool_call_id,
                    cover_call=False,
                    staged_terminal=None,
                ),
            )

    @timed_phase("staging")
    async def seal_call(
        self,
        *,
        tool_call_id: str,
        snapshot: invocation_secrets.InvocationPublicationSnapshot,
    ) -> None:
        """Durably cover a call whose terminal outcome will be synthesized."""

        async with self._lock:
            self._redactor = self._redactor.merged_with(snapshot.redactor)
            if snapshot.secret_scope_incomplete:
                self._unsafe_tool_call_ids.add(tool_call_id)
            await self._session_store.transform_checkpoint(
                self._session_id,
                self._checkpoint_transform(
                    tool_call_id=tool_call_id,
                    cover_call=True,
                    unsafe_scope=snapshot.secret_scope_incomplete,
                    staged_terminal=None,
                ),
            )

    @timed_phase("staging")
    async def stage_terminal(
        self,
        *,
        tool_call_id: str,
        event: Event,
        snapshot: invocation_secrets.InvocationPublicationSnapshot,
        hooks_state: Literal["pending", "finalized", "observational", "completed"],
    ) -> Event:
        """Persist a stable terminal event before redelivering caller cancellation."""

        stage_task = asyncio.create_task(
            self._stage_terminal_owned(
                tool_call_id=tool_call_id,
                event=event,
                snapshot=snapshot,
                hooks_state=hooks_state,
            )
        )
        outcome = await await_shielded_task_outcome(stage_task)
        if outcome.error is not None:
            raise outcome.error
        if outcome.result is None:  # pragma: no cover - owned task invariant
            raise RuntimeError("Staged terminal publication returned no durable event.")
        builder = current_builder()
        if builder is not None:
            builder.mark_staged(tool_call_id, outcome.result)
        if outcome.cancellation is not None:
            restore_task_cancellation_requests(
                outcome.cancellation_requests_consumed,
                cancellation=outcome.cancellation,
            )
            raise outcome.cancellation
        return outcome.result

    async def _stage_terminal_owned(
        self,
        *,
        tool_call_id: str,
        event: Event,
        snapshot: invocation_secrets.InvocationPublicationSnapshot,
        hooks_state: Literal["pending", "finalized", "observational", "completed"],
    ) -> Event:
        """Complete one cancellation-resistant durable staging operation."""

        if event.session_id != self._session_id:
            raise ValueError("Staged terminal event belongs to a different session.")
        async with self._lock:
            previous_redactor = self._redactor
            self._redactor = self._redactor.merged_with(snapshot.redactor)
            if snapshot.secret_scope_incomplete:
                self._unsafe_tool_call_ids.add(tool_call_id)
            estimated_bytes = _terminal_publication_work_estimate(event)
            projected, payload_bytes = await self._publication_governor.run_cpu(
                estimated_bytes,
                lambda: _project_and_size_staged_terminal_event(
                    event,
                    redactor=self._redactor,
                ),
            )
            self.validate_staged_payload(tool_call_id, payload_bytes)
            projected_payload = dict(projected.payload)
            for field_name in _TOOL_TERMINAL_TIMING_FIELDS:
                projected_payload.pop(field_name, None)
            projected = projected.model_copy(update={"payload": projected_payload})
            effect_completed_at = _normalized_event_timestamp(event.timestamp)
            staged_at = max(self._clock(), effect_completed_at)
            staged = tool_round_recovery.StagedToolCallTerminal(
                tool_call_id=tool_call_id,
                event=projected,
                hooks_state=hooks_state,
                payload_bytes=payload_bytes,
                effect_completed_at=effect_completed_at,
                staged_at=staged_at,
            )
            self._stage_attempted_event_ids.add(projected.id)
            try:
                transform = self._checkpoint_transform(
                    tool_call_id=tool_call_id,
                    cover_call=True,
                    unsafe_scope=snapshot.secret_scope_incomplete,
                    staged_terminal=staged,
                    reproject_existing=not previous_redactor.has_same_registry(self._redactor),
                )
                session = await self._session_store.load(self._session_id)
                if session is None:
                    raise RuntimeError("Terminal staging lost its session.")
                if (
                    session.instance_id != self._session_instance_id
                    or session.run_epoch != self._run_epoch
                ):
                    raise RuntimeError("Terminal staging lost its original session authority.")
                effect_owner = ToolEffectStateOwner(self._session_store)
                effect_record = await effect_owner.resolve_call(
                    session,
                    tool_round_id=self._tool_round_identity.tool_round_id,
                    tool_call_id=tool_call_id,
                )
                result_payload = projected.payload.get("result")
                result_evidence = (
                    result_payload.get("structured") if type(result_payload) is dict else None
                )
                unverified_output = (
                    dict(result_evidence)
                    if type(result_evidence) is dict
                    and "portable_result_evidence" in result_evidence
                    and result_evidence.get("durable_value_error_code")
                    in {"json_value_too_large", "too_many_json_nodes", "nesting_too_deep"}
                    else None
                )
                if (
                    effect_record is not None
                    and (
                        projected.payload.get("outcome_unknown") is True
                        or projected.payload.get("interrupted") is True
                    )
                    and await effect_owner.preserve_unresolved(
                        session,
                        tool_round_id=self._tool_round_identity.tool_round_id,
                        tool_call_ids=(tool_call_id,),
                        unverified_output=unverified_output,
                        failure_evidence=(
                            FailureEvidence.model_validate(projected.payload["failure_evidence"])
                            if "failure_evidence" in projected.payload
                            else FailureEvidence(
                                classification=(
                                    "timeout"
                                    if projected.payload.get("terminal_outcome")
                                    == "tool_execution_timeout"
                                    else "interruption"
                                    if projected.payload.get("interrupted") is True
                                    else "failure"
                                ),
                            )
                        ).model_copy(
                            update={
                                "session_id": session.id,
                                "run_epoch": session.run_epoch,
                                "terminal_event_id": None,
                            }
                        ),
                    )
                ):
                    self._stage_attempted_event_ids.discard(projected.id)
                    raise ToolEffectReconciliationRequired()
                if (
                    effect_record is not None
                    and (
                        projected.type
                        in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
                        or is_command_policy_refusal_terminal(projected)
                    )
                    and projected.payload.get("outcome_unknown") is not True
                    and projected.payload.get("interrupted") is not True
                ):
                    source_checkpoint = await self._session_store.load_checkpoint(self._session_id)
                    mutation = runtime_publication_checkpoint_mutation(
                        source_checkpoint,
                        transform(session, source_checkpoint),
                    )
                    await effect_owner.transition(
                        effect_record,
                        state="completed"
                        if projected.type is EventType.TOOL_CALL_COMPLETED
                        else "failed",
                        run_epoch=self._run_epoch,
                        terminal=ToolEffectTerminal(
                            event_id=projected.id,
                            result_digest=hashlib.sha256(
                                canonical_durable_json_bytes(
                                    projected.payload["result"],
                                    "effect_terminal_result",
                                )
                            ).hexdigest(),
                        ),
                        mutation=mutation,
                    )
                else:
                    await self._session_store.transform_checkpoint(self._session_id, transform)
            except BaseException:
                # A transform error may be either a definite rejection or a
                # lost acknowledgement after commit. Reconcile from the
                # authoritative checkpoint before the outer owner seals the
                # lease: proven absence is abortable, while uncertainty or a
                # durable stage keeps its pre-effect reservation fenced.
                with suppress(BaseException):
                    checkpoint = await self._session_store.load_checkpoint(self._session_id)
                    stored_stages = tool_round_recovery.checkpoint_staged_terminals(
                        checkpoint,
                        tool_round_identity=self._tool_round_identity,
                    )
                    stored = next(
                        (item for item in stored_stages if item.tool_call_id == tool_call_id),
                        None,
                    )
                    if stored is None:
                        self._stage_attempted_event_ids.discard(projected.id)
                    elif stored.event.id == projected.id:
                        self._record_durable_stage(stored)
                raise
            checkpoint = await self._session_store.load_checkpoint(self._session_id)
            stored_stages = tool_round_recovery.checkpoint_staged_terminals(
                checkpoint,
                tool_round_identity=self._tool_round_identity,
            )
            stored = next(
                (item for item in stored_stages if item.tool_call_id == tool_call_id),
                None,
            )
            if stored is None or stored.event.id != event.id:
                raise RuntimeError("Staged terminal acknowledgement conflicts with its event.")
            self._record_durable_stage(stored)
            return self.restore_staged_event_authority(stored.event)

    def _record_durable_stage(
        self,
        stored: tool_round_recovery.StagedToolCallTerminal,
    ) -> None:
        if (
            stored.payload_bytes is None
            or stored.effect_completed_at is None
            or stored.staged_at is None
        ):
            raise RuntimeError("Staged terminal acknowledgement lost size or timing evidence.")
        self._publication_governor.reconcile_stage(
            session_id=self._session_id,
            event_id=stored.event.id,
            payload_bytes=stored.payload_bytes,
            effect_completed_at=stored.effect_completed_at,
            tool_round_id=(
                self._tool_round_identity.tool_round_id if self._capacity_reserved else None
            ),
        )
        self._stage_attempted_event_ids.discard(stored.event.id)
        self._staged_event_ids.add(stored.event.id)

    @timed_phase("staging")
    async def record_projected_terminal(self, event: Event) -> Event:
        """Persist a public projection while retaining its current hook state."""

        return await self._persist_projected_terminal(event, hooks_completed=False)

    @timed_phase("staging")
    async def record_workspace_capture(self, event: Event) -> Event:
        """Persist final workspace-capture controls on an owned terminal stage."""

        return await self._persist_projected_terminal(event, hooks_completed=False)

    @timed_phase("staging")
    async def complete_terminal_hooks(self, event: Event) -> Event:
        """Persist the final hook projection and mark its hooks complete."""

        return await self._persist_projected_terminal(event, hooks_completed=True)

    async def _persist_projected_terminal(
        self,
        event: Event,
        *,
        hooks_completed: bool,
    ) -> Event:
        if event.session_id != self._session_id:
            raise ValueError("Projected terminal belongs to a different session.")
        async with self._lock:
            # Hook execution and tool-result projection already applied the
            # finalized round redactor.  Preparing that event again validates
            # the boundary while preserving the runtime-owned projection
            # authority attached to externalized artifact references.
            projected, payload_bytes = await self._publication_governor.run_cpu(
                _terminal_publication_work_estimate(event),
                lambda: _prepare_and_size_projected_terminal_event(
                    event,
                    redactor=self._redactor,
                ),
            )
            tool_call_id = projected.payload.get("tool_call_id")
            if type(tool_call_id) is not str:
                raise ValueError("Projected terminal lost its tool-call identity.")
            self.validate_staged_payload(tool_call_id, payload_bytes)
            await self._session_store.transform_checkpoint(
                self._session_id,
                (
                    tool_round_recovery.completed_staged_terminal_transform
                    if hooks_completed
                    else tool_round_recovery.projected_staged_terminal_transform
                )(
                    tool_round_identity=self._tool_round_identity,
                    event=projected,
                    payload_bytes=payload_bytes,
                ),
            )
            checkpoint = await self._session_store.load_checkpoint(self._session_id)
            stored_stages = tool_round_recovery.checkpoint_staged_terminals(
                checkpoint,
                tool_round_identity=self._tool_round_identity,
            )
            stored = next(
                (item for item in stored_stages if item.tool_call_id == tool_call_id),
                None,
            )
            if (
                stored is None
                or stored.event.id != projected.id
                or stored.payload_bytes != payload_bytes
                or (hooks_completed and stored.hooks_state != "completed")
                or (not hooks_completed and stored.event != projected)
            ):
                raise RuntimeError("Projected terminal acknowledgement conflicts with its stage.")
            self._record_durable_stage(stored)
            return self.restore_started_publication_authority(stored)

    def _checkpoint_transform(
        self,
        *,
        tool_call_id: str,
        cover_call: bool,
        staged_terminal: tool_round_recovery.StagedToolCallTerminal | None,
        unsafe_scope: bool = False,
        reproject_existing: bool = True,
    ) -> CheckpointTransform:
        identity = copy_tool_round_identity(self._tool_round_identity)
        redactor = self._redactor

        def transform(
            _session: Session,
            checkpoint: dict[str, Any] | None,
        ) -> dict[str, Any]:
            updated = (
                tool_round_recovery.checkpoint_with_assistant_publication_snapshot(
                    checkpoint,
                    tool_round_identity=identity,
                    tool_call_id=tool_call_id,
                    redactor=redactor,
                    unsafe_output=unsafe_scope,
                )
                if cover_call
                else tool_round_recovery.checkpoint_with_assistant_publication_redactor(
                    checkpoint,
                    tool_round_identity=identity,
                    tool_call_id=tool_call_id,
                    redactor=redactor,
                )
            )
            # The projection has changed the checkpoint. Admit that result once,
            # then share its owner between reading and replacing staged terminals.
            updated = copy_durable_json_object(updated, "checkpoint")
            owner_key, owner = tool_round_recovery._staged_terminal_owner_from_owned_checkpoint(
                updated, tool_round_identity=identity
            )
            existing_stages = owner.staged_terminals
            projected = (
                [
                    item.model_copy(
                        update={
                            "event": _project_staged_terminal_event(
                                item.event,
                                redactor=redactor,
                                trust_persisted_tool_result_authority=True,
                            )
                        },
                        deep=True,
                    )
                    for item in existing_stages
                ]
                if reproject_existing
                else existing_stages
            )
            if staged_terminal is not None:
                existing = next(
                    (
                        item
                        for item in projected
                        if item.tool_call_id == staged_terminal.tool_call_id
                    ),
                    None,
                )
                if existing is None:
                    projected.append(staged_terminal)
                elif existing.event.id != staged_terminal.event.id:
                    raise RuntimeError(
                        "Tool call already has conflicting staged terminal evidence."
                    )
                else:
                    projected = [
                        staged_terminal
                        if item.tool_call_id == staged_terminal.tool_call_id
                        else item
                        for item in projected
                    ]
            return tool_round_recovery._replace_owned_staged_terminals(
                updated, owner_key, owner, projected
            )

        return transform


def _prepare_tool_result_event(
    *,
    event: Event,
    result: ToolResult,
    redactor: SecretRedactor,
    runtime_tool: object | None = None,
    restore_terminal_result_controls: bool = True,
) -> tuple[Event, ToolResult]:
    argument_state = event.payload.get(tool_argument_publication.ARGUMENTS_STATE_FIELD)
    if argument_state is not None and (
        type(argument_state) is not str
        or argument_state not in tool_argument_publication.TERMINAL_ARGUMENT_STATES
    ):
        raise ValueError("Terminal tool event has an invalid argument publication state.")
    argument_projection: tool_argument_publication.ToolArgumentProjection | None = None
    effective_arguments: dict[str, Any] | None = None
    if argument_state is not None:
        argument_projection = tool_argument_publication.terminal_argument_projection(
            event.payload,
            legacy_arguments={},
        )
        raw_effective_arguments = event.payload.get("effective_arguments")
        if raw_effective_arguments is not None:
            if argument_projection.state == "unavailable":
                raw_effective_arguments = None
            elif type(raw_effective_arguments) is not dict:
                raise TypeError("Terminal effective_arguments must be an object.")
            else:
                projected_effective_arguments = redactor.redact_json(raw_effective_arguments)
                if type(projected_effective_arguments) is not dict:
                    raise AssertionError("Effective argument projection returned a non-object.")
                effective_arguments = projected_effective_arguments
        payload_without_argument_projection = dict(event.payload)
        payload_without_argument_projection.pop(tool_argument_publication.ARGUMENTS_FIELD, None)
        payload_without_argument_projection.pop(
            tool_argument_publication.ARGUMENTS_STATE_FIELD,
            None,
        )
        payload_without_argument_projection.pop("effective_arguments", None)
        event = event.model_copy(update={"payload": payload_without_argument_projection})
    result = ToolResult(
        content=result.content,
        structured=tool_results.restore_runtime_tool_result_control_authority(
            result.structured,
            event.payload,
            include_terminal_controls=restore_terminal_result_controls,
        ),
        artifacts=tool_results.strip_runtime_tool_result_projection_authority(result.artifacts),
        is_error=result.is_error,
    )
    event = web_access_results.attest_runtime_web_access_result(
        event,
        result,
        tool=runtime_tool,
    )
    event = shared_artifact_results.attest_runtime_shared_artifact_result(
        event,
        result,
        tool=runtime_tool,
    )
    event, result = _validate_and_synchronize_tool_result_event(
        event=event,
        result=result,
    )
    if _is_policy_denial_event(event):
        event, result = _redact_policy_denial_event(
            event=event,
            result=result,
            redactor=redactor,
        )
    else:
        event, result = tool_results.redact_tool_result_event(
            event=event,
            result=result,
            redactor=redactor,
            include_terminal_controls=restore_terminal_result_controls,
        )
    if argument_projection is not None:
        payload = dict(event.payload)
        payload.update(argument_projection.payload_fields())
        if effective_arguments is not None:
            payload["effective_arguments"] = effective_arguments
        event = event.model_copy(update={"payload": payload})
    event, result = _bound_policy_denial_event(event=event, result=result)
    return _validate_and_synchronize_tool_result_event(event=event, result=result)


def _validate_and_synchronize_tool_result_event(
    *,
    event: Event,
    result: ToolResult,
) -> tuple[Event, ToolResult]:
    validated_result = tool_results.normalize_tool_result(tool_results.validate_tool_result(result))
    payload = copy_durable_json_object(event.payload, "tool_result_event.payload")
    payload["result"] = copy_durable_json_value(
        validated_result.model_dump(mode="python"),
        "tool_result_event.result",
    )
    synchronized = copy_event(event.model_copy(update={"payload": payload}))
    return synchronized, validated_result


def _redact_tool_result_for_event(
    *,
    event: Event,
    result: ToolResult,
    redactor: SecretRedactor,
) -> ToolResult:
    if _is_policy_denial_event(event):
        return _redact_policy_denial_result(result, redactor)
    _, redacted_result = tool_results.redact_tool_result_event(
        event=event,
        result=result,
        redactor=redactor,
    )
    return redacted_result


def _is_policy_denial_event(event: Event) -> bool:
    return event.type == EventType.TOOL_CALL_BLOCKED and "denied_by" in event.payload


def _redact_policy_denial_event(
    *,
    event: Event,
    result: ToolResult,
    redactor: SecretRedactor,
) -> tuple[Event, ToolResult]:
    redacted_result = _redact_tool_result_for_event(
        event=event,
        result=result,
        redactor=redactor,
    )
    if not redactor.has_values:
        return event, redacted_result
    timing_attribution = tool_results.runtime_terminal_timing_attribution(event)
    payload: dict[str, Any] = {}
    for key, value in event.payload.items():
        if key == "result":
            continue
        if (
            key == EXECUTION_PROFILE_FINGERPRINT_FIELD
            and type(value) is str
            and event_payload_authority_is_runtime_generated(
                event,
                field_name=key,
                value=value,
            )
        ):
            payload[key] = value
        elif key in timing_attribution:
            payload[key] = timing_attribution[key]
        elif key in _POLICY_DENIAL_CONTROL_PAYLOAD_FIELDS:
            payload[key] = copy_json_value(value, key)
        else:
            payload[key] = redactor.redact_json(value)
    payload["result"] = redacted_result.model_dump()
    return event.model_copy(update={"payload": payload}), redacted_result


def _redact_policy_denial_result(
    result: ToolResult,
    redactor: SecretRedactor,
) -> ToolResult:
    if type(result) is not ToolResult:
        raise TypeError("Policy denial results must be ToolResult instances.")
    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    if not redactor.has_values:
        return result
    structured = result.structured
    if structured is not None:
        structured = {
            key: (
                copy_json_value(value, key)
                if key in _POLICY_DENIAL_CONTROL_RESULT_FIELDS
                else redactor.redact_json(value)
            )
            for key, value in structured.items()
        }
    return ToolResult(
        content=redactor.redact_text(result.content),
        structured=structured,
        artifacts=redactor.redact_json(result.artifacts),
        is_error=result.is_error,
    )


def _bound_policy_denial_event(*, event: Event, result: ToolResult) -> tuple[Event, ToolResult]:
    if event.type != EventType.TOOL_CALL_BLOCKED or "denied_by" not in event.payload:
        return event, result
    bounded_result = _bound_policy_denial_result(result)
    payload = dict(event.payload)
    reason = payload.get("reason")
    if type(reason) is not str:
        raise ValueError("`reason` must be a string.")
    payload["reason"] = _bound_policy_denial_text(require_nonblank(reason, "reason"))
    payload["result"] = bounded_result.model_dump()
    return event.model_copy(update={"payload": payload}), bounded_result


def _project_staged_terminal_event(
    event: Event,
    *,
    redactor: SecretRedactor,
    trust_persisted_tool_result_authority: bool = False,
) -> Event:
    """Progressively sanitize one private terminal without changing its identity."""

    controls, _references = tool_results.runtime_tool_event_boundary_controls(
        event.payload,
        include_terminal_controls=event.type
        in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED},
    )
    if trust_persisted_tool_result_authority:
        event = web_access_results.restore_persisted_web_access_result_authority(event)
        event = shared_artifact_results.restore_persisted_shared_artifact_result_authority(event)
        if "tool_result_projection" in controls:
            event = event_with_runtime_nested_payload_authority(
                event,
                _TOOL_RESULT_PROJECTION_PROVENANCE_PATH,
            )
    if type(event.payload.get("result")) is not dict:
        raise ValueError("Staged terminal event requires a tool result object.")
    if "tool_result_projection" in controls and (
        event_nested_payload_authority_is_runtime_generated(
            event,
            path=_TOOL_RESULT_PROJECTION_PROVENANCE_PATH,
            value=controls["tool_result_projection"]["policy_id"],
        )
    ):
        # The staging entrance already prepared the event once. Re-enter the
        # generic event boundary with the cumulative redactor so its validated
        # artifact projection and typed authorities survive later secrets.
        return prepare_runtime_event(event, redactor=redactor)
    raw_result = event.payload.get("result")
    if type(raw_result) is not dict:  # pragma: no cover - checked above
        raise AssertionError("Staged terminal result changed during projection.")
    result = tool_results.tool_result_from_payload(raw_result)
    projected, _ = _prepare_tool_result_event(
        event=event,
        result=result,
        redactor=redactor,
    )
    return copy_event(projected)


def _terminal_publication_work_estimate(event: Event) -> int:
    """Return a bounded-cost conservative estimate for scheduler admission."""

    raw_result = event.payload.get("result")
    content = raw_result.get("content") if type(raw_result) is dict else None
    structured = raw_result.get("structured") if type(raw_result) is dict else None
    artifacts = raw_result.get("artifacts") if type(raw_result) is dict else None
    # ``len`` is constant-time and four bytes per scalar is a safe UTF-8 upper
    # bound. Nested structured/artifact data cannot be measured in constant
    # time, so conservatively offload any non-empty value rather than walking
    # an untrusted graph on the event loop merely to join the fair queue.
    if structured not in (None, {}, []) or artifacts not in (None, [], {}):
        return TOOL_TERMINAL_PUBLICATION_SLICE_BYTES + 1
    return (len(content) * 4 if type(content) is str else 0) + 16 * 1024


def _normalized_event_timestamp(value: datetime) -> datetime:
    """Apply Cayu's legacy offset-less-as-UTC event timestamp convention."""

    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _durable_payload_utf8_size(payload: dict[str, Any]) -> int:
    limit = 2**63 - 1
    counter = JsonUtf8SizeCounter(limit, canonical_durable_numbers=True)
    if not counter.value(payload) or counter.encountered_unsupported_value:
        raise ValueError("Staged terminal payload has no bounded durable JSON size.")
    return limit - counter.remaining


def _project_and_size_staged_terminal_event(
    event: Event,
    *,
    redactor: SecretRedactor,
) -> tuple[Event, int]:
    projected = _project_staged_terminal_event(event, redactor=redactor)
    return projected, _durable_payload_utf8_size(projected.payload)


def _prepare_and_size_projected_terminal_event(
    event: Event,
    *,
    redactor: SecretRedactor,
) -> tuple[Event, int]:
    """Validate a revised durable stage and measure the exact replacement."""

    projected = prepare_runtime_event(event, redactor=redactor)
    return projected, _durable_payload_utf8_size(projected.payload)


def _staged_terminal_argument_projections(
    event: Event,
) -> tuple[
    tool_argument_publication.ToolArgumentProjection,
    tool_argument_publication.ToolArgumentProjection,
]:
    """Recover sealed public and hook arguments from a durable terminal stage."""

    projection = tool_argument_publication.terminal_argument_projection(
        event.payload,
        legacy_arguments={},
    )
    effective_arguments = event.payload.get("effective_arguments")
    if projection.state == "unavailable":
        if effective_arguments is not None:
            raise ValueError("Unavailable staged arguments cannot carry effective arguments.")
        return projection, projection
    if effective_arguments is None:
        return projection, projection
    if type(effective_arguments) is not dict:
        raise TypeError("Staged effective arguments must be an object.")
    return (
        projection,
        tool_argument_publication.ToolArgumentProjection(
            state="finalized",
            arguments=effective_arguments,
        ),
    )


def restore_staged_terminal_authority(
    event: Event,
    *,
    session_id: str,
    tool_round_identity: ToolRoundIdentity,
    tool_exposure: ResolvedToolExposureAuthority | None = None,
) -> Event:
    """Restore only typed authority erased by checkpoint serialization."""

    session_id = require_clean_nonblank(session_id, "session_id")
    identity = copy_tool_round_identity(tool_round_identity)
    if event.session_id != session_id or not identity.matches_payload(event.payload):
        raise RuntimeError("Staged terminal conflicts with its durable round owner.")
    tool_call_id = event.payload.get("tool_call_id")
    if type(tool_call_id) is not str or not tool_call_id:
        raise ValueError("Staged terminal lost its tool-call identity.")
    restored = event_with_runtime_generated_id(copy_event(event))
    additional_fields = ["tool_call_id"]
    if (
        restored.type is EventType.TOOL_CALL_BLOCKED
        and restored.payload.get("blocked_by") == "tool_exposure"
    ):
        if tool_exposure is None:
            raise RuntimeError("Staged tool-exposure terminal has no durable exposure owner.")
        if type(tool_exposure) is not ResolvedToolExposureAuthority:
            raise TypeError("tool_exposure must be a ResolvedToolExposureAuthority.")
        exposure = tool_exposure
        validate_tool_exposure_terminal_event(restored, tool_exposure=exposure)
        payload = dict(restored.payload)
        payload["profile_id"] = exposure.profile_id
        payload["exposure_fingerprint"] = exposure.fingerprint
        restored = restored.model_copy(update={"payload": payload})
        additional_fields.extend(("profile_id", "exposure_fingerprint"))
    restored = _event_with_tool_round_authority(restored, identity, *additional_fields)
    if restored.interaction_id is not None:
        restored = event_with_runtime_envelope_authority(restored, "interaction_id")
    controls, _references = tool_results.runtime_tool_event_boundary_controls(
        restored.payload,
        include_terminal_controls=restored.type
        in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED},
    )
    if "tool_result_projection" in controls:
        restored = event_with_runtime_nested_payload_authority(
            restored,
            _TOOL_RESULT_PROJECTION_PROVENANCE_PATH,
        )
    return restored


def _redactor_for_tool_calls(
    base: SecretRedactor,
    *,
    registered_agent: runtime_records.RegisteredAgentState,
    tool_calls: list[runtime_records.ToolCallRequest],
) -> SecretRedactor:
    redactor = base
    for tool_call in tool_calls:
        registered_tool = registered_agent.executable_tool(tool_call.name)
        if registered_tool is None:
            continue
        tool = registered_tool.tool
        if isinstance(tool, McpToolAdapter):
            redactor = redactor.merged_with(tool.toolset.secret_redactor)
    return redactor


async def _tool_terminal_payload_limits(
    registered_agent: runtime_records.RegisteredAgentState,
    tool_calls: Iterable[runtime_records.ToolCallRequest],
    *,
    publication_governor: ToolTerminalPublicationGovernor,
    runtime_hooks: Iterable[runtime_records.RegisteredRuntimeHook] = (),
) -> dict[str, int | None]:
    """Resolve frozen registration-time terminal bounds for round admission."""

    has_argument_modifying_hook = any(
        _runtime_hook_supports_phase(
            hook=registered_hook.hook,
            phase=RuntimeHookPhase.BEFORE_TOOL_CALL,
        )
        for registered_hook in (*runtime_hooks, *registered_agent.runtime_hooks)
    )
    limits: dict[str, int | None] = {}
    for tool_call in tool_calls:
        registered_tool = registered_agent.executable_tool(tool_call.name)
        if registered_tool is None:
            limits[tool_call.id] = None
            continue
        contract = ToolExecutionContract.model_validate(registered_tool.execution_contract)
        result_limit = contract.max_terminal_payload_bytes
        if result_limit is None:
            limits[tool_call.id] = None
            continue
        if registered_tool.publish_arguments and has_argument_modifying_hook:
            limits[tool_call.id] = None
            continue
        argument_bytes = 0
        if registered_tool.publish_arguments:
            argument_bytes = await publication_governor.run_cpu(
                (TOOL_TERMINAL_PUBLICATION_SLICE_BYTES + 1 if tool_call.arguments else 0),
                lambda tool_call=tool_call: _durable_payload_utf8_size(tool_call.arguments),
            )
        limits[tool_call.id] = result_limit + argument_bytes
    return limits
