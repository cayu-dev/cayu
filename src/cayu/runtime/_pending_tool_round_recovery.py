"""Recover pending tool rounds from exact durable evidence."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from functools import partial
from hashlib import sha256
from typing import Any

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    copy_json_value,
)
from cayu.approvals.tools import (
    PendingToolCallApproval,
    ToolPolicyEvidence,
)
from cayu.approvals.user_input import (
    user_input_lifecycle_authority_from_checkpoint,
)
from cayu.artifacts.base import ArtifactReadResult, ArtifactStore
from cayu.context.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
)
from cayu.events import (
    Event,
    EventType,
    copy_event,
)
from cayu.execution_profiles import (
    ExecutionProfileIdentity,
)
from cayu.execution_units import (
    ModelAttemptIdentity,
    ToolRoundIdentity,
    copy_tool_round_identity,
)
from cayu.messages import Message
from cayu.runtime import _approval_support as approval_support
from cayu.runtime import _invocation_secrets as invocation_secrets
from cayu.runtime import _runtime_records as runtime_records
from cayu.runtime import _tool_call_replay as tool_call_replay
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime import _tool_results as tool_results
from cayu.runtime import _tool_round_recovery as tool_round_recovery
from cayu.runtime import _transcript as transcript_helpers
from cayu.runtime._child_session_identity import (
    ChildSessionKind,
    child_session_id_prefix,
    generate_child_session_id,
)
from cayu.runtime._delegated_event_stream import _close_delegated_event_stream
from cayu.runtime._durable_subagents import (
    durable_subagent_submission_from_checkpoint,
    durable_subagent_submission_receipt_from_checkpoint,
    durable_subagent_submission_seed_from_checkpoint,
)
from cayu.runtime._durable_tool_round import (
    DeferredInteractionInput,
    DurableToolRound,
    InterruptedToolRoundRequest,
    _interrupted_tool_round_results,
)
from cayu.runtime._environment_lifecycle import (
    EnvironmentLifecycle,
)
from cayu.runtime._event_writer import (
    RuntimeEventWriter,
)
from cayu.runtime._foreground_subagent_recovery import ForegroundSubagentRecoveryRequired
from cayu.runtime._interruption_coordinator import (
    _PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY,
)
from cayu.runtime._invocation_lifecycle import (
    InvocationContext,
)
from cayu.runtime._isolated_tool_process import (
    isolated_tool_dispatch_authority_digests,
    isolated_tool_dispatch_authority_storage_key,
    isolated_tool_dispatch_record_matches,
    isolated_tool_dispatch_settlement_matches,
    isolated_tool_dispatch_settlement_storage_key,
    isolated_tool_dispatch_storage_key,
)
from cayu.runtime._run_limits import (
    SessionUsageTracker,
)
from cayu.runtime._session_control import (
    SessionControl,
)
from cayu.runtime._session_queries import query_all_sessions
from cayu.runtime._tool_effect_preparation_recovery import (
    requires_explicit_effect_continuation,
    settle_prepared_tool_effects,
)
from cayu.runtime._tool_effect_state import (
    ToolEffectReconciliationRequired,
    ToolEffectRecord,
    ToolEffectStateOwner,
    ToolEffectTerminal,
)
from cayu.runtime._tool_invocation.admission import (
    ToolApprovalRequired,
)
from cayu.runtime._tool_invocation.invocation import ToolInvocation
from cayu.runtime._tool_round_staging import (
    _redactor_for_tool_calls,
    _tool_terminal_payload_limits,
)
from cayu.runtime._workspace_observation_recovery import WorkspaceObservationRecovery
from cayu.sessions import _pending_approval_reader as pending_approval_reader
from cayu.sessions import _pending_tool_round as pending_rounds
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions import _staged_tool_terminal_reader as staged_terminal_reader
from cayu.sessions._invocation_lifecycle import (
    AdmittedInvocationBinding,
)
from cayu.sessions._terminal_evidence import (
    _INTERRUPTION_TYPE_OPERATOR_REQUESTED,
)
from cayu.sessions.base import (
    RuntimePublicationReceipt,
    SessionOperationPublication,
    SessionRunFenced,
    SessionRuntimePublicationConflict,
    SessionStore,
)
from cayu.sessions.queries import SessionOrder, SessionQuery
from cayu.sessions.records import (
    Session,
    SessionStatus,
    _queued_dispatch_session_instance_fingerprint,
)
from cayu.tools import _terminal_controls as tool_terminal_controls
from cayu.tools._runner import durable_runner_receipt_observer, durable_runner_recovery_authority
from cayu.tools.base import (
    DurableToolOperationConflict,
    DurableToolRecoveryAuthority,
    DurableToolRecoveryEvidence,
    DurableToolRecoveryIdentity,
    DurableToolRecoveryInspection,
    ToolEffect,
    ToolResult,
)
from cayu.tools.exposure import (
    validate_resolved_tool_exposure_authority,
)
from cayu.tools.policy import ToolPolicyDecision
from cayu.vaults.redaction import SecretRedactor


def _approval_interrupt_close_intent_matches(
    checkpoint: dict[str, Any] | None,
    *,
    pending_round: pending_rounds.PendingToolRound,
) -> bool:
    """Require exact durable proof before recovering a cleared approval as interrupted."""

    if checkpoint is None:
        return False
    interrupt_payload = checkpoint.get(_PENDING_SESSION_INTERRUPT_CHECKPOINT_KEY)
    if (
        type(interrupt_payload) is not dict
        or interrupt_payload.get("interruption_type") != _INTERRUPTION_TYPE_OPERATOR_REQUESTED
        or type(interrupt_payload.get("interruption_request_id")) is not str
        or not interrupt_payload["interruption_request_id"].strip()
        or interrupt_payload["interruption_request_id"].strip()
        != interrupt_payload["interruption_request_id"]
    ):
        return False
    intent = interrupt_payload.get(approval_support.APPROVAL_INTERRUPT_CLOSE_INTENT_KEY)
    if type(intent) is not dict:
        return False
    identity = pending_rounds.pending_tool_round_identity(pending_round)
    expected = {
        "tool_call_id": _pending_round_policy_gate_call_id(pending_round),
        **identity.payload(),
    }
    return (
        all(intent.get(key) == value for key, value in expected.items())
        and type(intent.get("approval_id")) is str
    )


def _pending_round_policy_gate_call_id(
    pending_round: pending_rounds.PendingToolRound,
) -> str | None:
    """Return the call that must own this round's visible policy gate."""

    for call in pending_round.tool_calls:
        if (
            pending_approval_reader.effective_tool_policy_evidence(call)
            is ToolPolicyEvidence.AUTHORITATIVE
            and call.policy_decision == ToolPolicyDecision.REQUIRE_APPROVAL.value
        ):
            return call.tool_call_id
    for call in pending_round.tool_calls:
        if (
            pending_approval_reader.effective_tool_policy_evidence(call)
            is ToolPolicyEvidence.AMBIGUOUS
        ):
            return call.tool_call_id
    return None


RegisteredAgentResolver = Callable[[str], runtime_records.RegisteredAgentState]

RegisteredEnvironmentResolver = Callable[[str | None], runtime_records.RegisteredEnvironment | None]


class _DurableArtifactRecoveryReader:
    """Expose exact artifact reads without handing extensions the raw store."""

    __slots__ = ("__artifact_store", "id")

    def __init__(self, artifact_store: ArtifactStore) -> None:
        if not isinstance(artifact_store, ArtifactStore):
            raise TypeError("Durable artifact recovery requires an ArtifactStore.")
        self.__artifact_store = artifact_store
        self.id = artifact_store.id

    async def read_bytes(
        self,
        artifact_id: str,
        *,
        max_bytes: int | None = None,
    ) -> ArtifactReadResult:
        return await self.__artifact_store.read_bytes(
            artifact_id,
            max_bytes=max_bytes,
        )


def _matches_recoverable_subagent_child(
    child: Session,
    *,
    idempotency_key: str,
    tool_call_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    parent_session: Session,
    registered_agent: runtime_records.RegisteredAgentState,
) -> bool:
    registered_tool = registered_agent.tools.get(tool_name)
    if registered_tool is None or registered_tool.child_session_recovery is None:
        return False
    generated_prefix = child_session_id_prefix(ChildSessionKind.SUBAGENT)
    generated_identity = child.id.startswith(generated_prefix)
    if generated_identity and child.id != generate_child_session_id(
        kind=ChildSessionKind.SUBAGENT,
        parent_session_id=parent_session.id,
        logical_spawn_id=idempotency_key,
    ):
        return False
    matched = registered_tool.child_session_recovery.matches_recoverable_child(
        child,
        parent_invocation=parent_session.invocation,
        parent_session_id=parent_session.id,
        causal_budget_id=parent_session.causal_budget_id,
        environment_name=parent_session.environment_name,
        tool_call_id=tool_call_id,
        idempotency_key=idempotency_key,
        arguments=arguments,
        require_fingerprint=generated_identity,
    )
    if type(matched) is not bool:
        raise TypeError("Child-session recovery matchers must return bool.")
    return matched


def _environment_name(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> str | None:
    if registered_environment is None:
        return None
    return registered_environment.spec.name


class PendingToolRoundRecovery:
    """Recover pending tool rounds from exact durable evidence."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        event_writer: RuntimeEventWriter,
        session_control: SessionControl[SessionUsageTracker],
        environment_lifecycle: EnvironmentLifecycle,
        tool_invocation: ToolInvocation,
        deferred_input: DeferredInteractionInput,
        workspace_observation_recovery: WorkspaceObservationRecovery,
        secret_redactor: SecretRedactor,
        clock: Callable[[], datetime],
        resolve_registered_agent: RegisteredAgentResolver,
        resolve_registered_environment: RegisteredEnvironmentResolver,
    ) -> None:
        self._session_store = session_store
        self._event_writer = event_writer
        self._session_control = session_control
        self._environment_lifecycle = environment_lifecycle
        self._tool_invocation = tool_invocation
        self._deferred_input = deferred_input
        self._workspace_observation_recovery = workspace_observation_recovery
        self._secret_redactor = secret_redactor
        self._clock = clock
        self._resolve_registered_agent = resolve_registered_agent
        self._resolve_registered_environment = resolve_registered_environment

    async def deliver_pending_tool_effect_uncertainty(self, session: Session) -> list[Event]:
        """Deliver already-committed uncertainty through the existing event writer."""
        _checkpoint, pending = await pending_round_reader.load_pending_tool_round(
            self._session_store,
            session.id,
        )
        if pending is None:
            return []
        events = await ToolEffectStateOwner(self._session_store).load_uncertainty_events(
            session,
            tool_round_id=pending.tool_round_id,
            tool_call_ids=tuple(call.tool_call_id for call in pending.tool_calls),
        )
        return await self._event_writer.fan_out_persisted(events)

    async def close_interrupted_tool_round(
        self,
        request: InterruptedToolRoundRequest,
    ) -> AsyncGenerator[Event, None]:
        """Close an interrupted round without replaying unfinished tools."""
        stream = self._close_interrupted_tool_round(request)
        async with _close_delegated_event_stream(stream) as owned_stream:
            async for event in owned_stream:
                yield event

    async def _close_interrupted_tool_round(
        self,
        request: InterruptedToolRoundRequest,
    ) -> AsyncGenerator[Event, None]:
        owner = DurableToolRound(
            session=request.session,
            tool_round_identity=request.tool_round_identity,
            session_store=self._session_store,
            event_writer=self._event_writer,
        )
        snapshot = await owner.prepare_interruption(
            request,
            materialize_deferred_input_if_present=self._deferred_input.materialize_if_present,
        )
        if snapshot is None:
            return
        tool_round_identity = copy_tool_round_identity(request.tool_round_identity)
        source_checkpoint = snapshot.checkpoint
        pending_round = snapshot.pending_round
        pending_tool_calls = snapshot.tool_calls
        expected_transcript_cursor = snapshot.expected_transcript_cursor
        # Settlement reloads the checkpoint; release the original snapshot holder.
        del snapshot
        (
            source_checkpoint,
            pending_round,
            workspace_events,
        ) = await self._workspace_observation_recovery.settle_tool_round_workspace_observations(
            session=request.session,
            registered_environment=request.registered_environment,
            execution_profile=request.execution_profile,
            invocation_context=request.invocation_context,
            checkpoint=source_checkpoint,
            pending_round=pending_round,
            staged_only=True,
        )
        for event in workspace_events:
            yield event
        if await ToolEffectStateOwner(self._session_store).preserve_unresolved(
            request.session,
            tool_round_id=tool_round_identity.tool_round_id,
            tool_call_ids=tuple(call.id for call in pending_tool_calls),
        ):
            from cayu.runtime._tool_effect_diagnostics import persist_cleanup_diagnostics

            effect_owner = ToolEffectStateOwner(self._session_store)
            dispatched = {}
            for call in pending_tool_calls:
                record = await effect_owner.resolve_call(
                    request.session,
                    tool_round_id=tool_round_identity.tool_round_id,
                    tool_call_id=call.id,
                )
                if record is not None and record.dispatch_id is not None:
                    dispatched[call.id] = record
            redactors = request.cancellation_redactors_by_id or {}
            artifacts_by_id = request.cancellation_artifacts_by_id
            if artifacts_by_id is not None:
                for call_id, artifacts in artifacts_by_id.items():
                    if not artifacts:
                        continue
                    record = dispatched.get(call_id)
                    diagnostic = await persist_cleanup_diagnostics(
                        store=self._session_store,
                        writer=self._event_writer,
                        records=(record,) if record is not None else tuple(dispatched.values()),
                        artifacts=artifacts,
                        redactor=self._secret_redactor.merged_with(
                            redactors.get(call_id, self._secret_redactor)
                        ),
                        attributed=record is not None,
                    )
                    if diagnostic is not None:
                        for event in await self._event_writer.fan_out_persisted([diagnostic]):
                            yield event
            elif request.cancellation_artifacts:
                redactor = self._secret_redactor
                for invocation_redactor in redactors.values():
                    redactor = redactor.merged_with(invocation_redactor)
                diagnostic = await persist_cleanup_diagnostics(
                    store=self._session_store,
                    writer=self._event_writer,
                    records=tuple(dispatched.values()),
                    artifacts=request.cancellation_artifacts,
                    redactor=redactor,
                    attributed=False,
                )
                if diagnostic is not None:
                    for event in await self._event_writer.fan_out_persisted([diagnostic]):
                        yield event
            for event in await self.deliver_pending_tool_effect_uncertainty(request.session):
                yield event
            return
        if any(call.tool_name == STRUCTURED_OUTPUT_TOOL_NAME for call in pending_round.tool_calls):
            async for event in self._recover_structured_output_tool_round(
                session=request.session,
                registered_agent=request.registered_agent,
                registered_environment=request.registered_environment,
                messages=request.messages,
                pending_round=pending_round,
                retry_allowed=False,
                expected_transcript_cursor=expected_transcript_cursor,
                execution_profile=request.execution_profile,
                invocation_context=request.invocation_context,
                interrupted=True,
            ):
                yield event
            return
        lifecycle_events = await self.load_tool_round_lifecycle_events(
            session_id=request.session.id,
            pending_round=pending_round,
        )
        recorded_outcomes, started_ids = tool_round_recovery.recorded_tool_outcomes(
            events=lifecycle_events,
            pending_round=pending_round,
        )
        (
            isolated_dispatched_ids,
            isolated_call_ids,
        ) = await self.isolated_tool_dispatch_ids(
            session=request.session,
            pending_round=pending_round,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
        )
        interrupted_results = _interrupted_tool_round_results(
            tool_calls=pending_tool_calls,
            completed_outcomes=list(recorded_outcomes.values()),
            tool_round_identity=tool_round_identity,
            registered_agent=request.registered_agent,
            isolated_dispatched_ids=isolated_dispatched_ids,
            cancellation_artifacts=request.cancellation_artifacts,
            cancellation_artifacts_by_id=request.cancellation_artifacts_by_id,
        )
        interrupted_results = await self.reattach_subagent_children_in_outcomes(
            session=request.session,
            registered_agent=request.registered_agent,
            tool_round_id=request.tool_round_identity.tool_round_id,
            outcomes=interrupted_results,
            source_checkpoint=source_checkpoint,
        )
        cancellation_redactors = request.cancellation_redactors_by_id or {}
        interrupted_results = [
            tool_results.redact_runtime_owned_tool_call_outcomes(
                [outcome],
                cancellation_redactors.get(outcome.call.id, self._secret_redactor),
            )[0]
            for outcome in interrupted_results
        ]
        async for event in self._publish_recovered_tool_outcomes(
            owner=owner,
            registered_agent=request.registered_agent,
            registered_environment=request.registered_environment,
            messages=request.messages,
            pending_round=pending_round,
            lifecycle_events=lifecycle_events,
            synthesized_outcomes=interrupted_results,
            effective_started_ids=(started_ids - isolated_call_ids) | isolated_dispatched_ids,
            expected_transcript_cursor=expected_transcript_cursor,
            execution_profile=request.execution_profile,
            invocation_context=request.invocation_context,
            interrupted=True,
        ):
            yield event

    async def has_recoverable_durable_tool_result(
        self,
        *,
        session: Session,
        tool_round_id: str,
        tool_call_id: str,
    ) -> bool:
        """Advisory, read-only readiness; actual recovery repeats all checks under its fence."""
        checkpoint = await self._session_store.load_checkpoint(session.id)
        pending = pending_round_reader.pending_tool_round_from_checkpoint(checkpoint)
        if (
            pending is None
            or pending.tool_round_id != tool_round_id
            or pending.source_run_epoch is None
            or pending.execution_profile_fingerprint is None
        ):
            return False
        call = next(
            (item for item in pending.tool_calls if item.tool_call_id == tool_call_id), None
        )
        if (
            call is None
            or call.policy_decision != ToolPolicyDecision.ALLOW.value
            or pending_approval_reader.effective_tool_policy_evidence(call)
            is not ToolPolicyEvidence.AUTHORITATIVE
        ):
            return False
        agent = self._resolve_registered_agent(session.agent_name)
        registered = agent.executable_tool(call.tool_name)
        if registered is None or not isinstance(
            registered.durable_tool_recovery, DurableToolRecoveryInspection
        ):
            return False
        environment = self._resolve_registered_environment(session.environment_name)
        allocation = None
        observer = None
        if environment is not None:
            allocation = (
                await self._environment_lifecycle.durable_live_allocation_fingerprint(
                    session_id=session.id, environment_name=environment.spec.name
                )
                if environment.factory_backed
                else environment.live_allocation_fingerprint
            )
            source = (
                environment.factory
                if environment.factory_backed
                else environment.environment.runner
            )
            observer = durable_runner_receipt_observer(source, redactor=self._secret_redactor)

        async def load_operation(key: str) -> dict[str, Any] | None:
            return await self._session_store.load_session_operation(session.id, key)

        request = approval_support.tool_call_request_from_pending(call)
        identity = DurableToolRecoveryIdentity(
            parent_session_id=session.id,
            parent_run_epoch=pending.source_run_epoch,
            execution_profile_fingerprint=pending.execution_profile_fingerprint,
            environment_name=session.environment_name,
            environment_allocation_fingerprint=allocation,
            model_step_id=pending.model_step_id,
            model_attempt_id=pending.model_attempt_id,
            tool_round_id=pending.tool_round_id,
            tool_call_id=call.tool_call_id,
            idempotency_key=tool_execution.tool_idempotency_key(
                session_id=session.id,
                tool_round_id=pending.tool_round_id,
                tool_call_id=call.tool_call_id,
            ),
            arguments=copy_json_value(request.arguments, "durable_recovery_inspection.arguments"),
        )
        return await registered.durable_tool_recovery.inspect_durable_tool_call(
            identity=identity, load_operation=load_operation, observe_receipt=observer
        )

    async def has_completed_tool_round_results(
        self,
        *,
        session_id: str,
        pending_round: pending_rounds.PendingToolRound,
    ) -> bool:
        """Advisory readiness for recorded settled effects, not execution authority."""
        if pending_round.policy_state != "planned" or any(
            call.policy_evidence is not ToolPolicyEvidence.AUTHORITATIVE
            or call.policy_decision != ToolPolicyDecision.ALLOW.value
            for call in pending_round.tool_calls
        ):
            return False
        events = await self.load_tool_round_lifecycle_events(
            session_id=session_id, pending_round=pending_round
        )
        outcomes, _ = tool_round_recovery.recorded_tool_outcomes(
            events=events, pending_round=pending_round
        )
        terminal_events = [
            *events,
            *(
                staged.event
                for staged in staged_terminal_reader.staged_terminal_records(pending_round)
            ),
        ]
        call_ids = [call.tool_call_id for call in pending_round.tool_calls]
        return all(call_id in outcomes for call_id in call_ids) and self.tool_terminals_are_settled(
            terminal_events, call_ids
        )

    async def has_settled_published_tool_round_results(
        self, *, session_id: str, receipt: RuntimePublicationReceipt
    ) -> bool:
        """Check recorded outcomes; a publication receipt alone does not prove settlement."""
        if receipt.kind != "tool-round":
            return False
        call_ids = receipt.intent.get("tool_call_ids")
        identity_fields = {
            key: receipt.intent.get(key)
            for key in ("model_step_id", "model_attempt_id", "tool_round_id")
        }
        if (
            type(call_ids) is not list
            or not call_ids
            or any(type(value) is not str or not value for value in call_ids)
            or len(set(call_ids)) != len(call_ids)
            or any(type(value) is not str or not value for value in identity_fields.values())
        ):
            return False
        identity = ToolRoundIdentity.model_validate(identity_fields)
        events = await self._session_store.load_tool_round_lifecycle_events_for_round(
            session_id, call_ids, tool_round_identity=identity
        )
        if any(not identity.matches_payload(event.payload) for event in events):
            return False
        return self.tool_terminals_are_settled(events, call_ids)

    @staticmethod
    def tool_terminals_are_settled(events: list[Event], call_ids: list[str]) -> bool:
        expected = set(call_ids)
        settled: set[str] = set()
        for event in events:
            if event.type not in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}:
                continue
            call_id = event.payload.get("tool_call_id")
            if type(call_id) is not str or call_id not in expected:
                return False
            controls = tool_terminal_controls.runtime_terminal_controls(event.payload)
            if (
                controls.get("outcome_unknown", False)
                or controls.get("manual_reconciliation_required", False)
                or event.payload.get("secret_scope_incomplete", False) is not False
            ):
                return False
            result_payload = event.payload.get("result")
            if type(result_payload) is not dict:
                return False
            result = tool_results.tool_result_from_payload(result_payload)
            if result.is_error is not (event.type is EventType.TOOL_CALL_FAILED):
                return False
            settled.add(call_id)
        return bool(expected) and settled == expected

    async def load_tool_round_lifecycle_events(
        self, *, session_id: str, pending_round: pending_rounds.PendingToolRound
    ) -> list[Event]:
        return await tool_round_recovery.load_tool_round_lifecycle_events(
            self._session_store, session_id=session_id, pending_round=pending_round
        )

    async def isolated_tool_dispatch_ids(
        self,
        *,
        session: Session,
        pending_round: pending_rounds.PendingToolRound,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
    ) -> tuple[set[str], set[str]]:
        """Load exact possible-dispatch evidence and all isolated call IDs."""

        identity = pending_rounds.pending_tool_round_identity(pending_round)
        dispatched_ids: set[str] = set()
        isolated_call_ids: set[str] = set()
        environment_allocation_fingerprint_loaded = False
        environment_allocation_fingerprint: str | None = None
        for call in pending_round.tool_calls:
            registered_tool = registered_agent.executable_tool(call.tool_name)
            if registered_tool is None:
                continue
            contract = registered_tool.execution_contract
            if type(contract) is not dict:
                raise RuntimeError("Registered tool execution contract is malformed.")
            if contract.get("boundary") != "posix_process":
                continue
            isolated_call_ids.add(call.tool_call_id)
            idempotency_key = tool_execution.tool_idempotency_key(
                session_id=session.id,
                tool_round_id=identity.tool_round_id,
                tool_call_id=call.tool_call_id,
            )
            storage_key = isolated_tool_dispatch_storage_key(
                session_id=session.id,
                model_step_id=identity.model_step_id,
                model_attempt_id=identity.model_attempt_id,
                tool_round_id=identity.tool_round_id,
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                idempotency_key=idempotency_key,
            )
            authority_storage_key = isolated_tool_dispatch_authority_storage_key(
                session_id=session.id,
                model_step_id=identity.model_step_id,
                model_attempt_id=identity.model_attempt_id,
                tool_round_id=identity.tool_round_id,
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                idempotency_key=idempotency_key,
            )
            settlement_storage_key = isolated_tool_dispatch_settlement_storage_key(
                session_id=session.id,
                model_step_id=identity.model_step_id,
                model_attempt_id=identity.model_attempt_id,
                tool_round_id=identity.tool_round_id,
                tool_call_id=call.tool_call_id,
                tool_name=call.tool_name,
                idempotency_key=idempotency_key,
            )
            record = await self._session_store.load_session_operation(
                session.id,
                storage_key,
            )
            authority_record = await self._session_store.load_session_operation(
                session.id,
                authority_storage_key,
            )
            settlement_record = await self._session_store.load_session_operation(
                session.id,
                settlement_storage_key,
            )
            if record is None:
                if authority_record is not None or settlement_record is not None:
                    raise RuntimeError("Isolated tool dispatch evidence has no preparation record.")
                continue
            if not environment_allocation_fingerprint_loaded:
                if registered_environment is None:
                    environment_allocation_fingerprint = None
                elif registered_environment.factory_backed:
                    environment_allocation_fingerprint = (
                        await self._environment_lifecycle.durable_live_allocation_fingerprint(
                            session_id=session.id,
                            environment_name=registered_environment.spec.name,
                        )
                    )
                else:
                    environment_allocation_fingerprint = (
                        registered_environment.live_allocation_fingerprint
                    )
                environment_allocation_fingerprint_loaded = True
            authority_digests = (
                None
                if pending_round.source_run_epoch is None
                or pending_round.execution_profile_fingerprint is None
                else isolated_tool_dispatch_authority_digests(
                    authority_record,
                    session_id=session.id,
                    parent_task_id=pending_round.task_id,
                    parent_run_epoch=pending_round.source_run_epoch,
                    model_step_id=identity.model_step_id,
                    model_attempt_id=identity.model_attempt_id,
                    tool_round_id=identity.tool_round_id,
                    tool_call_id=call.tool_call_id,
                    tool_name=call.tool_name,
                    idempotency_key=idempotency_key,
                    execution_profile_fingerprint=(pending_round.execution_profile_fingerprint),
                    environment_allocation_fingerprint=(environment_allocation_fingerprint),
                )
            )
            if (
                pending_round.source_run_epoch is None
                or pending_round.execution_profile_fingerprint is None
                or authority_digests is None
                or not isolated_tool_dispatch_record_matches(
                    record,
                    session_id=session.id,
                    parent_task_id=pending_round.task_id,
                    parent_run_epoch=pending_round.source_run_epoch,
                    model_step_id=identity.model_step_id,
                    model_attempt_id=identity.model_attempt_id,
                    tool_round_id=identity.tool_round_id,
                    tool_call_id=call.tool_call_id,
                    tool_name=call.tool_name,
                    idempotency_key=idempotency_key,
                    request_sha256=authority_digests[0],
                    effective_arguments_sha256=authority_digests[1],
                    execution_profile_fingerprint=(pending_round.execution_profile_fingerprint),
                    environment_allocation_fingerprint=(environment_allocation_fingerprint),
                )
            ):
                raise RuntimeError(
                    "Isolated tool dispatch evidence conflicts with its pending round."
                )
            if settlement_record is not None:
                if not isolated_tool_dispatch_settlement_matches(
                    settlement_record,
                    dispatch_record=record,
                ):
                    raise RuntimeError(
                        "Isolated tool dispatch settlement conflicts with its preparation."
                    )
                continue
            dispatched_ids.add(call.tool_call_id)
        return dispatched_ids, isolated_call_ids

    async def _recover_structured_output_tool_round(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        messages: list[Message],
        pending_round: pending_rounds.PendingToolRound,
        retry_allowed: bool,
        expected_transcript_cursor: int,
        execution_profile: ExecutionProfileIdentity | None,
        invocation_context: InvocationContext | None = None,
        interrupted: bool = False,
    ) -> AsyncGenerator[Event, None]:
        """Supply session recovery collaborators to the durable round owner."""
        owner = DurableToolRound(
            session=session,
            tool_round_identity=pending_rounds.pending_tool_round_identity(pending_round),
            session_store=self._session_store,
            event_writer=self._event_writer,
        )
        async with contextlib.aclosing(
            owner.recover_structured(
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                messages=messages,
                pending_round=pending_round,
                retry_allowed=retry_allowed,
                expected_transcript_cursor=expected_transcript_cursor,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
                redactor=self._secret_redactor,
                tool_redactor=self._secret_redactor,
                materialize_expected_deferred_input=self._deferred_input.materialize_expected,
                interrupted=interrupted,
            )
        ) as events:
            async for event in events:
                yield event

    async def reattach_subagent_children_in_outcomes(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        tool_round_id: str | None,
        outcomes: list[runtime_records.ToolCallOutcome],
        source_checkpoint: dict[str, Any] | None,
    ) -> list[runtime_records.ToolCallOutcome]:
        """Replace unfinished spawn outcomes with matching durable child references."""
        if tool_round_id is None or not outcomes:
            return outcomes
        children = await self.subagent_children_by_idempotency_key(session.id)
        reattached: list[runtime_records.ToolCallOutcome] = []
        for outcome in outcomes:
            idempotency_key = tool_execution.tool_idempotency_key(
                session_id=session.id,
                tool_round_id=tool_round_id,
                tool_call_id=outcome.call.id,
            )
            marker_backed = (
                durable_subagent_submission_seed_from_checkpoint(
                    source_checkpoint,
                    idempotency_key=idempotency_key,
                )
                is not None
                or durable_subagent_submission_from_checkpoint(
                    source_checkpoint,
                    idempotency_key=idempotency_key,
                )
                is not None
                or durable_subagent_submission_receipt_from_checkpoint(
                    source_checkpoint,
                    idempotency_key=idempotency_key,
                )
                is not None
            )
            if idempotency_key not in children and not marker_backed:
                reattached.append(outcome)
                continue
            recovery_arguments = await self.subagent_recovery_arguments(
                checkpoint=source_checkpoint,
                parent_session=session,
                tool_name=outcome.call.name,
                tool_round_id=tool_round_id,
                tool_call_id=outcome.call.id,
                idempotency_key=idempotency_key,
                fallback=outcome.call.arguments,
            )
            reconciled_result = await self.reconcile_subagent_child(
                children,
                idempotency_key=idempotency_key,
                tool_call_id=outcome.call.id,
                tool_name=outcome.call.name,
                tool_round_id=tool_round_id,
                arguments=recovery_arguments,
                parent_session=session,
                registered_agent=registered_agent,
            )
            result = reconciled_result
            if result is None:
                result = await self.reattached_subagent_result(
                    children,
                    idempotency_key,
                    parent_checkpoint=source_checkpoint,
                    tool_call_id=outcome.call.id,
                    tool_name=outcome.call.name,
                    tool_round_id=tool_round_id,
                    arguments=recovery_arguments,
                    parent_session=session,
                    registered_agent=registered_agent,
                )
            if result is not None and outcome.result.artifacts:
                result = result.model_copy(
                    update={
                        "artifacts": copy_json_value(
                            outcome.result.artifacts,
                            "reattached_subagent_artifacts",
                        )
                    },
                    deep=True,
                )
            reattached.append(
                outcome
                if result is None
                else runtime_records.ToolCallOutcome(call=outcome.call, result=result)
            )
        return reattached

    async def recover_pending_tool_round(
        self,
        *,
        session: Session,
        invocation_context: InvocationContext | None = None,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        messages: list[Message],
        execution_profile: ExecutionProfileIdentity | None = None,
        tail_message_count: int = 0,
        incomplete_recovery_claimed: bool = False,
        expected_transcript_cursor: int | None = None,
        admit_replay: Callable[[ModelAttemptIdentity], Awaitable[bool]] | None = None,
        elect_replay: bool = False,
    ) -> AsyncGenerator[Event, None]:
        """Repair one durable pending round strictly from recorded evidence.

        ``elect_replay`` lets a takeover of a dead running execution keep the round
        open for its continuation to replay, raising ``ToolCallReplayRequired``.
        Only a caller that handles that exception may pass it.
        """
        if invocation_context is not None:
            if type(invocation_context) is not InvocationContext or not isinstance(
                invocation_context.binding,
                AdmittedInvocationBinding,
            ):
                raise TypeError(
                    "Pending tool-round recovery requires an authenticated admitted context."
                )
            binding = invocation_context.binding
            if (
                binding.session_id != session.id
                or binding.session_instance_id != session.instance_id
                or binding.run_epoch != session.run_epoch
            ):
                raise SessionRunFenced(
                    "Pending tool-round recovery lost its frozen invocation binding."
                )
            if registered_agent is not invocation_context.registered_agent or (
                registered_environment is not invocation_context.registered_environment
            ):
                raise RuntimeError(
                    "Pending tool-round recovery substituted a registered collaborator."
                )
            if (
                execution_profile is not None
                and execution_profile is not invocation_context.profile
            ):
                raise RuntimeError(
                    "Pending tool-round recovery substituted its validated execution profile."
                )
            execution_profile = invocation_context.profile
        checkpoint, pending_round = await pending_round_reader.load_pending_tool_round(
            self._session_store,
            session.id,
            redactor=self._secret_redactor,
            consume_on_rejection=True,
            runtime_session=session,
        )
        if pending_round is None:
            return
        (
            checkpoint,
            pending_round,
            workspace_events,
        ) = await self._workspace_observation_recovery.settle_tool_round_workspace_observations(
            session=session,
            registered_environment=registered_environment,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
            checkpoint=checkpoint,
            pending_round=pending_round,
        )
        for event in workspace_events:
            yield event
        for event in await settle_prepared_tool_effects(
            store=self._session_store,
            writer=self._event_writer,
            session=session,
            pending=pending_round,
            profile=execution_profile,
        ):
            yield event
        await ToolEffectStateOwner(self._session_store).preserve_unresolved(
            session,
            tool_round_id=pending_round.tool_round_id,
            tool_call_ids=tuple(call.tool_call_id for call in pending_round.tool_calls),
        )
        for event in await self.deliver_pending_tool_effect_uncertainty(session):
            yield event
        if incomplete_recovery_claimed:
            for call in pending_round.tool_calls:
                effect = await ToolEffectStateOwner(self._session_store).resolve_call(
                    session,
                    tool_round_id=pending_round.tool_round_id,
                    tool_call_id=call.tool_call_id,
                )
                if effect is not None and requires_explicit_effect_continuation(
                    effect, pending_round
                ):
                    # Incomplete recovery does not run the model continuation.
                    # Closing this round would make exact receipt replay appear
                    # consumed before its explicit continuation has started.
                    # Targeted preparations also wait for continuation; do not
                    # rejoin their grants under an interruption-only claim.
                    # Ordinary non-dispatch terminals can close deterministically.
                    raise ToolEffectReconciliationRequired()
        if expected_transcript_cursor is None:
            expected_transcript_cursor = await self._session_store.load_transcript_cursor(
                session.id
            )
        environment_name = _environment_name(registered_environment)
        if pending_round.agent_name != registered_agent.spec.name:
            raise RuntimeError(
                f"Pending tool round belongs to a different agent: {pending_round.agent_name}."
            )
        if pending_round.environment_name != environment_name:
            raise RuntimeError(
                "Pending tool round belongs to a different environment: "
                f"{pending_round.environment_name}."
            )
        if pending_round.tool_exposure is not None:
            validate_resolved_tool_exposure_authority(
                pending_round.tool_exposure,
                registered_agent.tool_capabilities,
                catalogue_revision=registered_agent.tool_catalogue.revision,
            )
        (
            pending_round,
            _resolved_tool_calls,
            targeted_resolution_events,
        ) = await self._tool_invocation.admission.resolve_targeted_calls(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            pending_round=pending_round,
            invocation_context=invocation_context,
        )
        for targeted_resolution_event in targeted_resolution_events:
            yield targeted_resolution_event
        if targeted_resolution_events:
            checkpoint = await self._session_store.load_checkpoint(session.id)
        registered_tool_names = registered_agent.executable_tool_names
        durable_policy_decisions = frozenset(decision.value for decision in ToolPolicyDecision)
        ambiguous_interrupt_close_intent = (
            session.status == SessionStatus.INTERRUPTING
            and _approval_interrupt_close_intent_matches(
                checkpoint,
                pending_round=pending_round,
            )
        )
        invalid_planned_calls = [
            call
            for call in pending_round.tool_calls
            if pending_round.policy_state == "planned"
            and call.tool_name in registered_tool_names
            and not (
                (
                    call.policy_evidence is ToolPolicyEvidence.AUTHORITATIVE
                    and call.policy_decision in durable_policy_decisions
                )
                or call.policy_evidence is ToolPolicyEvidence.UNREGISTERED
                or call.policy_evidence is ToolPolicyEvidence.UNEXPOSED
                or (
                    call.policy_evidence is ToolPolicyEvidence.AMBIGUOUS
                    and ambiguous_interrupt_close_intent
                )
                or (
                    call.policy_evidence is None
                    and call.policy_decision in durable_policy_decisions
                )
            )
        ]
        if invalid_planned_calls:
            raise RuntimeError(
                "Policy-planned pending tool round has no authoritative decision for "
                f"registered call {invalid_planned_calls[0].tool_call_id}."
            )
        pending_tool_calls = tool_round_recovery.pending_round_tool_calls(pending_round)
        tool_round_identity = pending_rounds.pending_tool_round_identity(pending_round)
        if await transcript_helpers.tool_round_has_result_messages(
            self._session_store,
            session.id,
            pending_tool_calls,
            tool_round_identity=tool_round_identity,
        ):
            raise SessionRuntimePublicationConflict(
                "The durable transcript already closes the pending tool round without "
                "its atomic checkpoint publication."
            )
        insert_at = len(messages) - tail_message_count
        if insert_at < 0:
            raise RuntimeError("Pending tool round recovery received an invalid tail size.")
        if any(call.tool_name == STRUCTURED_OUTPUT_TOOL_NAME for call in pending_round.tool_calls):
            async for event in self._recover_structured_output_tool_round(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                messages=messages,
                pending_round=pending_round,
                retry_allowed=session.status == SessionStatus.RUNNING,
                expected_transcript_cursor=expected_transcript_cursor,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
            ):
                yield event
            return
        legacy_policy_plan_is_ambiguous = pending_round.policy_context_version is None and any(
            call.tool_name in registered_tool_names
            and call.policy_decision not in durable_policy_decisions
            for call in pending_round.tool_calls
        )
        legacy_policy_plan_requires_approval = (
            pending_round.policy_context_version is None
            and any(
                call.tool_name in registered_tool_names
                and call.policy_decision == ToolPolicyDecision.REQUIRE_APPROVAL.value
                for call in pending_round.tool_calls
            )
            and pending_approval_reader.pending_approval_from_checkpoint(
                checkpoint,
                redactor=self._secret_redactor,
            )
            is None
        )
        if pending_round.policy_state == "unplanned" and (
            pending_round.policy_context_version == 1
            or legacy_policy_plan_is_ambiguous
            or legacy_policy_plan_requires_approval
        ):
            # A raw round proves only that model output was durably staged. It
            # cannot prove whether a stateful policy had already returned before
            # the process stopped. Replaying authorize() and accepting ALLOW
            # could erase an earlier REQUIRE_APPROVAL result, so recovery
            # constructs a fail-closed manual gate without invoking policy.
            #
            # Legacy unversioned rounds are treated the same way whenever a
            # registered call lacks a complete recognized decision. Absence of
            # a decision is not positive authorization. Unversioned rounds with
            # complete decisions retain those authoritative legacy outcomes.
            if (
                session.status
                not in {
                    SessionStatus.RUNNING,
                    SessionStatus.INTERRUPTING,
                }
                and not incomplete_recovery_claimed
            ):
                raise RuntimeError(
                    "Pending tool round has no durable policy plan; resume it under a "
                    "claimed run fence before recovering tool results."
                )
            replanned_tool_calls = [
                approval_support.tool_call_request_from_pending(call)
                for call in pending_round.tool_calls
            ]
            policy_plan = await self._tool_invocation.admission.fail_closed_recovery_policy_plan(
                session=session,
                registered_agent=registered_agent,
                tool_calls=replanned_tool_calls,
                request_metadata=pending_round.request_metadata,
                durable_tool_calls=(
                    pending_round.tool_calls
                    if pending_round.policy_context_version is None
                    else None
                ),
                tool_exposure=pending_round.tool_exposure,
            )
            if policy_plan.pending_approval is not None:
                approval_plan = policy_plan.pending_approval
                (
                    approval,
                    approval_events,
                ) = await self._tool_invocation.admission.pause_for_approval(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call=approval_plan.call,
                    tool_calls=approval_plan.calls,
                    policy_outcomes=approval_plan.policy_outcomes,
                    active_taint_by_id=policy_plan.active_taint_labels,
                    task_id=pending_round.task_id,
                    policy_result=approval_plan.policy_result,
                    structured_output=pending_round.structured_output,
                    thinking=pending_round.thinking,
                    max_steps=pending_round.max_steps,
                    limits=pending_round.limits,
                    budget_limits=pending_round.budget_limits,
                    retry_policy=pending_round.retry_policy,
                    tool_round_identity=pending_rounds.pending_tool_round_identity(pending_round),
                    deferred_messages=messages[insert_at:],
                    recovered=True,
                )
                for approval_event in approval_events:
                    yield approval_event
                raise ToolApprovalRequired(approval)
            pending_round = await self._tool_invocation.admission.checkpoint_tool_round_policy_plan(
                session=session,
                registered_agent=registered_agent,
                tool_calls=replanned_tool_calls,
                policy_outcomes=policy_plan.outcomes,
                active_taint_by_id=policy_plan.active_taint_labels,
                tool_round_identity=pending_rounds.pending_tool_round_identity(pending_round),
                recovered=True,
            )
            checkpoint = await self._session_store.load_checkpoint(session.id)
        approval_required_calls = [
            call
            for call in pending_round.tool_calls
            if pending_approval_reader.effective_tool_policy_evidence(call)
            is ToolPolicyEvidence.AUTHORITATIVE
            and call.policy_decision == ToolPolicyDecision.REQUIRE_APPROVAL.value
        ]
        if approval_required_calls:
            current_checkpoint = await self._session_store.load_checkpoint(session.id)
            paired_approval = pending_approval_reader.pending_approval_from_checkpoint(
                current_checkpoint,
                redactor=self._secret_redactor,
            )
            if paired_approval is None:
                if not (
                    session.status == SessionStatus.INTERRUPTING
                    and _approval_interrupt_close_intent_matches(
                        current_checkpoint,
                        pending_round=pending_round,
                    )
                ):
                    raise RuntimeError(
                        "Policy-planned REQUIRE_APPROVAL round has no matching pending approval."
                    )
            elif (
                paired_approval.tool_round_id != pending_round.tool_round_id
                or paired_approval.tool_call_id != approval_required_calls[0].tool_call_id
            ):
                raise RuntimeError(
                    "Policy-planned REQUIRE_APPROVAL round has no matching pending approval."
                )
            else:
                raise RuntimeError(
                    "Pending tool approval must be resolved before recovering its tool round."
                )
        (
            lifecycle_events,
            recorded_outcomes,
            effective_started_ids,
        ) = await self._recorded_tool_round_evidence(
            session=session,
            pending_round=pending_round,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
        )
        replay_identities = await self._select_tool_call_replay(
            session=session,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            pending_round=pending_round,
            recorded_outcomes=recorded_outcomes,
            effective_started_ids=effective_started_ids,
            incomplete_recovery_claimed=incomplete_recovery_claimed,
            elect_replay=elect_replay,
            invocation_context=invocation_context,
            admit_replay=admit_replay,
            lifecycle_events=lifecycle_events,
        )
        if replay_identities:
            assert invocation_context is not None and admit_replay is not None
            async for event in self._replay_tool_calls(
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                pending_round=pending_round,
                identities=replay_identities,
                admit_replay=admit_replay,
                invocation_context=invocation_context,
            ):
                yield event
            # The replayed terminals are staged on the round; publication below
            # closes it from them exactly as from any other staged outcome.
            checkpoint, pending_round = await pending_round_reader.load_pending_tool_round(
                self._session_store,
                session.id,
                redactor=self._secret_redactor,
                consume_on_rejection=True,
                runtime_session=session,
            )
            if (
                pending_round is None
                or pending_rounds.pending_tool_round_identity(pending_round) != tool_round_identity
            ):
                raise RuntimeError("Replayed tool round changed before its publication.")
            # Publication must judge the round by the evidence the replay added.
            (
                lifecycle_events,
                recorded_outcomes,
                effective_started_ids,
            ) = await self._recorded_tool_round_evidence(
                session=session,
                pending_round=pending_round,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
            )
        subagent_children: dict[str, Session | None] = {}
        subagent_recovery_checkpoint: dict[str, Any] | None = None
        if any(
            recorded_outcomes.get(call.tool_call_id) is None for call in pending_round.tool_calls
        ):
            subagent_children = await self.subagent_children_by_idempotency_key(session.id)
            subagent_recovery_checkpoint = await self._session_store.load_checkpoint(session.id)
        owner = DurableToolRound(
            session=session,
            tool_round_identity=tool_round_identity,
            session_store=self._session_store,
            event_writer=self._event_writer,
        )
        synthesized_outcomes, confirmed_native_effect_records = await owner.recover_outcomes(
            registered_agent=registered_agent,
            pending_round=pending_round,
            recorded_outcomes=recorded_outcomes,
            effective_started_ids=effective_started_ids,
            reconcile_call=partial(
                self._reconcile_recovered_tool_call,
                session=session,
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                environment_name=environment_name,
                pending_round=pending_round,
                execution_profile=execution_profile,
                effective_started_ids=effective_started_ids,
                subagent_children=subagent_children,
                subagent_recovery_checkpoint=subagent_recovery_checkpoint,
                ambiguous_interrupt_close_intent=ambiguous_interrupt_close_intent,
            ),
            redactor=self._secret_redactor,
        )
        async for event in self._publish_recovered_tool_outcomes(
            owner=owner,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            messages=messages,
            pending_round=pending_round,
            lifecycle_events=lifecycle_events,
            synthesized_outcomes=synthesized_outcomes,
            effective_started_ids=effective_started_ids,
            expected_transcript_cursor=expected_transcript_cursor,
            execution_profile=execution_profile,
            invocation_context=invocation_context,
            confirmed_native_effect_records=confirmed_native_effect_records,
        ):
            yield event

    async def _recorded_tool_round_evidence(
        self,
        *,
        session: Session,
        pending_round: pending_rounds.PendingToolRound,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
    ) -> tuple[list[Event], dict[str, runtime_records.ToolCallOutcome], set[str]]:
        """Return the round's lifecycle events, recorded outcomes and started calls."""

        lifecycle_events = await self.load_tool_round_lifecycle_events(
            session_id=session.id,
            pending_round=pending_round,
        )
        recorded_outcomes, started_ids = tool_round_recovery.recorded_tool_outcomes(
            events=lifecycle_events,
            pending_round=pending_round,
        )
        (
            isolated_dispatched_ids,
            isolated_call_ids,
        ) = await self.isolated_tool_dispatch_ids(
            session=session,
            pending_round=pending_round,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
        )
        effective_started_ids = (started_ids - isolated_call_ids) | isolated_dispatched_ids
        return lifecycle_events, recorded_outcomes, effective_started_ids

    async def _select_tool_call_replay(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        pending_round: pending_rounds.PendingToolRound,
        recorded_outcomes: Mapping[str, runtime_records.ToolCallOutcome | None],
        effective_started_ids: set[str],
        incomplete_recovery_claimed: bool,
        elect_replay: bool,
        invocation_context: InvocationContext | None,
        admit_replay: Callable[[ModelAttemptIdentity], Awaitable[bool]] | None,
        lifecycle_events: Sequence[Event],
    ) -> dict[str, tuple[str | None, str | None]]:
        """Elect a takeover's interrupted calls for replay, or start an elected replay.

        A takeover whose caller opted in keeps the round open and raises; the
        continuation that follows returns each call's original
        ``(approval_id, input_id)`` for the replay, which checks current policy per
        call and marks the record ``dispatched`` just before its first dispatch. Any
        other recovery, one that finds the replay already dispatched, or one that
        computes different calls than were elected closes the round as before.
        """
        from cayu.runtime._abandoned_session_recovery import abandoned_running_execution

        if not self._session_control.execution_presence.config.replay_interrupted_tool_calls:
            return {}
        if incomplete_recovery_claimed:
            if not elect_replay or not abandoned_running_execution(session):
                return {}
        elif (
            session.status is not SessionStatus.RUNNING
            or invocation_context is None
            or admit_replay is None
        ):
            return {}
        staged_ids = {
            staged.tool_call_id
            for staged in staged_terminal_reader.staged_terminal_records(pending_round)
        }
        unfinished_ids = {
            call.tool_call_id
            for call in pending_round.tool_calls
            if recorded_outcomes.get(call.tool_call_id) is None
            and call.tool_call_id not in staged_ids
        }
        publication = pending_round.assistant_publication
        call_ids = tool_call_replay.replayable_calls(
            pending_round=pending_round,
            registered_agent=registered_agent,
            secret_resolution_scope=invocation_secrets.continuation_secret_resolution_scope(
                "unknown" if publication is None else publication.secret_resolution_scope,
                registered_environment,
            ),
            unfinished_call_ids=unfinished_ids,
            started_call_ids=effective_started_ids,
        )
        if not call_ids:
            return {}
        identities = tool_call_replay.started_dispatch_identities(
            lifecycle_events,
            session_id=session.id,
            tool_round_id=pending_round.tool_round_id,
            call_ids=call_ids,
        )
        if identities is None:
            return {}
        if incomplete_recovery_claimed:
            record = await tool_call_replay.load_record(
                self._session_store, session, pending_round.tool_round_id
            )
            if record is not None and record[0] == "dispatched":
                return {}
            if record is None:
                await tool_call_replay.elect(
                    self._session_store, session, pending_round.tool_round_id, call_ids
                )
            raise tool_call_replay.ToolCallReplayRequired(
                "Interrupted NONE/IDEMPOTENT tool calls will be replayed by the continuation."
            )
        record = await tool_call_replay.load_record(
            self._session_store, session, pending_round.tool_round_id
        )
        if record is None or record[0] != "elected":
            return {}
        if record[1] != call_ids:
            # The round's evidence changed since the takeover elected it. Spend the
            # replay so no later takeover elects it again, and close the round.
            await tool_call_replay.begin_dispatch(
                self._session_store, session, pending_round.tool_round_id, record[1]
            )
            return {}
        return identities

    async def _replay_tool_calls(
        self,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        pending_round: pending_rounds.PendingToolRound,
        identities: Mapping[str, tuple[str | None, str | None]],
        admit_replay: Callable[[ModelAttemptIdentity], Awaitable[bool]],
        invocation_context: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        """Dispatch elected calls that current policy allows and stage their terminals.

        A call that current policy denies or would pause for approval is not replayed:
        its original attempt started and its outcome stays unknown, so the round
        closes it with the ordinary unknown-outcome result.
        """

        identity = pending_rounds.pending_tool_round_identity(pending_round)
        pending_by_id = {call.tool_call_id: call for call in pending_round.tool_calls}
        tool_calls = [
            approval_support.tool_call_request_from_pending(pending_by_id[call_id])
            for call_id in identities
        ]
        # Redaction and payload limits cover the whole round, as in any continuation,
        # so a replayed result is never redacted less than its siblings.
        round_tool_calls = [
            approval_support.tool_call_request_from_pending(call)
            for call in pending_round.tool_calls
        ]
        round_owner = DurableToolRound.for_continuation(
            session=session,
            tool_round_identity=identity,
            session_store=self._session_store,
            event_writer=self._event_writer,
            registered_agent=registered_agent,
            registered_environment=registered_environment,
            environment_name=environment_name,
            tool_calls=tool_calls,
            task_id=pending_round.task_id,
            execution_profile=invocation_context.profile,
            invocation_context=invocation_context,
            redactor=_redactor_for_tool_calls(
                self._secret_redactor,
                registered_agent=registered_agent,
                tool_calls=round_tool_calls,
            ),
            tool_exposure=pending_round.tool_exposure,
            publication_governor=self._tool_invocation.terminals.governor,
            clock=self._clock,
            emit_result=self._tool_invocation.terminals.publish_result,
            emit_terminal=self._tool_invocation.terminals.emit_staged,
            defer_terminals=True,
            terminal_payload_limits=await _tool_terminal_payload_limits(
                registered_agent,
                round_tool_calls,
                publication_governor=self._tool_invocation.terminals.governor,
                runtime_hooks=invocation_context.runtime_hooks,
            ),
            pause_authority={},
            idempotency_options={},
        )
        await round_owner.admit()
        # Calls whose tool body the replay actually entered.
        invoked_ids: set[str] = set()

        async def stage_replayed(
            event: Event,
            outcome: runtime_records.ToolCallOutcome,
            allow_modification: bool,
            publish_before_hooks: bool,
            snapshot: invocation_secrets.InvocationPublicationSnapshot,
        ) -> Event:
            # Mark only a terminal from a replay that ran the tool, not one that a
            # hook blocked or short-circuited or that dispatch checks refused.
            if event.payload.get("tool_call_id") in invoked_ids:
                event = event.model_copy(
                    update={"payload": {**event.payload, "replayed_after_recovery": True}}
                )
            return await round_owner.stage_terminal(
                event, outcome, allow_modification, publish_before_hooks, snapshot
            )

        call_ids = tuple(identities)
        dispatched = False
        try:
            for tool_call in tool_calls:
                await self._session_control.raise_if_interrupted(session.id)
                if not await admit_replay(
                    ModelAttemptIdentity(
                        model_step_id=identity.model_step_id,
                        model_attempt_id=identity.model_attempt_id,
                    )
                ):
                    break
                pending_call = pending_by_id[tool_call.id]
                decision = await self._tool_invocation.admission.authorize(
                    session=session,
                    registered_agent=registered_agent,
                    registered_environment=registered_environment,
                    tool_call=tool_call,
                    request_metadata=pending_round.request_metadata,
                    taint_labels=approval_support.taint_labels_from_pending_tool_call(pending_call),
                )
                if decision.decision is not ToolPolicyDecision.ALLOW:
                    continue
                if not dispatched:
                    # Spent only once a call is about to run: a crash from here on
                    # never replays again, while an earlier failure keeps the replay.
                    await tool_call_replay.begin_dispatch(
                        self._session_store, session, pending_round.tool_round_id, call_ids
                    )
                    dispatched = True
                approval_id, input_id = identities[tool_call.id]
                async for event, _outcome in round_owner.timed_continuation_dispatch(
                    self._tool_invocation.execute(
                        session=session,
                        registered_agent=registered_agent,
                        registered_environment=registered_environment,
                        tool_call=tool_call,
                        request_metadata=pending_round.request_metadata,
                        budget_limits=tuple(pending_round.budget_limits or ()),
                        task_id=pending_round.task_id,
                        model_step=pending_round.model_step,
                        execution_profile=invocation_context.profile,
                        invocation_context=invocation_context,
                        check_policy=True,
                        policy_result=decision,
                        emit_started=False,
                        approval_id=approval_id,
                        input_id=input_id,
                        tool_exposure=pending_round.tool_exposure,
                        policy_output_secret_resolution_scope="static",
                        tool_round_identity=identity,
                        taint_labels=approval_support.taint_labels_from_pending_tool_call(
                            pending_call
                        ),
                        deferred_terminal_stager=stage_replayed,
                        deferred_terminal_capture_recorder=round_owner.record_workspace_capture,
                        resolved_redactor_observer=round_owner.record_redactor,
                        publication_snapshot_observer=round_owner.record_publication_snapshot,
                        tool_invocation_observer=invoked_ids.add,
                    )
                ):
                    yield event
            if not dispatched:
                # Nothing was allowed to run; the round closes below, so spend it.
                await tool_call_replay.begin_dispatch(
                    self._session_store, session, pending_round.tool_round_id, call_ids
                )
        finally:
            round_owner.finish_dispatch()
            round_owner.finish_continuation_timing()

    async def _reconcile_recovered_tool_call(
        self,
        pending_tool_call: PendingToolCallApproval,
        tool_call: runtime_records.ToolCallRequest,
        *,
        session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        environment_name: str | None,
        pending_round: pending_rounds.PendingToolRound,
        execution_profile: ExecutionProfileIdentity | None,
        effective_started_ids: set[str],
        subagent_children: dict[str, Session | None],
        subagent_recovery_checkpoint: dict[str, Any] | None,
        ambiguous_interrupt_close_intent: bool,
    ) -> tuple[ToolResult | None, ToolEffectRecord | None]:
        """Resolve one missing call through existing native, effect and child authority.

        A missing result permits ordinary unknown-outcome synthesis only after
        external-effect fences have passed. This operation never dispatches a tool.
        """
        expected_idempotency_key = tool_execution.tool_idempotency_key(
            session_id=session.id,
            tool_round_id=pending_round.tool_round_id,
            tool_call_id=pending_tool_call.tool_call_id,
        )
        recovery_arguments = await self.subagent_recovery_arguments(
            checkpoint=subagent_recovery_checkpoint,
            parent_session=session,
            tool_name=pending_tool_call.tool_name,
            tool_round_id=pending_round.tool_round_id,
            tool_call_id=pending_tool_call.tool_call_id,
            idempotency_key=expected_idempotency_key,
            fallback=tool_call.arguments,
        )
        registered_tool = registered_agent.executable_tool(pending_tool_call.tool_name)
        effect_record = None
        if registered_tool is not None and registered_tool.effect is ToolEffect.EXTERNAL:
            effect_record = await ToolEffectStateOwner(self._session_store).resolve_call(
                session,
                tool_round_id=pending_round.tool_round_id,
                tool_call_id=pending_tool_call.tool_call_id,
            )
            if effect_record is None:
                if (
                    ambiguous_interrupt_close_intent
                    and pending_tool_call.tool_call_id not in effective_started_ids
                ):
                    # The exact approval-close intent is written by the
                    # approval-pause publisher before round dispatch. It
                    # positively proves this round was stopped at that gate;
                    # absence of a journal alone is never sufficient proof.
                    # A start or effect record would contradict that proof
                    # and must retain the ordinary reconciliation fence.
                    return (
                        tool_round_recovery.unknown_recovered_tool_result(
                            pending_tool_call=pending_tool_call,
                            pending_round=pending_round,
                            started=False,
                        ),
                        None,
                    )
                raise ToolEffectReconciliationRequired()
        result: ToolResult | None = None
        confirmed_effect_record: ToolEffectRecord | None = None
        if registered_tool is not None and registered_tool.durable_tool_recovery is not None:

            async def load_durable_tool_operation(
                storage_key: str,
            ) -> dict[str, Any] | None:
                return await self._session_store.load_session_operation(
                    session.id,
                    storage_key,
                )

            async def compare_and_set_durable_tool_operation(
                storage_key: str,
                expected: dict[str, Any] | None,
                desired: dict[str, Any],
                secondary_records: Mapping[str, dict[str, Any]],
            ) -> dict[str, Any]:
                expected_copy = (
                    None
                    if expected is None
                    else copy_durable_json_object(
                        expected,
                        "durable_tool_recovery.expected",
                    )
                )
                desired_copy = copy_durable_json_object(
                    desired,
                    "durable_tool_recovery.desired",
                )
                secondary_copy = {
                    key: copy_durable_json_object(
                        value,
                        f"durable_tool_recovery.secondary[{key!r}]",
                    )
                    for key, value in secondary_records.items()
                }
                if storage_key in secondary_copy:
                    raise ValueError("Durable tool recovery cannot duplicate its primary key.")

                def publish(
                    current_session: Session,
                    checkpoint: dict[str, Any] | None,
                    current: dict[str, Any] | None,
                ) -> SessionOperationPublication:
                    if (
                        current_session.id != session.id
                        or current_session.run_epoch != session.run_epoch
                    ):
                        raise SessionRunFenced(
                            "Durable tool recovery lost its parent run authority."
                        )
                    if current != expected_copy:
                        raise DurableToolOperationConflict(
                            "Durable tool recovery state changed before publication."
                        )
                    return SessionOperationPublication(
                        checkpoint={} if checkpoint is None else checkpoint,
                        operation_records={storage_key: desired_copy, **secondary_copy},
                    )

                await self._session_store.publish_session_operation(
                    session.id,
                    idempotency_key=storage_key,
                    operation_transform=publish,
                    events=[],
                    expected_statuses={session.status},
                    expected_run_epoch=session.run_epoch,
                )
                return copy_durable_json_object(
                    desired_copy,
                    "durable_tool_recovery.result",
                )

            recovery_artifact_store = (
                None
                if registered_environment is None
                else registered_environment.environment.artifact_store
            )
            recovery_runner = (
                None
                if registered_environment is None
                else registered_environment.environment.runner
            )
            runner_resource_identity, reconcile_runner_operation = (
                durable_runner_recovery_authority(recovery_runner)
            )
            recovery_authority = DurableToolRecoveryAuthority(
                agent_name=registered_agent.spec.name,
                environment_name=environment_name,
                workspace=(
                    None
                    if registered_environment is None
                    else registered_environment.environment.workspace
                ),
                artifact_reader=(
                    None
                    if recovery_artifact_store is None
                    else _DurableArtifactRecoveryReader(recovery_artifact_store)
                ),
                compare_and_set_operation=compare_and_set_durable_tool_operation,
                runner_resource_identity=runner_resource_identity,
                reconcile_runner_operation=reconcile_runner_operation,
                reconcile_runner_receipt=durable_runner_receipt_observer(
                    recovery_runner, redactor=self._secret_redactor
                ),
            )

            evidence = await registered_tool.durable_tool_recovery.reconcile_durable_tool_call(
                parent_session_id=session.id,
                parent_run_epoch=(
                    pending_round.source_run_epoch
                    if pending_round.source_run_epoch is not None
                    else session.run_epoch
                ),
                execution_profile_fingerprint=(
                    None if execution_profile is None else execution_profile.fingerprint
                ),
                environment_name=environment_name,
                environment_allocation_fingerprint=(
                    None
                    if registered_environment is None
                    else registered_environment.live_allocation_fingerprint
                ),
                model_step_id=pending_round.model_step_id,
                model_attempt_id=pending_round.model_attempt_id,
                tool_round_id=pending_round.tool_round_id,
                tool_call_id=pending_tool_call.tool_call_id,
                idempotency_key=expected_idempotency_key,
                arguments=copy_json_value(
                    tool_call.arguments,
                    "durable_tool_recovery.arguments",
                ),
                started=pending_tool_call.tool_call_id in effective_started_ids,
                load_operation=load_durable_tool_operation,
                recovery_authority=recovery_authority,
            )
            if evidence is not None:
                if type(evidence) is not DurableToolRecoveryEvidence:
                    raise TypeError("Durable tool recovery requires explicit typed evidence.")
                evidence = DurableToolRecoveryEvidence(evidence.disposition, evidence.result)
                if (
                    registered_tool.effect is ToolEffect.EXTERNAL
                    and evidence.disposition != "confirmed"
                ):
                    raise ToolEffectReconciliationRequired()
                result = evidence.result
                if effect_record is not None:
                    confirmed_effect_record = effect_record
        reconciled_result = None
        if result is None:
            reconciled_result = await self.reconcile_subagent_child(
                subagent_children,
                idempotency_key=expected_idempotency_key,
                tool_call_id=pending_tool_call.tool_call_id,
                tool_name=pending_tool_call.tool_name,
                tool_round_id=pending_round.tool_round_id,
                arguments=recovery_arguments,
                parent_session=session,
                registered_agent=registered_agent,
            )
            result = reconciled_result
            if result is not None and effect_record is not None:
                # The child recovery contract returns a ToolResult only for
                # a durably settled submission, not an unverified child.
                confirmed_effect_record = effect_record
        if result is None:
            result = await self.reattached_subagent_result(
                subagent_children,
                expected_idempotency_key,
                parent_checkpoint=subagent_recovery_checkpoint,
                tool_call_id=pending_tool_call.tool_call_id,
                tool_name=pending_tool_call.tool_name,
                tool_round_id=pending_round.tool_round_id,
                arguments=recovery_arguments,
                parent_session=session,
                registered_agent=registered_agent,
            )
            if result is not None and effect_record is not None:
                child = subagent_children.get(expected_idempotency_key)
                if (
                    child is None
                    or child.status not in tool_round_recovery._SUBAGENT_RECOVERY_TERMINAL_STATUSES
                ):
                    raise ToolEffectReconciliationRequired()
                confirmed_effect_record = effect_record
        if (
            result is None
            and registered_tool is not None
            and registered_tool.effect is ToolEffect.EXTERNAL
        ):
            # Neither a missing journal nor an unsuccessful lookup proves
            # an external call safe to synthesize or redispatch. Only the
            # typed journal confirmation above can supply its terminal.
            raise ToolEffectReconciliationRequired()
        return result, confirmed_effect_record

    async def _publish_recovered_tool_outcomes(
        self,
        *,
        owner: DurableToolRound,
        registered_agent: runtime_records.RegisteredAgentState,
        registered_environment: runtime_records.RegisteredEnvironment | None,
        messages: list[Message],
        pending_round: pending_rounds.PendingToolRound,
        lifecycle_events: list[Event],
        synthesized_outcomes: list[runtime_records.ToolCallOutcome],
        effective_started_ids: set[str],
        expected_transcript_cursor: int,
        execution_profile: ExecutionProfileIdentity | None,
        invocation_context: InvocationContext | None,
        interrupted: bool = False,
        confirmed_native_effect_records: Mapping[str, ToolEffectRecord] | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Supply recovery collaborators to the durable round publication owner."""
        async with contextlib.aclosing(
            owner.publish_recovered(
                registered_agent=registered_agent,
                registered_environment=registered_environment,
                messages=messages,
                pending_round=pending_round,
                lifecycle_events=lifecycle_events,
                synthesized_outcomes=synthesized_outcomes,
                effective_started_ids=effective_started_ids,
                expected_transcript_cursor=expected_transcript_cursor,
                execution_profile=execution_profile,
                invocation_context=invocation_context,
                redactor=self._secret_redactor,
                publication_governor=self._tool_invocation.terminals.governor,
                clock=self._clock,
                runtime_hooks=self._tool_invocation.hooks.registrations,
                emit_result=self._tool_invocation.terminals.publish_result,
                emit_terminal=self._tool_invocation.terminals.emit_staged,
                emit_native_terminal=self.emit_confirmed_native_tool_terminal,
                materialize_expected_deferred_input=self._deferred_input.materialize_expected,
                interrupted=interrupted,
                confirmed_native_effect_records=confirmed_native_effect_records,
            )
        ) as events:
            async for event in events:
                yield event

    async def subagent_children_by_idempotency_key(
        self,
        parent_session_id: str,
    ) -> dict[str, Session | None]:
        children: dict[str, Session | None] = {}
        sessions = await query_all_sessions(
            self._session_store,
            SessionQuery(
                parent_session_id=parent_session_id,
                order_by=SessionOrder.CREATED_AT_ASC,
            ),
        )
        for child in sessions:
            idempotency_key = tool_round_recovery.subagent_child_idempotency_key(child)
            if idempotency_key is not None:
                # Contradictory durable claims must not make recovery attach
                # whichever child happened to be listed last.
                children[idempotency_key] = None if idempotency_key in children else child
        return children

    async def emit_confirmed_native_tool_terminal(
        self, *, session: Session, record: ToolEffectRecord, event: Event
    ) -> Event:
        """Select native evidence and its terminal event in the existing effect transaction."""
        prepared = self._event_writer.prepare(event)
        await ToolEffectStateOwner(self._session_store).transition(
            record,
            state="failed" if prepared.type is EventType.TOOL_CALL_FAILED else "completed",
            run_epoch=session.run_epoch,
            terminal=ToolEffectTerminal(
                event_id=prepared.id,
                result_digest=sha256(
                    canonical_durable_json_bytes(prepared.payload["result"], "native_result")
                ).hexdigest(),
            ),
            events=(prepared,),
        )
        await self._event_writer.fan_out_persisted([prepared])
        return copy_event(prepared)

    async def subagent_recovery_arguments(
        self,
        *,
        checkpoint: dict[str, Any] | None,
        parent_session: Session,
        tool_name: str,
        tool_round_id: str,
        tool_call_id: str,
        idempotency_key: str,
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        """Restore post-hook arguments from the exact store-owned dispatched call."""

        effect = await ToolEffectStateOwner(self._session_store).resolve_call(
            parent_session, tool_round_id=tool_round_id, tool_call_id=tool_call_id
        )
        effect_arguments = None
        if effect is not None:
            if (
                effect.intent.tool_name != tool_name
                or effect.intent.idempotency_key != idempotency_key
            ):
                raise RuntimeError("Child recovery effect conflicts with its tool call.")
            effect_arguments = effect.child_recovery_arguments

        seed = durable_subagent_submission_seed_from_checkpoint(
            checkpoint,
            idempotency_key=idempotency_key,
        )
        if seed is None:
            intent = durable_subagent_submission_from_checkpoint(
                checkpoint,
                idempotency_key=idempotency_key,
            )
            if intent is not None:
                raise RuntimeError(
                    "Durable subagent submission intent has no effective-argument seed."
                )
            copied = copy_json_value(
                fallback if effect_arguments is None else effect_arguments,
                "subagent_recovery.arguments",
            )
            if type(copied) is not dict:
                raise TypeError("Subagent recovery arguments must be an object.")
            return copied
        if (
            seed.parent_session_id != parent_session.id
            or seed.parent_session_instance_fingerprint
            != _queued_dispatch_session_instance_fingerprint(parent_session)
            or seed.tool_name != tool_name
            or seed.tool_round_id != tool_round_id
            or seed.tool_call_id != tool_call_id
            or seed.idempotency_key != idempotency_key
        ):
            raise RuntimeError(
                "Durable subagent effective-argument authority conflicts with its tool call."
            )
        copied = copy_json_value(
            seed.effective_arguments,
            "durable_subagent_recovery.effective_arguments",
        )
        if type(copied) is not dict:
            raise AssertionError("Durable subagent effective arguments must be an object.")
        if effect_arguments is not None and copied != effect_arguments:
            raise RuntimeError("Child recovery argument authorities conflict.")
        return copied

    @staticmethod
    async def reconcile_subagent_child(
        children: dict[str, Session | None],
        *,
        idempotency_key: str,
        tool_call_id: str,
        tool_name: str,
        tool_round_id: str,
        arguments: dict[str, Any],
        parent_session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
    ) -> ToolResult | None:
        if idempotency_key in children and children[idempotency_key] is None:
            # Multiple children already claim this exact spawn identity. A
            # repair callback must not erase that contradiction by returning
            # whichever deterministic child it can load independently.
            return None
        registered_tool = registered_agent.tools.get(tool_name)
        if registered_tool is None or registered_tool.child_session_recovery is None:
            return None
        matcher = registered_tool.child_session_recovery
        reconciled = await matcher.reconcile_recoverable_child(
            children.get(idempotency_key),
            parent_session=parent_session,
            tool_name=tool_name,
            tool_round_id=tool_round_id,
            tool_call_id=tool_call_id,
            idempotency_key=idempotency_key,
            arguments=copy_json_value(arguments, "subagent_recovery.arguments"),
        )
        if reconciled is not None and type(reconciled) not in {Session, ToolResult}:
            raise TypeError(
                "Child-session reconciliation must return a Session, ToolResult, or None."
            )
        if type(reconciled) is ToolResult:
            return reconciled.model_copy(deep=True)
        if type(reconciled) is Session:
            existing = children.get(idempotency_key)
            if existing is not None and existing.id != reconciled.id:
                children[idempotency_key] = None
            else:
                children[idempotency_key] = reconciled
        return None

    async def reattached_subagent_result(
        self,
        children: dict[str, Session | None],
        idempotency_key: str,
        *,
        parent_checkpoint: dict[str, Any] | None,
        tool_call_id: str,
        tool_name: str,
        tool_round_id: str,
        arguments: dict[str, Any],
        parent_session: Session,
        registered_agent: runtime_records.RegisteredAgentState,
    ) -> ToolResult | None:
        child = children.get(idempotency_key)
        if child is None or not _matches_recoverable_subagent_child(
            child,
            idempotency_key=idempotency_key,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
            parent_session=parent_session,
            registered_agent=registered_agent,
        ):
            return None
        matcher = registered_agent.tools[tool_name].child_session_recovery
        assert matcher is not None
        subagent_metadata = child.metadata.get("subagent")
        child_action_pending = False
        from cayu.runtime._foreground_child_wait import owned_delegated_wait

        if (
            type(subagent_metadata) is dict
            and subagent_metadata.get("mode") == "foreground"
            and child.status == SessionStatus.INTERRUPTED
        ):
            from cayu.sessions.pending_actions import pending_action_evidence_round_from_checkpoint

            child_checkpoint = await self._session_store.load_checkpoint(child.id)
            child_action_pending = (
                await owned_delegated_wait(
                    self._session_store, child=child, checkpoint=child_checkpoint
                )
                is not None
            )
            # Validate the entire topology before deciding that an interrupted
            # child is terminal. Approval/input pauses remain child-owned and
            # cannot be converted into a recovered parent tool result.
            pending_action_evidence_round_from_checkpoint(child_checkpoint)
            child_action_pending = (
                child_action_pending
                or pending_approval_reader.pending_approval_from_checkpoint(child_checkpoint)
                is not None
                or user_input_lifecycle_authority_from_checkpoint(
                    child_checkpoint, current_run_epoch=child.run_epoch
                )[0]
                is not None
            )
        if type(subagent_metadata) is dict and subagent_metadata.get("mode") == "foreground":
            from cayu.runtime._foreground_child_wait import (
                ForegroundChildActionRequired,
                observe_foreground_child_wait,
                retain_foreground_child_wait,
            )

            effect = await ToolEffectStateOwner(self._session_store).resolve_call(
                parent_session, tool_round_id=tool_round_id, tool_call_id=tool_call_id
            )
            if effect is not None:
                recovered_wait = await observe_foreground_child_wait(
                    self._session_store,
                    parent=parent_session,
                    intent=effect.intent,
                    matcher=matcher,
                    arguments=arguments,
                )
                if recovered_wait is not None:
                    await retain_foreground_child_wait(
                        self._session_store,
                        parent=parent_session,
                        effect=effect,
                        wait=recovered_wait,
                    )
                    raise ForegroundChildActionRequired(recovered_wait)
        if (
            type(subagent_metadata) is dict
            and subagent_metadata.get("mode") == "foreground"
            and (
                child_action_pending
                or child.status
                in {
                    SessionStatus.PENDING,
                    SessionStatus.RUNNING,
                    SessionStatus.INTERRUPTING,
                }
            )
        ):
            raise ForegroundSubagentRecoveryRequired(
                child_session_id=child.id, tool_round_id=tool_round_id, tool_call_id=tool_call_id
            )
        from cayu.sessions._foreground_child_checkpoint import (
            foreground_child_state_from_checkpoint,
        )

        wait, selected = foreground_child_state_from_checkpoint(parent_checkpoint)
        if wait is not None and wait.parent_effect.idempotency_key == idempotency_key:
            if selected is None:
                raise ForegroundSubagentRecoveryRequired(
                    child_session_id=child.id,
                    tool_round_id=tool_round_id,
                    tool_call_id=tool_call_id,
                )
            effect = await ToolEffectStateOwner(self._session_store).resolve_call(
                parent_session, tool_round_id=tool_round_id, tool_call_id=tool_call_id
            )
            outcome = await self._session_store.summarize_outcome(child.id)
            terminal_event = (
                None if outcome.terminal_event is None else outcome.terminal_event.event
            )
            if (
                selected.wait != wait
                or effect is None
                or effect.intent != wait.parent_effect
                or child.id != wait.child_session_id
                or child.instance_id != wait.child_session_instance_id
                or child.run_epoch != selected.child_released_run_epoch
                or terminal_event is None
                or terminal_event.id != selected.event_id
                or terminal_event.type != selected.event_type
                or sha256(
                    canonical_durable_json_bytes(
                        terminal_event.model_dump(mode="json"), "foreground_child_terminal"
                    )
                ).hexdigest()
                != selected.event_digest
            ):
                raise RuntimeError("Foreground recovery conflicts with its selected child outcome.")
        from cayu.runtime._foreground_subagent_recovery import project_authenticated_child_result

        return await project_authenticated_child_result(
            matcher,
            child,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_round_id=tool_round_id,
        )
