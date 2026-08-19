from __future__ import annotations

import asyncio
import traceback
import warnings
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from tests.core._execution_profile_fixtures import profiled_session_identity
from tests.provider_traceback_assertions import is_cayu_source_filename

from cayu import SQLiteBudgetLedger, SQLiteSessionStore
from cayu.core import (
    AgentSpec,
    Event,
    EventType,
    Message,
    ThinkingConfig,
    ThinkingPart,
)
from cayu.core.billing import BillingIdentity
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.providers import (
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    ProviderOperationAdapter,
    ProviderOperationCancellationSupport,
    ProviderOperationConnection,
    ProviderOperationMode,
    ProviderOperationSnapshot,
    ProviderOperationStartIdempotencySupport,
    ProviderOperationStartRecoveryRequest,
    ProviderOperationStartRequest,
    ProviderOperationState,
    ProviderOperationStatus,
)
from cayu.runtime import (
    AllowAllToolPolicy,
    AlwaysRequireApprovalToolPolicy,
    BudgetLedger,
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    CayuApp,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    IncompleteSessionsRecoveryRequest,
    InMemoryBudgetLedger,
    InMemoryEventSink,
    InMemorySessionStore,
    InteractionStatus,
    InteractionSummaryEvidence,
    InterruptSessionRequest,
    ResolutionActor,
    ResumeRequest,
    RetryPolicy,
    RunLimits,
    RunRequest,
    RuntimeHook,
    RuntimeHookContext,
    SessionRunFenced,
    SessionStatus,
    SessionStatusConflict,
    SessionStore,
    ToolApprovalDecision,
    ToolApprovalRequest,
)
from cayu.runtime import _model_step_executor as model_step_executor
from cayu.runtime import _recovery_coordinator as recovery_coordinator_module
from cayu.runtime import _session_engine as session_engine_module
from cayu.runtime._model_errors import _BillingIdentityResolutionCancelled
from cayu.runtime._model_step_executor import ModelCompletionRecoveryContext
from cayu.runtime._recovery_coordinator import ModelCompletionManualRecoveryRequired
from cayu.runtime.budgets import (
    _effective_budget_limit_id,
    budget_settlement_event_id,
    budget_settlement_id,
    request_budget_limits_for_session,
)
from cayu.runtime.costs import ModelPrice, PriceBook
from cayu.runtime.execution_profiles import (
    ExecutionProfileIdentity,
    active_invocation_execution_profile_from_checkpoint,
    checkpoint_with_active_invocation_execution_profile,
)
from cayu.runtime.execution_units import ModelAttemptIdentity
from cayu.runtime.provider_operations import (
    ProviderOperationAccountingStatus,
    ProviderOperationCancellationStatus,
    ProviderOperationEvidenceError,
    ProviderOperationInspectionStatus,
    ProviderOperationPendingDisposition,
    ProviderOperationResolutionAction,
    ProviderOperationResolutionConflict,
    ProviderOperationResolutionRequest,
    ProviderOperationUnavailableReason,
    commit_provider_operation_progress,
    inspect_provider_operation,
    load_pending_provider_operation_disposition,
    load_provider_operation_resolution,
    provider_operation_progress_envelope,
    provider_operation_progress_event_id,
    provider_operation_resolution_outcome_event_id,
    resolve_provider_operation_stage,
)
from cayu.runtime.sessions import ModelCompletionStageRequest, _deactivate_session_run_fence
from cayu.runtime.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    StructuredOutputSpec,
)
from cayu.runtime.usage import SessionUsageSummary
from cayu.vaults import SecretRedactor


class _OfflineOperationAdapter(ProviderOperationAdapter):
    def __init__(
        self,
        status: ProviderOperationStatus,
        *,
        events: tuple[ModelStreamEvent, ...] | None = None,
    ) -> None:
        self.status = status
        self.state = ProviderOperationState(
            operation_id="response_offline_123",
            stream_protocol="responses-v1",
            recovery_metadata={"cursor": 0},
        )
        self.start_calls = 0
        self.retrieve_calls: list[ProviderOperationState] = []
        self.events = events
        self.start_events: tuple[ModelStreamEvent, ...] | None = None

    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        del request
        self.start_calls += 1
        if self.start_events is None:
            raise AssertionError("offline recovery must not submit a replacement operation")

        async def events() -> AsyncIterator[ModelStreamEvent]:
            if self.start_events is None:  # pragma: no cover - captured above
                raise AssertionError("start events disappeared")
            for cursor, event in enumerate(self.start_events, start=1):
                if event.recovery_metadata is None:
                    event = ModelStreamEvent.model_validate(
                        {
                            **event.model_dump(mode="python"),
                            "recovery_metadata": {"cursor": cursor},
                        }
                    )
                yield event

        return ProviderOperationConnection(
            state=ProviderOperationState(
                operation_id=f"response_resumed_{self.start_calls}",
                stream_protocol="responses-v1",
                recovery_metadata={"cursor": 0},
            ),
            status=ProviderOperationStatus.IN_PROGRESS,
            events=events(),
        )

    async def retrieve(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        self.retrieve_calls.append(state)
        events = self.events or (
            ModelStreamEvent.text_delta("finished while offline"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 3, "output_tokens": 4},
                }
            ),
        )
        return ProviderOperationSnapshot(
            state=self.state,
            status=self.status,
            events=events if self.status is ProviderOperationStatus.COMPLETED else (),
        )

    async def reconnect(self, state: ProviderOperationState) -> ProviderOperationConnection:
        raise AssertionError("offline terminal retrieval must not reconnect a stream")

    async def cancel(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        raise AssertionError("offline recovery must not cancel the operation")


class _SimulatedProcessLoss(BaseException):
    pass


class _BudgetedOfflineOperationAdapter(_OfflineOperationAdapter):
    def __init__(self) -> None:
        super().__init__(ProviderOperationStatus.IN_PROGRESS)

    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        del request
        self.start_calls += 1

        async def events() -> AsyncIterator[ModelStreamEvent]:
            raise _SimulatedProcessLoss("worker lost after provider operation publication")
            yield  # pragma: no cover - keeps this an async generator

        return ProviderOperationConnection(
            state=self.state,
            status=ProviderOperationStatus.IN_PROGRESS,
            events=events(),
        )


class _OfflineOperationProvider(ModelProvider):
    name = "offline-operation"

    def __init__(
        self,
        status: ProviderOperationStatus,
        *,
        events: tuple[ModelStreamEvent, ...] | None = None,
    ) -> None:
        self.adapter = _OfflineOperationAdapter(status, events=events)

    @property
    def provider_operation_mode(self) -> ProviderOperationMode:
        return ProviderOperationMode.BACKGROUND

    @property
    def provider_operations(self) -> ProviderOperationAdapter:
        return self.adapter

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        raise AssertionError("offline recovery must not use the synchronous stream")
        yield  # pragma: no cover


class _FailOnceFallbackBillingProvider(_OfflineOperationProvider):
    name = "fail-once-fallback-billing"

    def __init__(self) -> None:
        super().__init__(ProviderOperationStatus.UNAVAILABLE)
        self.billing_calls = 0

    async def billing_identity_for_request(
        self,
        request: ModelRequest,
    ) -> BillingIdentity | None:
        del request
        self.billing_calls += 1
        if self.billing_calls == 1:
            raise RuntimeError("fallback billing lookup failed before provider dispatch")
        return None


class _BlockingFallbackBillingProvider(_OfflineOperationProvider):
    name = "blocking-fallback-billing"

    def __init__(self) -> None:
        super().__init__(ProviderOperationStatus.UNAVAILABLE)
        self.billing_calls = 0
        self.billing_entered = asyncio.Event()

    async def billing_identity_for_request(
        self,
        request: ModelRequest,
    ) -> BillingIdentity | None:
        del request
        self.billing_calls += 1
        if self.billing_calls == 1:
            self.billing_entered.set()
            await asyncio.Event().wait()
        return None


class _BudgetedOfflineOperationProvider(_OfflineOperationProvider):
    name = "budgeted-offline-operation"

    def __init__(self) -> None:
        self.adapter = _BudgetedOfflineOperationAdapter()


class _IdempotentAmbiguousStartAdapter(_OfflineOperationAdapter):
    def __init__(self) -> None:
        super().__init__(ProviderOperationStatus.COMPLETED)
        self.start_keys: list[str] = []
        self.recovery_keys: list[str] = []

    @property
    def start_idempotency_support(self) -> ProviderOperationStartIdempotencySupport:
        return ProviderOperationStartIdempotencySupport.EXACT

    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        self.start_calls += 1
        self.start_keys.append(request.idempotency_key)
        raise _SimulatedProcessLoss(
            "worker lost after provider acceptance but before identity publication"
        )

    async def recover_start(
        self,
        request: ProviderOperationStartRecoveryRequest,
    ) -> ProviderOperationConnection:
        self.recovery_keys.append(request.idempotency_key)

        async def events() -> AsyncIterator[ModelStreamEvent]:
            if False:  # pragma: no cover - defines an empty async iterator
                yield ModelStreamEvent.text_delta("")

        return ProviderOperationConnection(
            state=self.state,
            status=ProviderOperationStatus.COMPLETED,
            events=events(),
        )


class _IdempotentAmbiguousStartProvider(_OfflineOperationProvider):
    name = "idempotent-ambiguous-start"

    def __init__(self) -> None:
        self.adapter = _IdempotentAmbiguousStartAdapter()


class _UnsupportedAmbiguousStartAdapter(_OfflineOperationAdapter):
    def __init__(self) -> None:
        super().__init__(ProviderOperationStatus.IN_PROGRESS)

    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        if self.start_calls == 0:
            self.start_calls += 1
            raise _SimulatedProcessLoss(
                "worker lost after an unsupported start may have reached the provider"
            )
        return await _OfflineOperationAdapter.start(self, request)


class _UnsupportedAmbiguousStartProvider(_OfflineOperationProvider):
    name = "unsupported-ambiguous-start"

    def __init__(self) -> None:
        self.adapter = _UnsupportedAmbiguousStartAdapter()


class _ResumableBudgetedOfflineOperationAdapter(_BudgetedOfflineOperationAdapter):
    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        if self.start_calls == 0:
            return await super().start(request)
        return await _OfflineOperationAdapter.start(self, request)


class _ResumableBudgetedOfflineOperationProvider(_OfflineOperationProvider):
    name = "resumable-budgeted-offline-operation"

    def __init__(self) -> None:
        self.adapter = _ResumableBudgetedOfflineOperationAdapter()


class _LoseFirstRecoverySettlementAcknowledgement(InMemoryBudgetLedger):
    def __init__(self) -> None:
        super().__init__(reservation_ttl_seconds=None)
        self.reconcile_calls = 0

    async def reconcile(self, **kwargs):
        reconciliation = await super().reconcile(**kwargs)
        self.reconcile_calls += 1
        if self.reconcile_calls == 1:
            raise RuntimeError("provider recovery settlement acknowledgement lost after commit")
        return reconciliation


class _BlockingReservationLoadLedger(InMemoryBudgetLedger):
    def __init__(self) -> None:
        super().__init__(reservation_ttl_seconds=None)
        self.load_entered = asyncio.Event()
        self.load_release = asyncio.Event()

    async def load_reservation(self, reservation_id: str):
        self.load_entered.set()
        await self.load_release.wait()
        return await super().load_reservation(reservation_id)


class _ConcurrentResolutionReservationLoadLedger(InMemoryBudgetLedger):
    def __init__(self) -> None:
        super().__init__(reservation_ttl_seconds=None)
        self._remaining_blocked_loads = 0
        self.concurrent_loads_entered = asyncio.Event()
        self.concurrent_loads_release = asyncio.Event()

    def block_next_two_loads(self) -> None:
        self._remaining_blocked_loads = 2

    async def load_reservation(self, reservation_id: str):
        if self._remaining_blocked_loads > 0:
            self._remaining_blocked_loads -= 1
            if self._remaining_blocked_loads == 0:
                self.concurrent_loads_entered.set()
            await self.concurrent_loads_release.wait()
        return await super().load_reservation(reservation_id)


class _BlockingReconciliationLedger(InMemoryBudgetLedger):
    def __init__(self) -> None:
        super().__init__(reservation_ttl_seconds=None)
        self.reconcile_entered = asyncio.Event()
        self.reconcile_release = asyncio.Event()

    async def reconcile(self, **kwargs):
        self.reconcile_entered.set()
        await self.reconcile_release.wait()
        return await super().reconcile(**kwargs)


class _CancellableBudgetedOfflineOperationAdapter(_BudgetedOfflineOperationAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_calls: list[ProviderOperationState] = []

    @property
    def cancellation_support(self) -> ProviderOperationCancellationSupport:
        return ProviderOperationCancellationSupport.SUPPORTED

    async def cancel(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        self.cancel_calls.append(state)
        self.status = ProviderOperationStatus.CANCELLED
        return ProviderOperationSnapshot(state=state, status=self.status)


class _CancellableBudgetedOfflineOperationProvider(_OfflineOperationProvider):
    name = "cancellable-budgeted-offline-operation"

    def __init__(self) -> None:
        self.adapter = _CancellableBudgetedOfflineOperationAdapter()


class _LiveCancellableBudgetedOperationAdapter(_CancellableBudgetedOfflineOperationAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.stream_started = asyncio.Event()

    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        del request
        self.start_calls += 1

        async def events() -> AsyncIterator[ModelStreamEvent]:
            self.stream_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")
            yield  # pragma: no cover - keeps this an async generator

        return ProviderOperationConnection(
            state=self.state,
            status=ProviderOperationStatus.IN_PROGRESS,
            events=events(),
        )


class _LiveCancellableBudgetedOperationProvider(_OfflineOperationProvider):
    name = "live-cancellable-budgeted-operation"

    def __init__(self) -> None:
        self.adapter = _LiveCancellableBudgetedOperationAdapter()


class _CancellableOfflineOperationAdapter(_OfflineOperationAdapter):
    def __init__(self) -> None:
        super().__init__(ProviderOperationStatus.IN_PROGRESS)
        self.cancel_calls: list[ProviderOperationState] = []

    @property
    def cancellation_support(self) -> ProviderOperationCancellationSupport:
        return ProviderOperationCancellationSupport.SUPPORTED

    async def cancel(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        self.cancel_calls.append(state)
        self.status = ProviderOperationStatus.CANCELLED
        return ProviderOperationSnapshot(state=state, status=self.status)


class _CancellableOfflineOperationProvider(_OfflineOperationProvider):
    name = "cancellable-offline-operation"

    def __init__(self) -> None:
        self.adapter = _CancellableOfflineOperationAdapter()


class _BlockingCancellationAdapter(_CancellableOfflineOperationAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.cancel_entered = asyncio.Event()
        self.cancel_release = asyncio.Event()
        self.cancel_exited = asyncio.Event()

    async def cancel(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        self.cancel_calls.append(state)
        self.cancel_entered.set()
        try:
            await self.cancel_release.wait()
            self.status = ProviderOperationStatus.CANCELLED
            return ProviderOperationSnapshot(state=state, status=self.status)
        finally:
            self.cancel_exited.set()


class _BlockingCancellationProvider(_OfflineOperationProvider):
    name = "blocking-cancellation"

    def __init__(self) -> None:
        self.adapter = _BlockingCancellationAdapter()


class _CompletionWinsCancellationAdapter(_CancellableOfflineOperationAdapter):
    async def cancel(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        self.cancel_calls.append(state)
        self.status = ProviderOperationStatus.COMPLETED
        return await self.retrieve(state)


class _CompletionWinsCancellationProvider(_OfflineOperationProvider):
    name = "completion-wins-cancellation"

    def __init__(self) -> None:
        self.adapter = _CompletionWinsCancellationAdapter()


class _LostCancellationAcknowledgementAdapter(_CancellableOfflineOperationAdapter):
    async def cancel(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        self.cancel_calls.append(state)
        self.status = ProviderOperationStatus.CANCELLED
        raise TimeoutError("provider cancelled but acknowledgement was lost")


class _LostCancellationAcknowledgementProvider(_OfflineOperationProvider):
    name = "lost-cancellation-acknowledgement"

    def __init__(self) -> None:
        self.adapter = _LostCancellationAcknowledgementAdapter()


class _FenceOnCancellationRequestStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.fenced = False

    async def publish_checkpoint_and_events(self, session_id: str, **kwargs):
        events = kwargs["events"]
        if (
            any(event.type is EventType.PROVIDER_OPERATION_CANCEL_REQUESTED for event in events)
            and not self.fenced
        ):
            self.fenced = True
            fenced = await self.fence_stalled_run(
                session_id,
                statuses={SessionStatus.INTERRUPTING},
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
            assert fenced is not None
        return await super().publish_checkpoint_and_events(session_id, **kwargs)


class _CommitThenRaiseInterruptionClaimStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.interruption_claim_committed = False

    async def fence_run_and_transform_checkpoint(self, session_id: str, **kwargs):
        result = await super().fence_run_and_transform_checkpoint(session_id, **kwargs)
        if not self.interruption_claim_committed and kwargs["statuses"] == {
            SessionStatus.INTERRUPTING
        }:
            self.interruption_claim_committed = True
            raise ConnectionError("interruption claim acknowledgement lost after commit")
        return result


class _CrashAfterCancellationEventStore(InMemorySessionStore):
    def __init__(self, crash_after: EventType) -> None:
        super().__init__()
        self.crash_after = crash_after
        self.crashed = False

    async def publish_checkpoint_and_events(self, session_id: str, **kwargs):
        result = await super().publish_checkpoint_and_events(session_id, **kwargs)
        if not self.crashed and any(event.type is self.crash_after for event in kwargs["events"]):
            self.crashed = True
            raise _SimulatedProcessLoss("worker died after durable cancellation claim")
        return result


class _FailCancellationClaimHeartbeatStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.fail_heartbeat = False
        self.heartbeat_failed = asyncio.Event()

    async def publish_checkpoint_and_events(self, session_id: str, **kwargs):
        events = kwargs["events"]
        if any(event.type is EventType.PROVIDER_OPERATION_CANCEL_REQUESTED for event in events):
            self.fail_heartbeat = True
        elif self.fail_heartbeat and not events:
            self.fail_heartbeat = False
            self.heartbeat_failed.set()
            raise OSError("transient cancellation-claim heartbeat failure")
        return await super().publish_checkpoint_and_events(session_id, **kwargs)


class _DelayCancellationResolutionAcknowledgementStore(InMemorySessionStore):
    async def publish_checkpoint_and_events(self, session_id: str, **kwargs):
        result = await super().publish_checkpoint_and_events(session_id, **kwargs)
        if any(
            event.type is EventType.PROVIDER_OPERATION_CANCEL_RESOLVED for event in kwargs["events"]
        ):
            await asyncio.sleep(0.03)
        return result


class _BlockingInterruptionTransitionStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.transition_entered = asyncio.Event()
        self.transition_release = asyncio.Event()

    async def transition_status_and_checkpoint(self, session_id: str, **kwargs):
        self.transition_entered.set()
        await self.transition_release.wait()
        return await super().transition_status_and_checkpoint(session_id, **kwargs)


class _RecordingInterruptedProfileHook(RuntimeHook):
    name = "offline-interruption-profile-hook"

    def __init__(self) -> None:
        self.execution_profiles: list[ExecutionProfileIdentity | None] = []

    async def after_session_interrupted(self, context: RuntimeHookContext) -> None:
        self.execution_profiles.append(context.execution_profile)


class _IdentityAwareOfflineProvider(_OfflineOperationProvider):
    def __init__(self, identity: BillingIdentity) -> None:
        super().__init__(ProviderOperationStatus.COMPLETED)
        self.identity = identity

    def billing_identity_for_completion(
        self,
        identity: BillingIdentity | None,
        payload: dict,
    ) -> BillingIdentity | None:
        del payload
        assert identity == self.identity
        return identity


class _LookupTool(Tool):
    spec = ToolSpec(
        name="lookup",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        effect=ToolEffect.NONE,
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx, args
        return ToolResult(content="lookup complete")


class _RecordingLookupTool(_LookupTool):
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx
        self.calls.append(args)
        return ToolResult(content="lookup complete")


def _budget_policy(provider_name: str) -> BudgetPolicy:
    return BudgetPolicy(
        limits=(
            BudgetLimit(
                scope="app",
                max_estimated_cost=Decimal("10"),
                pricing=PriceBook(
                    prices=(
                        ModelPrice.fixed(
                            provider_name=provider_name,
                            model="fake-model",
                            input_per_million=Decimal("1"),
                            output_per_million=Decimal("2"),
                        ),
                    )
                ),
                reservation=BudgetReservation(
                    max_input_tokens=1_000_000,
                    max_output_tokens=1_000_000,
                ),
            ),
        )
    )


async def _stage_offline_operation(
    store: SessionStore,
    *,
    session_id: str,
    provider: _OfflineOperationProvider,
    recovery_context: dict | None = None,
    started_at: datetime | None = None,
    prior_events: tuple[Event, ...] = (),
    tools: tuple[Tool, ...] = (),
    step: int = 1,
) -> Message:
    user_message = Message.text("user", "finish this while no worker is attached")
    interaction_id = f"interaction-{session_id}"
    started_event_id = f"{session_id}:interaction-started"
    started_at = datetime.now(UTC) if started_at is None else started_at
    started_event = Event(
        id=started_event_id,
        type=EventType.INTERACTION_STARTED,
        session_id=session_id,
        interaction_id=interaction_id,
        timestamp=started_at,
        agent_name="assistant",
        payload=InteractionSummaryEvidence(
            status=InteractionStatus.ACTIVE,
            start_event_id=started_event_id,
            started_at=started_at,
        ).model_dump(mode="json"),
    )
    session_identity = profiled_session_identity(
        provider_name=provider.name,
        model="fake-model",
        direct_tools=(
            {
                "name": tool.name,
                "description": tool.description,
                "schema": tool.schema,
                "parallel_safe": tool.spec.parallel_safe,
                "effect": tool.spec.effect.value,
            }
            for tool in tools
        ),
    )
    execution_profile = session_identity.execution_profile
    assert execution_profile is not None
    session = await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[user_message],
        ),
        identity=session_identity,
        interaction_started_event=started_event,
        interaction_source_messages=[user_message],
        checkpoint_transform=lambda current_session, checkpoint: (
            checkpoint_with_active_invocation_execution_profile(
                checkpoint,
                session_id=current_session.id,
                interaction_id=interaction_id,
                run_epoch=current_session.run_epoch,
                profile=execution_profile,
            )
        ),
    )
    await store.replace_initial_transcript_messages(
        session_id,
        [user_message],
        [user_message],
        interaction_id=interaction_id,
    )
    if prior_events:
        await store.append_events(session_id, list(prior_events))
    identity = ModelAttemptIdentity(
        model_step_id="mstep_" + "a" * 32,
        model_attempt_id="matt_" + "b" * 32,
    )
    stage_id = f"{identity.model_step_id}:dispatch:0"
    intent = {
        "schema_version": 1,
        "purpose": "assistant-turn",
        **identity.payload(),
        "logical_step_id": identity.model_step_id,
        "provider_name": provider.name,
        "requested_model": "fake-model",
        "source_transcript_cursor": 1,
        "request_fingerprint": "c" * 64,
    }
    typed_recovery_context = ModelCompletionRecoveryContext.model_validate(
        {} if recovery_context is None else recovery_context
    ).model_copy(update={"execution_profile_fingerprint": execution_profile.fingerprint})
    intent["recovery_context"] = typed_recovery_context.model_dump(mode="json")
    await store.prepare_model_completion_stage(
        session_id,
        request=ModelCompletionStageRequest(
            stage_id=stage_id,
            logical_step_id=identity.model_step_id,
            dispatch_ordinal=0,
            intent=intent,
            reservation_ids=(),
        ),
        expected_statuses={session.status},
        expected_run_epoch=session.run_epoch,
        expected_transcript_cursor=1,
    )
    await store.append_events(
        session_id,
        [
            Event(
                type=EventType.MODEL_STARTED,
                session_id=session_id,
                interaction_id=interaction_id,
                agent_name="assistant",
                payload={
                    "provider": provider.name,
                    "model": "fake-model",
                    "step": step,
                    "attempt": 1,
                    "max_attempts": 1,
                    **identity.payload(),
                },
            ),
            Event(
                type=EventType.PROVIDER_OPERATION_STARTED,
                session_id=session_id,
                interaction_id=interaction_id,
                agent_name="assistant",
                payload={
                    "provider": provider.name,
                    "model": "fake-model",
                    "step": step,
                    "attempt": 1,
                    "max_attempts": 1,
                    **identity.payload(),
                    "source_run_epoch": session.run_epoch,
                    "start_id": f"provider-operation:{identity.model_attempt_id}",
                    "state_version": provider.adapter.state.version,
                    "operation_id": provider.adapter.state.operation_id,
                    "stream_protocol": provider.adapter.state.stream_protocol,
                    "status": ProviderOperationStatus.IN_PROGRESS.value,
                    "recovery_metadata": {"cursor": 0},
                },
            ),
        ],
    )
    await store.release_run_fence(session_id)
    return user_message


async def _prepare_explicit_fallback_resolution(
    store: SessionStore,
    *,
    session_id: str,
    provider: _OfflineOperationProvider,
    recovery_context: dict | None = None,
) -> tuple[CayuApp, ProviderOperationResolutionRequest]:
    await _stage_offline_operation(
        store,
        session_id=session_id,
        provider=provider,
        recovery_context=recovery_context,
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    await app.recover_incomplete_session(
        IncompleteSessionRecoveryRequest(
            session_id=session_id,
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )
    )
    interrupted = await store.load(session_id)
    active = await store.load_active_model_completion_stage(session_id)
    assert interrupted is not None
    assert interrupted.status is SessionStatus.INTERRUPTED
    assert active is not None
    return app, ProviderOperationResolutionRequest(
        session_id=session_id,
        stage_id=active.stage.stage_id,
        expected_run_epoch=interrupted.run_epoch,
        action=ProviderOperationResolutionAction.FALLBACK_RETRY,
        reason="provider operation unavailable; accept explicit fallback",
    )


@pytest.mark.parametrize(
    "field_name",
    [
        "resolved_run_epoch",
        "source_step",
        "source_dispatch_ordinal",
        "target_dispatch_ordinal",
    ],
)
def test_provider_operation_resolution_rejects_boolean_epoch_and_disposition_authority(
    field_name: str,
) -> None:
    with pytest.raises(ValueError):
        ProviderOperationResolutionRequest(
            session_id="strict-resolution-session",
            stage_id="strict-resolution-stage",
            expected_run_epoch=True,
            action=ProviderOperationResolutionAction.FAIL,
        )

    pending = {
        "session_id": "strict-resolution-session",
        "stage_id": "strict-resolution-stage",
        "resolution_id": "strict-resolution-id",
        "request_digest": "a" * 64,
        "action": ProviderOperationResolutionAction.FALLBACK_RETRY,
        "resolved_run_epoch": 1,
        "logical_step_id": "strict-model-step",
        "source_step": 1,
        "source_dispatch_ordinal": 0,
        "target_dispatch_ordinal": 1,
        "execution_profile_fingerprint": "b" * 64,
    }
    pending[field_name] = True
    with pytest.raises(ValueError):
        ProviderOperationPendingDisposition.model_validate(pending)


@pytest.mark.parametrize(
    "field_name",
    ["preparation_digest", "model_attempt_id", "source_run_epoch"],
)
def test_pending_provider_resolution_rejects_conflicting_source_stage(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        session_id = f"conflicting-resolution-source-{field_name}"
        provider = _OfflineOperationProvider(ProviderOperationStatus.UNAVAILABLE)
        await _stage_offline_operation(store, session_id=session_id, provider=provider)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )
        interrupted = await store.load(session_id)
        active = await store.load_active_model_completion_stage(session_id)
        assert interrupted is not None
        assert active is not None
        await resolve_provider_operation_stage(
            store,
            ProviderOperationResolutionRequest(
                session_id=session_id,
                stage_id=active.stage.stage_id,
                expected_run_epoch=interrupted.run_epoch,
                action=ProviderOperationResolutionAction.FALLBACK_RETRY,
            ),
            redactor=SecretRedactor(),
        )

        original_load = store.load_model_completion_stage

        async def load_conflicting_stage(loaded_session_id: str, stage_id: str):
            stage = await original_load(loaded_session_id, stage_id)
            assert stage is not None
            if field_name == "model_attempt_id":
                return stage.model_copy(
                    update={
                        "intent": {
                            **stage.intent,
                            "model_attempt_id": "matt_" + "c" * 32,
                        }
                    }
                )
            if field_name == "preparation_digest":
                return stage.model_copy(update={"preparation_digest": "d" * 64})
            return stage.model_copy(update={"source_run_epoch": stage.source_run_epoch + 1})

        monkeypatch.setattr(store, "load_model_completion_stage", load_conflicting_stage)
        with pytest.raises(
            ProviderOperationEvidenceError,
            match="conflicts with its source stage",
        ):
            await load_pending_provider_operation_disposition(store, session_id)

    asyncio.run(scenario())


def test_pending_fail_resolution_rejects_conflicting_terminal_event() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        session_id = "conflicting-resolution-terminal"
        provider = _OfflineOperationProvider(ProviderOperationStatus.UNAVAILABLE)
        await _stage_offline_operation(store, session_id=session_id, provider=provider)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )
        interrupted = await store.load(session_id)
        active = await store.load_active_model_completion_stage(session_id)
        assert interrupted is not None
        assert active is not None
        request = ProviderOperationResolutionRequest(
            session_id=session_id,
            stage_id=active.stage.stage_id,
            expected_run_epoch=interrupted.run_epoch,
            action=ProviderOperationResolutionAction.FAIL,
        )
        accepted = await resolve_provider_operation_stage(
            store,
            request,
            redactor=SecretRedactor(),
        )
        await store.append_event(
            session_id,
            Event(
                id=provider_operation_resolution_outcome_event_id(
                    accepted.record.resolution_id,
                    "session_failed",
                ),
                type="custom.provider-resolution-terminal-collision",
                session_id=session_id,
            ),
        )

        with pytest.raises(
            ProviderOperationEvidenceError,
            match="contradictory event identity",
        ):
            _ = [event async for event in app.resolve_provider_operation(request)]
        assert await load_pending_provider_operation_disposition(store, session_id) is not None
        unchanged = await store.load(session_id)
        assert unchanged is not None
        assert unchanged.status is SessionStatus.INTERRUPTED
        assert provider.adapter.start_calls == 0

    asyncio.run(scenario())


async def stage_provider_resolution_process_loss(
    store: SessionStore,
    *,
    action: ProviderOperationResolutionAction,
    after_status_transition: bool,
) -> tuple[str, _OfflineOperationProvider]:
    """Accept a disposition, then stop at one durable crash boundary."""

    crash_point = "after-transition" if after_status_transition else "after-commit"
    session_id = f"provider-resolution-{action.value}-{crash_point}"
    provider = _OfflineOperationProvider(ProviderOperationStatus.UNAVAILABLE)
    await _stage_offline_operation(store, session_id=session_id, provider=provider)
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    await app.recover_incomplete_session(
        IncompleteSessionRecoveryRequest(
            session_id=session_id,
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )
    )
    interrupted = await store.load(session_id)
    active = await store.load_active_model_completion_stage(session_id)
    assert interrupted is not None
    assert interrupted.status is SessionStatus.INTERRUPTED
    assert active is not None
    request = ProviderOperationResolutionRequest(
        session_id=session_id,
        stage_id=active.stage.stage_id,
        expected_run_epoch=interrupted.run_epoch,
        action=action,
        reason=f"simulate process loss {crash_point}",
    )

    if not after_status_transition:
        await resolve_provider_operation_stage(
            store,
            request,
            redactor=SecretRedactor(),
        )
    else:
        stream = app.resolve_provider_operation(request)
        stop_event = (
            EventType.INTERACTION_RESUMED
            if action is ProviderOperationResolutionAction.FALLBACK_RETRY
            else EventType.INTERACTION_FAILED
        )
        while True:
            event = await anext(stream)
            if event.type is stop_event:
                break
        await stream.aclose()
        _deactivate_session_run_fence(session_id)

    pending = await load_pending_provider_operation_disposition(store, session_id)
    assert pending is not None
    await store.transform_checkpoint(
        session_id,
        session_engine_module._replace_checkpoint_preserving_runtime_state(
            {
                "provider_operation_pending_resolution_disposition": {"record_type": "stale"},
                "provider_operation_fallback_dispatch_ordinals": {"stale": 99},
                "application_state": {"retained": True},
            }
        ),
    )
    assert await load_pending_provider_operation_disposition(store, session_id) == pending
    checkpoint = await store.load_checkpoint(session_id)
    assert checkpoint is not None
    assert checkpoint["application_state"] == {"retained": True}
    ordinals = checkpoint.get("provider_operation_fallback_dispatch_ordinals")
    if action is ProviderOperationResolutionAction.FALLBACK_RETRY:
        assert ordinals == {pending[0].logical_step_id: pending[0].target_dispatch_ordinal}
    else:
        assert ordinals is None
    return session_id, provider


async def assert_provider_resolution_process_loss_recovery(
    store: SessionStore,
    *,
    session_id: str,
    provider: _OfflineOperationProvider,
    action: ProviderOperationResolutionAction,
) -> None:
    """Prove a fresh coordinator finishes the accepted disposition exactly once."""

    if action is ProviderOperationResolutionAction.FALLBACK_RETRY:
        provider.adapter.start_events = (
            ModelStreamEvent.text_delta("recovered accepted fallback"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    before_recovery = await store.load(session_id)
    assert before_recovery is not None
    if before_recovery.status in {
        SessionStatus.INTERRUPTED,
        SessionStatus.FAILED,
        SessionStatus.COMPLETED,
    }:
        with pytest.raises(
            RuntimeError,
            match="accepted provider-operation resolution pending",
        ):
            _ = [
                event
                async for event in app.resume(
                    ResumeRequest(
                        session_id=session_id,
                        messages=[Message.text("user", "do not bypass the accepted resolution")],
                    )
                )
            ]
        assert await load_pending_provider_operation_disposition(store, session_id) is not None
        assert provider.adapter.start_calls == 0
    recovered_page = await app.recover_incomplete_sessions(
        IncompleteSessionsRecoveryRequest(
            statuses={
                SessionStatus.INTERRUPTED,
                SessionStatus.RUNNING,
                SessionStatus.FAILED,
            },
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )
    )
    recovered = next(result for result in recovered_page.results if result.session_id == session_id)

    assert (
        IncompleteSessionRecoveryAction.REPAIRED_PROVIDER_OPERATION_RESOLUTION in recovered.actions
    )
    assert await load_pending_provider_operation_disposition(store, session_id) is None
    session = await store.load(session_id)
    assert session is not None
    events = await store.load_events(session_id)
    if action is ProviderOperationResolutionAction.FALLBACK_RETRY:
        assert session.status is SessionStatus.COMPLETED
        assert provider.adapter.start_calls == 1
        model_started = [event for event in events if event.type is EventType.MODEL_STARTED]
        assert len(model_started) == 2
        assert (
            model_started[1].payload["model_step_id"] == model_started[0].payload["model_step_id"]
        )
        assert (
            model_started[1].payload["model_attempt_id"]
            != model_started[0].payload["model_attempt_id"]
        )
    else:
        assert session.status is SessionStatus.FAILED
        assert provider.adapter.start_calls == 0
        assert sum(event.type is EventType.SESSION_FAILED for event in events) == 1

    replay_page = await app.recover_incomplete_sessions(
        IncompleteSessionsRecoveryRequest(
            statuses={SessionStatus.COMPLETED, SessionStatus.FAILED},
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )
    )
    assert not replay_page.results
    assert provider.adapter.start_calls == (
        1 if action is ProviderOperationResolutionAction.FALLBACK_RETRY else 0
    )


async def _commit_partial_provider_progress(
    store: SessionStore,
    *,
    session_id: str,
    provider: _OfflineOperationProvider,
) -> None:
    active = await store.load_active_model_completion_stage(session_id)
    session = await store.load(session_id)
    assert active is not None
    assert session is not None
    stage = active.stage
    model_attempt_id = stage.intent["model_attempt_id"]
    assert isinstance(model_attempt_id, str)
    identity = ModelAttemptIdentity(
        model_step_id=stage.logical_step_id,
        model_attempt_id=model_attempt_id,
    )
    stream_event = ModelStreamEvent.text_delta(
        "partial output",
        recovery_metadata={"cursor": 1},
    )
    envelope = provider_operation_progress_envelope(provider.adapter.state, stream_event)
    event = Event(
        id=provider_operation_progress_event_id(stage.stage_id, 1),
        type=EventType.MODEL_TEXT_DELTA,
        session_id=session_id,
        interaction_id=f"interaction-{session_id}",
        agent_name="assistant",
        payload={
            "provider": provider.name,
            "model": "fake-model",
            "step": 1,
            "attempt": 1,
            "max_attempts": 1,
            **identity.payload(),
            "source_run_epoch": stage.source_run_epoch,
            "provider_operation_progress": envelope.model_dump(mode="json"),
        },
    )
    await commit_provider_operation_progress(
        store,
        stage=stage,
        model_attempt_identity=identity,
        current_state=provider.adapter.state,
        stream_event=stream_event,
        event=event,
        expected_run_epoch=session.run_epoch,
    )


async def assert_offline_provider_operation_recovery(store: SessionStore) -> None:
    provider = _OfflineOperationProvider(ProviderOperationStatus.COMPLETED)
    user_message = await _stage_offline_operation(
        store,
        session_id="offline-completed",
        provider=provider,
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    await app.recover_incomplete_session(
        IncompleteSessionRecoveryRequest(
            session_id="offline-completed",
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )
    )
    await app.recover_incomplete_session(
        IncompleteSessionRecoveryRequest(session_id="offline-completed")
    )

    assert provider.adapter.start_calls == 0
    assert provider.adapter.retrieve_calls == [provider.adapter.state]
    transcript = await store.load_transcript("offline-completed")
    assert transcript[0] == user_message
    assert transcript[1].content[0].text == "finished while offline"
    events = await store.load_events("offline-completed")
    completed_events = [event for event in events if event.type is EventType.MODEL_COMPLETED]
    assert len(completed_events) == 1
    assert completed_events[0].payload["usage_metrics"]["input_tokens"] == 3
    assert completed_events[0].payload["usage_metrics"]["output_tokens"] == 4
    assert await store.load_active_model_completion_stage("offline-completed") is None
    inspection = await inspect_provider_operation(store, "offline-completed")
    assert inspection.status is ProviderOperationInspectionStatus.PROVIDER_OPERATION_RECONCILED


async def assert_terminal_session_fails_closed_with_active_provider_operation(
    store: SessionStore,
) -> None:
    session_id = "terminal-with-active-provider-operation"
    provider = _OfflineOperationProvider(ProviderOperationStatus.COMPLETED)
    await _stage_offline_operation(
        store,
        session_id=session_id,
        provider=provider,
    )
    await store.update_status(session_id, SessionStatus.COMPLETED)
    await store.append_event(
        session_id,
        Event(type=EventType.SESSION_COMPLETED, session_id=session_id),
    )

    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    with pytest.raises(
        ModelCompletionManualRecoveryRequired,
        match="terminal session retains an active model-completion stage",
    ):
        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )

    assert provider.adapter.retrieve_calls == []
    assert await store.load_active_model_completion_stage(session_id) is not None


async def assert_terminal_session_fails_closed_without_active_provider(
    store: SessionStore,
) -> None:
    session_id = "terminal-with-unregistered-active-provider"
    provider = _OfflineOperationProvider(ProviderOperationStatus.IN_PROGRESS)
    await _stage_offline_operation(
        store,
        session_id=session_id,
        provider=provider,
    )
    await store.update_status(session_id, SessionStatus.COMPLETED)
    await store.append_event(
        session_id,
        Event(type=EventType.SESSION_COMPLETED, session_id=session_id),
    )

    app = CayuApp(session_store=store, enable_logging=False)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    with pytest.raises(KeyError, match="Provider not registered"):
        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )

    assert provider.adapter.retrieve_calls == []
    assert await store.load_active_model_completion_stage(session_id) is not None


@pytest.mark.parametrize("batched", [False, True])
def test_unregistered_agent_recovery_leaves_terminal_interaction_and_stage_untouched(
    batched: bool,
) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _OfflineOperationProvider(ProviderOperationStatus.IN_PROGRESS)
        session_id = f"terminal-unregistered-agent-{'batch' if batched else 'single'}"
        await _stage_offline_operation(store, session_id=session_id, provider=provider)
        await store.update_status(session_id, SessionStatus.COMPLETED)
        await store.append_event(
            session_id,
            Event(type=EventType.SESSION_COMPLETED, session_id=session_id),
        )

        before_session = await store.load(session_id)
        before_checkpoint = await store.load_checkpoint(session_id)
        before_stage = await store.load_active_model_completion_stage(session_id)
        before_events = await store.load_events(session_id)

        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        if batched:
            page = await app.recover_incomplete_sessions(
                IncompleteSessionsRecoveryRequest(
                    statuses={SessionStatus.COMPLETED},
                    inactive_before=datetime.now(UTC) + timedelta(seconds=1),
                )
            )
            assert len(page.results) == 1
            result = page.results[0]
        else:
            result = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    inactive_before=datetime.now(UTC) + timedelta(seconds=1),
                )
            )

        assert result.actions == (IncompleteSessionRecoveryAction.SKIPPED_UNREGISTERED_AGENT,)
        assert result.events == ()
        assert await store.load(session_id) == before_session
        assert await store.load_checkpoint(session_id) == before_checkpoint
        assert await store.load_active_model_completion_stage(session_id) == before_stage
        assert await store.load_events(session_id) == before_events

    asyncio.run(scenario())


async def assert_pending_provider_operation_later_completes(
    store: SessionStore,
    *,
    initial_status: ProviderOperationStatus,
) -> None:
    provider = _OfflineOperationProvider(initial_status)
    user_message = await _stage_offline_operation(
        store,
        session_id=f"offline-{initial_status.value}-then-completed",
        provider=provider,
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    session_id = f"offline-{initial_status.value}-then-completed"

    await app.recover_incomplete_session(
        IncompleteSessionRecoveryRequest(
            session_id=session_id,
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )
    )
    pending = await store.load(session_id)
    assert pending is not None
    assert pending.status in {SessionStatus.PENDING, SessionStatus.RUNNING}
    assert await store.load_active_model_completion_stage(session_id) is not None

    provider.adapter.status = ProviderOperationStatus.COMPLETED
    await app.recover_incomplete_session(IncompleteSessionRecoveryRequest(session_id=session_id))
    transcript = await store.load_transcript(session_id)
    assert transcript[0] == user_message
    assert transcript[1].content[0].text == "finished while offline"
    events = await store.load_events(session_id)
    assert sum(event.type is EventType.MODEL_COMPLETED for event in events) == 1
    assert provider.adapter.start_calls == 0
    assert provider.adapter.retrieve_calls == [provider.adapter.state] * 2


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_completed_operation_is_retrieved_and_published_exactly_once(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / "offline-recovery.sqlite3")
        )
        await assert_offline_provider_operation_recovery(store)
        if isinstance(store, SQLiteSessionStore):
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("provider_status", "expected_reason"),
    [
        (ProviderOperationStatus.FAILED, ProviderOperationUnavailableReason.FAILED),
        (ProviderOperationStatus.EXPIRED, ProviderOperationUnavailableReason.EXPIRED),
        (ProviderOperationStatus.CANCELLED, ProviderOperationUnavailableReason.CANCELLED),
        (ProviderOperationStatus.UNAVAILABLE, ProviderOperationUnavailableReason.UNAVAILABLE),
    ],
)
def test_terminal_provider_operation_requires_explicit_resolution_without_redispatch(
    provider_status: ProviderOperationStatus,
    expected_reason: ProviderOperationUnavailableReason,
) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        session_id = f"offline-{provider_status.value}"
        provider = _OfflineOperationProvider(provider_status)
        await _stage_offline_operation(store, session_id=session_id, provider=provider)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )

        session = await store.load(session_id)
        assert session is not None
        assert session.status is SessionStatus.INTERRUPTED
        assert provider.adapter.start_calls == 0
        assert provider.adapter.retrieve_calls == [provider.adapter.state]
        assert await store.load_active_model_completion_stage(session_id) is not None
        inspection = await inspect_provider_operation(store, session_id)
        assert inspection.status is ProviderOperationInspectionStatus.PROVIDER_OPERATION_UNAVAILABLE
        assert inspection.recovery_reason is expected_reason
        assert inspection.duplicate_request_risk is (
            expected_reason is ProviderOperationUnavailableReason.UNAVAILABLE
        )
        assert inspection.allowed_resolutions == ("fallback_retry", "fail")
        events = await store.load_events(session_id)
        assert (
            sum(event.type is EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED for event in events)
            == 1
        )
        assert EventType.MODEL_RETRY not in {event.type for event in events}

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("retrieval_outcome", "expected_reason"),
    [
        ("unavailable", ProviderOperationUnavailableReason.UNAVAILABLE),
        ("malformed", ProviderOperationUnavailableReason.MALFORMED),
        ("wrong_provider", ProviderOperationUnavailableReason.WRONG_PROVIDER),
    ],
)
def test_invalid_provider_retrieval_outcomes_require_typed_resolution(
    retrieval_outcome: str,
    expected_reason: ProviderOperationUnavailableReason,
) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        session_id = f"offline-retrieval-{retrieval_outcome}"
        provider = _OfflineOperationProvider(ProviderOperationStatus.COMPLETED)

        async def retrieve(state: ProviderOperationState):
            provider.adapter.retrieve_calls.append(state)
            if retrieval_outcome == "unavailable":
                raise RuntimeError("provider retrieval endpoint unavailable")
            if retrieval_outcome == "malformed":
                return {"unexpected": "response"}
            return ProviderOperationSnapshot(
                state=ProviderOperationState(
                    operation_id="different-provider-operation",
                    stream_protocol=state.stream_protocol,
                    recovery_metadata=state.recovery_metadata,
                ),
                status=ProviderOperationStatus.COMPLETED,
                events=(),
            )

        provider.adapter.retrieve = retrieve  # type: ignore[method-assign]
        await _stage_offline_operation(
            store,
            session_id=session_id,
            provider=provider,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )

        session = await store.load(session_id)
        assert session is not None
        assert session.status is SessionStatus.INTERRUPTED
        inspection = await inspect_provider_operation(store, session_id)
        assert inspection.recovery_reason is expected_reason
        assert inspection.allowed_resolutions == ("fallback_retry", "fail")
        assert inspection.duplicate_request_risk
        assert provider.adapter.start_calls == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("reconnect_outcome", "expected_reason", "duplicate_request_risk"),
    [
        ("unavailable", ProviderOperationUnavailableReason.UNAVAILABLE, True),
        ("malformed_connection", ProviderOperationUnavailableReason.MALFORMED, True),
        ("wrong_provider", ProviderOperationUnavailableReason.WRONG_PROVIDER, True),
        ("malformed_event", ProviderOperationUnavailableReason.MALFORMED, True),
        ("failed", ProviderOperationUnavailableReason.FAILED, False),
    ],
)
def test_partial_progress_reconnect_failures_require_typed_resolution(
    reconnect_outcome: str,
    expected_reason: ProviderOperationUnavailableReason,
    duplicate_request_risk: bool,
) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        session_id = f"offline-reconnect-{reconnect_outcome}"
        provider = _OfflineOperationProvider(ProviderOperationStatus.IN_PROGRESS)
        await _stage_offline_operation(
            store,
            session_id=session_id,
            provider=provider,
        )
        await _commit_partial_provider_progress(
            store,
            session_id=session_id,
            provider=provider,
        )

        async def reconnect(state: ProviderOperationState):
            if reconnect_outcome == "unavailable":
                raise RuntimeError("provider reconnect endpoint unavailable")
            if reconnect_outcome == "malformed_connection":
                return {"unexpected": "response"}

            async def events() -> AsyncIterator[ModelStreamEvent]:
                if reconnect_outcome == "malformed_event":
                    yield ModelStreamEvent.model_construct(
                        type="not-a-model-stream-event",
                        delta="",
                        payload={},
                        recovery_metadata={"cursor": 2},
                    )

            connection_state = state
            if reconnect_outcome == "wrong_provider":
                connection_state = ProviderOperationState(
                    operation_id="different-provider-operation",
                    stream_protocol=state.stream_protocol,
                    recovery_metadata=state.recovery_metadata,
                )
            return ProviderOperationConnection(
                state=connection_state,
                status=(
                    ProviderOperationStatus.FAILED
                    if reconnect_outcome == "failed"
                    else ProviderOperationStatus.IN_PROGRESS
                ),
                events=events(),
            )

        provider.adapter.reconnect = reconnect  # type: ignore[method-assign]
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )

        session = await store.load(session_id)
        assert session is not None
        assert session.status is SessionStatus.INTERRUPTED
        inspection = await inspect_provider_operation(store, session_id)
        assert inspection.recovery_reason is expected_reason
        assert inspection.duplicate_request_risk is duplicate_request_risk
        assert inspection.allowed_resolutions == ("fallback_retry", "fail")
        assert provider.adapter.start_calls == 0
        events_after_recovery = await store.load_events(session_id)
        required = [
            event
            for event in events_after_recovery
            if event.type is EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED
        ]
        assert len(required) == 1
        assert required[0].payload["recovery_reason"] == expected_reason.value
        interrupted = next(
            event for event in events_after_recovery if event.type is EventType.SESSION_INTERRUPTED
        )
        assert interrupted.payload["duplicate_request_risk"] is duplicate_request_risk

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "entrance",
    ["recover_start", "retrieve", "reconnect", "reconnect_stream"],
)
def test_provider_recovery_treats_child_cancellation_as_unavailable(
    entrance: str,
) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        session_id = f"provider-recovery-child-cancellation-{entrance}"
        if entrance == "recover_start":
            provider: _OfflineOperationProvider = _IdempotentAmbiguousStartProvider()

            def runtime() -> CayuApp:
                app = CayuApp(session_store=store, enable_logging=False)
                app.register_provider(provider, default=True)
                app.register_agent(AgentSpec(name="assistant", model="fake-model"))
                return app

            with pytest.raises(_SimulatedProcessLoss):
                async for _event in runtime().run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "recover one exact provider start")],
                    )
                ):
                    pass

            async def recover_start(
                _request: ProviderOperationStartRecoveryRequest,
            ) -> ProviderOperationConnection:
                raise asyncio.CancelledError("provider child cancelled start recovery")

            provider.adapter.recover_start = recover_start  # type: ignore[method-assign]
            app = runtime()
        else:
            provider = _OfflineOperationProvider(ProviderOperationStatus.IN_PROGRESS)
            await _stage_offline_operation(store, session_id=session_id, provider=provider)
            if entrance in {"reconnect", "reconnect_stream"}:
                await _commit_partial_provider_progress(
                    store,
                    session_id=session_id,
                    provider=provider,
                )

                async def reconnect(
                    state: ProviderOperationState,
                ) -> ProviderOperationConnection:
                    if entrance == "reconnect":
                        raise asyncio.CancelledError("provider child cancelled reconnect")

                    async def events() -> AsyncIterator[ModelStreamEvent]:
                        raise asyncio.CancelledError("provider child cancelled reconnect stream")
                        yield  # pragma: no cover - keeps this an async generator

                    return ProviderOperationConnection(
                        state=state,
                        status=ProviderOperationStatus.IN_PROGRESS,
                        events=events(),
                    )

                provider.adapter.reconnect = reconnect  # type: ignore[method-assign]
            else:

                async def retrieve(
                    _state: ProviderOperationState,
                ) -> ProviderOperationSnapshot:
                    raise asyncio.CancelledError("provider child cancelled retrieval")

                provider.adapter.retrieve = retrieve  # type: ignore[method-assign]
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        recovery_task = asyncio.create_task(
            app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    inactive_before=datetime.now(UTC) + timedelta(seconds=1),
                )
            )
        )
        await recovery_task

        assert recovery_task.cancelling() == 0
        assert not recovery_task.cancelled()
        session = await store.load(session_id)
        assert session is not None
        assert session.status is SessionStatus.INTERRUPTED
        inspection = await inspect_provider_operation(store, session_id)
        assert inspection.recovery_reason is ProviderOperationUnavailableReason.UNAVAILABLE
        assert inspection.allowed_resolutions == ("fallback_retry", "fail")
        events = await store.load_events(session_id)
        assert (
            sum(event.type is EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED for event in events)
            == 1
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("adapter_replaces_cancellation", [False, True])
def test_provider_recovery_preserves_real_task_cancellation(
    adapter_replaces_cancellation: bool,
) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        session_id = "provider-recovery-real-caller-cancellation"
        provider = _OfflineOperationProvider(ProviderOperationStatus.IN_PROGRESS)
        await _stage_offline_operation(store, session_id=session_id, provider=provider)
        retrieval_started = asyncio.Event()

        async def retrieve(
            _state: ProviderOperationState,
        ) -> ProviderOperationSnapshot:
            retrieval_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if adapter_replaces_cancellation:
                    raise RuntimeError("provider replaced caller cancellation") from None
                raise
            raise AssertionError("cancelled retrieval unexpectedly continued")

        provider.adapter.retrieve = retrieve  # type: ignore[method-assign]
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        recovery_task = asyncio.create_task(
            app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    inactive_before=datetime.now(UTC) + timedelta(seconds=1),
                )
            )
        )
        await retrieval_started.wait()

        recovery_task.cancel("cancel provider recovery owner")
        assert recovery_task.cancelling() == 1
        with pytest.raises(asyncio.CancelledError):
            await recovery_task

        assert recovery_task.cancelled()
        events = await store.load_events(session_id)
        assert EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED not in {
            event.type for event in events
        }

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("action", "after_status_transition"),
    [
        (ProviderOperationResolutionAction.FALLBACK_RETRY, False),
        (ProviderOperationResolutionAction.FALLBACK_RETRY, True),
        (ProviderOperationResolutionAction.FAIL, False),
        (ProviderOperationResolutionAction.FAIL, True),
    ],
)
def test_sqlite_provider_resolution_process_loss_finishes_disposition(
    tmp_path: Path,
    action: ProviderOperationResolutionAction,
    after_status_transition: bool,
) -> None:
    async def scenario() -> None:
        path = tmp_path / f"provider-resolution-{action.value}-{after_status_transition}.sqlite3"
        store = SQLiteSessionStore(path)
        try:
            session_id, provider = await stage_provider_resolution_process_loss(
                store,
                action=action,
                after_status_transition=after_status_transition,
            )
        finally:
            await store.close()

        reopened = SQLiteSessionStore(path)
        try:
            await assert_provider_resolution_process_loss_recovery(
                reopened,
                session_id=session_id,
                provider=provider,
                action=action,
            )
        finally:
            await reopened.close()

    asyncio.run(scenario())


def test_sqlite_fallback_pre_dispatch_failure_remains_recoverable_after_restart(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        path = tmp_path / "provider-resolution-pre-dispatch-failure.sqlite3"
        provider = _FailOnceFallbackBillingProvider()
        store = SQLiteSessionStore(path)
        session_id = "provider-resolution-pre-dispatch-failure"
        try:
            await _stage_offline_operation(
                store,
                session_id=session_id,
                provider=provider,
                recovery_context={"max_steps": 2},
            )
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    inactive_before=datetime.now(UTC) + timedelta(seconds=1),
                )
            )
            interrupted = await store.load(session_id)
            active = await store.load_active_model_completion_stage(session_id)
            assert interrupted is not None
            assert interrupted.status is SessionStatus.INTERRUPTED
            assert active is not None
            request = ProviderOperationResolutionRequest(
                session_id=session_id,
                stage_id=active.stage.stage_id,
                expected_run_epoch=interrupted.run_epoch,
                action=ProviderOperationResolutionAction.FALLBACK_RETRY,
                reason="retry after exact continuation became unavailable",
            )

            with pytest.raises(
                RuntimeError,
                match="Model provider billing identity resolution failed",
            ):
                _ = [event async for event in app.resolve_provider_operation(request)]

            failed_attempt = await store.load(session_id)
            assert failed_attempt is not None
            assert failed_attempt.status is SessionStatus.RUNNING
            assert await load_pending_provider_operation_disposition(store, session_id) is not None
            assert await store.load_active_model_completion_stage(session_id) is None
            assert provider.adapter.start_calls == 0
            assert not any(
                event.type is EventType.SESSION_FAILED
                for event in await store.load_events(session_id)
            )
        finally:
            await store.close()

        reopened = SQLiteSessionStore(path)
        try:
            provider.adapter.start_events = (
                ModelStreamEvent.text_delta("recovered after pre-dispatch failure"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            )
            restarted = CayuApp(session_store=reopened, enable_logging=False)
            restarted.register_provider(provider, default=True)
            restarted.register_agent(AgentSpec(name="assistant", model="fake-model"))
            recovered = await restarted.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    inactive_before=datetime.now(UTC) + timedelta(seconds=1),
                )
            )

            assert (
                IncompleteSessionRecoveryAction.REPAIRED_PROVIDER_OPERATION_RESOLUTION
                in recovered.actions
            )
            assert await load_pending_provider_operation_disposition(reopened, session_id) is None
            completed = await reopened.load(session_id)
            assert completed is not None
            assert completed.status is SessionStatus.COMPLETED
            assert provider.billing_calls == 2
            assert provider.adapter.start_calls == 1
            events = await reopened.load_events(session_id)
            assert sum(event.type is EventType.SESSION_FAILED for event in events) == 0
            assert sum(event.type is EventType.PROVIDER_OPERATION_STARTING for event in events) == 1
        finally:
            await reopened.close()

    asyncio.run(scenario())


def test_explicit_fallback_resolution_is_fenced_idempotent_and_dispatches_one_new_attempt() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        session_id = "offline-fallback-resolution"
        provider = _OfflineOperationProvider(ProviderOperationStatus.EXPIRED)
        user_message = await _stage_offline_operation(
            store,
            session_id=session_id,
            provider=provider,
            recovery_context={"max_steps": 2},
            step=2,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )
        interrupted = await store.load(session_id)
        active = await store.load_active_model_completion_stage(session_id)
        assert interrupted is not None
        assert active is not None
        request = ProviderOperationResolutionRequest(
            session_id=session_id,
            stage_id=active.stage.stage_id,
            expected_run_epoch=interrupted.run_epoch,
            action=ProviderOperationResolutionAction.FALLBACK_RETRY,
            reason="provider operation expired; accept duplicate-request risk",
        )

        with pytest.raises(SessionRunFenced):
            _ = [
                event
                async for event in app.resolve_provider_operation(
                    request.model_copy(update={"expected_run_epoch": interrupted.run_epoch + 1})
                )
            ]

        provider.adapter.start_events = (
            ModelStreamEvent.text_delta("retried after explicit fallback"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        )
        events = [event async for event in app.resolve_provider_operation(request)]

        assert events[0].type is EventType.PROVIDER_OPERATION_RESOLVED
        assert events[0].payload["resolution_action"] == "fallback_retry"
        assert events[0].payload["recovery_reason"] == "expired"
        assert provider.adapter.start_calls == 1
        assert await store.load_active_model_completion_stage(session_id) is None
        transcript = await store.load_transcript(session_id)
        assert transcript[0] == user_message
        assert transcript[1].content[0].text == "retried after explicit fallback"
        stored_events = await store.load_events(session_id)
        model_attempt_ids = {
            event.payload["model_attempt_id"]
            for event in stored_events
            if event.type is EventType.MODEL_STARTED
        }
        assert len(model_attempt_ids) == 2
        model_started = [event for event in stored_events if event.type is EventType.MODEL_STARTED]
        assert [event.payload["step"] for event in model_started] == [2, 2]
        assert (
            model_started[0].payload["model_step_id"] == model_started[1].payload["model_step_id"]
        )
        assert (
            sum(event.type is EventType.PROVIDER_OPERATION_RESOLVED for event in stored_events) == 1
        )

        replay = [event async for event in app.resolve_provider_operation(request)]
        assert [event.type for event in replay] == [EventType.PROVIDER_OPERATION_RESOLVED]
        assert provider.adapter.start_calls == 1
        with pytest.raises(ProviderOperationResolutionConflict):
            _ = [
                event
                async for event in app.resolve_provider_operation(
                    request.model_copy(update={"action": ProviderOperationResolutionAction.FAIL})
                )
            ]

    asyncio.run(scenario())


def test_closing_fallback_before_dispatch_preserves_recoverable_disposition() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        session_id = "fallback-stream-close-before-dispatch"
        provider = _OfflineOperationProvider(ProviderOperationStatus.UNAVAILABLE)
        app, request = await _prepare_explicit_fallback_resolution(
            store,
            session_id=session_id,
            provider=provider,
            recovery_context={"max_steps": 2},
        )
        provider.adapter.start_events = (
            ModelStreamEvent.text_delta("recovered after stream closure"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        )

        stream = app.resolve_provider_operation(request)
        assert (await anext(stream)).type is EventType.PROVIDER_OPERATION_RESOLVED
        assert (await anext(stream)).type is EventType.INTERACTION_RESUMED
        await stream.aclose()

        preserved = await store.load(session_id)
        assert preserved is not None
        assert preserved.status is SessionStatus.RUNNING
        assert await load_pending_provider_operation_disposition(store, session_id) is not None
        assert provider.adapter.start_calls == 0

        recovered = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )

        assert (
            IncompleteSessionRecoveryAction.REPAIRED_PROVIDER_OPERATION_RESOLUTION
            in recovered.actions
        )
        assert await load_pending_provider_operation_disposition(store, session_id) is None
        completed = await store.load(session_id)
        assert completed is not None
        assert completed.status is SessionStatus.COMPLETED
        assert provider.adapter.start_calls == 1

    asyncio.run(scenario())


def test_cancelling_fallback_before_dispatch_preserves_recoverable_disposition() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        session_id = "fallback-cancel-before-dispatch"
        provider = _BlockingFallbackBillingProvider()
        app, request = await _prepare_explicit_fallback_resolution(
            store,
            session_id=session_id,
            provider=provider,
            recovery_context={"max_steps": 2},
        )
        provider.adapter.start_events = (
            ModelStreamEvent.text_delta("recovered after cancellation"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        )

        async def drain_resolution() -> list[Event]:
            return [event async for event in app.resolve_provider_operation(request)]

        resolution_task = asyncio.create_task(drain_resolution())
        await provider.billing_entered.wait()

        replay = [event async for event in app.resolve_provider_operation(request)]
        assert [event.type for event in replay] == [EventType.PROVIDER_OPERATION_RESOLVED]
        assert not resolution_task.done()
        assert provider.billing_calls == 1
        assert provider.adapter.start_calls == 0

        resolution_task.cancel("cancel fallback before provider dispatch")
        assert resolution_task.cancelling() == 1
        with pytest.raises(asyncio.CancelledError):
            await resolution_task
        assert resolution_task.cancelled()

        preserved = await store.load(session_id)
        assert preserved is not None
        assert preserved.status is SessionStatus.RUNNING
        assert await load_pending_provider_operation_disposition(store, session_id) is not None
        assert provider.adapter.start_calls == 0

        recovered = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )

        assert (
            IncompleteSessionRecoveryAction.REPAIRED_PROVIDER_OPERATION_RESOLUTION
            in recovered.actions
        )
        assert await load_pending_provider_operation_disposition(store, session_id) is None
        completed = await store.load(session_id)
        assert completed is not None
        assert completed.status is SessionStatus.COMPLETED
        assert provider.billing_calls == 2
        assert provider.adapter.start_calls == 1

    asyncio.run(scenario())


def test_concurrent_exact_fallback_replay_does_not_start_a_second_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        session_id = "concurrent-exact-fallback-resolution"
        provider = _BlockingFallbackBillingProvider()
        app, request = await _prepare_explicit_fallback_resolution(
            store,
            session_id=session_id,
            provider=provider,
            recovery_context={"max_steps": 2},
        )
        provider.adapter.start_events = (
            ModelStreamEvent.text_delta("recovered after concurrent resolution"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        )
        original_transition = app._recovery_coordinator._transition_recovery_session_to_running
        transition_call_count = 0
        both_callers_ready = asyncio.Event()
        release_transitions = asyncio.Event()

        async def blocked_transition(*args: Any, **kwargs: Any):
            nonlocal transition_call_count
            transition_call_count += 1
            if transition_call_count == 2:
                both_callers_ready.set()
            await release_transitions.wait()
            return await original_transition(*args, **kwargs)

        monkeypatch.setattr(
            app._recovery_coordinator,
            "_transition_recovery_session_to_running",
            blocked_transition,
        )

        async def collect_resolution() -> list[Event]:
            return [event async for event in app.resolve_provider_operation(request)]

        first = asyncio.create_task(collect_resolution())
        second = asyncio.create_task(collect_resolution())
        await both_callers_ready.wait()
        release_transitions.set()
        await provider.billing_entered.wait()
        done, pending = await asyncio.wait(
            {first, second},
            timeout=1,
            return_when=asyncio.FIRST_COMPLETED,
        )

        assert len(done) == 1
        assert len(pending) == 1
        replay_events = done.pop().result()
        assert [event.type for event in replay_events] == [EventType.PROVIDER_OPERATION_RESOLVED]
        assert transition_call_count == 2
        assert provider.billing_calls == 1
        assert provider.adapter.start_calls == 0

        owner = pending.pop()
        owner.cancel("stop the winning fallback owner")
        assert owner.cancelling() == 1
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert owner.cancelled()

        recovered = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )
        assert (
            IncompleteSessionRecoveryAction.REPAIRED_PROVIDER_OPERATION_RESOLUTION
            in recovered.actions
        )
        assert await load_pending_provider_operation_disposition(store, session_id) is None
        completed = await store.load(session_id)
        assert completed is not None
        assert completed.status is SessionStatus.COMPLETED
        assert provider.billing_calls == 2
        assert provider.adapter.start_calls == 1

    asyncio.run(scenario())


def test_concurrent_exact_fail_replay_observes_in_progress_terminalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        session_id = "concurrent-exact-fail-resolution"
        provider = _OfflineOperationProvider(ProviderOperationStatus.UNAVAILABLE)
        await _stage_offline_operation(store, session_id=session_id, provider=provider)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )
        interrupted = await store.load(session_id)
        active = await store.load_active_model_completion_stage(session_id)
        assert interrupted is not None
        assert active is not None
        request = ProviderOperationResolutionRequest(
            session_id=session_id,
            stage_id=active.stage.stage_id,
            expected_run_epoch=interrupted.run_epoch,
            action=ProviderOperationResolutionAction.FAIL,
            reason="fail once despite concurrent exact delivery",
        )
        terminalization_started = asyncio.Event()
        release_terminalization = asyncio.Event()
        original_fail = app._recovery_coordinator._fail_provider_operation

        async def blocked_fail(request):
            async for event in original_fail(request):
                yield event
                if event.type is EventType.INTERACTION_FAILED:
                    terminalization_started.set()
                    await release_terminalization.wait()

        monkeypatch.setattr(
            app._recovery_coordinator,
            "_fail_provider_operation",
            blocked_fail,
        )

        async def collect_resolution() -> list[Event]:
            return [event async for event in app.resolve_provider_operation(request)]

        owner = asyncio.create_task(collect_resolution())
        await terminalization_started.wait()
        replay = await collect_resolution()
        assert [event.type for event in replay] == [EventType.PROVIDER_OPERATION_RESOLVED]
        assert not owner.done()

        release_terminalization.set()
        owner_events = await owner
        assert EventType.SESSION_FAILED in {event.type for event in owner_events}
        durable = await store.load_events(session_id)
        assert sum(event.type is EventType.MODEL_ERROR for event in durable) == 1
        assert sum(event.type is EventType.INTERACTION_FAILED for event in durable) == 1
        assert sum(event.type is EventType.SESSION_FAILED for event in durable) == 1
        assert await load_pending_provider_operation_disposition(store, session_id) is None
        assert provider.adapter.start_calls == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("state_check_fails", [False, True])
def test_fallback_grouped_billing_cancellation_is_detached_before_publication(
    state_check_fails: bool,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canary = "fallback-grouped-billing-secret-canary-0123456789"

    class CredentialBearingProvider(_OfflineOperationProvider):
        name = "credential-bearing-fallback"

        def __repr__(self) -> str:
            return f"CredentialBearingProvider(api_key={canary!r})"

    class GroupedCancellationRun:
        def __init__(self) -> None:
            self.entered = asyncio.Event()

        def execute(self, **kwargs):
            del kwargs

            async def events():
                self.entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    raise BaseExceptionGroup(
                        f"provider-bearing cancellation near {canary}",
                        [
                            _BillingIdentityResolutionCancelled(f"billing cancelled near {canary}"),
                            RuntimeError(f"cleanup failed near {canary}"),
                        ],
                    ) from None
                yield None, None

            return events()

    def traceback_frames(
        captured: traceback.TracebackException,
    ) -> list[traceback.FrameSummary]:
        frames = list(captured.stack)
        for child in captured.exceptions or ():
            frames.extend(traceback_frames(child))
        return frames

    async def scenario() -> tuple[BaseException, int, bool]:
        store = InMemorySessionStore()
        provider = CredentialBearingProvider(ProviderOperationStatus.UNAVAILABLE)
        app, request = await _prepare_explicit_fallback_resolution(
            store,
            session_id="fallback-grouped-billing-cancellation",
            provider=provider,
            recovery_context={"max_steps": 2},
        )
        if state_check_fails:

            async def fail_interruption_state_check(_session_id: str) -> bool:
                raise RuntimeError(f"store check failed near {canary}")

            monkeypatch.setattr(
                app._session_control,
                "interrupt_requested",
                fail_interruption_state_check,
            )
        grouped_run = GroupedCancellationRun()
        monkeypatch.setattr(
            app._model_step_executor,
            "create_run",
            lambda **_kwargs: grouped_run,
        )

        async def collect_resolution() -> list[Event]:
            return [event async for event in app.resolve_provider_operation(request)]

        task = asyncio.create_task(collect_resolution())
        await grouped_run.entered.wait()
        task.cancel("cancel grouped fallback")
        cancelling = task.cancelling()
        with pytest.raises((BaseExceptionGroup, RuntimeError)) as raised:
            await task
        return raised.value, cancelling, task.cancelled()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        failure, cancelling, cancelled = asyncio.run(scenario())
    captured_output = capsys.readouterr()
    assert cancelling == 1
    assert not cancelled
    if state_check_fails:
        assert type(failure) is RuntimeError
        assert str(failure) == (
            "Session interruption state check failed after provider billing cancellation"
        )
    else:
        assert isinstance(failure, BaseExceptionGroup)
    assert failure.__cause__ is None
    assert failure.__context__ is None
    captured = traceback.TracebackException.from_exception(failure, capture_locals=True)
    cayu_frames = [
        frame for frame in traceback_frames(captured) if is_cayu_source_filename(frame.filename)
    ]
    retained = "\n".join(
        (
            str(failure),
            repr(failure),
            repr(vars(failure)),
            repr([(frame.name, frame.locals) for frame in cayu_frames]),
            repr(caught),
            caplog.text,
            captured_output.out,
            captured_output.err,
        )
    )
    assert canary not in retained


def test_operator_interrupt_before_fallback_dispatch_supersedes_pending_retry() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        session_id = "fallback-operator-interrupt-before-dispatch"
        provider = _BlockingFallbackBillingProvider()
        app, request = await _prepare_explicit_fallback_resolution(
            store,
            session_id=session_id,
            provider=provider,
            recovery_context={"max_steps": 2},
        )

        async def drain_resolution() -> list[Event]:
            return [event async for event in app.resolve_provider_operation(request)]

        resolution_task = asyncio.create_task(drain_resolution())
        await provider.billing_entered.wait()
        interruption_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id=session_id,
                    reason="operator supersedes accepted fallback",
                )
            )
        ]
        resolution_events = await resolution_task

        assert not resolution_task.cancelled()
        assert EventType.SESSION_INTERRUPTED in {
            event.type for event in (*resolution_events, *interruption_events)
        }
        assert await load_pending_provider_operation_disposition(store, session_id) is None
        interrupted = await store.load(session_id)
        assert interrupted is not None
        assert interrupted.status is SessionStatus.INTERRUPTED
        events = await store.load_events(session_id)
        terminal = [
            event
            for event in events
            if event.type is EventType.SESSION_INTERRUPTED
            and event.payload.get("interruption_type") == "operator_requested"
        ]
        assert len(terminal) == 1
        assert provider.adapter.start_calls == 0

    asyncio.run(scenario())


def test_fallback_limit_outcome_recovers_marker_clear_acknowledgement_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        session_id = "fallback-limit-marker-ack-loss"
        provider = _OfflineOperationProvider(ProviderOperationStatus.UNAVAILABLE)
        started_at = datetime.now(UTC) - timedelta(seconds=10)
        app, request = await _prepare_explicit_fallback_resolution(
            store,
            session_id=session_id,
            provider=provider,
            recovery_context={
                "schema_version": 1,
                "max_steps": 2,
                "limits": RunLimits(
                    max_elapsed_seconds=1,
                    scope="run",
                ).model_dump(mode="json"),
                "run_limit_accounting": {
                    "schema_version": 1,
                    "started_at": started_at.isoformat(),
                    "baseline": SessionUsageSummary(session_id=session_id).model_dump(mode="json"),
                    "run_budget_authorities": [],
                },
            },
        )
        original_clear = recovery_coordinator_module.clear_pending_provider_operation_disposition
        failed_once = False

        async def lose_first_clear_acknowledgement(session_store, pending) -> None:
            nonlocal failed_once
            if not failed_once:
                failed_once = True
                raise ConnectionError("pending disposition clear acknowledgement lost")
            await original_clear(session_store, pending)

        monkeypatch.setattr(
            recovery_coordinator_module,
            "clear_pending_provider_operation_disposition",
            lose_first_clear_acknowledgement,
        )
        observed: list[Event] = []
        with pytest.raises(ConnectionError, match="acknowledgement lost"):
            async for event in app.resolve_provider_operation(request):
                observed.append(event)

        assert EventType.SESSION_LIMIT_REACHED in {event.type for event in observed}
        assert EventType.INTERACTION_INTERRUPTED in {event.type for event in observed}
        assert EventType.SESSION_INTERRUPTED in {event.type for event in observed}
        interrupted = await store.load(session_id)
        assert interrupted is not None
        assert interrupted.status is SessionStatus.INTERRUPTED
        assert await load_pending_provider_operation_disposition(store, session_id) is not None
        assert provider.adapter.start_calls == 0

        monkeypatch.setattr(
            recovery_coordinator_module,
            "clear_pending_provider_operation_disposition",
            original_clear,
        )
        recovered = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )

        assert (
            IncompleteSessionRecoveryAction.REPAIRED_PROVIDER_OPERATION_RESOLUTION
            in recovered.actions
        )
        assert await load_pending_provider_operation_disposition(store, session_id) is None
        assert provider.adapter.start_calls == 0

    asyncio.run(scenario())


def test_closing_fallback_at_limit_event_recovers_typed_limit_outcome() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        session_id = "fallback-limit-stream-close"
        provider = _OfflineOperationProvider(ProviderOperationStatus.UNAVAILABLE)
        started_at = datetime.now(UTC) - timedelta(seconds=10)
        app, request = await _prepare_explicit_fallback_resolution(
            store,
            session_id=session_id,
            provider=provider,
            recovery_context={
                "schema_version": 1,
                "max_steps": 2,
                "limits": RunLimits(
                    max_elapsed_seconds=1,
                    scope="run",
                ).model_dump(mode="json"),
                "run_limit_accounting": {
                    "schema_version": 1,
                    "started_at": started_at.isoformat(),
                    "baseline": SessionUsageSummary(session_id=session_id).model_dump(mode="json"),
                    "run_budget_authorities": [],
                },
            },
        )

        stream = app.resolve_provider_operation(request)
        while (await anext(stream)).type is not EventType.SESSION_LIMIT_REACHED:
            pass
        await stream.aclose()

        assert await load_pending_provider_operation_disposition(store, session_id) is None
        interrupted = await store.load(session_id)
        assert interrupted is not None
        assert interrupted.status is SessionStatus.INTERRUPTED
        events = await store.load_events(session_id)
        assert sum(event.type is EventType.SESSION_LIMIT_REACHED for event in events) == 1
        assert (
            sum(
                event.type is EventType.SESSION_INTERRUPTED
                and event.payload.get("interruption_type") == "limit_reached"
                for event in events
            )
            == 1
        )
        assert provider.adapter.start_calls == 0

    asyncio.run(scenario())


def test_new_resolution_rejects_already_failed_session_without_partial_acceptance() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        session_id = "failed-session-new-provider-resolution"
        provider = _OfflineOperationProvider(ProviderOperationStatus.UNAVAILABLE)
        app, request = await _prepare_explicit_fallback_resolution(
            store,
            session_id=session_id,
            provider=provider,
        )
        await store.update_status(session_id, SessionStatus.FAILED)

        with pytest.raises(
            SessionStatusConflict,
            match="requires an interrupted session",
        ):
            _ = [event async for event in app.resolve_provider_operation(request)]

        assert await load_pending_provider_operation_disposition(store, session_id) is None
        assert await store.load_active_model_completion_stage(session_id) is not None
        events = await store.load_events(session_id)
        assert EventType.PROVIDER_OPERATION_RESOLVED not in {event.type for event in events}
        assert provider.adapter.start_calls == 0

    asyncio.run(scenario())


def test_public_resolution_copy_rejects_mutation_without_diagnostic_leakage(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class SecretCanary:
        def __repr__(self) -> str:
            return "provider-resolution-copy-secret"

        __str__ = __repr__

    async def scenario() -> BaseException:
        store = InMemorySessionStore()
        session_id = "provider-resolution-mutated-public-request"
        provider = _OfflineOperationProvider(ProviderOperationStatus.UNAVAILABLE)
        await _stage_offline_operation(store, session_id=session_id, provider=provider)
        request = ProviderOperationResolutionRequest(
            session_id=session_id,
            stage_id="unreached-stage",
            expected_run_epoch=0,
            action=ProviderOperationResolutionAction.FAIL,
            resolved_by=ResolutionActor(subject="operator"),
        )
        assert request.resolved_by is not None
        object.__setattr__(request.resolved_by, "subject", SecretCanary())
        app = CayuApp(session_store=store, enable_logging=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises((TypeError, ValueError)) as raised:
                _ = [event async for event in app.resolve_provider_operation(request)]
        assert caught == []
        return raised.value

    error = asyncio.run(scenario())
    captured = capsys.readouterr()
    combined_diagnostics = "\n".join(
        (str(error), repr(error), caplog.text, captured.out, captured.err)
    )
    assert "provider-resolution-copy-secret" not in combined_diagnostics


def test_successful_provider_resolution_redacts_audit_fields_before_persistence_and_sinks(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def scenario() -> str:
        canary = "provider-resolution-audit-secret-canary-0123456789"
        store = InMemorySessionStore()
        sink = InMemoryEventSink()
        session_id = "provider-resolution-secret-audit-fields"
        provider = _OfflineOperationProvider(ProviderOperationStatus.UNAVAILABLE)
        await _stage_offline_operation(store, session_id=session_id, provider=provider)
        app = CayuApp(
            session_store=store,
            event_sinks=[sink],
            secret_redactor=SecretRedactor(canary),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )
        interrupted = await store.load(session_id)
        active = await store.load_active_model_completion_stage(session_id)
        assert interrupted is not None
        assert active is not None
        request = ProviderOperationResolutionRequest(
            session_id=session_id,
            stage_id=active.stage.stage_id,
            expected_run_epoch=interrupted.run_epoch,
            action=ProviderOperationResolutionAction.FAIL,
            reason=f"operator reason contains {canary}",
            metadata={"note": {"credential": canary}},
            resolved_by=ResolutionActor(
                subject=f"operator-{canary}",
                tenant=f"tenant-{canary}",
                claims={"credential": canary},
            ),
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            emitted = [event async for event in app.resolve_provider_operation(request)]
        assert caught == []
        stored_resolution = await load_provider_operation_resolution(
            store,
            session_id,
            active.stage.stage_id,
        )
        assert stored_resolution is not None
        durable_events = await store.load_events(session_id)
        combined = repr(
            (
                [event.model_dump(mode="json") for event in emitted],
                stored_resolution.record.model_dump(mode="json"),
                [event.model_dump(mode="json") for event in durable_events],
                [event.model_dump(mode="json") for event in sink.events],
            )
        )
        assert "[REDACTED_SECRET]" in combined
        assert canary not in combined
        return canary

    canary = asyncio.run(scenario())
    captured = capsys.readouterr()
    assert canary not in "\n".join((caplog.text, captured.out, captured.err))


@pytest.mark.parametrize("key_location", ["metadata", "actor_claims"])
def test_provider_resolution_rejects_secret_bearing_audit_keys_before_persistence(
    key_location: str,
) -> None:
    async def scenario() -> None:
        canary = "provider-resolution-audit-key-secret-0123456789"
        store = InMemorySessionStore()
        session_id = f"provider-resolution-secret-key-{key_location}"
        provider = _OfflineOperationProvider(ProviderOperationStatus.UNAVAILABLE)
        await _stage_offline_operation(store, session_id=session_id, provider=provider)
        app = CayuApp(
            session_store=store,
            secret_redactor=SecretRedactor(canary),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )
        interrupted = await store.load(session_id)
        active = await store.load_active_model_completion_stage(session_id)
        assert interrupted is not None
        assert active is not None
        metadata = {f"secret-{canary}": "value"} if key_location == "metadata" else {}
        actor_claims = {f"secret-{canary}": "value"} if key_location == "actor_claims" else {}
        request = ProviderOperationResolutionRequest(
            session_id=session_id,
            stage_id=active.stage.stage_id,
            expected_run_epoch=interrupted.run_epoch,
            action=ProviderOperationResolutionAction.FAIL,
            metadata=metadata,
            resolved_by=ResolutionActor(subject="operator", claims=actor_claims),
        )

        with pytest.raises(ValueError, match="contains a workload secret"):
            _ = [event async for event in app.resolve_provider_operation(request)]

        assert (
            await load_provider_operation_resolution(
                store,
                session_id,
                active.stage.stage_id,
            )
            is None
        )
        assert EventType.PROVIDER_OPERATION_RESOLVED not in {
            event.type for event in await store.load_events(session_id)
        }

    asyncio.run(scenario())


def test_explicit_fail_resolution_terminalizes_without_provider_redispatch() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        session_id = "offline-fail-resolution"
        provider = _OfflineOperationProvider(ProviderOperationStatus.UNAVAILABLE)
        await _stage_offline_operation(
            store,
            session_id=session_id,
            provider=provider,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )
        interrupted = await store.load(session_id)
        active = await store.load_active_model_completion_stage(session_id)
        assert interrupted is not None
        assert active is not None
        request = ProviderOperationResolutionRequest(
            session_id=session_id,
            stage_id=active.stage.stage_id,
            expected_run_epoch=interrupted.run_epoch,
            action=ProviderOperationResolutionAction.FAIL,
            reason="provider operation unavailable; fail this model attempt",
        )

        events = [event async for event in app.resolve_provider_operation(request)]

        assert [event.type for event in events] == [
            EventType.PROVIDER_OPERATION_RESOLVED,
            EventType.MODEL_ERROR,
            EventType.INTERACTION_FAILED,
            EventType.SESSION_FAILED,
        ]
        assert events[1].payload["error_type"] == "provider_operation_unavailable"
        assert events[1].payload["recovery_reason"] == "unavailable"
        assert events[0].payload["duplicate_request_risk"] is True
        assert events[3].payload["failure_type"] == "provider_operation_unavailable"
        failed = await store.load(session_id)
        assert failed is not None
        assert failed.status is SessionStatus.FAILED
        assert provider.adapter.start_calls == 0
        assert await store.load_active_model_completion_stage(session_id) is None

        replay = [event async for event in app.resolve_provider_operation(request)]
        assert [event.type for event in replay] == [EventType.PROVIDER_OPERATION_RESOLVED]
        stored_events = await store.load_events(session_id)
        assert sum(event.type is EventType.MODEL_ERROR for event in stored_events) == 1
        assert sum(event.type is EventType.INTERACTION_FAILED for event in stored_events) == 1
        assert sum(event.type is EventType.SESSION_FAILED for event in stored_events) == 1
        assert provider.adapter.start_calls == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_terminal_session_does_not_skip_active_provider_operation(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store: SessionStore = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / "terminal-provider-recovery.sqlite3")
        )
        try:
            await assert_terminal_session_fails_closed_with_active_provider_operation(store)
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


def test_terminal_session_with_unregistered_active_provider_fails_closed() -> None:
    asyncio.run(
        assert_terminal_session_fails_closed_without_active_provider(InMemorySessionStore())
    )


def test_exact_start_idempotency_recovers_ambiguous_acceptance_without_new_request() -> None:
    async def scenario() -> None:
        session_id = "idempotent-ambiguous-start"
        store = InMemorySessionStore()
        provider = _IdempotentAmbiguousStartProvider()

        def runtime() -> CayuApp:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            return app

        with pytest.raises(_SimulatedProcessLoss):
            async for _event in runtime().run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[
                        Message.text("user", f"{index}:" + "x" * 65_000) for index in range(9)
                    ],
                )
            ):
                pass

        active = await store.load_active_model_completion_stage(session_id)
        assert active is not None
        assert active.stage.intent["provider_operation_start"] == {
            "schema_version": 1,
            "idempotency_support": "exact",
            "idempotency_key": active.stage.intent["provider_operation_start"]["idempotency_key"],
        }
        before = await inspect_provider_operation(store, session_id)
        assert before.status is ProviderOperationInspectionStatus.AMBIGUOUS_SUBMISSION
        assert provider.adapter.start_calls == 1

        result = await runtime().recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )

        assert result.session_id == session_id
        assert provider.adapter.start_calls == 1
        assert provider.adapter.recovery_keys == provider.adapter.start_keys
        assert await store.load_active_model_completion_stage(session_id) is None
        transcript = await store.load_transcript(session_id)
        assert transcript[-1].content[0].text == "finished while offline"
        events = await store.load_events(session_id)
        started = [event for event in events if event.type is EventType.PROVIDER_OPERATION_STARTED]
        assert len(started) == 1
        assert started[0].payload["idempotent_start_recovery"] is True
        assert EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED not in {
            event.type for event in events
        }

    asyncio.run(scenario())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "action",
    [
        ProviderOperationResolutionAction.FALLBACK_RETRY,
        ProviderOperationResolutionAction.FAIL,
    ],
)
def test_unsupported_start_process_loss_requires_explicit_resolution_after_restart(
    store_kind: str,
    action: ProviderOperationResolutionAction,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        session_id = f"unsupported-ambiguous-start-{store_kind}-{action.value}"
        path = tmp_path / f"{session_id}.sqlite3"
        provider = _UnsupportedAmbiguousStartProvider()

        def runtime(store: SessionStore) -> CayuApp:
            app = CayuApp(session_store=store, enable_logging=False)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            return app

        first_store: SessionStore = (
            InMemorySessionStore() if store_kind == "memory" else SQLiteSessionStore(path)
        )
        with pytest.raises(_SimulatedProcessLoss):
            async for _event in runtime(first_store).run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "do not silently retry")],
                )
            ):
                pass
        staged = await first_store.load_active_model_completion_stage(session_id)
        assert staged is not None
        assert staged.stage.intent["provider_operation_start"] == {
            "schema_version": 1,
            "idempotency_support": "unsupported",
            "idempotency_key": staged.stage.intent["provider_operation_start"]["idempotency_key"],
        }
        assert provider.adapter.start_calls == 1
        if isinstance(first_store, SQLiteSessionStore):
            await first_store.close()
            store: SessionStore = SQLiteSessionStore(path)
        else:
            store = first_store

        try:
            await runtime(store).recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=session_id,
                    inactive_before=datetime.now(UTC) + timedelta(seconds=1),
                )
            )

            interrupted = await store.load(session_id)
            active = await store.load_active_model_completion_stage(session_id)
            assert interrupted is not None
            assert interrupted.status is SessionStatus.INTERRUPTED
            assert active is not None
            assert provider.adapter.start_calls == 1
            inspection = await inspect_provider_operation(store, session_id)
            assert inspection.status is ProviderOperationInspectionStatus.AMBIGUOUS_SUBMISSION
            assert inspection.duplicate_request_risk is True
            assert inspection.allowed_resolutions == ("fallback_retry", "fail")
            durable_before_resolution = await store.load_events(session_id)
            assert (
                sum(
                    event.type is EventType.PROVIDER_OPERATION_RECOVERY_REQUIRED
                    for event in durable_before_resolution
                )
                == 1
            )
            assert EventType.INTERACTION_FAILED not in {
                event.type for event in durable_before_resolution
            }

            if action is ProviderOperationResolutionAction.FALLBACK_RETRY:
                provider.adapter.start_events = (
                    ModelStreamEvent.text_delta("explicit fallback only"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                )
            resolved = [
                event
                async for event in runtime(store).resolve_provider_operation(
                    ProviderOperationResolutionRequest(
                        session_id=session_id,
                        stage_id=active.stage.stage_id,
                        expected_run_epoch=interrupted.run_epoch,
                        action=action,
                        reason="operator accepts the explicit disposition",
                    )
                )
            ]

            assert resolved[0].type is EventType.PROVIDER_OPERATION_RESOLVED
            final = await store.load(session_id)
            assert final is not None
            if action is ProviderOperationResolutionAction.FALLBACK_RETRY:
                assert provider.adapter.start_calls == 2
                assert final.status is SessionStatus.COMPLETED
                transcript = await store.load_transcript(session_id)
                assert transcript[-1].content[0].text == "explicit fallback only"
            else:
                assert provider.adapter.start_calls == 1
                assert final.status is SessionStatus.FAILED
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


async def assert_budgeted_offline_provider_operation_recovery(
    store: SessionStore,
    ledger: BudgetLedger,
    *,
    expect_settlement_acknowledgement_loss: bool = False,
) -> None:
    provider = _BudgetedOfflineOperationProvider()
    budget_policy = _budget_policy(provider.name)
    app = CayuApp(
        session_store=store,
        budget_ledger=ledger,
        budget_policy=budget_policy,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    first_events: list[Event] = []
    with pytest.raises(_SimulatedProcessLoss):
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="offline-budget-recovery",
                messages=[Message.text("user", "finish after the worker is replaced")],
                limits=RunLimits(max_total_tokens=100, scope="run"),
                budget_limits=(
                    BudgetLimit(
                        scope="run",
                        max_estimated_cost=Decimal("10"),
                        pricing=PriceBook(
                            prices=(
                                ModelPrice.fixed(
                                    provider_name=provider.name,
                                    model="fake-model",
                                    input_per_million=Decimal("1"),
                                    output_per_million=Decimal("2"),
                                ),
                            )
                        ),
                    ),
                ),
            )
        ):
            first_events.append(event)
    assert EventType.PROVIDER_OPERATION_STARTED in {event.type for event in first_events}

    stage = await store.load_active_model_completion_stage("offline-budget-recovery")
    assert stage is not None
    assert len(stage.stage.reservation_ids) == 1
    recovery_context = ModelCompletionRecoveryContext.model_validate(
        stage.stage.intent["recovery_context"]
    )
    assert recovery_context.run_limit_accounting is not None
    assert recovery_context.run_limit_accounting.baseline == SessionUsageSummary(
        session_id="offline-budget-recovery"
    )
    assert recovery_context.run_limit_accounting.started_at.tzinfo is not None
    assert len(recovery_context.run_limit_accounting.run_budget_authorities) == 1
    assert (
        recovery_context.run_limit_accounting.run_budget_authorities[0].started_at
        == recovery_context.run_limit_accounting.started_at
    )
    provider.adapter.status = ProviderOperationStatus.COMPLETED

    request = IncompleteSessionRecoveryRequest(
        session_id="offline-budget-recovery",
        inactive_before=datetime.now(UTC) + timedelta(seconds=1),
    )
    if expect_settlement_acknowledgement_loss:
        with pytest.raises(RuntimeError, match="acknowledgement lost"):
            await app.recover_incomplete_session(request)
    else:
        await app.recover_incomplete_session(request)
    await app.recover_incomplete_session(
        IncompleteSessionRecoveryRequest(session_id="offline-budget-recovery")
    )

    events = await store.load_events("offline-budget-recovery")
    reserved = [event for event in events if event.type is EventType.BUDGET_RESERVED]
    reconciled = [event for event in events if event.type is EventType.BUDGET_RECONCILED]
    completed = [event for event in events if event.type is EventType.MODEL_COMPLETED]
    assert len(reserved) == 1
    assert len(reconciled) == 1
    assert len(completed) == 1
    assert reconciled[0].payload["reservation_id"] == stage.stage.reservation_ids[0]
    assert reconciled[0].payload["actual_amount"] == "0.000011"
    assert len(completed[0].payload["budget_settlements"]) == 1
    assert provider.adapter.start_calls == 1
    assert provider.adapter.retrieve_calls == [provider.adapter.state]


async def assert_offline_provider_operation_reuses_run_limit_accounting(
    store: SessionStore,
    *,
    limit_kind: str,
    approval_limits: RunLimits | None = None,
    approval_budget_limits: tuple[BudgetLimit, ...] | None = None,
) -> None:
    session_id = f"offline-run-limit-{limit_kind}"
    provider = _OfflineOperationProvider(
        ProviderOperationStatus.COMPLETED,
        events=(
            ModelStreamEvent.tool_call(
                id="run-limited-call",
                name="lookup",
                arguments={"query": "must not execute"},
            ),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "tool_calls",
                    "usage": {"input_tokens": 6, "output_tokens": 0},
                }
            ),
        ),
    )
    started_at = datetime.now(UTC) - timedelta(seconds=10)
    limits = {
        "tokens": RunLimits(max_total_tokens=10, scope="run"),
        "tools": RunLimits(max_tool_calls=1, scope="run"),
        "elapsed": RunLimits(max_elapsed_seconds=1, scope="run"),
        "cost": RunLimits(scope="run"),
    }[limit_kind]
    budget_limits: tuple[BudgetLimit, ...] = ()
    run_budget_authorities: list[dict[str, object]] = []
    if limit_kind == "cost":
        budget_limits = (
            BudgetLimit(
                scope="run",
                max_estimated_cost=Decimal("0.000007"),
                pricing=PriceBook(
                    prices=(
                        ModelPrice.fixed(
                            provider_name=provider.name,
                            model="fake-model",
                            input_per_million=Decimal("1"),
                            output_per_million=Decimal("1"),
                        ),
                    )
                ),
            ),
        )
        effective_limit = request_budget_limits_for_session(
            limits=budget_limits,
            agent_name="assistant",
            causal_budget_id=session_id,
        )[0]
        run_budget_authorities = [
            {
                "budget_limit_id": _effective_budget_limit_id(effective_limit),
                "currency": "USD",
                "started_at": started_at.isoformat(),
            }
        ]
    recovery_context = {
        "schema_version": 1,
        "limits": limits.model_dump(mode="json"),
        "budget_limits": [limit.model_dump(mode="json") for limit in budget_limits],
        "run_limit_accounting": {
            "schema_version": 1,
            "started_at": started_at.isoformat(),
            "baseline": SessionUsageSummary(session_id=session_id).model_dump(mode="json"),
            "run_budget_authorities": run_budget_authorities,
        },
    }
    prior_events: tuple[Event, ...] = ()
    if limit_kind in {"tokens", "cost"}:
        prior_events = (
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id=session_id,
                agent_name="assistant",
                payload={
                    "usage_metrics": {
                        "provider_name": provider.name,
                        "model": "fake-model",
                        "input_tokens": 5,
                        "output_tokens": 0,
                        "total_tokens": 5,
                    },
                },
            ),
        )
    elif limit_kind == "tools":
        prior_events = tuple(
            Event(
                type=EventType.TOOL_CALL_STARTED,
                session_id=session_id,
                agent_name="assistant",
                tool_name="earlier-tool",
                payload={"tool_call_id": f"earlier-call-{index}"},
            )
            for index in range(2)
        )
    tool = _RecordingLookupTool()
    await _stage_offline_operation(
        store,
        session_id=session_id,
        provider=provider,
        recovery_context=recovery_context,
        started_at=started_at,
        prior_events=prior_events,
        tools=(tool,),
    )

    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[tool],
        tool_policy=AlwaysRequireApprovalToolPolicy(),
    )

    recovery = await app.recover_incomplete_session(
        IncompleteSessionRecoveryRequest(
            session_id=session_id,
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )
    )
    assert recovery.pending_approval_id is not None
    checkpoint = await store.load_checkpoint(session_id)
    pending = app._pending_tool_approval_from_checkpoint(checkpoint)
    assert pending is not None

    resolution_events = [
        event
        async for event in app.resolve_tool_approval(
            ToolApprovalRequest(
                session_id=session_id,
                approval_id=pending.approval_id,
                tool_round_id=pending.tool_round_id,
                tool_call_id=pending.tool_call_id,
                decision=ToolApprovalDecision.APPROVE,
                limits=approval_limits,
                budget_limits=approval_budget_limits,
            )
        )
    ]

    limit_event = next(
        event for event in resolution_events if event.type is EventType.SESSION_LIMIT_REACHED
    )
    if limit_kind == "cost":
        assert limit_event.payload["limit"] == "estimated_cost"
        assert limit_event.payload["actual"] == "0.000011"
    assert all(
        "run_limit_accounting" not in event.model_dump_json()
        for event in (*recovery.events, *resolution_events)
    )
    assert tool.calls == []
    assert provider.adapter.start_calls == 0
    assert provider.adapter.retrieve_calls == [provider.adapter.state]


@pytest.mark.parametrize("limit_kind", ["tokens", "tools", "elapsed", "cost"])
@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_offline_recovery_reuses_original_run_limit_accounting(
    limit_kind: str,
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store: SessionStore = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / f"run-limit-{limit_kind}.sqlite3")
        )
        try:
            await assert_offline_provider_operation_reuses_run_limit_accounting(
                store,
                limit_kind=limit_kind,
            )
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
@pytest.mark.parametrize("override_kind", ["limits", "budget_limits"])
def test_offline_recovery_field_override_preserves_other_run_accounting(
    store_kind: str,
    override_kind: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store: SessionStore = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / "run-budget-limit-override.sqlite3")
        )
        try:
            await assert_offline_provider_operation_reuses_run_limit_accounting(
                store,
                limit_kind="cost" if override_kind == "limits" else "tokens",
                approval_limits=(
                    RunLimits(max_total_tokens=1_000, scope="run")
                    if override_kind == "limits"
                    else None
                ),
                approval_budget_limits=() if override_kind == "budget_limits" else None,
            )
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_offline_recovery_reuses_original_budget_reservation_and_pricing(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        if store_kind == "memory":
            store = InMemorySessionStore()
            ledger = InMemoryBudgetLedger(reservation_ttl_seconds=None)
        else:
            store = SQLiteSessionStore(tmp_path / "budget-recovery-session.sqlite3")
            ledger = SQLiteBudgetLedger(
                tmp_path / "budget-recovery-ledger.sqlite3",
                reservation_ttl_seconds=None,
            )
        try:
            await assert_budgeted_offline_provider_operation_recovery(store, ledger)
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()
                await ledger.close()

    asyncio.run(scenario())


def test_provider_recovery_replays_a_lost_settlement_acknowledgement_exactly_once() -> None:
    async def scenario() -> None:
        ledger = _LoseFirstRecoverySettlementAcknowledgement()
        await assert_budgeted_offline_provider_operation_recovery(
            InMemorySessionStore(),
            ledger,
            expect_settlement_acknowledgement_loss=True,
        )
        assert ledger.reconcile_calls == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "action",
    [
        ProviderOperationResolutionAction.FALLBACK_RETRY,
        ProviderOperationResolutionAction.FAIL,
    ],
)
def test_operator_resolution_conservatively_settles_original_provider_reservation(
    action: ProviderOperationResolutionAction,
) -> None:
    async def scenario() -> None:
        session_id = f"budgeted-provider-resolution-{action.value}"
        store = InMemorySessionStore()
        ledger = InMemoryBudgetLedger(reservation_ttl_seconds=None)
        provider = _ResumableBudgetedOfflineOperationProvider()

        def runtime() -> CayuApp:
            app = CayuApp(
                session_store=store,
                budget_ledger=ledger,
                budget_policy=_budget_policy(provider.name),
                enable_logging=False,
            )
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            return app

        with pytest.raises(_SimulatedProcessLoss):
            async for _event in runtime().run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "settle before the operator disposition")],
                )
            ):
                pass
        active = await store.load_active_model_completion_stage(session_id)
        assert active is not None
        [reservation_id] = active.stage.reservation_ids
        reserved = await ledger.load_reservation(reservation_id)
        assert reserved is not None
        assert reserved.status == "active"

        provider.adapter.status = ProviderOperationStatus.UNAVAILABLE
        await runtime().recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )
        interrupted = await store.load(session_id)
        assert interrupted is not None
        assert interrupted.status is SessionStatus.INTERRUPTED
        if action is ProviderOperationResolutionAction.FALLBACK_RETRY:
            provider.adapter.start_events = (
                ModelStreamEvent.text_delta("completed after conservative settlement"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            )

        emitted = [
            event
            async for event in runtime().resolve_provider_operation(
                ProviderOperationResolutionRequest(
                    session_id=session_id,
                    stage_id=active.stage.stage_id,
                    expected_run_epoch=interrupted.run_epoch,
                    action=action,
                    reason="operator selected a disposition with unknown provider usage",
                )
            )
        ]

        assert emitted[0].type is EventType.PROVIDER_OPERATION_RESOLVED
        assert emitted[1].type is EventType.BUDGET_RECONCILED
        settled = await ledger.load_reservation(reservation_id)
        assert settled is not None
        assert settled.status == "reconciled"
        settlement = await ledger.load_settlement(budget_settlement_id(reservation_id))
        assert settlement is not None
        assert settlement.reconciliation.actual_amount == reserved.reserved_amount
        assert settlement.reconciliation.settlement_kind == "conservative"
        assert await load_pending_provider_operation_disposition(store, session_id) is None
        durable = await store.load_events(session_id)
        assert (
            sum(
                event.type is EventType.BUDGET_RECONCILED
                and event.payload.get("reservation_id") == reservation_id
                for event in durable
            )
            == 1
        )
        if action is ProviderOperationResolutionAction.FALLBACK_RETRY:
            assert provider.adapter.start_calls == 2
            replacement_start = next(
                index
                for index, event in enumerate(durable)
                if event.type is EventType.PROVIDER_OPERATION_STARTING
                and event.payload.get("model_attempt_id") != active.stage.intent["model_attempt_id"]
            )
            settlement_index = next(
                index
                for index, event in enumerate(durable)
                if event.type is EventType.BUDGET_RECONCILED
            )
            assert settlement_index < replacement_start
        else:
            assert provider.adapter.start_calls == 1
            final = await store.load(session_id)
            assert final is not None
            assert final.status is SessionStatus.FAILED

    asyncio.run(scenario())


def test_concurrent_exact_resolution_replays_one_budget_settlement() -> None:
    async def scenario() -> None:
        session_id = "concurrent-budgeted-provider-resolution"
        store = InMemorySessionStore()
        ledger = _ConcurrentResolutionReservationLoadLedger()
        provider = _ResumableBudgetedOfflineOperationProvider()
        app = CayuApp(
            session_store=store,
            budget_ledger=ledger,
            budget_policy=_budget_policy(provider.name),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        with pytest.raises(_SimulatedProcessLoss):
            async for _event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "settle one exact concurrent disposition")],
                )
            ):
                pass
        active = await store.load_active_model_completion_stage(session_id)
        assert active is not None
        [reservation_id] = active.stage.reservation_ids
        provider.adapter.status = ProviderOperationStatus.UNAVAILABLE
        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )
        interrupted = await store.load(session_id)
        assert interrupted is not None
        request = ProviderOperationResolutionRequest(
            session_id=session_id,
            stage_id=active.stage.stage_id,
            expected_run_epoch=interrupted.run_epoch,
            action=ProviderOperationResolutionAction.FAIL,
            reason="accept one exact concurrent failure disposition",
        )
        ledger.block_next_two_loads()

        async def collect_resolution() -> list[Event]:
            return [event async for event in app.resolve_provider_operation(request)]

        first = asyncio.create_task(collect_resolution())
        second = asyncio.create_task(collect_resolution())
        await asyncio.wait_for(ledger.concurrent_loads_entered.wait(), timeout=1)
        ledger.concurrent_loads_release.set()
        first_events, second_events = await asyncio.gather(first, second)

        assert first_events[0].type is EventType.PROVIDER_OPERATION_RESOLVED
        assert second_events[0].type is EventType.PROVIDER_OPERATION_RESOLVED
        settlement = await ledger.load_settlement(budget_settlement_id(reservation_id))
        assert settlement is not None
        assert settlement.reconciliation.settled_at == first_events[0].timestamp
        durable = await store.load_events(session_id)
        assert (
            sum(
                event.type is EventType.BUDGET_RECONCILED
                and event.payload.get("reservation_id") == reservation_id
                for event in durable
            )
            == 1
        )
        assert sum(event.type is EventType.SESSION_FAILED for event in durable) == 1
        assert await load_pending_provider_operation_disposition(store, session_id) is None
        assert provider.adapter.start_calls == 1

    asyncio.run(scenario())


def test_operator_resolution_recovers_lost_budget_settlement_acknowledgement() -> None:
    async def scenario() -> None:
        session_id = "provider-resolution-budget-acknowledgement-loss"
        store = InMemorySessionStore()
        ledger = _LoseFirstRecoverySettlementAcknowledgement()
        provider = _ResumableBudgetedOfflineOperationProvider()

        def runtime() -> CayuApp:
            app = CayuApp(
                session_store=store,
                budget_ledger=ledger,
                budget_policy=_budget_policy(provider.name),
                enable_logging=False,
            )
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            return app

        with pytest.raises(_SimulatedProcessLoss):
            async for _event in runtime().run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "recover settlement acknowledgement loss")],
                )
            ):
                pass
        active = await store.load_active_model_completion_stage(session_id)
        assert active is not None
        [reservation_id] = active.stage.reservation_ids
        provider.adapter.status = ProviderOperationStatus.UNAVAILABLE
        await runtime().recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )
        interrupted = await store.load(session_id)
        assert interrupted is not None
        request = ProviderOperationResolutionRequest(
            session_id=session_id,
            stage_id=active.stage.stage_id,
            expected_run_epoch=interrupted.run_epoch,
            action=ProviderOperationResolutionAction.FAIL,
            reason="finish failure after settlement acknowledgement loss",
        )

        with pytest.raises(RuntimeError, match="acknowledgement lost"):
            _ = [event async for event in runtime().resolve_provider_operation(request)]

        committed = await ledger.load_reservation(reservation_id)
        assert committed is not None
        assert committed.status == "reconciled"
        retained = await ledger.load_settlement(budget_settlement_id(reservation_id))
        assert retained is not None
        assert retained.event_published is False
        assert await load_pending_provider_operation_disposition(store, session_id) is not None

        replay = [event async for event in runtime().resolve_provider_operation(request)]

        assert replay[0].type is EventType.PROVIDER_OPERATION_RESOLVED
        assert EventType.BUDGET_RECONCILED in {event.type for event in replay}
        assert replay[-1].type is EventType.SESSION_FAILED
        assert ledger.reconcile_calls == 1
        repaired = await ledger.load_settlement(budget_settlement_id(reservation_id))
        assert repaired is not None
        assert repaired.event_published is True
        assert await load_pending_provider_operation_disposition(store, session_id) is None
        assert (
            sum(
                event.type is EventType.BUDGET_RECONCILED
                and event.payload.get("reservation_id") == reservation_id
                for event in await store.load_events(session_id)
            )
            == 1
        )

    asyncio.run(scenario())


def test_competing_provider_recovery_workers_settle_original_reservation_once() -> None:
    async def scenario() -> None:
        session_id = "offline-budget-competing-recovery"
        store = InMemorySessionStore()
        ledger = InMemoryBudgetLedger(reservation_ttl_seconds=None)
        provider = _BudgetedOfflineOperationProvider()
        budget_policy = _budget_policy(provider.name)

        def runtime() -> CayuApp:
            app = CayuApp(
                session_store=store,
                budget_ledger=ledger,
                budget_policy=budget_policy,
                enable_logging=False,
            )
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="fake-model"))
            return app

        with pytest.raises(_SimulatedProcessLoss):
            async for _event in runtime().run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "recover exactly once")],
                )
            ):
                pass
        stage = await store.load_active_model_completion_stage(session_id)
        assert stage is not None
        [reservation_id] = stage.stage.reservation_ids
        provider.adapter.status = ProviderOperationStatus.COMPLETED
        request = IncompleteSessionRecoveryRequest(
            session_id=session_id,
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )

        results = await asyncio.gather(
            runtime().recover_incomplete_session(request),
            runtime().recover_incomplete_session(request),
            return_exceptions=True,
        )

        assert all(not isinstance(result, BaseException) for result in results), results
        assert provider.adapter.retrieve_calls == [provider.adapter.state]
        durable = await store.load_events(session_id)
        matching = [
            event
            for event in durable
            if event.type is EventType.BUDGET_RECONCILED
            and event.payload.get("reservation_id") == reservation_id
        ]
        assert len(matching) == 1
        assert sum(event.type is EventType.MODEL_COMPLETED for event in durable) == 1
        settlement = await ledger.load_settlement(budget_settlement_id(reservation_id))
        assert settlement is not None
        assert settlement.reservation_id == reservation_id
        assert settlement.event_published is True

    asyncio.run(scenario())


def test_interruption_after_worker_loss_cancels_the_durable_provider_operation() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _CancellableOfflineOperationProvider()
        await _stage_offline_operation(
            store,
            session_id="offline-provider-cancellation",
            provider=provider,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="offline-provider-cancellation",
                    reason="cancel after worker loss",
                )
            )
        ]

        assert [event.type for event in events] == [EventType.SESSION_INTERRUPTED]
        assert provider.adapter.cancel_calls == [provider.adapter.state]
        durable_events = await store.load_events("offline-provider-cancellation")
        assert [
            event.payload["cancellation_status"]
            for event in durable_events
            if event.type
            in {
                EventType.PROVIDER_OPERATION_CANCEL_REQUESTED,
                EventType.PROVIDER_OPERATION_CANCEL_RESOLVED,
            }
        ] == ["requested", "cancelled"]
        inspection = await inspect_provider_operation(store, "offline-provider-cancellation")
        assert inspection.cancellation_status is ProviderOperationCancellationStatus.CANCELLED
        assert inspection.accounting_status is ProviderOperationAccountingStatus.NOT_APPLICABLE

    asyncio.run(scenario())


def test_worker_loss_interruption_reports_unsupported_provider_cancellation() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _OfflineOperationProvider(ProviderOperationStatus.IN_PROGRESS)
        await _stage_offline_operation(
            store,
            session_id="offline-provider-cancellation-unsupported",
            provider=provider,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="offline-provider-cancellation-unsupported",
                    reason="interrupt locally even when provider cancellation is unavailable",
                )
            )
        ]

        assert [event.type for event in events] == [EventType.SESSION_INTERRUPTED]
        inspection = await inspect_provider_operation(
            store,
            "offline-provider-cancellation-unsupported",
        )
        assert inspection.cancellation_status is ProviderOperationCancellationStatus.UNSUPPORTED
        assert inspection.accounting_status is ProviderOperationAccountingStatus.NOT_APPLICABLE

    asyncio.run(scenario())


def test_interruption_claim_acknowledgement_loss_releases_committed_epoch() -> None:
    async def scenario() -> None:
        session_id = "offline-provider-interruption-claim-ack-loss"
        store = _CommitThenRaiseInterruptionClaimStore()
        provider = _OfflineOperationProvider(ProviderOperationStatus.IN_PROGRESS)
        await _stage_offline_operation(store, session_id=session_id, provider=provider)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        with pytest.raises(
            ConnectionError,
            match="interruption claim acknowledgement lost after commit",
        ):
            async for _event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id=session_id,
                    reason="lose the interruption claim acknowledgement",
                )
            ):
                pass

        assert store.interruption_claim_committed is True
        interrupted = await store.load(session_id)
        interrupted_profile = active_invocation_execution_profile_from_checkpoint(
            await store.load_checkpoint(session_id)
        )
        assert interrupted is not None
        assert interrupted.status is SessionStatus.INTERRUPTING
        assert interrupted_profile is not None
        assert interrupted_profile.run_epoch == interrupted.run_epoch - 1

        replacement = await store.fence_stalled_run(
            session_id,
            statuses={SessionStatus.INTERRUPTING},
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )
        assert replacement is not None
        await store.release_run_fence(session_id)

    asyncio.run(scenario())


def test_budgeted_unsupported_cancellation_retains_then_settles_original_reservation() -> None:
    async def scenario() -> None:
        session_id = "offline-budget-cancellation-unsupported"
        store = InMemorySessionStore()
        ledger = InMemoryBudgetLedger(reservation_ttl_seconds=None)
        provider = _ResumableBudgetedOfflineOperationProvider()
        app = CayuApp(
            session_store=store,
            budget_ledger=ledger,
            budget_policy=_budget_policy(provider.name),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        with pytest.raises(_SimulatedProcessLoss):
            async for _event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "continue outside this worker")],
                )
            ):
                pass
        stage = await store.load_active_model_completion_stage(session_id)
        assert stage is not None
        [reservation_id] = stage.stage.reservation_ids

        async for _event in app.interrupt_session(
            InterruptSessionRequest(
                session_id=session_id,
                reason="interrupt locally without claiming provider cancellation",
            )
        ):
            pass

        retained = await ledger.load_reservation(reservation_id)
        assert retained is not None
        assert retained.status == "active"
        assert await store.load_active_model_completion_stage(session_id) is not None
        interrupted = await inspect_provider_operation(store, session_id)
        assert interrupted.cancellation_status is ProviderOperationCancellationStatus.UNSUPPORTED
        assert interrupted.accounting_status is ProviderOperationAccountingStatus.RESERVED
        interrupted_session = await store.load(session_id)
        interrupted_profile = active_invocation_execution_profile_from_checkpoint(
            await store.load_checkpoint(session_id)
        )
        assert interrupted_session is not None
        assert interrupted_profile is not None
        assert interrupted_profile.run_epoch == interrupted_session.run_epoch - 1

        provider.adapter.status = ProviderOperationStatus.COMPLETED
        provider.adapter.start_events = (
            ModelStreamEvent.text_delta("continued after recovery"),
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ),
        )
        resumed = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "continue")],
                )
            )
        ]
        assert resumed[-1].type is EventType.SESSION_COMPLETED

        settled = await ledger.load_reservation(reservation_id)
        assert settled is not None
        assert settled.status == "reconciled"
        durable = await store.load_events(session_id)
        matching = [
            event
            for event in durable
            if event.type is EventType.BUDGET_RECONCILED
            and event.payload.get("reservation_id") == reservation_id
        ]
        assert len(matching) == 1
        assert provider.adapter.retrieve_calls == [provider.adapter.state]

    asyncio.run(scenario())


def test_stale_worker_loss_owner_cannot_cancel_or_resolve_provider_operation() -> None:
    async def scenario() -> None:
        store = _FenceOnCancellationRequestStore()
        provider = _CancellableOfflineOperationProvider()
        await _stage_offline_operation(
            store,
            session_id="offline-provider-cancellation-fenced",
            provider=provider,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        with pytest.raises(SessionRunFenced):
            async for _event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="offline-provider-cancellation-fenced",
                    reason="race a replacement owner",
                )
            ):
                pass

        assert store.fenced is True
        assert provider.adapter.cancel_calls == []
        cancellation_events = [
            event
            for event in await store.load_events("offline-provider-cancellation-fenced")
            if event.type
            in {
                EventType.PROVIDER_OPERATION_CANCEL_REQUESTED,
                EventType.PROVIDER_OPERATION_CANCEL_RESOLVED,
            }
        ]
        assert cancellation_events == []

    asyncio.run(scenario())


def test_offline_interruption_keeps_provider_frozen_across_status_transition() -> None:
    async def scenario() -> None:
        store = _BlockingInterruptionTransitionStore()
        provider = _CancellableOfflineOperationProvider()
        replacement_provider = _CancellableOfflineOperationProvider()
        hook = _RecordingInterruptedProfileHook()
        session_id = "offline-interruption-frozen-provider"
        await _stage_offline_operation(store, session_id=session_id, provider=provider)
        active_profile = active_invocation_execution_profile_from_checkpoint(
            await store.load_checkpoint(session_id)
        )
        assert active_profile is not None

        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            runtime_hooks=[hook],
        )
        replacement_app = CayuApp(enable_logging=False)
        replacement_app.register_provider(replacement_provider, default=True)

        async def interrupt() -> list[Event]:
            return [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(
                        session_id=session_id,
                        reason="freeze the admitted provider runtime",
                    )
                )
            ]

        interrupt_task = asyncio.create_task(interrupt())
        await asyncio.wait_for(store.transition_entered.wait(), timeout=1)
        app._providers[provider.name] = replacement_app._providers[provider.name]
        store.transition_release.set()

        assert [event.type for event in await interrupt_task] == [
            EventType.SESSION_INTERRUPTED,
            EventType.HOOK_STARTED,
            EventType.HOOK_COMPLETED,
        ]
        assert provider.adapter.cancel_calls == [provider.adapter.state]
        assert replacement_provider.adapter.cancel_calls == []
        assert hook.execution_profiles == [active_profile.profile]
        assert hook.execution_profiles[0] is not None

    asyncio.run(scenario())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_cancellation_claim_blocks_epoch_takeover_during_provider_call(
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store: SessionStore = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / "cancellation-claim.sqlite3")
        )
        provider = _BlockingCancellationProvider()
        session_id = "offline-provider-cancellation-claimed"
        await _stage_offline_operation(store, session_id=session_id, provider=provider)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        async def interrupt() -> list[Event]:
            return [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(
                        session_id=session_id,
                        reason="hold durable cancellation ownership",
                    )
                )
            ]

        interrupt_task = asyncio.create_task(interrupt())
        await asyncio.wait_for(provider.adapter.cancel_entered.wait(), timeout=1)
        before = await store.load(session_id)
        assert before is not None
        fenced = await store.fence_stalled_run(
            session_id,
            statuses={SessionStatus.INTERRUPTING},
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )
        assert fenced is None
        during = await store.load(session_id)
        assert during is not None
        assert during.run_epoch == before.run_epoch

        provider.adapter.cancel_release.set()
        assert [event.type for event in await interrupt_task] == [EventType.SESSION_INTERRUPTED]
        try:
            inspection = await inspect_provider_operation(store, session_id)
            assert inspection.cancellation_status is ProviderOperationCancellationStatus.CANCELLED
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


def test_cancellation_claim_heartbeat_failure_stops_the_active_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_step_executor,
        "_PROVIDER_OPERATION_CANCELLATION_CLAIM_LEASE",
        timedelta(milliseconds=50),
    )
    monkeypatch.setattr(
        model_step_executor,
        "_PROVIDER_OPERATION_CANCELLATION_CLAIM_HEARTBEAT_SECONDS",
        0.005,
    )

    async def scenario() -> None:
        store = _FailCancellationClaimHeartbeatStore()
        provider = _BlockingCancellationProvider()
        session_id = "provider-cancellation-heartbeat-lost"
        await _stage_offline_operation(store, session_id=session_id, provider=provider)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        async def interrupt() -> None:
            async for _event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id=session_id,
                    reason="fail cancellation ownership heartbeat",
                )
            ):
                pass

        interrupt_task = asyncio.create_task(interrupt())
        await asyncio.wait_for(provider.adapter.cancel_entered.wait(), timeout=1)
        await asyncio.wait_for(store.heartbeat_failed.wait(), timeout=1)
        await asyncio.wait_for(provider.adapter.cancel_exited.wait(), timeout=1)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(interrupt_task, timeout=1)

        await asyncio.sleep(0.06)
        fenced = await store.fence_stalled_run(
            session_id,
            statuses={SessionStatus.INTERRUPTING},
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )
        assert fenced is not None

    asyncio.run(scenario())


def test_successful_claim_release_does_not_trigger_heartbeat_ownership_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_step_executor,
        "_PROVIDER_OPERATION_CANCELLATION_CLAIM_LEASE",
        timedelta(milliseconds=50),
    )
    monkeypatch.setattr(
        model_step_executor,
        "_PROVIDER_OPERATION_CANCELLATION_CLAIM_HEARTBEAT_SECONDS",
        0.005,
    )

    async def scenario() -> None:
        store = _DelayCancellationResolutionAcknowledgementStore()
        provider = _CancellableOfflineOperationProvider()
        session_id = "provider-cancellation-release-heartbeat"
        await _stage_offline_operation(store, session_id=session_id, provider=provider)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id=session_id,
                    reason="release cancellation ownership normally",
                )
            )
        ]

        assert [event.type for event in events] == [EventType.SESSION_INTERRUPTED]
        current = await store.load(session_id)
        assert current is not None
        assert current.status is SessionStatus.INTERRUPTED
        assert provider.adapter.cancel_calls == [provider.adapter.state]

    asyncio.run(scenario())


def test_provider_completion_winning_cancellation_is_reconciled_before_interruption() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _CompletionWinsCancellationProvider()
        session_id = "offline-provider-completion-wins-cancel"
        await _stage_offline_operation(store, session_id=session_id, provider=provider)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        emitted = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id=session_id,
                    reason="provider completed before cancellation took effect",
                )
            )
        ]

        durable = await store.load_events(session_id)
        assert provider.adapter.cancel_calls == [provider.adapter.state]
        assert sum(event.type is EventType.MODEL_COMPLETED for event in durable) == 1
        completed = next(event for event in durable if event.type is EventType.MODEL_COMPLETED)
        assert completed.interaction_id is not None, completed.model_dump(mode="json")
        assert await store.load_active_model_completion_stage(session_id) is None
        assert EventType.SESSION_INTERRUPTED in {event.type for event in emitted}
        inspection = await inspect_provider_operation(store, session_id)
        assert inspection.cancellation_status is ProviderOperationCancellationStatus.COMPLETED
        assert inspection.status is ProviderOperationInspectionStatus.PROVIDER_OPERATION_RECONCILED

        interrupted_session = await store.load(session_id)
        interrupted_profile = active_invocation_execution_profile_from_checkpoint(
            await store.load_checkpoint(session_id)
        )
        assert interrupted_session is not None
        assert interrupted_profile is not None
        assert interrupted_profile.run_epoch == interrupted_session.run_epoch - 1

        provider.adapter.start_events = (
            ModelStreamEvent.text_delta("new invocation after interruption"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        )
        resumed = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "continue after interruption")],
                )
            )
        ]
        assert resumed[-1].type is EventType.SESSION_COMPLETED
        assert provider.adapter.start_calls == 1

    asyncio.run(scenario())


def test_lost_cancellation_acknowledgement_remains_truthfully_unconfirmed() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _LostCancellationAcknowledgementProvider()
        session_id = "offline-provider-cancellation-ack-loss"
        await _stage_offline_operation(store, session_id=session_id, provider=provider)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id=session_id,
                    reason="provider acknowledgement may be lost",
                )
            )
        ]

        assert [event.type for event in events] == [EventType.SESSION_INTERRUPTED]
        assert provider.adapter.cancel_calls == [provider.adapter.state]
        inspection = await inspect_provider_operation(store, session_id)
        assert inspection.cancellation_status is ProviderOperationCancellationStatus.FAILED
        assert inspection.accounting_status is ProviderOperationAccountingStatus.NOT_APPLICABLE
        resolved = [
            event
            for event in await store.load_events(session_id)
            if event.type is EventType.PROVIDER_OPERATION_CANCEL_RESOLVED
        ]
        assert len(resolved) == 1
        assert resolved[0].payload["error_type"] == "CancellationUnconfirmed"
        assert await store.load_active_model_completion_stage(session_id) is not None

    asyncio.run(scenario())


def test_confirmed_offline_cancellation_settles_the_original_reservation() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        ledger = InMemoryBudgetLedger(reservation_ttl_seconds=None)
        provider = _CancellableBudgetedOfflineOperationProvider()
        budget_policy = _budget_policy(provider.name)
        app = CayuApp(
            session_store=store,
            budget_ledger=ledger,
            budget_policy=budget_policy,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        with pytest.raises(_SimulatedProcessLoss):
            async for _event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="offline-budget-cancellation",
                    messages=[Message.text("user", "cancel after the worker is lost")],
                )
            ):
                pass
        stage = await store.load_active_model_completion_stage("offline-budget-cancellation")
        assert stage is not None

        async for _event in app.interrupt_session(
            InterruptSessionRequest(
                session_id="offline-budget-cancellation",
                reason="cancel the exact provider job",
            )
        ):
            pass

        events = await store.load_events("offline-budget-cancellation")
        reserved = [event for event in events if event.type is EventType.BUDGET_RESERVED]
        reconciled = [event for event in events if event.type is EventType.BUDGET_RECONCILED]
        assert len(reserved) == 1
        assert len(reconciled) == 1
        assert reconciled[0].payload["reservation_id"] == stage.stage.reservation_ids[0]
        assert reconciled[0].payload["settlement_kind"] == "conservative"
        assert provider.adapter.cancel_calls == [provider.adapter.state]
        inspection = await inspect_provider_operation(store, "offline-budget-cancellation")
        assert inspection.cancellation_status is ProviderOperationCancellationStatus.CANCELLED
        assert inspection.accounting_status is ProviderOperationAccountingStatus.SETTLED
        assert inspection.reservation_count == 1

        unrelated: list[Event] = []
        for index in range(33):
            reservation_id = f"bres_unrelated_{index:02}"
            settlement_id = budget_settlement_id(reservation_id)
            unrelated.append(
                reconciled[0].model_copy(
                    update={
                        "id": budget_settlement_event_id(settlement_id),
                        "payload": {
                            **reconciled[0].payload,
                            "reservation_id": reservation_id,
                            "settlement_id": settlement_id,
                        },
                    },
                    deep=True,
                )
            )
        await store.append_events("offline-budget-cancellation", unrelated)

        after_unrelated = await inspect_provider_operation(
            store,
            "offline-budget-cancellation",
        )
        assert after_unrelated.accounting_status is ProviderOperationAccountingStatus.SETTLED
        assert after_unrelated.reservation_count == 1

    asyncio.run(scenario())


def test_cancellation_claim_blocks_epoch_takeover_during_budget_settlement() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        ledger = _BlockingReservationLoadLedger()
        provider = _CancellableBudgetedOfflineOperationProvider()
        budget_policy = _budget_policy(provider.name)
        app = CayuApp(
            session_store=store,
            budget_ledger=ledger,
            budget_policy=budget_policy,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "offline-budget-cancellation-claimed"

        with pytest.raises(_SimulatedProcessLoss):
            async for _event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "cancel after the worker is lost")],
                )
            ):
                pass

        async def interrupt() -> None:
            async for _event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id=session_id,
                    reason="retain settlement ownership",
                )
            ):
                pass

        interrupt_task = asyncio.create_task(interrupt())
        await asyncio.wait_for(ledger.load_entered.wait(), timeout=1)
        before = await store.load(session_id)
        assert before is not None
        fenced = await store.fence_stalled_run(
            session_id,
            statuses={SessionStatus.INTERRUPTING},
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )
        assert fenced is None
        during = await store.load(session_id)
        assert during is not None
        assert during.run_epoch == before.run_epoch

        ledger.load_release.set()
        await interrupt_task
        events = await store.load_events(session_id)
        assert sum(event.type is EventType.BUDGET_RECONCILED for event in events) == 1

    asyncio.run(scenario())


def test_live_cancellation_claim_blocks_takeover_through_budget_settlement() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        ledger = _BlockingReconciliationLedger()
        provider = _LiveCancellableBudgetedOperationProvider()
        app = CayuApp(
            session_store=store,
            budget_ledger=ledger,
            budget_policy=_budget_policy(provider.name),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        session_id = "live-budget-cancellation-claimed"

        async def run_session() -> list[Event]:
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "cancel and settle exactly once")],
                    )
                )
            ]

        run_task = asyncio.create_task(run_session())
        await asyncio.wait_for(provider.adapter.stream_started.wait(), timeout=1)

        async def interrupt() -> list[Event]:
            return [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(
                        session_id=session_id,
                        reason="retain settlement ownership",
                    )
                )
            ]

        interrupt_task = asyncio.create_task(interrupt())
        await asyncio.wait_for(ledger.reconcile_entered.wait(), timeout=1)
        before = await store.load(session_id)
        assert before is not None
        fenced = await store.fence_stalled_run(
            session_id,
            statuses={SessionStatus.INTERRUPTING},
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )
        assert fenced is None
        during = await store.load(session_id)
        assert during is not None
        assert during.run_epoch == before.run_epoch

        ledger.reconcile_release.set()
        run_events = await run_task
        interrupt_events = await interrupt_task
        durable = await store.load_events(session_id)
        assert EventType.SESSION_INTERRUPTED in {event.type for event in run_events}
        assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
        assert sum(event.type is EventType.BUDGET_RECONCILED for event in durable) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "crash_after",
    [
        EventType.PROVIDER_OPERATION_CANCEL_REQUESTED,
        EventType.PROVIDER_OPERATION_CANCEL_RESOLVED,
    ],
)
def test_expired_cancellation_claim_allows_worker_loss_takeover(
    crash_after: EventType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        model_step_executor,
        "_PROVIDER_OPERATION_CANCELLATION_CLAIM_LEASE",
        timedelta(milliseconds=10),
    )
    monkeypatch.setattr(
        model_step_executor,
        "_PROVIDER_OPERATION_CANCELLATION_CLAIM_HEARTBEAT_SECONDS",
        3600.0,
    )

    async def scenario() -> None:
        def fixed_clock() -> datetime:
            return datetime(2020, 1, 1, tzinfo=UTC)

        store = _CrashAfterCancellationEventStore(crash_after)
        session_id = f"expired-cancellation-claim-{crash_after.value}"
        if crash_after is EventType.PROVIDER_OPERATION_CANCEL_RESOLVED:
            provider = _CancellableBudgetedOfflineOperationProvider()
            ledger = InMemoryBudgetLedger(reservation_ttl_seconds=None)
            app = CayuApp(
                session_store=store,
                budget_ledger=ledger,
                budget_policy=_budget_policy(provider.name),
                clock=fixed_clock,
                enable_logging=False,
            )
        else:
            provider = _CancellableOfflineOperationProvider()
            app = CayuApp(
                session_store=store,
                clock=fixed_clock,
                enable_logging=False,
            )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        if crash_after is EventType.PROVIDER_OPERATION_CANCEL_RESOLVED:
            with pytest.raises(_SimulatedProcessLoss):
                async for _event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "lose worker before cancellation")],
                    )
                ):
                    pass
        else:
            await _stage_offline_operation(
                store,
                session_id=session_id,
                provider=provider,
                started_at=fixed_clock(),
            )

        with pytest.raises(_SimulatedProcessLoss):
            async for _event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id=session_id,
                    reason="simulate process loss under cancellation ownership",
                )
            ):
                pass

        await asyncio.sleep(0.02)
        recovery = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )
        assert recovery.status is SessionStatus.INTERRUPTED
        current = await store.load(session_id)
        assert current is not None
        assert current.status is SessionStatus.INTERRUPTED
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        assert "__cayu_provider_operation_cancellation_claim_v1__" not in checkpoint
        expected_cancel_calls = (
            2 if crash_after is EventType.PROVIDER_OPERATION_CANCEL_RESOLVED else 1
        )
        assert provider.adapter.cancel_calls == [provider.adapter.state] * expected_cancel_calls
        if crash_after is EventType.PROVIDER_OPERATION_CANCEL_RESOLVED:
            durable = await store.load_events(session_id)
            assert sum(event.type is EventType.BUDGET_RECONCILED for event in durable) == 1
            inspection = await inspect_provider_operation(store, session_id)
            assert inspection.accounting_status is ProviderOperationAccountingStatus.SETTLED

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "initial_status",
    [ProviderOperationStatus.QUEUED, ProviderOperationStatus.IN_PROGRESS],
)
@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_pending_operation_later_completes_without_losing_publication_eligibility(
    initial_status: ProviderOperationStatus,
    store_kind: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        store = (
            InMemorySessionStore()
            if store_kind == "memory"
            else SQLiteSessionStore(tmp_path / f"{initial_status.value}.sqlite3")
        )
        await assert_pending_provider_operation_later_completes(
            store,
            initial_status=initial_status,
        )
        if isinstance(store, SQLiteSessionStore):
            await store.close()

    asyncio.run(scenario())


def test_recovery_rejects_an_operation_after_provider_output_was_accepted() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _OfflineOperationProvider(ProviderOperationStatus.COMPLETED)
        await _stage_offline_operation(
            store,
            session_id="offline-partial-output",
            provider=provider,
        )
        await store.append_events(
            "offline-partial-output",
            [
                Event(
                    type=EventType.MODEL_TEXT_DELTA,
                    session_id="offline-partial-output",
                    agent_name="assistant",
                    payload={
                        "delta": "already accepted",
                        "model_step_id": "mstep_" + "a" * 32,
                        "model_attempt_id": "matt_" + "b" * 32,
                    },
                )
            ],
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        with pytest.raises(ModelCompletionManualRecoveryRequired):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id="offline-partial-output",
                    inactive_before=datetime.now(UTC) + timedelta(seconds=1),
                )
            )

        assert provider.adapter.retrieve_calls == []
        assert await store.load_transcript("offline-partial-output") == [
            Message.text("user", "finish this while no worker is attached")
        ]

    asyncio.run(scenario())


def test_offline_recovery_preserves_request_billing_identity() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        identity = BillingIdentity(
            provider_name="offline-operation",
            resource_id="offline-model",
            request_evidence={"region": "us-test-1"},
        )
        provider = _IdentityAwareOfflineProvider(identity)
        context = ModelCompletionRecoveryContext(
            billing_identity=identity,
        )
        await _stage_offline_operation(
            store,
            session_id="offline-billing-identity",
            provider=provider,
            recovery_context=context.model_dump(mode="json"),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="offline-billing-identity",
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )

        completed = next(
            event
            for event in await store.load_events("offline-billing-identity")
            if event.type is EventType.MODEL_COMPLETED
        )
        assert completed.payload["billing_identity"] == identity.model_dump(mode="json")

    asyncio.run(scenario())


def test_offline_recovery_honors_hidden_thinking_transcript_policy() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _OfflineOperationProvider(
            ProviderOperationStatus.COMPLETED,
            events=(
                ModelStreamEvent.thinking("private reasoning"),
                ModelStreamEvent.text_delta("public answer"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ),
        )
        context = ModelCompletionRecoveryContext(
            thinking=ThinkingConfig(include_in_transcript=False),
        )
        await _stage_offline_operation(
            store,
            session_id="offline-hidden-thinking",
            provider=provider,
            recovery_context=context.model_dump(mode="json"),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="offline-hidden-thinking",
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )

        assistant = (await store.load_transcript("offline-hidden-thinking"))[1]
        assert not any(isinstance(part, ThinkingPart) for part in assistant.content)
        assert assistant.content[0].text == "public answer"

    asyncio.run(scenario())


def test_offline_recovery_accepts_hidden_thinking_only_completion() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _OfflineOperationProvider(
            ProviderOperationStatus.COMPLETED,
            events=(
                ModelStreamEvent.thinking("private reasoning"),
                ModelStreamEvent.completed({"finish_reason": "length"}),
            ),
        )
        context = ModelCompletionRecoveryContext(
            thinking=ThinkingConfig(include_in_transcript=False),
        )
        session_id = "offline-hidden-thinking-only"
        user_message = await _stage_offline_operation(
            store,
            session_id=session_id,
            provider=provider,
            recovery_context=context.model_dump(mode="json"),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )
        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )

        assert provider.adapter.start_calls == 0
        assert provider.adapter.retrieve_calls == [provider.adapter.state]
        assert await store.load_transcript(session_id) == [user_message]
        events = await store.load_events(session_id)
        completed_events = [event for event in events if event.type is EventType.MODEL_COMPLETED]
        assert len(completed_events) == 1
        assert completed_events[0].payload["transcript_cursor"] == 1
        assert await store.load_active_model_completion_stage(session_id) is None
        inspection = await inspect_provider_operation(store, session_id)
        assert inspection.status is ProviderOperationInspectionStatus.PROVIDER_OPERATION_RECONCILED

    asyncio.run(scenario())


def test_offline_recovery_restores_structured_output_tool_contract() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _OfflineOperationProvider(
            ProviderOperationStatus.COMPLETED,
            events=(
                ModelStreamEvent.tool_call(
                    id="structured-output-call",
                    name=STRUCTURED_OUTPUT_TOOL_NAME,
                    arguments={"output": {"answer": "recovered"}},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ),
        )
        spec = StructuredOutputSpec(
            name="answer",
            json_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        )
        context = ModelCompletionRecoveryContext(
            request_metadata={"traceparent": "00-" + "1" * 32 + "-" + "2" * 16 + "-01"},
            structured_output=spec,
            max_steps=3,
            limits=RunLimits(max_tool_calls=2),
            retry_policy=RetryPolicy(max_attempts=2),
            structured_output_attempt=1,
        )
        await _stage_offline_operation(
            store,
            session_id="offline-structured-output",
            provider=provider,
            recovery_context=context.model_dump(mode="json"),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        result = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="offline-structured-output",
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )

        event_types = {event.type for event in result.events}
        assert EventType.STRUCTURED_OUTPUT_VALIDATING in event_types
        assert EventType.STRUCTURED_OUTPUT_VALIDATED in event_types
        assert EventType.TOOL_CALL_STARTED not in event_types

    asyncio.run(scenario())


def test_offline_recovery_preserves_an_ordinary_tool_call_during_structured_output() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _OfflineOperationProvider(
            ProviderOperationStatus.COMPLETED,
            events=(
                ModelStreamEvent.tool_call(
                    id="ordinary-tool-call",
                    name="lookup",
                    arguments={"query": "recovered"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ),
        )
        spec = StructuredOutputSpec(
            name="answer",
            json_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
            max_retries=1,
        )
        context = ModelCompletionRecoveryContext(
            structured_output=spec,
            structured_output_attempt=2,
        )
        session_id = "offline-structured-output-ordinary-tool"
        await _stage_offline_operation(
            store,
            session_id=session_id,
            provider=provider,
            recovery_context=context.model_dump(mode="json"),
            tools=(_LookupTool(),),
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="assistant", model="fake-model"),
            tools=[_LookupTool()],
            tool_policy=AllowAllToolPolicy(),
        )

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=session_id,
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )

        events = await store.load_events(session_id)
        completed = next(event for event in events if event.type is EventType.MODEL_COMPLETED)
        assert completed.payload["step_classification"]["type"] == "continue"
        assert EventType.STRUCTURED_OUTPUT_VALIDATING not in {event.type for event in events}
        stage = await store.load_model_completion_stage(
            session_id,
            "mstep_" + "a" * 32 + ":dispatch:0",
        )
        assert stage is not None
        assert stage.publication is not None
        pending_operation = next(
            operation
            for operation in stage.publication.mutation.operations
            if operation.key == "pending_tool_round"
        )
        assert pending_operation.value["structured_output_retries"] == 1
        checkpoint = await store.load_checkpoint(session_id)
        assert checkpoint is not None
        checkpoint.pop("pending_tool_approval")
        pending_round = checkpoint["pending_tool_round"]
        pending_round["policy_state"] = "planned"
        pending_round["policy_context_version"] = 1
        for call in pending_round["tool_calls"]:
            call["policy_evidence"] = "authoritative"
            call["policy_decision"] = "allow"
        await store.checkpoint(session_id, checkpoint)

        provider.adapter.start_events = (
            ModelStreamEvent.tool_call(
                id="invalid-finalizer-after-ordinary-tool",
                name=STRUCTURED_OUTPUT_TOOL_NAME,
                arguments={"output": {"wrong": "value"}},
            ),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        )
        resumed_events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id=session_id,
                    messages=[Message.text("user", "continue")],
                    structured_output=spec,
                )
            )
        ]
        failed = [
            event for event in resumed_events if event.type is EventType.STRUCTURED_OUTPUT_FAILED
        ]
        assert [event.payload["attempt"] for event in failed] == [2]
        assert EventType.STRUCTURED_OUTPUT_RETRY not in {event.type for event in resumed_events}
        assert provider.adapter.start_calls == 1

    asyncio.run(scenario())


def test_queued_operation_remains_recoverable_without_redispatch() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _OfflineOperationProvider(ProviderOperationStatus.QUEUED)
        user_message = await _stage_offline_operation(
            store,
            session_id="offline-queued",
            provider=provider,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="offline-queued",
                inactive_before=datetime.now(UTC) + timedelta(seconds=1),
            )
        )
        for _ in range(4):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id="offline-queued")
            )

        assert provider.adapter.start_calls == 0
        assert provider.adapter.retrieve_calls == [provider.adapter.state] * 5
        assert await store.load_transcript("offline-queued") == [user_message]
        assert await store.load_active_model_completion_stage("offline-queued") is not None
        inspection = await inspect_provider_operation(store, "offline-queued")
        assert inspection.status is ProviderOperationInspectionStatus.RECONNECT_SCHEDULED

    asyncio.run(scenario())


def test_resume_retrieves_queued_operation_then_relinquishes_run_fence() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _OfflineOperationProvider(ProviderOperationStatus.QUEUED)
        await _stage_offline_operation(
            store,
            session_id="resume-offline-queued",
            provider=provider,
        )
        await store.update_status("resume-offline-queued", SessionStatus.INTERRUPTED)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))

        events = [
            event
            async for event in app.resume(
                ResumeRequest(
                    session_id="resume-offline-queued",
                    messages=[Message.text("user", "continue when the operation finishes")],
                )
            )
        ]

        assert provider.adapter.start_calls == 0
        assert provider.adapter.retrieve_calls == [provider.adapter.state]
        assert EventType.SESSION_FAILED not in {event.type for event in events}
        assert EventType.PROVIDER_OPERATION_RECONNECT_STARTED in {event.type for event in events}
        persisted = await store.load("resume-offline-queued")
        assert persisted is not None
        assert persisted.status is SessionStatus.INTERRUPTED
        assert await store.load_active_model_completion_stage("resume-offline-queued") is not None

    asyncio.run(scenario())


def test_competing_offline_recovery_workers_converge_on_one_completion() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _OfflineOperationProvider(ProviderOperationStatus.COMPLETED)
        await _stage_offline_operation(
            store,
            session_id="competing-offline-recovery",
            provider=provider,
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        request = IncompleteSessionRecoveryRequest(
            session_id="competing-offline-recovery",
            inactive_before=datetime.now(UTC) + timedelta(seconds=1),
        )

        await asyncio.gather(
            app.recover_incomplete_session(request),
            app.recover_incomplete_session(request),
        )

        assert provider.adapter.start_calls == 0
        assert provider.adapter.retrieve_calls == [provider.adapter.state]
        events = await store.load_events("competing-offline-recovery")
        assert sum(event.type is EventType.MODEL_COMPLETED for event in events) == 1
        assert len(await store.load_transcript("competing-offline-recovery")) == 2

    asyncio.run(scenario())
