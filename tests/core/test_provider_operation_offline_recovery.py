from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

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
    IncompleteSessionRecoveryRequest,
    InMemoryBudgetLedger,
    InMemorySessionStore,
    InteractionStatus,
    InteractionSummaryEvidence,
    InterruptSessionRequest,
    ResumeRequest,
    RetryPolicy,
    RunLimits,
    RunRequest,
    SessionIdentity,
    SessionRunFenced,
    SessionStatus,
    SessionStore,
    ToolApprovalDecision,
    ToolApprovalRequest,
)
from cayu.runtime import _model_step_executor as model_step_executor
from cayu.runtime._model_step_executor import ModelCompletionRecoveryContext
from cayu.runtime._recovery_coordinator import ModelCompletionManualRecoveryRequired
from cayu.runtime.budgets import (
    _effective_budget_limit_id,
    budget_settlement_event_id,
    budget_settlement_id,
    request_budget_limits_for_session,
)
from cayu.runtime.costs import ModelPrice, PriceBook
from cayu.runtime.execution_units import ModelAttemptIdentity
from cayu.runtime.provider_operations import (
    ProviderOperationAccountingStatus,
    ProviderOperationCancellationStatus,
    ProviderOperationInspectionStatus,
    inspect_provider_operation,
)
from cayu.runtime.sessions import ModelCompletionStageRequest
from cayu.runtime.structured_output import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    StructuredOutputSpec,
)
from cayu.runtime.usage import SessionUsageSummary


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


class _BudgetedOfflineOperationProvider(_OfflineOperationProvider):
    name = "budgeted-offline-operation"

    def __init__(self) -> None:
        self.adapter = _BudgetedOfflineOperationAdapter()


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
    session = await store.create(
        RunRequest(
            agent_name="assistant",
            session_id=session_id,
            messages=[user_message],
        ),
        identity=SessionIdentity(provider_name=provider.name, model="fake-model"),
        interaction_started_event=started_event,
        interaction_source_messages=[user_message],
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
    if recovery_context is not None:
        intent["recovery_context"] = recovery_context
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
                    "step": 1,
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
                    "step": 1,
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
    await _stage_offline_operation(
        store,
        session_id=session_id,
        provider=provider,
        recovery_context=recovery_context,
        started_at=started_at,
        prior_events=prior_events,
    )

    tool = _RecordingLookupTool()
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
