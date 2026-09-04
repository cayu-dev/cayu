from __future__ import annotations

import asyncio
import contextvars
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Any
from uuid import UUID

import pytest

import cayu.runtime.budgets as budgets_module
from cayu import CayuConfig, RunDefaults
from cayu.core import AgentSpec, Event, EventType, Message
from cayu.core.tools import Tool, ToolContext, ToolResult, ToolSpec
from cayu.providers import (
    ModelContextOverflowError,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
)
from cayu.runtime import (
    AlwaysRequireApprovalToolPolicy,
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    CayuApp,
    InMemoryBudgetLedger,
    InMemorySessionStore,
    ModelPrice,
    PriceBook,
    RecentTurnsContextPolicy,
    RetryPolicy,
    RunRequest,
    SessionStatus,
    ToolApprovalDecision,
    ToolApprovalRequest,
)
from cayu.runtime.context import (
    ContextBuildResult,
    ContextRecallTelemetry,
    ContextRequest,
    RuntimeManagedContextPolicy,
)
from cayu.runtime.costs import Provenance
from cayu.runtime.sessions import SessionRunFenced
from cayu.vaults import REDACTED_SECRET, SecretRedactor


class _UsageProvider(ModelProvider):
    name = "identity-budget"

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.calls += 1
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 1_000_000,
                    "output_tokens": 0,
                    "total_tokens": 1_000_000,
                },
            }
        )


class _NoUsageProvider(ModelProvider):
    name = "identity-budget"

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.calls += 1
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _OverflowThenNoUsageProvider(ModelProvider):
    name = "identity-budget"

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.calls += 1
        if self.calls == 1:
            raise ModelContextOverflowError(
                "context too large",
                provider=self.name,
                status_code=400,
                error_code="context_length_exceeded",
            )
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _ApprovalUsageProvider(ModelProvider):
    name = "identity-budget"

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.calls += 1
        if self.calls == 1:
            yield ModelStreamEvent.tool_call(
                id="call_1",
                name="record",
                arguments={},
            )
            yield ModelStreamEvent.completed(
                {
                    "finish_reason": "tool_calls",
                    "usage": {
                        "input_tokens": 1_000_000,
                        "output_tokens": 0,
                        "total_tokens": 1_000_000,
                    },
                }
            )
            return
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _RecordingTool(Tool):
    spec = ToolSpec(name="record", input_schema={"type": "object"})

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx, args
        self.calls += 1
        return ToolResult(content="recorded")


def _price_book() -> PriceBook:
    return PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="identity-budget",
                model="identity-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
            ),
        )
    )


def _collect_events(
    *,
    session_id: str,
    budget_policy: BudgetPolicy | None = None,
    budget_limits: tuple[BudgetLimit, ...] = (),
) -> list[Event]:
    app = CayuApp(budget_policy=budget_policy, enable_logging=False)
    app.register_provider(_UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="identity-model"))

    async def collect() -> list[Event]:
        return [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "answer")],
                    budget_limits=budget_limits,
                )
            )
        ]

    return asyncio.run(collect())


async def _collect_app_events(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]


async def _collect_approval_events(
    app: CayuApp,
    request: ToolApprovalRequest,
) -> list[Event]:
    return [event async for event in app.resolve_tool_approval(request)]


def _model_attempt_payload(event: Event) -> dict[str, object]:
    return {
        "model_step_id": event.payload.get("model_step_id"),
        "model_attempt_id": event.payload.get("model_attempt_id"),
    }


@pytest.mark.parametrize(
    ("provider_event", "terminal_type"),
    [
        (
            ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "model_step_id": f"mstep_{'a' * 32}",
                    "model_attempt_id": f"matt_{'b' * 32}",
                    "tool_round_id": f"tround_{'c' * 32}",
                    "budget_limit_id": f"blim_{'d' * 64}",
                    "reservation_id": f"bres_{'e' * 32}",
                }
            ),
            EventType.MODEL_COMPLETED,
        ),
        (
            ModelStreamEvent(
                type="error",
                payload={
                    "error": "provider failed",
                    "model_step_id": f"mstep_{'a' * 32}",
                    "model_attempt_id": f"matt_{'b' * 32}",
                    "tool_round_id": f"tround_{'c' * 32}",
                    "budget_limit_id": f"blim_{'d' * 64}",
                    "reservation_id": f"bres_{'e' * 32}",
                },
            ),
            EventType.MODEL_ERROR,
        ),
    ],
)
def test_runtime_replaces_all_provider_supplied_execution_identity(
    provider_event: ModelStreamEvent,
    terminal_type: EventType,
) -> None:
    class SpoofingProvider(ModelProvider):
        name = "identity-budget"

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            yield provider_event

    app = CayuApp(enable_logging=False)
    app.register_provider(SpoofingProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="identity-model"))

    events = asyncio.run(
        _collect_app_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id=f"sess_provider_identity_{terminal_type.value}",
                messages=[Message.text("user", "answer")],
            ),
        )
    )

    terminal = next(event for event in events if event.type == terminal_type)
    assert terminal.payload["model_step_id"] != provider_event.payload["model_step_id"]
    assert terminal.payload["model_attempt_id"] != provider_event.payload["model_attempt_id"]
    for field_name in ("tool_round_id", "budget_limit_id", "reservation_id"):
        assert field_name not in terminal.payload


def test_runtime_replaces_all_custom_recall_telemetry_identity() -> None:
    forged_identity = {
        "model_step_id": f"mstep_{'a' * 32}",
        "model_attempt_id": f"matt_{'b' * 32}",
        "tool_round_id": f"tround_{'c' * 32}",
        "budget_limit_id": f"blim_{'d' * 64}",
        "reservation_id": f"bres_{'e' * 32}",
    }

    class SpoofingContextPolicy(RuntimeManagedContextPolicy):
        async def build_with_checkpoint(
            self,
            request: ContextRequest,
            *,
            checkpoint: dict[str, Any] | None,
        ) -> ContextBuildResult:
            del checkpoint
            return ContextBuildResult(
                messages=request.messages,
                recall_telemetry=[
                    ContextRecallTelemetry(
                        event_type=EventType.AUTOMATIC_RECALL_STARTED,
                        payload=forged_identity,
                    )
                ],
            )

    app = CayuApp(enable_logging=False)
    app.register_provider(_NoUsageProvider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="identity-model"),
        context_policy=SpoofingContextPolicy(),
    )

    events = asyncio.run(
        _collect_app_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_custom_recall_identity",
                messages=[Message.text("user", "answer")],
            ),
        )
    )

    telemetry = next(event for event in events if event.type == EventType.AUTOMATIC_RECALL_STARTED)
    assert telemetry.payload["model_step_id"] != forged_identity["model_step_id"]
    for field_name in (
        "model_attempt_id",
        "tool_round_id",
        "budget_limit_id",
        "reservation_id",
    ):
        assert field_name not in telemetry.payload


def test_post_completion_app_budget_check_retains_the_completed_attempt() -> None:
    events = _collect_events(
        session_id="sess_post_completion_app_budget_identity",
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("10"),
                    pricing=_price_book(),
                ),
            )
        ),
    )

    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    checks = [event for event in events if event.type == EventType.BUDGET_CHECKED]

    assert len(checks) == 2
    assert checks[0].payload["model_step_id"] == completed.payload["model_step_id"]
    assert "model_attempt_id" not in checks[0].payload
    assert _model_attempt_payload(checks[1]) == _model_attempt_payload(completed)
    assert checks[0].payload["budget_limit_id"] == checks[1].payload["budget_limit_id"]


def test_post_completion_request_budget_notification_retains_the_completed_attempt() -> None:
    events = _collect_events(
        session_id="sess_post_completion_request_budget_identity",
        budget_limits=(
            BudgetLimit(
                scope="session",
                max_estimated_cost=Decimal("1"),
                pricing=_price_book(),
                action="notify",
            ),
        ),
    )

    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    reached = [event for event in events if event.type == EventType.BUDGET_LIMIT_REACHED]

    assert len(reached) == 1
    assert reached[0].payload["actual"] == "1"
    assert _model_attempt_payload(reached[0]) == _model_attempt_payload(completed)


def test_approval_resume_budget_notification_retains_the_originating_attempt() -> None:
    store = InMemorySessionStore()
    provider = _ApprovalUsageProvider()
    tool = _RecordingTool()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="identity-model"),
        tools=[tool],
        tool_policy=AlwaysRequireApprovalToolPolicy(),
    )
    budget_limit = BudgetLimit(
        scope="session",
        max_estimated_cost=Decimal("0.5"),
        pricing=_price_book(),
        action="notify",
    )

    paused = asyncio.run(
        _collect_app_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_approval_budget_identity",
                messages=[Message.text("user", "record once")],
                budget_limits=(budget_limit,),
            ),
        )
    )
    approval_requested = next(
        event for event in paused if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )

    resumed = asyncio.run(
        _collect_approval_events(
            app,
            ToolApprovalRequest(
                session_id="sess_approval_budget_identity",
                approval_id=approval_requested.payload["approval_id"],
                tool_round_id=approval_requested.payload["tool_round_id"],
                tool_call_id=approval_requested.payload["tool_call_id"],
                decision=ToolApprovalDecision.APPROVE,
                budget_limits=(budget_limit,),
            ),
        )
    )
    approval_budget_notification = next(
        event for event in resumed if event.type == EventType.BUDGET_LIMIT_REACHED
    )

    assert provider.calls == 2
    assert tool.calls == 1
    assert _model_attempt_payload(approval_budget_notification) == _model_attempt_payload(
        approval_requested
    )


def _reservation_limit() -> BudgetLimit:
    return BudgetLimit(
        scope="app",
        max_estimated_cost=Decimal("10"),
        pricing=_price_book(),
        reservation=BudgetReservation(
            max_input_tokens=1_000_000,
            max_output_tokens=0,
        ),
    )


def test_reservation_identity_matching_a_workload_secret_fails_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_uuid = UUID("12345678-1234-5678-1234-567812345678")
    reservation_id = f"bres_{fixed_uuid.hex}"
    monkeypatch.setattr(budgets_module, "uuid4", lambda: fixed_uuid)
    provider = _UsageProvider()
    store = InMemorySessionStore()
    ledger = InMemoryBudgetLedger()
    app = CayuApp(
        session_store=store,
        budget_ledger=ledger,
        budget_policy=BudgetPolicy(limits=(_reservation_limit(),)),
        secret_redactor=SecretRedactor(reservation_id),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="identity-model"))

    events = asyncio.run(
        _collect_app_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_secret_reservation_identity",
                messages=[Message.text("user", "answer")],
            ),
        )
    )
    persisted = asyncio.run(store.load_events("sess_secret_reservation_identity"))

    assert provider.calls == 0
    assert not any(event.type == EventType.BUDGET_RESERVED for event in persisted)
    assert not any(
        event.payload.get("reservation_id") in {reservation_id, REDACTED_SECRET}
        for event in [*events, *persisted]
    )
    assert ledger._records == {}
    assert ledger._settlements == {}


def test_context_overflow_terminalizes_the_reserved_attempt_for_inspection() -> None:
    provider = _OverflowThenNoUsageProvider()
    app = CayuApp(
        budget_policy=BudgetPolicy(limits=(_reservation_limit(),)),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="identity-model"),
        context_overflow_policy=RecentTurnsContextPolicy(max_user_turns=1),
    )

    events = asyncio.run(
        _collect_app_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_context_overflow_budget_identity",
                messages=[
                    Message.text("user", "old request"),
                    Message.text("user", "new request"),
                ],
            ),
        )
    )
    inspection = asyncio.run(
        app.session_store.inspect_summary("sess_context_overflow_budget_identity")
    )

    reservations = [event for event in events if event.type == EventType.BUDGET_RESERVED]
    reconciliations = [event for event in events if event.type == EventType.BUDGET_RECONCILED]
    model_errors = [event for event in events if event.type == EventType.MODEL_ERROR]
    completions = [event for event in events if event.type == EventType.MODEL_COMPLETED]

    assert provider.calls == 2
    assert len(reservations) == len(reconciliations) == 2
    assert len(model_errors) == len(completions) == 1
    assert _model_attempt_payload(reservations[0]) == _model_attempt_payload(model_errors[0])
    assert _model_attempt_payload(reconciliations[0]) == _model_attempt_payload(model_errors[0])
    assert _model_attempt_payload(reservations[1]) == _model_attempt_payload(completions[0])
    assert _model_attempt_payload(reconciliations[1]) == _model_attempt_payload(completions[0])
    interaction_ids = {
        event.interaction_id
        for event in [*reservations, *reconciliations, *model_errors, *completions]
    }
    assert None not in interaction_ids
    assert len(interaction_ids) == 1
    assert inspection.budget.cost_state == "unpriced"
    assert inspection.budget.amount is None
    assert inspection.budget.currency is None


def test_runtime_rejects_reservation_identity_rewritten_by_custom_ledger() -> None:
    class RewritingReservationLedger(InMemoryBudgetLedger):
        async def reserve(self, **kwargs):
            result = await super().reserve(**kwargs)
            assert result.record is not None
            rewritten_record = result.record.model_copy(
                update={"model_attempt_id": f"matt_{'e' * 32}"}
            )
            return result.model_copy(
                update={
                    "model_attempt_id": rewritten_record.model_attempt_id,
                    "record": rewritten_record,
                }
            )

    provider = _UsageProvider()
    app = CayuApp(
        budget_policy=BudgetPolicy(limits=(_reservation_limit(),)),
        budget_ledger=RewritingReservationLedger(),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="identity-model"))

    events = asyncio.run(
        _collect_app_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_rewritten_reservation_identity",
                messages=[Message.text("user", "answer")],
            ),
        )
    )

    assert provider.calls == 0
    assert EventType.BUDGET_RESERVED not in {event.type for event in events}
    assert events[-1].type == EventType.SESSION_FAILED
    assert (
        events[-1].payload["error"]
        == "Budget ledger reservation result changed its requested identity."
    )


def test_runtime_rejects_reservation_amount_rewritten_by_custom_ledger() -> None:
    class UnderReservingLedger(InMemoryBudgetLedger):
        async def reserve(self, **kwargs):
            result = await super().reserve(**kwargs)
            assert result.record is not None
            rewritten_record = result.record.model_copy(update={"reserved_amount": Decimal("0")})
            return result.model_copy(
                update={
                    "requested": Decimal("0"),
                    "record": rewritten_record,
                }
            )

    provider = _UsageProvider()
    app = CayuApp(
        budget_policy=BudgetPolicy(limits=(_reservation_limit(),)),
        budget_ledger=UnderReservingLedger(),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="identity-model"))

    events = asyncio.run(
        _collect_app_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_rewritten_reservation_amount",
                messages=[Message.text("user", "answer")],
            ),
        )
    )

    assert provider.calls == 0
    assert EventType.BUDGET_RESERVED not in {event.type for event in events}
    assert events[-1].type == EventType.SESSION_FAILED
    assert (
        events[-1].payload["error"]
        == "Budget ledger reservation result changed its requested amount."
    )


def test_runtime_rejects_accepted_reservation_above_configured_maximum() -> None:
    class OverCapLedger(InMemoryBudgetLedger):
        async def reserve(self, **kwargs):
            result = await super().reserve(**kwargs)
            assert result.accepted is True
            return result.model_copy(update={"actual": result.maximum + Decimal("1")})

    provider = _UsageProvider()
    app = CayuApp(
        budget_policy=BudgetPolicy(limits=(_reservation_limit(),)),
        budget_ledger=OverCapLedger(),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="identity-model"))

    events = asyncio.run(
        _collect_app_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_accepted_reservation_over_cap",
                messages=[Message.text("user", "answer")],
            ),
        )
    )

    assert provider.calls == 0
    assert EventType.BUDGET_RESERVED not in {event.type for event in events}
    assert events[-1].type == EventType.SESSION_FAILED
    assert (
        events[-1].payload["error"]
        == "Accepted budget reservation violated its configured maximum."
    )


def test_runtime_rejects_duplicate_reservation_ids_before_provider_dispatch() -> None:
    class DuplicateReservationIdLedger(InMemoryBudgetLedger):
        first_reservation_id: str | None = None

        async def reserve(self, **kwargs):
            result = await super().reserve(**kwargs)
            assert result.record is not None
            if self.first_reservation_id is None:
                self.first_reservation_id = result.record.reservation_id
                return result
            return result.model_copy(
                update={
                    "record": result.record.model_copy(
                        update={"reservation_id": self.first_reservation_id}
                    )
                }
            )

    provider = _UsageProvider()
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                _reservation_limit(),
                _reservation_limit(),
            )
        ),
        budget_ledger=DuplicateReservationIdLedger(),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="identity-model"))

    events = asyncio.run(
        _collect_app_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_duplicate_reservation_identity",
                messages=[Message.text("user", "answer")],
            ),
        )
    )

    assert provider.calls == 0
    assert sum(event.type == EventType.BUDGET_RESERVED for event in events) == 1
    assert sum(event.type == EventType.BUDGET_RESERVATION_RELEASED for event in events) == 1
    assert EventType.BUDGET_RECONCILED not in {event.type for event in events}
    assert events[-1].type == EventType.SESSION_FAILED
    assert events[-1].payload["error"] == "Budget ledger reused a reservation identity."


def test_runtime_rejects_reservation_id_reuse_across_retry_dispatches() -> None:
    class ReusingReservationIdLedger(InMemoryBudgetLedger):
        first_reservation_id: str | None = None

        async def reserve(self, **kwargs):
            result = await super().reserve(**kwargs)
            assert result.record is not None
            if self.first_reservation_id is None:
                self.first_reservation_id = result.record.reservation_id
                return result
            return result.model_copy(
                update={
                    "record": result.record.model_copy(
                        update={"reservation_id": self.first_reservation_id}
                    )
                }
            )

    class RetryOnceProvider(ModelProvider):
        name = "identity-budget"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            self.calls += 1
            if self.calls == 1:
                raise ModelProviderError(
                    "temporary provider failure",
                    provider=self.name,
                    retryable=True,
                )
            yield ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 1_000_000,
                        "output_tokens": 0,
                        "total_tokens": 1_000_000,
                    },
                }
            )

    provider = RetryOnceProvider()
    app = CayuApp(
        config=CayuConfig(
            run=RunDefaults(retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0))
        ),
        budget_policy=BudgetPolicy(limits=(_reservation_limit(),)),
        budget_ledger=ReusingReservationIdLedger(),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="identity-model"))

    events = asyncio.run(
        _collect_app_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_reused_retry_reservation_identity",
                messages=[Message.text("user", "answer")],
            ),
        )
    )

    assert provider.calls == 1
    assert sum(event.type == EventType.BUDGET_RESERVED for event in events) == 1
    assert sum(event.type == EventType.BUDGET_RECONCILED for event in events) == 1
    assert events[-1].type == EventType.SESSION_FAILED
    assert events[-1].payload["error"] == "temporary provider failure"


def test_runtime_rejects_reservation_id_reuse_across_logical_model_steps() -> None:
    class ReusingReservationIdLedger(InMemoryBudgetLedger):
        first_reservation_id: str | None = None

        def __init__(self) -> None:
            super().__init__(reservation_ttl_seconds=None)

        async def reserve(self, **kwargs):
            result = await super().reserve(**kwargs)
            assert result.record is not None
            if self.first_reservation_id is None:
                self.first_reservation_id = result.record.reservation_id
                return result
            return result.model_copy(
                update={
                    "record": result.record.model_copy(
                        update={"reservation_id": self.first_reservation_id}
                    )
                }
            )

    class TwoStepProvider(ModelProvider):
        name = "identity-budget"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            self.calls += 1
            if self.calls == 1:
                yield ModelStreamEvent.tool_call(
                    id="call_1",
                    name="record",
                    arguments={},
                )
                yield ModelStreamEvent.completed(
                    {
                        "finish_reason": "tool_calls",
                        "usage": {
                            "input_tokens": 1_000_000,
                            "output_tokens": 0,
                            "total_tokens": 1_000_000,
                        },
                    }
                )
                return
            yield ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 1_000_000,
                        "output_tokens": 0,
                        "total_tokens": 1_000_000,
                    },
                }
            )

    class RecordingTool(Tool):
        spec = ToolSpec(name="record", input_schema={"type": "object"})

        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            del ctx, args
            self.calls += 1
            return ToolResult(content="recorded")

    provider = TwoStepProvider()
    tool = RecordingTool()
    app = CayuApp(
        budget_policy=BudgetPolicy(limits=(_reservation_limit(),)),
        budget_ledger=ReusingReservationIdLedger(),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="identity-model"),
        tools=[tool],
    )

    events = asyncio.run(
        _collect_app_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_reused_later_step_reservation_identity",
                messages=[Message.text("user", "record once")],
            ),
        )
    )

    assert provider.calls == 1
    assert tool.calls == 1
    assert sum(event.type == EventType.BUDGET_RESERVED for event in events) == 1
    assert sum(event.type == EventType.BUDGET_RECONCILED for event in events) == 1
    assert events[-1].type == EventType.SESSION_FAILED
    assert events[-1].payload["error"] == "Budget ledger reused a reservation identity."


def test_runtime_rejects_reservation_id_reuse_across_sessions() -> None:
    class ReusingReservationIdLedger(InMemoryBudgetLedger):
        first_reservation_id: str | None = None

        def __init__(self) -> None:
            super().__init__(reservation_ttl_seconds=None)

        async def reserve(self, **kwargs):
            result = await super().reserve(**kwargs)
            assert result.record is not None
            if self.first_reservation_id is None:
                self.first_reservation_id = result.record.reservation_id
                return result
            return result.model_copy(
                update={
                    "record": result.record.model_copy(
                        update={"reservation_id": self.first_reservation_id}
                    )
                }
            )

    async def scenario() -> tuple[_UsageProvider, list[Event], list[Event]]:
        provider = _UsageProvider()
        app = CayuApp(
            budget_policy=BudgetPolicy(limits=(_reservation_limit(),)),
            budget_ledger=ReusingReservationIdLedger(),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="identity-model"))
        first_events = await _collect_app_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_reused_reservation_first",
                messages=[Message.text("user", "first")],
            ),
        )
        second_events = await _collect_app_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_reused_reservation_second",
                messages=[Message.text("user", "second")],
            ),
        )
        return provider, first_events, second_events

    provider, first_events, second_events = asyncio.run(scenario())

    assert provider.calls == 1
    assert sum(event.type == EventType.BUDGET_RESERVED for event in first_events) == 1
    assert EventType.BUDGET_RESERVED not in {event.type for event in second_events}
    assert second_events[-1].type == EventType.SESSION_FAILED
    assert second_events[-1].payload["error"] == "Budget ledger reused a reservation identity."


@pytest.mark.parametrize(
    "shared_session_store",
    [True, False],
    ids=["shared-session-store", "split-session-stores"],
)
def test_concurrent_reservation_collision_does_not_release_the_winner(
    shared_session_store: bool,
) -> None:
    class BlockingProvider(_UsageProvider):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            del request
            self.calls += 1
            self.started.set()
            await self.release.wait()
            yield ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 1_000_000,
                        "output_tokens": 0,
                        "total_tokens": 1_000_000,
                    },
                }
            )

    class CoordinatedDuplicateLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__(reservation_ttl_seconds=None)
            self.first_reservation_id: str | None = None
            self.second_reserved = asyncio.Event()
            self.release_second_result = asyncio.Event()

        async def reserve(self, **kwargs):
            result = await super().reserve(**kwargs)
            assert result.record is not None
            if kwargs["session_id"] == "sess_collision_winner":
                self.first_reservation_id = result.record.reservation_id
                await self.second_reserved.wait()
                return result
            self.second_reserved.set()
            await self.release_second_result.wait()
            assert self.first_reservation_id is not None
            return result.model_copy(
                update={
                    "record": result.record.model_copy(
                        update={"reservation_id": self.first_reservation_id}
                    )
                }
            )

    async def scenario() -> tuple[list[Event], list[Event], bool, BlockingProvider]:
        shared_store = InMemorySessionStore()
        ledger = CoordinatedDuplicateLedger()
        provider = BlockingProvider()
        policy = BudgetPolicy(limits=(_reservation_limit(),))

        def app() -> CayuApp:
            runtime = CayuApp(
                session_store=(shared_store if shared_session_store else InMemorySessionStore()),
                budget_policy=policy,
                budget_ledger=ledger,
                enable_logging=False,
            )
            runtime.register_provider(provider, default=True)
            runtime.register_agent(AgentSpec(name="assistant", model="identity-model"))
            return runtime

        winner_task = asyncio.create_task(
            _collect_app_events(
                app(),
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_collision_winner",
                    messages=[Message.text("user", "winner")],
                ),
            )
        )
        loser_task = asyncio.create_task(
            _collect_app_events(
                app(),
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_collision_loser",
                    messages=[Message.text("user", "loser")],
                ),
            )
        )
        await asyncio.wait_for(provider.started.wait(), timeout=5)
        ledger.release_second_result.set()
        loser_events = await asyncio.wait_for(loser_task, timeout=5)
        assert ledger.first_reservation_id is not None
        winner_still_active = await ledger.heartbeat(reservation_id=ledger.first_reservation_id)
        provider.release.set()
        winner_events = await asyncio.wait_for(winner_task, timeout=5)
        return winner_events, loser_events, winner_still_active, provider

    winner_events, loser_events, winner_still_active, provider = asyncio.run(scenario())

    assert winner_still_active is True
    assert provider.calls == 1
    assert any(event.type == EventType.BUDGET_RECONCILED for event in winner_events)
    assert winner_events[-1].type == EventType.SESSION_COMPLETED
    assert EventType.BUDGET_RESERVATION_RELEASED not in {event.type for event in loser_events}
    assert loser_events[-1].type == EventType.SESSION_FAILED
    assert loser_events[-1].payload["error"] == "Budget ledger reused a reservation identity."


def test_stale_run_cannot_claim_or_release_a_reserved_identity() -> None:
    class TakeoverBeforeClaimStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.replacement_epoch: int | None = None

        async def claim_budget_reservation_identity(
            self,
            *,
            reservation_id: str,
            publication_session_id: str,
            publication_id: str,
        ) -> None:
            async def replace_owner() -> None:
                replacement = await self.fence_stalled_run(
                    publication_session_id,
                    statuses={SessionStatus.RUNNING},
                    inactive_for_seconds=0,
                )
                assert replacement is not None
                self.replacement_epoch = replacement.run_epoch

            await asyncio.create_task(
                replace_owner(),
                context=contextvars.Context(),
            )
            await super().claim_budget_reservation_identity(
                reservation_id=reservation_id,
                publication_session_id=publication_session_id,
                publication_id=publication_id,
            )

    class RecordingLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__(reservation_ttl_seconds=None)
            self.reservation_id: str | None = None

        async def reserve(self, **kwargs):
            result = await super().reserve(**kwargs)
            assert result.record is not None
            self.reservation_id = result.record.reservation_id
            return result

    async def scenario() -> tuple[TakeoverBeforeClaimStore, RecordingLedger, _UsageProvider]:
        store = TakeoverBeforeClaimStore()
        ledger = RecordingLedger()
        provider = _UsageProvider()
        app = CayuApp(
            session_store=store,
            budget_policy=BudgetPolicy(limits=(_reservation_limit(),)),
            budget_ledger=ledger,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="identity-model"))

        with pytest.raises(
            SessionRunFenced,
            match="(?:Session run epoch no longer owns|Invocation command lost its run epoch)",
        ):
            await _collect_app_events(
                app,
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_fenced_reservation_claim",
                    messages=[Message.text("user", "answer")],
                ),
            )
        assert ledger.reservation_id is not None
        assert await ledger.heartbeat(reservation_id=ledger.reservation_id)
        return store, ledger, provider

    store, ledger, provider = asyncio.run(scenario())

    assert store.replacement_epoch is not None
    assert ledger.reservation_id is not None
    assert provider.calls == 0


def test_runtime_uses_registry_without_scanning_reservation_history() -> None:
    class NoReservationHistoryScanStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def query_events(self, query=None):
            if (
                query is not None
                and query.session_id is None
                and query.event_type == EventType.BUDGET_RESERVED
            ):
                raise AssertionError("reservation admission scanned the full event history")
            return await super().query_events(query)

    class ReusingHistoricalReservationIdLedger(InMemoryBudgetLedger):
        def __init__(self, reservation_id: str) -> None:
            super().__init__(reservation_ttl_seconds=None)
            self._reservation_id = reservation_id
            self.reserve_calls = 0

        async def reserve(self, **kwargs):
            self.reserve_calls += 1
            result = await super().reserve(**kwargs)
            assert result.record is not None
            return result.model_copy(
                update={
                    "record": result.record.model_copy(
                        update={"reservation_id": self._reservation_id}
                    )
                }
            )

    async def scenario() -> tuple[list[Event], list[Event], _UsageProvider]:
        store = NoReservationHistoryScanStore()
        policy = BudgetPolicy(limits=(_reservation_limit(),))
        first_provider = _UsageProvider()
        first_app = CayuApp(
            session_store=store,
            budget_policy=policy,
            enable_logging=False,
        )
        first_app.register_provider(first_provider, default=True)
        first_app.register_agent(AgentSpec(name="assistant", model="identity-model"))
        first_request = RunRequest(
            agent_name="assistant",
            session_id="sess_reconstructed_reservation_identity_first",
            messages=[Message.text("user", "answer")],
        )
        first_events = await _collect_app_events(first_app, first_request)
        reservation_id = next(
            event.payload["reservation_id"]
            for event in first_events
            if event.type == EventType.BUDGET_RESERVED
        )

        second_ledger = ReusingHistoricalReservationIdLedger(reservation_id)
        second_provider = _UsageProvider()
        second_app = CayuApp(
            session_store=store,
            budget_policy=policy,
            budget_ledger=second_ledger,
            enable_logging=False,
        )
        second_app.register_provider(second_provider, default=True)
        second_app.register_agent(AgentSpec(name="assistant", model="identity-model"))
        second_events = await _collect_app_events(
            second_app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_reconstructed_reservation_identity_second",
                messages=[Message.text("user", "answer")],
            ),
        )
        assert second_ledger.reserve_calls == 1
        return first_events, second_events, second_provider

    first_events, second_events, second_provider = asyncio.run(scenario())

    assert sum(event.type == EventType.BUDGET_RESERVED for event in first_events) == 1
    assert second_provider.calls == 0
    assert EventType.BUDGET_RESERVED not in {event.type for event in second_events}
    assert second_events[-1].type == EventType.SESSION_FAILED
    assert second_events[-1].payload["error"] == "Budget ledger reused a reservation identity."


def test_runtime_detaches_effective_limit_before_custom_ledger_reservation() -> None:
    forged_limit_id = f"blim_{'e' * 64}"

    class MutatingLimitLedger(InMemoryBudgetLedger):
        async def reserve(self, *, limit, **kwargs):
            object.__setattr__(limit, "budget_limit_id", forged_limit_id)
            return await super().reserve(limit=limit, **kwargs)

    provider = _UsageProvider()
    app = CayuApp(
        budget_policy=BudgetPolicy(limits=(_reservation_limit(),)),
        budget_ledger=MutatingLimitLedger(),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="identity-model"))

    events = asyncio.run(
        _collect_app_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_mutated_effective_limit",
                messages=[Message.text("user", "answer")],
            ),
        )
    )

    assert provider.calls == 0
    assert EventType.BUDGET_RESERVED not in {event.type for event in events}
    assert events[-1].type == EventType.SESSION_FAILED
    assert (
        events[-1].payload["error"]
        == "Budget ledger reservation result changed its requested identity."
    )


def test_runtime_rejects_settlement_identity_rewritten_by_custom_ledger() -> None:
    class RewritingSettlementLedger(InMemoryBudgetLedger):
        async def reconcile(self, **kwargs):
            reconciliation = await super().reconcile(**kwargs)
            return reconciliation.model_copy(update={"model_attempt_id": f"matt_{'e' * 32}"})

    provider = _UsageProvider()
    app = CayuApp(
        budget_policy=BudgetPolicy(limits=(_reservation_limit(),)),
        budget_ledger=RewritingSettlementLedger(),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="identity-model"))

    events = asyncio.run(
        _collect_app_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_rewritten_settlement_identity",
                messages=[Message.text("user", "answer")],
            ),
        )
    )

    assert provider.calls == 1
    assert EventType.MODEL_COMPLETED in {event.type for event in events}
    assert EventType.BUDGET_RECONCILED not in {event.type for event in events}
    assert events[-1].type == EventType.SESSION_FAILED
    assert events[-1].payload["error"] == "Budget ledger settlement changed its requested outcome."


def test_runtime_rejects_settlement_id_rewritten_by_custom_ledger() -> None:
    class RewritingSettlementLedger(InMemoryBudgetLedger):
        async def reconcile(self, **kwargs):
            reconciliation = await super().reconcile(**kwargs)
            return reconciliation.model_copy(update={"settlement_id": "forged-custom-settlement"})

    provider = _UsageProvider()
    app = CayuApp(
        budget_policy=BudgetPolicy(limits=(_reservation_limit(),)),
        budget_ledger=RewritingSettlementLedger(),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="identity-model"))

    events = asyncio.run(
        _collect_app_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_rewritten_settlement_id",
                messages=[Message.text("user", "answer")],
            ),
        )
    )

    assert provider.calls == 1
    assert EventType.MODEL_COMPLETED in {event.type for event in events}
    assert EventType.BUDGET_RECONCILED not in {event.type for event in events}
    assert events[-1].type == EventType.SESSION_FAILED
    assert events[-1].payload["error"] == "Budget ledger settlement changed its requested outcome."


def test_runtime_detaches_reservation_record_before_custom_ledger_reconciliation() -> None:
    forged_attempt_id = f"matt_{'e' * 32}"

    class MutatingSettlementLedger(InMemoryBudgetLedger):
        retained_record = None

        async def reserve(self, **kwargs):
            result = await super().reserve(**kwargs)
            assert result.record is not None
            self.retained_record = result.record
            return result

        async def reconcile(self, **kwargs):
            reconciliation = await super().reconcile(**kwargs)
            assert self.retained_record is not None
            object.__setattr__(
                self.retained_record,
                "model_attempt_id",
                forged_attempt_id,
            )
            return reconciliation.model_copy(update={"model_attempt_id": forged_attempt_id})

    provider = _UsageProvider()
    app = CayuApp(
        budget_policy=BudgetPolicy(limits=(_reservation_limit(),)),
        budget_ledger=MutatingSettlementLedger(),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="identity-model"))

    events = asyncio.run(
        _collect_app_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_mutated_retained_reservation_reconciliation",
                messages=[Message.text("user", "answer")],
            ),
        )
    )

    reserved = next(event for event in events if event.type == EventType.BUDGET_RESERVED)
    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert provider.calls == 1
    assert _model_attempt_payload(reserved) == _model_attempt_payload(completed)
    assert reserved.payload["model_attempt_id"] != forged_attempt_id
    assert EventType.BUDGET_RECONCILED not in {event.type for event in events}
    assert events[-1].type == EventType.SESSION_FAILED
    assert events[-1].payload["error"] == "Budget ledger settlement changed its requested outcome."


def test_runtime_detaches_reservation_record_before_custom_ledger_release() -> None:
    forged_attempt_id = f"matt_{'e' * 32}"

    class MutatingReleaseLedger(InMemoryBudgetLedger):
        retained_record = None

        async def reserve(self, **kwargs):
            result = await super().reserve(**kwargs)
            assert result.record is not None
            self.retained_record = result.record
            return result

        async def heartbeat(self, *, reservation_id: str) -> bool:
            del reservation_id
            return False

        async def release(self, **kwargs):
            reconciliation = await super().release(**kwargs)
            assert self.retained_record is not None
            object.__setattr__(
                self.retained_record,
                "model_attempt_id",
                forged_attempt_id,
            )
            return reconciliation.model_copy(update={"model_attempt_id": forged_attempt_id})

    provider = _UsageProvider()
    app = CayuApp(
        budget_policy=BudgetPolicy(limits=(_reservation_limit(),)),
        budget_ledger=MutatingReleaseLedger(),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="identity-model"))

    events = asyncio.run(
        _collect_app_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_mutated_retained_reservation_release",
                messages=[Message.text("user", "answer")],
            ),
        )
    )

    reserved = next(event for event in events if event.type == EventType.BUDGET_RESERVED)
    assert provider.calls == 0
    assert reserved.payload["model_attempt_id"] != forged_attempt_id
    assert EventType.BUDGET_RESERVATION_RELEASED not in {event.type for event in events}
    assert events[-1].type == EventType.SESSION_FAILED
    assert events[-1].payload["error"].startswith("Budget reservation lease was lost")


def test_runtime_rejects_settlement_amount_rewritten_by_custom_ledger() -> None:
    class RewritingSettlementLedger(InMemoryBudgetLedger):
        async def reconcile(self, **kwargs):
            reconciliation = await super().reconcile(**kwargs)
            return reconciliation.model_copy(update={"released_amount": Decimal("9")})

    provider = _UsageProvider()
    app = CayuApp(
        budget_policy=BudgetPolicy(limits=(_reservation_limit(),)),
        budget_ledger=RewritingSettlementLedger(),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="identity-model"))

    events = asyncio.run(
        _collect_app_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_rewritten_settlement_amount",
                messages=[Message.text("user", "answer")],
            ),
        )
    )

    assert provider.calls == 1
    assert EventType.MODEL_COMPLETED in {event.type for event in events}
    assert EventType.BUDGET_RECONCILED not in {event.type for event in events}
    assert events[-1].type == EventType.SESSION_FAILED
    assert events[-1].payload["error"] == "Budget ledger settlement changed its requested outcome."


def test_runtime_rejects_custom_ledger_pricing_for_unpriced_completion() -> None:
    class ForgedPricingLedger(InMemoryBudgetLedger):
        async def reconcile(self, **kwargs):
            reconciliation = await super().reconcile(**kwargs)
            return reconciliation.model_copy(
                update={
                    "pricing_provider_name": "forged-provider",
                    "pricing_model": "forged-model",
                    "pricing_match": "exact",
                    "pricing_provenance": Provenance(
                        source="forged-ledger",
                        url="https://example.invalid/forged",
                        as_of="2099-01-01",
                    ),
                }
            )

    provider = _NoUsageProvider()
    app = CayuApp(
        budget_policy=BudgetPolicy(limits=(_reservation_limit(),)),
        budget_ledger=ForgedPricingLedger(),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="identity-model"))

    events = asyncio.run(
        _collect_app_events(
            app,
            RunRequest(
                agent_name="assistant",
                session_id="sess_forged_unpriced_reconciliation",
                messages=[Message.text("user", "answer")],
            ),
        )
    )

    completed = next(event for event in events if event.type == EventType.MODEL_COMPLETED)
    assert provider.calls == 1
    assert completed.payload.get("usage") is None
    assert EventType.BUDGET_RECONCILED not in {event.type for event in events}
    assert events[-1].type == EventType.SESSION_FAILED
    assert (
        events[-1].payload["error"]
        == "Budget ledger settlement changed its requested pricing evidence."
    )
