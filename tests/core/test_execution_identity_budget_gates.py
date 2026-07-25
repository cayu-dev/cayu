from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal

from cayu.core import AgentSpec, Event, EventType, Message
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    CayuApp,
    InMemoryBudgetLedger,
    ModelPrice,
    PriceBook,
    RunRequest,
)


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


def _model_attempt_payload(event: Event) -> dict[str, object]:
    return {
        "model_step_id": event.payload.get("model_step_id"),
        "model_attempt_id": event.payload.get("model_attempt_id"),
    }


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
