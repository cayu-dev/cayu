"""Verified, bounded live-provider budget contract."""

from __future__ import annotations

import asyncio
import json
import os
from decimal import Decimal
from typing import Any

from _live_checks import require
from cayu import (
    AgentSpec,
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    CayuApp,
    Event,
    EventType,
    InMemoryBudgetLedger,
    Message,
    ModelPrice,
    PriceBook,
    RunRequest,
)
from cayu.providers import ModelProvider, OpenAIProvider

PROVIDER_NAME = "openai"
DEFAULT_MODEL = "gpt-5.6-luna"
MAX_INPUT_TOKENS = 1_024
MAX_OUTPUT_TOKENS = 16
OPENAI_PROVIDER_OPTIONS = {"openai": {"max_output_tokens": MAX_OUTPUT_TOKENS}}
INPUT_PER_MILLION = Decimal("1")
OUTPUT_PER_MILLION = Decimal("1")
# (1,024 input + 16 output) x $1 / 1M = $0.00104: one reservation fills the cap,
# so a second reservation must fail before another provider request can start.
MAX_ESTIMATED_COST = Decimal("0.00104")
EVIDENCE_PREFIX = "CAYU_NIGHTLY_EVIDENCE="


async def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY to run the live budget contract.")
    model = os.environ.get("CAYU_BUDGET_OPENAI_MODEL", DEFAULT_MODEL)
    evidence = await _run_contract(
        provider=OpenAIProvider(),
        provider_name=PROVIDER_NAME,
        model=model,
        provider_options=OPENAI_PROVIDER_OPTIONS,
    )
    print(f"{EVIDENCE_PREFIX}{json.dumps(evidence, sort_keys=True)}")


async def _run_contract(
    *,
    provider: ModelProvider,
    provider_name: str,
    model: str,
    provider_options: dict[str, Any],
) -> dict[str, object]:
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name=provider_name,
                model=model,
                input_per_million=INPUT_PER_MILLION,
                output_per_million=OUTPUT_PER_MILLION,
            ),
        )
    )
    app = CayuApp(
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=MAX_ESTIMATED_COST,
                    pricing=pricing,
                    reservation=BudgetReservation(
                        max_input_tokens=MAX_INPUT_TOKENS,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                    ),
                ),
            )
        ),
        budget_ledger=InMemoryBudgetLedger(),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(
            name="budget-live-assistant",
            model=model,
            provider_name=provider_name,
            provider_options=provider_options,
        )
    )

    first_events = await _collect_events(
        app,
        RunRequest(
            agent_name="budget-live-assistant",
            session_id="budget_live_first",
            max_steps=1,
            messages=[Message.text("user", "Reply briefly with the word bounded.")],
        ),
    )
    second_events = await _collect_events(
        app,
        RunRequest(
            agent_name="budget-live-assistant",
            session_id="budget_live_second",
            max_steps=1,
            messages=[Message.text("user", "This provider call must be blocked by the budget.")],
        ),
    )
    return _validate_contract(
        first_events,
        second_events,
        provider_name=provider_name,
        model=model,
    )


async def _collect_events(app: CayuApp, request: RunRequest) -> list[Event]:
    events: list[Event] = []
    async for event in app.run(request):
        events.append(event)
        print(event.type, event.payload)
    return events


def _validate_contract(
    first_events: list[Event],
    second_events: list[Event],
    *,
    provider_name: str,
    model: str,
) -> dict[str, object]:
    require(first_events[-1].type == EventType.SESSION_COMPLETED, "first session did not complete")
    require(
        second_events[-1].type == EventType.SESSION_INTERRUPTED,
        "second session was not interrupted by the budget",
    )

    reserved = _one_event(first_events, EventType.BUDGET_RESERVED)
    completed = _one_event(first_events, EventType.MODEL_COMPLETED)
    reconciled = _one_event(first_events, EventType.BUDGET_RECONCILED)
    _one_event(second_events, EventType.BUDGET_RESERVATION_FAILED)
    require(
        all(event.type != EventType.MODEL_STARTED for event in second_events),
        "second provider call started before budget enforcement",
    )

    usage = completed.payload.get("usage_metrics")
    if not isinstance(usage, dict):
        raise RuntimeError("model.completed did not contain normalized usage")
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        require(
            isinstance(usage.get(key), int),
            f"model.completed usage field {key!r} is missing",
        )
    require(usage["total_tokens"] > 0, "provider reported zero total tokens")

    reserved_amount = Decimal(str(reserved.payload.get("actual")))
    actual_amount = Decimal(str(reconciled.payload.get("actual_amount")))
    expected_actual_amount = (
        Decimal(usage["input_tokens"]) * INPUT_PER_MILLION
        + Decimal(usage["output_tokens"]) * OUTPUT_PER_MILLION
    ) / Decimal(1_000_000)
    require(
        reserved_amount == MAX_ESTIMATED_COST,
        "reserved amount did not match the configured cap",
    )
    require(
        actual_amount == expected_actual_amount,
        "reconciled amount did not match usage under the configured pricing",
    )
    require(
        Decimal(0) < actual_amount <= reserved_amount,
        "reconciled amount was outside the reserved bound",
    )

    attempted = sum(
        event.type == EventType.MODEL_STARTED for event in [*first_events, *second_events]
    )
    completed_calls = sum(
        event.type == EventType.MODEL_COMPLETED for event in [*first_events, *second_events]
    )
    require(attempted == 1, f"expected one attempted provider call, got {attempted}")
    require(completed_calls == 1, f"expected one completed provider call, got {completed_calls}")

    return {
        "provider": provider_name,
        "model": model,
        "currency": "USD",
        "max_estimated_cost": str(MAX_ESTIMATED_COST),
        "reserved_amount": str(reserved_amount),
        "actual_estimated_cost": str(actual_amount),
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "provider_calls_attempted": attempted,
        "provider_calls_completed": completed_calls,
        "enforcement": "second_reservation_rejected_before_provider",
    }


def _one_event(events: list[Event], event_type: EventType) -> Event:
    matching = [event for event in events if event.type == event_type]
    require(len(matching) == 1, f"expected one {event_type}, got {len(matching)}")
    return matching[0]


if __name__ == "__main__":
    asyncio.run(main())
