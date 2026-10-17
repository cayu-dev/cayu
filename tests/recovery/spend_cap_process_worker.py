"""One deployment process spending against an app-wide dollar cap.

The parent test starts two of these against the same session store and budget
ledger, releases them together, and checks the combined spend.
"""

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.budgets import (
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    ModelPrice,
    PriceBook,
)
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.sessions.requests import RunRequest
from cayu.storage.budget_ledger import SQLiteBudgetLedger
from cayu.storage.budget_postgres import PostgresBudgetLedger
from cayu.storage.migrations import SchemaMode
from cayu.storage.postgres import PostgresSessionStore
from cayu.storage.sqlite import SQLiteSessionStore

# Every model call uses, and reserves, exactly $0.25 of input.
QUARTER_DOLLAR_TOKENS = 250_000
DEPLOYMENT_CAP_USD = Decimal("0.75")


def deployment_prices() -> PriceBook:
    return PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="gateway",
                model="frontier",
                match="exact",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("1"),
            ),
        )
    )


def deployment_cap() -> BudgetLimit:
    """The identical limit every process configures."""

    return BudgetLimit.usd(
        DEPLOYMENT_CAP_USD,
        scope="app",
        pricing=deployment_prices(),
        reservation=BudgetReservation(max_input_tokens=QUARTER_DOLLAR_TOKENS, max_output_tokens=0),
    )


def open_stores(
    backend: str, address: str
) -> tuple[SQLiteSessionStore | PostgresSessionStore, SQLiteBudgetLedger | PostgresBudgetLedger]:
    if backend == "sqlite":
        root = Path(address)
        return (
            SQLiteSessionStore(root / "sessions.sqlite"),
            SQLiteBudgetLedger(root / "ledger.sqlite"),
        )
    return (
        PostgresSessionStore(address, schema_mode=SchemaMode.CREATE),
        PostgresBudgetLedger(address, schema_mode=SchemaMode.CREATE),
    )


class _QuarterDollarProvider(ModelProvider):
    name = "gateway"

    def __init__(self) -> None:
        self.requests = 0

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests += 1
        # Keep the reservation active long enough for the other process to race it.
        await asyncio.sleep(0.2)
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": QUARTER_DOLLAR_TOKENS,
                    "output_tokens": 0,
                    "total_tokens": QUARTER_DOLLAR_TOKENS,
                },
            }
        )


async def main() -> None:
    config = json.loads(sys.stdin.readline())
    sessions, ledger = open_stores(config["backend"], config["address"])
    provider = _QuarterDollarProvider()
    try:
        app = CayuApp(
            session_store=sessions,
            budget_ledger=ledger,
            budget_policy=BudgetPolicy(limits=(deployment_cap(),)),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker", model="frontier"))
        print("ready", flush=True)
        go = Path(config["go"])
        while not go.exists():
            await asyncio.sleep(0.01)
        completed = stopped = 0
        for index in range(config["sessions"]):
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="worker",
                        session_id=f"{config['name']}-{index}",
                        messages=[Message.text("user", "work")],
                    )
                )
            ]
            terminal = events[-1].type
            if terminal == EventType.SESSION_COMPLETED:
                completed += 1
            elif terminal == EventType.SESSION_INTERRUPTED and any(
                event.type == EventType.BUDGET_RESERVATION_FAILED
                or event.type == EventType.BUDGET_LIMIT_REACHED
                for event in events
            ):
                stopped += 1
            else:
                raise AssertionError(f"unexpected terminal event {terminal}")
        print(
            json.dumps(
                {
                    "provider_calls": provider.requests,
                    "completed": completed,
                    "stopped": stopped,
                }
            ),
            flush=True,
        )
    finally:
        await sessions.close()
        await ledger.close()


if __name__ == "__main__":
    asyncio.run(main())
