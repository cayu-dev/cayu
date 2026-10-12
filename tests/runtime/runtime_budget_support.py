"""Budget support."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Literal

from cayu.budgets.base import BudgetLimit, BudgetWindow
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent


class BlockingBudgetProvider(ModelProvider):
    name = "fake"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        yield ModelStreamEvent.completed(
            {
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 250_000,
                    "output_tokens": 0,
                    "total_tokens": 250_000,
                },
            }
        )


def fake_budget_limit(
    max_estimated_cost: str,
    *,
    scope: Literal["session", "run"] = "session",
    allow_unpriced: bool = False,
    window: BudgetWindow | str | None = None,
    action: Literal["interrupt", "notify"] = "interrupt",
) -> BudgetLimit:
    return BudgetLimit(
        max_estimated_cost=Decimal(max_estimated_cost),
        window=BudgetWindow.all_time() if window is None else window,
        pricing=PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="fake",
                    model="fake-model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("10"),
                ),
            )
        ),
        scope=scope,
        allow_unpriced=allow_unpriced,
        action=action,
    )
