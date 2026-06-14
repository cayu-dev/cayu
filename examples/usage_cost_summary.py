from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal

from cayu import (
    AgentSpec,
    CayuApp,
    Message,
    ModelPricing,
    PricingCatalog,
    RunRequest,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent


class UsageProvider(ModelProvider):
    """Deterministic provider that emits OpenAI-shaped usage counters."""

    name = "openai"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed(
            {
                "model": "gpt-5.5-2026-04-23",
                "usage": {
                    "input_tokens": 1200,
                    "input_tokens_details": {"cached_tokens": 800},
                    "output_tokens": 100,
                    "output_tokens_details": {"reasoning_tokens": 12},
                    "total_tokens": 1300,
                },
            }
        )


async def main() -> None:
    app = CayuApp()
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-5.5"))

    async for event in app.run(
        RunRequest(
            agent_name="assistant",
            session_id="demo_usage_cost",
            messages=[Message.text("user", "Summarize usage.")],
        )
    ):
        print(event.type, event.payload)

    usage = await app.get_session_usage("demo_usage_cost")
    print("input_tokens", usage.usage.input_tokens)
    print("output_tokens", usage.usage.output_tokens)
    print("cached_input_tokens", usage.usage.cache.cached_input_tokens)

    pricing = PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="openai",
                model="gpt-5.5",
                match="prefix",
                input_per_million=Decimal("2.00"),
                output_per_million=Decimal("8.00"),
                cache_read_input_per_million=Decimal("0.50"),
            ),
        )
    )
    cost = await app.get_session_cost("demo_usage_cost", pricing)
    print("estimated_cost", cost.total_cost)
    print("unpriced_model_steps", cost.unpriced_model_steps)


if __name__ == "__main__":
    asyncio.run(main())
