from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal

from cayu import (
    AgentSpec,
    CayuApp,
    Message,
    ModelPrice,
    PriceBook,
    RetryPolicy,
    RunRequest,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent


class ReportProvider(ModelProvider):
    """Deterministic provider that retries once, then emits OpenAI-shaped usage."""

    name = "openai"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) == 1:
            raise TimeoutError("stream idle timeout")

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
    app = CayuApp(enable_logging=False)
    app.register_provider(ReportProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="gpt-5.5"))

    async for event in app.run(
        RunRequest(
            agent_name="assistant",
            session_id="demo_usage_cost",
            messages=[Message.text("user", "Summarize usage.")],
            retry_policy=RetryPolicy(max_attempts=2, initial_delay_s=0.0),
        )
    ):
        print(event.type, event.payload)

    usage = await app.get_session_usage("demo_usage_cost")

    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
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

    print("usage_input_tokens", usage.usage.input_tokens)
    print("usage_output_tokens", usage.usage.output_tokens)
    print("usage_cache_read_tokens", usage.usage.cache.read_tokens)
    print("usage_cached_input_tokens", usage.usage.cache.cached_input_tokens)
    print("estimated_cost", cost.total_cost)
    print("unpriced_model_steps", cost.unpriced_model_steps)


if __name__ == "__main__":
    asyncio.run(main())
