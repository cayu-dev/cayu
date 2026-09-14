from __future__ import annotations

import asyncio

import pytest
from examples.runtime_auxiliary_inference import MODEL, Summarize

from cayu import AgentSpec, CayuApp, Message, ModelStreamEvent, RunRequest, ScriptedModelProvider
from cayu.budgets.pricing import ModelPrice, PriceBook

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi.testclient import TestClient

from cayu.server import ServerConfig, create_server


@pytest.mark.parametrize("observed", [False, True])
def test_sessions_summary_preserves_managed_auxiliary_accounting(observed):
    app = CayuApp(enable_logging=False)
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    name="summarize", id="parent", arguments={"text": "Long input"}
                ),
                ModelStreamEvent.completed({"usage": {"input_tokens": 2, "output_tokens": 1}}),
            ],
            [
                ModelStreamEvent.text_delta("Short summary"),
                ModelStreamEvent.completed(
                    {"usage": {"input_tokens": 5, "output_tokens": 2}} if observed else {}
                ),
            ],
            [
                ModelStreamEvent.text_delta("Finished"),
                ModelStreamEvent.completed({"usage": {"input_tokens": 2, "output_tokens": 1}}),
            ],
        ]
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model=MODEL), tools=[Summarize()])

    async def run():
        async for _ in app.run(
            RunRequest(
                session_id="auxiliary-summary",
                agent_name="assistant",
                messages=[Message.text("user", "Summarize")],
            )
        ):
            pass

    asyncio.run(run())
    assert len(provider.requests) == 3
    pricing = PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name=provider.name,
                model=MODEL,
                input_per_million=1_000_000,
                output_per_million=1_000_000,
            ),
        )
    )
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        response = client.post(
            "/api/sessions/summary", json={"pricing": pricing.model_dump(mode="json")}
        )
    assert response.status_code == 200
    summary = response.json()
    usage = summary["usage"]
    assert usage["model_steps"] == 2
    assert usage["unmeasured_model_attempts"] == int(not observed)
    assert usage["session_summaries"][0]["unmeasured_model_attempts"] == int(not observed)
    cost = summary["cost"]
    assert cost["model_steps"] == cost["priced_model_steps"] == 2
    assert cost["unpriced_model_steps"] == 0
    assert cost["auxiliary_attempts"] == 1
    assert cost["unpriced_auxiliary_attempts"] == int(not observed)
    assert cost["session_costs"][0]["unpriced_auxiliary_attempts"] == int(not observed)
    assert sum(item["auxiliary_attempt"] for item in cost["line_items"]) == 1
