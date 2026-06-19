"""Filtered session summary example.

Usage:
    uv sync --extra dev
    PYTHONPATH=src .venv/bin/python examples/session_labels_summary.py
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.server import create_server


class UsageProvider(ModelProvider):
    name = "fake"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed(
            {
                "usage": {
                    "input_tokens": 10,
                    "input_tokens_details": {"cached_tokens": 4},
                    "output_tokens": 2,
                }
            }
        )


async def seed_sessions(app: CayuApp) -> None:
    for session_id, labels in (
        (
            "invoice_ap_q2",
            {
                "organization": "org_123",
                "project": "ap_q2",
                "workflow": "invoice-review",
            },
        ),
        (
            "invoice_research",
            {
                "organization": "org_123",
                "project": "research",
                "workflow": "invoice-review",
            },
        ),
        (
            "other_org_ap_q2",
            {
                "organization": "org_999",
                "project": "ap_q2",
                "workflow": "invoice-review",
            },
        ),
    ):
        async for _event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                labels=labels,
                messages=[Message.text("user", "Summarize invoice status.")],
            )
        ):
            pass


def main() -> None:
    app = CayuApp(enable_logging=False)
    app.register_provider(UsageProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    asyncio.run(seed_sessions(app))

    client = TestClient(create_server(app))
    response = client.post(
        "/api/sessions/summary",
        params=[
            ("label", "organization=org_123"),
            ("label_selector", "project in (ap_q2,research)"),
            ("order_by", "created_at_asc"),
        ],
        json={
            "pricing": {
                "prices": [
                    {
                        "provider_name": "fake",
                        "model": "fake-model",
                        "input_per_million": "1",
                        "output_per_million": "2",
                        "cache_read_input_per_million": "0.25",
                    }
                ]
            }
        },
    )
    response.raise_for_status()
    body = response.json()

    print("session_count", body["session_count"])
    print("session_ids", [item["session"]["id"] for item in body["sessions"]])
    print("usage_total_tokens", body["usage"]["usage"]["total_tokens"])
    print("estimated_cost", body["cost"]["total_cost"])


if __name__ == "__main__":
    main()
