"""Reusable HTTP server test configuration, providers and helpers."""

from __future__ import annotations

# ruff: noqa: E402
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")
httpx = pytest.importorskip("httpx")

import asyncio
import json
from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from cayu.applications import CayuApp
from cayu.events import Event
from cayu.messages import Message
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent, UsageDialect
from cayu.server import ServerConfig, create_server
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest

_LOCAL_SERVER_CONFIG = ServerConfig.local_development()


class OneShotProvider(ModelProvider):
    name = "fake"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class UsageProvider(ModelProvider):
    name = "fake"
    usage_dialect = UsageDialect.OPENAI

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


def _aggregate_usage_json(
    *,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    cache_read_tokens: int,
    cached_input_tokens: int,
    uncached_input_tokens: int,
) -> dict:
    return {
        "input_tokens": str(input_tokens),
        "output_tokens": str(output_tokens),
        "total_tokens": str(total_tokens),
        "reasoning_output_tokens": "0",
        "cache": {
            "read_tokens": str(cache_read_tokens),
            "write_tokens": "0",
            "write_5m_tokens": "0",
            "write_1h_tokens": "0",
            "write_unknown_ttl_tokens": "0",
            "cached_input_tokens": str(cached_input_tokens),
            "uncached_input_tokens": str(uncached_input_tokens),
        },
    }


async def _collect_run(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]


def _price_book_payload(
    *,
    provider_name: str = "fake",
    model: str = "fake-model",
    input_per_million: str = "1",
    output_per_million: str = "1",
    cache_read_input_per_million: str | None = None,
    standard: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    tier: dict[str, object] = {
        "input_per_million": input_per_million,
        "output_per_million": output_per_million,
    }
    if cache_read_input_per_million is not None:
        tier["cache_read_input_per_million"] = cache_read_input_per_million
    return {
        "price_book_version": "test",
        "generated_at": "2026-07-13",
        "prices": [
            {
                "provider_name": provider_name,
                "model": model,
                "schedules": [
                    {
                        "pricing": {"standard": standard or [tier]},
                        "provenance": {
                            "source": "official",
                            "url": "https://example.com/pricing",
                            "as_of": "2026-07-13",
                        },
                    }
                ],
            }
        ],
    }


def _lifecycle_store_and_client(seed) -> tuple[InMemorySessionStore, TestClient]:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store)
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    asyncio.run(seed(store))
    return store, client


def _create_session(store: InMemorySessionStore, session_id: str, **kwargs):
    return store.create(
        RunRequest(
            agent_name="builder",
            session_id=session_id,
            messages=[Message.text("user", "x")],
            **kwargs,
        ),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )


def _sse_frames(response) -> list[dict]:
    """Collect SSE frames as dicts with optional `id`, `event`, and parsed `data`."""
    frames: list[dict] = []
    current: dict = {}
    for line in response.iter_lines():
        if not line.strip():
            if current:
                frames.append(current)
                current = {}
            continue
        if line.startswith("id:"):
            current["id"] = line[len("id:") :].strip()
        elif line.startswith("event:"):
            current["event"] = line[len("event:") :].strip()
        elif line.startswith("data:"):
            current["data"] = json.loads(line[len("data:") :].strip())
    if current:
        frames.append(current)
    return frames


def _sse_events(client: TestClient, path: str, body: dict) -> list[dict]:
    with client.stream("POST", path, json=body) as response:
        assert response.status_code == 200
        return [frame["data"] for frame in _sse_frames(response) if "data" in frame]
