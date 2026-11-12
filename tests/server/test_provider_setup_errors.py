"""A run or answer this server's providers can't serve says why, with a 503 or a 409."""

from __future__ import annotations

import asyncio

import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.core.test_resource_access_http import Policy
from tests.server.test_server_inbox import _as, _authenticate, _items, _respond, _settle

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent, ProviderNotConfigured
from cayu.server import ServerConfig, create_server
from cayu.server.resource_access import create_resource_router
from cayu.sessions.memory import InMemorySessionStore
from cayu.tools.user_input import UserInputTool

_SETUP = "No model provider is set. Set CAYU_PROVIDER to openai or anthropic."


class _Unselected(ScriptedModelProvider):
    """Like a scaffold's placeholder: it stands in for a provider nobody selected."""

    def __init__(self) -> None:
        super().__init__([], name="unconfigured")

    def preflight_model_target(self, *, model: str) -> None:
        raise ProviderNotConfigured(_SETUP)


def _app(provider, store=None, **options) -> CayuApp:
    app = CayuApp(session_store=store or InMemorySessionStore(), enable_logging=False, **options)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="asker", model="ask"), tools=[UserInputTool()])
    return app


def test_a_run_without_a_provider_is_a_503_with_the_fix() -> None:
    server = create_server(_app(_Unselected()), config=ServerConfig.local_development())
    with TestClient(server) as client:
        response = client.post("/api/run", json={"agent": "asker", "prompt": "hello"})

    assert response.status_code == 503, response.text
    assert response.json()["detail"] == _SETUP


def test_a_scoped_run_without_a_provider_is_a_503_with_the_fix() -> None:
    async def run() -> httpx.Response:
        app = _app(_Unselected(), resource_access_policy=Policy())

        async def authenticate(request):
            return "alice"

        server = FastAPI()
        server.include_router(create_resource_router(app, authenticate=authenticate))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server), base_url="http://test"
        ) as client:
            return await client.post(
                "/resources/runs",
                json={
                    "agent_name": "asker",
                    "messages": [Message.text("user", "hello").model_dump(mode="json")],
                    "labels": {"organization": "acme"},
                },
            )

    response = asyncio.run(run())
    assert response.status_code == 503, response.text
    assert response.json()["detail"] == _SETUP


def _asked(store: InMemorySessionStore) -> None:
    """A question asked under a provider named ``scripted``, in another process."""

    async def run() -> None:
        app = _app(ScriptedModelProvider(response_factory=_respond), store)
        async with app:
            async for _event in app.run(
                RunRequest(
                    agent_name="asker",
                    session_id="asked",
                    messages=[Message.text("user", "go")],
                )
            ):
                pass

    asyncio.run(run())


def test_answering_a_session_whose_provider_this_server_lacks_is_a_409_sentence() -> None:
    store = InMemorySessionStore()
    _asked(store)
    other = ScriptedModelProvider(response_factory=_respond, name="other")
    app = _app(other, store)
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        [item] = _items(client, "priya")
        refused = _settle(client, item["id"], "priya", decision="answer", answer="staging")
        resumed = client.post(
            "/api/resume", json={"session_id": "asked", "prompt": "hi"}, headers=_as("priya")
        )

    sentence = "This session was started with provider `scripted`, which this server doesn't have."
    assert refused.status_code == 409, refused.text
    detail = refused.json()["detail"]
    assert detail["code"] == "not_settleable"
    assert detail["message"].startswith(sentence)
    assert detail["item"]["status"] == "pending"
    assert resumed.status_code == 409, resumed.text
    assert resumed.json()["detail"].startswith(sentence)


def test_a_question_across_a_changed_profile_says_why_it_cant_adopt() -> None:
    store = InMemorySessionStore()
    _asked(store)
    # Same name, but a response factory's identity is opaque: another process's
    # provider is a different profile, as a scripted stand-in's was.
    app = _app(ScriptedModelProvider(response_factory=_respond), store)
    with TestClient(create_server(app, config=ServerConfig.protected(_authenticate))) as client:
        [item] = _items(client, "priya")
        refused = _settle(client, item["id"], "priya", decision="answer", answer="staging")

    assert refused.status_code == 409, refused.text
    detail = refused.json()["detail"]
    assert detail["code"] == "not_settleable"
    assert "a question can't continue under the update" in detail["message"]
    assert "dismiss it" in detail["message"]
    assert detail["item"]["status"] == "pending"


def test_a_missing_provider_a_request_names_isnt_blamed_on_the_session() -> None:
    async def run() -> httpx.Response:
        replies = [[ModelStreamEvent.text_delta("ok"), ModelStreamEvent.completed({})]]
        app = _app(ScriptedModelProvider(replies), resource_access_policy=Policy())
        access = await app.access("alice")
        async for _event in access.run(
            RunRequest(
                agent_name="asker",
                session_id="own",
                messages=[Message.text("user", "hello")],
                labels={"organization": "acme"},
            )
        ):
            pass

        async def authenticate(request):
            return "alice"

        server = FastAPI()
        server.include_router(create_resource_router(app, authenticate=authenticate))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server), base_url="http://test"
        ) as client:
            return await client.post(
                "/resources/resume",
                json={
                    "session_id": "own",
                    "messages": [Message.text("user", "again").model_dump(mode="json")],
                    "target": {"provider_name": "typo", "model": "example"},
                },
            )

    response = asyncio.run(run())
    assert response.status_code == 409, response.text
    assert response.json()["detail"] == "This server has no provider `typo`."
