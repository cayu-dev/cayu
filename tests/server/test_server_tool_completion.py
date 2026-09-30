from __future__ import annotations

import asyncio
import json

import pytest
from tests.core.test_tool_completion import FinalTool, app_for, call, request
from tests.core.test_tool_round_publication_failure_matrix import _TwoCallProvider

from cayu import EventType, InMemorySessionStore, SessionStatus
from cayu.server import ServerConfig, create_server

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


def test_http_run_and_resume_forward_tool_completion():
    provider = _TwoCallProvider([call(), call()])
    app = app_for(InMemorySessionStore(), provider, FinalTool())
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        for endpoint in ("run", "resume"):
            body = {
                "session_id": "final-tool-http",
                "prompt": "Help",
                "tool_completion": {"tool_names": ["ask_customer"]},
            }
            if endpoint == "run":
                body["agent"] = "support"
            response = client.post(f"/api/{endpoint}", json=body)
            assert response.status_code == 200, response.text
            completions = [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: ") and '"session.completed"' in line
            ]
            assert len(completions) == 1, response.text
            assert completions[0]["payload"]["reason"] == "host_rendered_tool"
            assert (
                completions[0]["payload"]["tool_completion"]["call"]["tool_name"] == "ask_customer"
            )
        assert len(provider.requests) == 2


@pytest.mark.parametrize("policy_state", ["omitted", "none", "configured"])
def test_http_pending_resume_preserves_completion_controls(policy_state):
    class PublicationFailureStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        terminal_interaction_publication_version = 1
        durable_model_terminalization_version = 1

        def __init__(self):
            super().__init__()
            self.failed = False

        async def publish_runtime_publication(self, session_id, **kwargs):
            if not self.failed and kwargs["request"].kind == "tool-round":
                self.failed = True
                raise RuntimeError("Injected failure before tool-round publication.")
            return await super().publish_runtime_publication(session_id, **kwargs)

    store = PublicationFailureStore()
    provider = _TwoCallProvider([call(), call()])
    tool = FinalTool()
    app = app_for(store, provider, tool)

    async def pause():
        events = [
            event
            async for event in app.run(request(tool_completion={"tool_names": ["ask_customer"]}))
        ]
        session = await store.load("s")
        checkpoint = await store.load_checkpoint("s")
        assert session.status == SessionStatus.FAILED
        assert checkpoint["pending_tool_round"]
        assert events[-1].type == EventType.SESSION_FAILED
        assert tool.calls == len(provider.requests) == 1
        return session, checkpoint, await store.load_events("s")

    before = asyncio.run(pause())
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        body = {"session_id": "s", "prompt": "Continue"}
        if policy_state != "omitted":
            body["tool_completion"] = (
                {"tool_names": ["ask_customer"]} if policy_state == "configured" else None
            )
        response = client.post("/api/resume", json=body)
        if policy_state == "none":
            assert response.status_code == 409, response.text
            assert "tool_completion cannot change during a pending continuation" in response.text
            assert tool.calls == len(provider.requests) == 1

            async def unchanged():
                assert (
                    await store.load("s"),
                    await store.load_checkpoint("s"),
                    await store.load_events("s"),
                ) == before

            asyncio.run(unchanged())
            del body["tool_completion"]
            response = client.post("/api/resume", json=body)
        assert response.status_code == 200, response.text
        completions = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and '"session.completed"' in line
        ]
        assert completions, response.text
        assert completions[-1]["payload"]["reason"] == "host_rendered_tool"
        assert completions[-1]["payload"]["tool_completion"]["call"]["tool_name"] == "ask_customer"
        # One call belongs to the recovered round and one to the new user turn.
        assert tool.calls == len(provider.requests) == 2
    assert asyncio.run(store.load("s")).status == SessionStatus.COMPLETED
