"""HTTP resumes preserve omitted controls and explicit retry-policy overrides."""

from __future__ import annotations

import json

import pytest

from cayu import AgentSpec, CayuApp, ModelStreamEvent, ScriptedModelProvider
from cayu.runtime._model_step_executor import model_completion_recovery_context_from_stage
from cayu.sessions._model_completion_publication import model_step_publication_from_checkpoint
from cayu.sessions.base import InMemorySessionStore


@pytest.mark.parametrize("override", ["omitted", "null", "default", "same", "changed"])
def test_http_resume_inherits_only_omitted_retry_policy(override):
    pytest.importorskip("fastapi")
    pytest.importorskip("sse_starlette")
    from fastapi.testclient import TestClient

    from cayu.server import ServerConfig, create_server

    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    provider = ScriptedModelProvider(
        [[ModelStreamEvent.text_delta("ok"), ModelStreamEvent.completed()] for _ in range(2)]
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="scripted"))
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        started = client.post(
            "/api/run",
            json={
                "session_id": "http-controls",
                "prompt": "first",
                "max_steps": 18,
                "limits": {"max_tool_calls": 7},
                "retry_policy": {"max_attempts": 2},
            },
        )
        assert started.status_code == 200, started.text
        assert len(provider.requests) == 1
        body = {"session_id": "http-controls", "prompt": "again"}
        if override != "omitted":
            body["retry_policy"] = {
                "null": None,
                "default": {},
                "same": {"max_attempts": 2},
                "changed": {"max_attempts": 3},
            }[override]
        resumed = client.post("/api/resume", json=body)
        if override in {"omitted", "same"}:
            assert resumed.status_code == 200, resumed.text
            events = [
                json.loads(line[len("data:") :].strip())
                for line in resumed.text.splitlines()
                if line.startswith("data:")
            ]
            assert events[-1]["type"] == "session.completed"
            assert len(provider.requests) == 2
            assert client.portal is not None
            pointer = model_step_publication_from_checkpoint(
                client.portal.call(store.load_checkpoint, "http-controls")
            )
            stage = client.portal.call(
                store.load_model_completion_stage, "http-controls", pointer.stage_id
            )
            context = model_completion_recovery_context_from_stage(stage)
            assert context.max_steps == 18
            assert context.limits.max_tool_calls == 7
            assert context.retry_policy.max_attempts == 2
        else:
            assert resumed.status_code == 409, resumed.text
            assert "finalization" in resumed.json()["detail"]
            assert len(provider.requests) == 1
