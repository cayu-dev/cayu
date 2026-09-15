"""HTTP run/resume preserve the policy through admission, SSE and reconstruction."""

from __future__ import annotations

import asyncio
import json

import pytest
from tests.core.test_model_failover_recovery import _RecoveryProvider
from tests.core.test_model_failover_stages import _StageMemoryStore, _StageSQLiteStore

from cayu import AgentSpec, CayuApp, EventType


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_http_run_and_reconstructed_resume_preserve_fallback(monkeypatch, tmp_path, backend):
    pytest.importorskip("fastapi")
    pytest.importorskip("sse_starlette")
    from fastapi.testclient import TestClient

    from cayu.server import ServerConfig, create_server

    path = tmp_path / "http.sqlite"
    store = _StageMemoryStore() if backend == "memory" else _StageSQLiteStore(path)
    policy = {
        "fallbacks": [{"provider_name": "backup", "model": "large"}],
        "max_total_attempts": 2,
    }
    try:
        for resumed in (False, True):
            app = CayuApp(session_store=store, enable_logging=False)
            primary, backup = _RecoveryProvider("primary"), _RecoveryProvider("backup")
            app.register_provider(primary, default=True)
            app.register_provider(backup)
            app.register_agent(AgentSpec(name="assistant", model="small"))
            with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
                assert client.portal is not None
                if not resumed:
                    invalid = client.post(
                        "/api/run",
                        json={
                            "prompt": "answer",
                            "session_id": "invalid",
                            "failover": {**policy, "max_total_attempts": True},
                        },
                    )
                    assert invalid.status_code == 422
                    assert not primary.requests and not backup.requests
                    assert client.portal.call(store.load, "invalid") is None
                with client.stream(
                    "POST",
                    "/api/resume" if resumed else "/api/run",
                    json={
                        "prompt": "continue" if resumed else "answer",
                        "session_id": "http-failover",
                        "retry_policy": {"max_attempts": 1},
                        **({} if resumed else {"failover": policy}),
                    },
                ) as response:
                    assert response.status_code == 200
                    events = [
                        json.loads(line[len("data:") :].strip())
                        for line in response.iter_lines()
                        if line.startswith("data:")
                    ]
                assert events[-1]["type"] == "session.completed", events[-1]
                assert len(primary.requests) == int(not resumed)
                assert len(backup.requests) == 1 and backup.requests[0].model == "large"
                assert sum(event["type"] == "model.failover.selected" for event in events) == (
                    0 if resumed else 2
                )
                checkpoint = client.portal.call(store.load_checkpoint, "http-failover")
                assert checkpoint is not None
                assert checkpoint["model_failover"]["candidate_index"] == 1
                assert checkpoint["model_failover"]["attempts_used"] == (1 if resumed else 2)
                persisted = client.portal.call(store.load_events, "http-failover")
                assert persisted[-1].type is EventType.SESSION_COMPLETED
                session = client.portal.call(store.load, "http-failover")
                assert session is not None and session.provider_name == "primary"
            if isinstance(store, _StageSQLiteStore):
                asyncio.run(store.close())
                store = _StageSQLiteStore(path)
    finally:
        if isinstance(store, _StageSQLiteStore):
            asyncio.run(store.close())
