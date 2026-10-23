from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient
from tests.server.server_test_support import _LOCAL_SERVER_CONFIG, OneShotProvider

from cayu import AgentSpec, CayuApp
from cayu.server import create_server
from cayu.vaults.redaction import SecretRedactor


def test_http_title_is_persisted_and_redacted():
    app = CayuApp(secret_redactor=SecretRedactor("title-secret"), enable_logging=False)
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    with client.stream(
        "POST",
        "/api/run",
        json={
            "session_id": "metadata-http-run",
            "prompt": "hello",
            "title": "Review title-secret",
        },
    ) as response:
        assert response.status_code == 200
        frames = list(response.iter_lines())
        assert frames
    assert (
        client.get("/api/contract").json()["capabilities"]["surfaces"]["session_display_metadata"][
            "read"
        ]["enabled"]
        is True
    )
    result = client.get("/api/sessions/metadata-http-run")
    assert result.status_code == 200
    assert result.json()["metadata"]["cayu:title"].startswith("Review ")
    assert "title-secret" not in result.text
    asyncio.run(app.aclose())


def test_http_invalid_title_does_not_create_session():
    app = CayuApp(enable_logging=False)
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    response = client.post(
        "/api/run",
        json={
            "session_id": "metadata-http-invalid",
            "prompt": "hello",
            "title": "HTTP_SECRET_CANARY" + "x" * 257,
        },
    )
    assert response.status_code == 422
    assert "HTTP_SECRET_CANARY" not in response.text
    assert asyncio.run(app.session_store.load("metadata-http-invalid")) is None
    asyncio.run(app.aclose())


def test_sqlite_metadata_writer_and_overview_share_the_contract(tmp_path):
    from cayu.storage import SQLiteSessionStore

    store = SQLiteSessionStore(tmp_path / "metadata-overview.db")
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(OneShotProvider(), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    with client.stream(
        "POST",
        "/api/run",
        json={
            "session_id": "metadata-overview",
            "prompt": "hello",
            "title": "Review accounts",
        },
    ) as response:
        assert response.status_code == 200
        list(response.iter_lines())
    asyncio.run(store.set_summary("metadata-overview", "Two entries need review", tone="attention"))
    asyncio.run(store.update_metadata("metadata-overview", {"ticket": "42"}))
    response = client.get(
        "/api/sessions", params={"include": "overview", "ids": "metadata-overview"}
    )
    assert response.status_code == 200
    row = response.json()["overviews"][0]
    assert (row["title"], row["summary"], row["summary_tone"]) == (
        "Review accounts",
        "Two entries need review",
        "attention",
    )
    assert row["state"] == "completed"
    assert client.get("/api/sessions/metadata-overview").json()["metadata"]["ticket"] == "42"
    asyncio.run(app.aclose())


def test_display_metadata_schema_uses_canonical_limits_and_tones():
    from cayu.session_metadata import (
        MAX_SESSION_SUMMARY_LENGTH,
        MAX_SESSION_TITLE_LENGTH,
        SUMMARY_TONES,
    )

    app = CayuApp(enable_logging=False)
    client = TestClient(create_server(app, config=_LOCAL_SERVER_CONFIG))
    schemas = client.get("/openapi.json").json()["components"]["schemas"]
    run = schemas["RunBody"]["properties"]
    overview = schemas["SessionOverviewResponse"]["properties"]
    assert run["title"]["anyOf"][0]["maxLength"] == MAX_SESSION_TITLE_LENGTH
    assert overview["title"]["anyOf"][0]["maxLength"] == MAX_SESSION_TITLE_LENGTH
    assert overview["summary"]["anyOf"][0]["maxLength"] == MAX_SESSION_SUMMARY_LENGTH
    assert set(overview["summary_tone"]["anyOf"][0]["enum"]) == SUMMARY_TONES
    asyncio.run(app.aclose())
