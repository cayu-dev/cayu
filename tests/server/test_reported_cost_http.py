import base64
from datetime import timedelta

import httpx
import pytest
from pydantic import SecretStr

pytest.importorskip("fastapi")
pytest.importorskip("sse_starlette")

from tests.core.test_gateway_provider import chunk, provider_for, response, wire

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.server import ServerConfig, create_server
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.vaults.redaction import SecretRedactor


@pytest.mark.anyio
@pytest.mark.parametrize(
    "cost,status", [(None, "pending"), ("0.000000000", "reported"), ("0.001234567", "reported")]
)
async def test_public_report_retains_cost_without_exposing_private_identity(tmp_path, cost, status):
    canary = "request-only-private-canary"
    usage = {"cost": cost, "cost_status": status, "cost_currency": "USD"}
    provider = provider_for(
        lambda _: response(
            wire(chunk({"content": "OK"}, id=canary), chunk(finish="stop", id=canary, usage=usage)),
            content_type="text/event-stream",
        )
    )
    path = tmp_path / "runtime.sqlite"
    codec = PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(
            active_key_id="test",
            keys={
                "test": SecretStr(
                    base64.urlsafe_b64encode(bytes(range(32))).decode("ascii").rstrip("=")
                )
            },
        )
    )
    store = SQLiteSessionStore(path, public_authority_alias_codec=codec)
    app = CayuApp(session_store=store)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="example/model"))
    try:
        events = [
            e
            async for e in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="reported",
                    messages=[Message.text("user", "Say OK")],
                )
            )
        ]
        assert events[-1].type is EventType.SESSION_COMPLETED
        completed = next(e for e in events if e.type is EventType.MODEL_COMPLETED)
    finally:
        await provider.aclose()
        await store.close()
    store = SQLiteSessionStore(path, public_authority_alias_codec=codec)
    try:
        app = CayuApp(session_store=store, secret_redactor=SecretRedactor([canary]))
        server = create_server(app, config=ServerConfig.local_development())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server), base_url="http://runtime"
        ) as client:
            value = await client.post(
                "/api/usage/rollup",
                json={
                    "start_at": (completed.timestamp - timedelta(seconds=1)).isoformat(),
                    "end_at": (completed.timestamp + timedelta(seconds=1)).isoformat(),
                },
            )
        assert value.status_code == 200, value.text
        assert canary not in value.text
        row = value.json()["reported_costs"]["records"][0]
        assert row["cost"] == cost and row["status"] == status
        assert value.json()["cost"] is None  # No PriceBook; never use reported cost as estimate.
    finally:
        await store.close()
