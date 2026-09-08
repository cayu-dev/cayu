"""Application purpose drift fails at authenticated HTTP before new authority."""

import asyncio
import time

import httpx
import pytest
from tests.core.test_browser_control import operator_purpose
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_control_publisher import publication_fixture
from tests.server.test_browser_control_server import transport

from cayu import BrowserControlConfig, CayuApp
from cayu.runtime._browser_control_checkpoint import browser_control_checkpoint_read_scope
from cayu.runtime._browser_control_publisher import BrowserControlPublisher
from cayu.server import ServerConfig, create_server
from cayu.server.auth import BasicAuth


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "change", [None, {"code": "support"}, {"expected_origins": ("https://different.test",)}]
)
def test_authenticated_discovery_rejects_reconstructed_application_purpose(
    tmp_path, backend, change
):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, bootstrap):
            record = await BrowserControlPublisher(store).publish(bootstrap)
            with browser_control_checkpoint_read_scope(record.identity.session_id):
                before = await store.load_checkpoint(record.identity.session_id)
            policy = Policy(True)
            purpose = operator_purpose()
            if change is not None:
                purpose = purpose.model_copy(update=change)
            app = CayuApp(
                enable_logging=False,
                session_store=store._store,
                browser_control=BrowserControlConfig(
                    policy=policy,
                    purpose=purpose,
                    guest_endpoint="wss://operator.test/api/browser-control/guest",
                ),
            )
            server = create_server(
                app,
                config=ServerConfig.protected(
                    BasicAuth(username="operator", password="password"),
                    browser_control=transport(),
                ),
            )
            runtime = app._browser_control_runtime
            assert runtime is not None
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=server),
                    base_url="https://operator.test",
                    auth=httpx.BasicAuth("operator", "password"),
                ) as client:
                    root = "/api/browser-control"
                    authenticated = await client.post(root + "/operator-session")
                    assert authenticated.status_code == 200
                    headers = {
                        "X-Cayu-Browser-Operator": authenticated.json()["operator_session_token"]
                    }
                    result = await client.get(root + "/sessions/session", headers=headers)
                    if change is None:
                        assert result.status_code == 200
                        assert result.json()["browsers"][0][
                            "identity"
                        ] == record.identity.model_dump(mode="json")
                        assert policy.requests
                    else:
                        assert result.status_code == 403
                        assert not policy.requests
                        now = int(time.time() * 1000)
                        rejected = await client.post(
                            root + "/takeover",
                            headers=headers,
                            json={
                                "identity": record.identity.model_dump(mode="json"),
                                "request_id": "bt_" + "1" * 32,
                                "expected_record_revision": record.revision,
                                "expected_control_epoch": record.control_epoch,
                                "pages": [
                                    {"page_id": "page", "revision": "revision", "control_epoch": 1}
                                ],
                                "purpose_code": purpose.code,
                                "requested_at_ms": now,
                                "expires_at_ms": now + 10_000,
                                "maximum_until_ms": now + 20_000,
                                "checkpoint_consent": "deny",
                            },
                        )
                        assert rejected.status_code == 403
                        assert not policy.requests
                with browser_control_checkpoint_read_scope(record.identity.session_id):
                    assert await store.load_checkpoint(record.identity.session_id) == before
                assert not runtime.service._owners
            finally:
                assert await runtime.service.drain()

    asyncio.run(scenario())
