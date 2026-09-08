"""HTTP cancellation leaves an exact store write owned by server shutdown."""

import asyncio
from datetime import UTC, datetime

import httpx
import pytest
from tests.core.test_browser_control import operator_purpose
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_control_publisher import publication_fixture
from tests.server.test_browser_control_server import transport

from cayu import BrowserControlConfig, CayuApp
from cayu.runtime._browser_control_publisher import BrowserControlPublisher
from cayu.server import ServerConfig, create_server
from cayu.server.auth import BasicAuth


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_shutdown_joins_takeover_write_after_http_caller_cancellation(
    tmp_path, monkeypatch, backend
):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, bootstrap):
            record = await BrowserControlPublisher(store).publish(bootstrap)
            app = CayuApp(
                enable_logging=False,
                session_store=store._store,
                clock=lambda: datetime(1970, 1, 1, 0, 0, 1, tzinfo=UTC),
                browser_control=BrowserControlConfig(
                    purpose=operator_purpose(),
                    policy=Policy(True),
                    guest_endpoint="wss://operator.test/api/browser-control/guest",
                ),
            )
            runtime = app._browser_control_runtime
            assert runtime is not None
            entered, release, ready, stop, draining = (asyncio.Event() for _ in range(5))
            writes = []
            publisher = runtime.coordinator._publisher
            publish = publisher._store.publish_session_operation_guarded_with_store_time
            drain = publisher.drain

            async def blocked_write(*args, **kwargs):
                writes.append(1)
                entered.set()
                await release.wait()
                return await publish(*args, **kwargs)

            async def observed_drain(**kwargs):
                draining.set()
                return await drain(**kwargs)

            monkeypatch.setattr(
                publisher._store, "publish_session_operation_guarded_with_store_time", blocked_write
            )
            monkeypatch.setattr(publisher, "drain", observed_drain)
            server = create_server(
                app,
                config=ServerConfig.protected(
                    BasicAuth(username="operator", password="password"), browser_control=transport()
                ),
            )

            async def lifespan():
                async with server.router.lifespan_context(server):
                    ready.set()
                    await stop.wait()

            lifecycle = asyncio.create_task(lifespan())
            caller = None
            try:
                await asyncio.wait_for(ready.wait(), 5)
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=server),
                    base_url="https://operator.test",
                    auth=httpx.BasicAuth("operator", "password"),
                ) as client:
                    root = "/api/browser-control"
                    authenticated = await client.post(root + "/operator-session")
                    assert authenticated.status_code == 200
                    caller = asyncio.create_task(
                        client.post(
                            root + "/takeover",
                            headers={
                                "X-Cayu-Browser-Operator": authenticated.json()[
                                    "operator_session_token"
                                ]
                            },
                            json={
                                "identity": record.identity.model_dump(mode="json"),
                                "request_id": "bt_" + "1" * 32,
                                "expected_record_revision": record.revision,
                                "expected_control_epoch": record.control_epoch,
                                "pages": [
                                    {"page_id": "page", "revision": "revision", "control_epoch": 1}
                                ],
                                "purpose_code": "login",
                                "requested_at_ms": 1000,
                                "expires_at_ms": 2000,
                                "maximum_until_ms": 5000,
                                "checkpoint_consent": "deny",
                            },
                        )
                    )
                    await asyncio.wait_for(entered.wait(), 5)
                    caller.cancel("HTTP caller disconnected")
                    with pytest.raises(asyncio.CancelledError):
                        await caller
                    assert caller.cancelled() and caller.cancelling() == 1
                    assert publisher.pending_count == 1
                stop.set()
                await asyncio.wait_for(draining.wait(), 5)
                assert not lifecycle.done()
                assert writes == [1]
                release.set()
                await asyncio.wait_for(lifecycle, 5)
                assert publisher._closing
                assert all(entry.task.done() for entry in publisher._pending.values())
                _, current = await runtime.coordinator._load(record.identity)
                assert current.state == "takeover_requested"
                assert current.revision == record.revision + 1
                assert (
                    current.request is not None and current.request.request_id == "bt_" + "1" * 32
                )
                assert writes == [1]
            finally:
                release.set()
                stop.set()
                if caller is not None:
                    if not caller.done():
                        caller.cancel()
                    await asyncio.gather(caller, return_exceptions=True)
                await asyncio.gather(lifecycle, return_exceptions=True)

    asyncio.run(scenario())
