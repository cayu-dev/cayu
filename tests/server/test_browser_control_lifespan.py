"""Real server shutdown closes idle operator sockets before lifecycle drain."""

import asyncio

import httpx
import pytest
from tests.core.test_browser_control import operator_purpose
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_control_publisher import publication_fixture
from tests.core.test_browser_control_transport import control_tls as _control_tls
from tests.server._browser_control_tls_server import browser_control_tls_server
from tests.server.test_browser_control_server import transport
from websockets.asyncio.client import connect
from websockets.typing import Origin, Subprotocol

from cayu import BrowserControlConfig, CayuApp
from cayu.runtime._browser_control_checkpoint import browser_control_checkpoint_read_scope
from cayu.runtime._browser_control_publisher import BrowserControlPublisher
from cayu.server import ServerConfig, create_server
from cayu.server.auth import BasicAuth

control_tls = _control_tls


@pytest.mark.parametrize("raises", [False, True])
def test_browser_drain_failure_does_not_skip_existing_cleanup_owners(monkeypatch, raises):
    async def scenario():
        app = CayuApp(
            enable_logging=False,
            browser_control=BrowserControlConfig(
                purpose=operator_purpose(),
                policy=Policy(True),
                guest_endpoint="wss://operator.test/api/browser-control/guest",
            ),
        )
        runtime = app._browser_control_runtime
        assert runtime is not None
        calls = []
        primary = RuntimeError("Browser drain failed.")

        async def browser_drain(**kwargs):
            calls.append("browser")
            if raises:
                raise primary
            return False

        async def environment_drain(**kwargs):
            calls.append("environment")
            return True

        monkeypatch.setattr(runtime.service, "drain", browser_drain)
        monkeypatch.setattr(app, "drain_environment_cleanups", environment_drain)
        server = create_server(
            app,
            config=ServerConfig.protected(
                BasicAuth(username="operator", password="password"), browser_control=transport()
            ),
        )
        with pytest.raises(RuntimeError) as failure:
            async with server.router.lifespan_context(server):
                pass
        assert calls == ["browser", "environment"]
        assert not runtime.coordinator._publisher._closing
        if raises:
            assert failure.value is primary
        else:
            assert str(failure.value) == "Browser control shutdown remains unsettled."

    asyncio.run(scenario())


@pytest.mark.parametrize("cleanup_fatal", [False, True])
def test_lifespan_cancellation_runs_cleanup_and_preserves_control_signal(
    monkeypatch, cleanup_fatal
):
    class CleanupSignal(BaseException):
        pass

    async def scenario():
        app = CayuApp(
            enable_logging=False,
            browser_control=BrowserControlConfig(
                purpose=operator_purpose(),
                policy=Policy(True),
                guest_endpoint="wss://operator.test/api/browser-control/guest",
            ),
        )
        runtime = app._browser_control_runtime
        assert runtime is not None
        entered, cleanup, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        cancellations = []
        fatal = CleanupSignal("Cleanup process-control signal.")

        async def browser_drain(**kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError as failure:
                cancellations.append(failure)
                raise

        async def environment_drain(**kwargs):
            cleanup.set()
            await release.wait()
            if cleanup_fatal:
                raise fatal
            return True

        monkeypatch.setattr(runtime.service, "drain", browser_drain)
        monkeypatch.setattr(app, "drain_environment_cleanups", environment_drain)
        server = create_server(
            app,
            config=ServerConfig.protected(
                BasicAuth(username="operator", password="password"), browser_control=transport()
            ),
        )

        async def lifecycle():
            async with server.router.lifespan_context(server):
                pass

        task = asyncio.create_task(lifecycle())
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert task.cancel("shutdown interrupted")
            await asyncio.wait_for(cleanup.wait(), 5)
            assert task.cancelling() == 1
            assert not task.done()
            release.set()
            if cleanup_fatal:
                with pytest.raises(CleanupSignal) as raised:
                    await task
                assert raised.value is fatal
                assert fatal.__context__ is cancellations[0]
                assert not task.cancelled()
            else:
                try:
                    await task
                except asyncio.CancelledError as failure:
                    assert failure is cancellations[0]
                    assert failure.__cause__ is None and failure.__context__ is None
                else:
                    pytest.fail("Lifespan cancellation was lost.")
                assert task.cancelled()
            assert task.cancelling() == 1
            assert len(cancellations) == 1
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_tls_shutdown_closes_idle_viewer_before_application_drain(
    tmp_path, control_tls, backend, monkeypatch
):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, bootstrap):
            record = await BrowserControlPublisher(store).publish(bootstrap)
            with browser_control_checkpoint_read_scope(record.identity.session_id):
                checkpoint = await store.load_checkpoint(record.identity.session_id)
            app = CayuApp(
                enable_logging=False,
                session_store=store._store,
                browser_control=BrowserControlConfig(
                    purpose=operator_purpose(),
                    policy=Policy(True),
                    guest_endpoint="wss://operator.test/api/browser-control/guest",
                ),
            )
            runtime = app._browser_control_runtime
            assert runtime is not None
            drain_entry_viewers = []
            original_drain = runtime.service.drain

            async def observe_drain(**kwargs):
                drain_entry_viewers.append(len(runtime.service._viewers))
                return await original_drain(**kwargs)

            monkeypatch.setattr(runtime.service, "drain", observe_drain)
            server = create_server(
                app,
                config=ServerConfig.protected(
                    BasicAuth(username="operator", password="password"),
                    browser_control=transport(),
                ),
            )
            connection = None
            try:
                async with browser_control_tls_server(server, tmp_path, lifespan="on") as port:
                    async with httpx.AsyncClient(
                        base_url=f"https://127.0.0.1:{port}",
                        verify=control_tls[1],
                        trust_env=False,
                        auth=httpx.BasicAuth("operator", "password"),
                    ) as client:
                        root = "/api/browser-control"
                        authenticated = await client.post(root + "/operator-session")
                        assert authenticated.status_code == 200
                        ticket = await client.post(
                            root + "/view-ticket",
                            headers={
                                "X-Cayu-Browser-Operator": authenticated.json()[
                                    "operator_session_token"
                                ]
                            },
                            json={
                                "identity": record.identity.model_dump(mode="json"),
                                "expected_record_revision": record.revision,
                                "page": {
                                    "page_id": "page",
                                    "revision": "revision",
                                    "control_epoch": 1,
                                },
                            },
                        )
                        assert ticket.status_code == 200
                    connection = await connect(
                        f"wss://127.0.0.1:{port}/api/browser-control/viewer",
                        ssl=control_tls[1],
                        origin=Origin("https://operator.test"),
                        subprotocols=[Subprotocol("cayu.browser-view.v1")],
                        proxy=None,
                        compression=None,
                        open_timeout=5,
                        close_timeout=2,
                    )
                    await connection.send(ticket.json()["ticket"])
                    assert await asyncio.wait_for(connection.recv(), 5) == "ready"
                    assert len(runtime.service._viewers) == 1
                    assert not runtime.service._closed
                    # Do not initiate client close. Exiting the server context
                    # must stop the actual socket before app lifespan completes.
                await asyncio.wait_for(connection.wait_closed(), 5)
                assert connection.close_code == 1012
                assert drain_entry_viewers == [0]
                assert runtime.service._closed
                assert not runtime.service._viewers
                assert not runtime.service.channels._tasks
                with browser_control_checkpoint_read_scope(record.identity.session_id):
                    assert await store.load_checkpoint(record.identity.session_id) == checkpoint
            finally:
                if connection is not None:
                    await connection.close()
                assert await runtime.service.drain()

    asyncio.run(scenario())
