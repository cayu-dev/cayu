"""Real TLS control framing into the native daemon, without Chromium input yet."""

from __future__ import annotations

import asyncio
import json
from http import HTTPStatus

import pytest
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_control_coordinator import coordinator, intent_for
from tests.core.test_browser_control_guest import takeover_material
from tests.core.test_browser_control_publisher import publication_fixture
from tests.core.test_browser_control_transport import control_tls as _control_tls
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.typing import Subprotocol

from cayu.runtime._browser_control_authorization import BrowserControlPermissionDenied
from cayu.runtime._browser_control_bootstrap import BrowserGuestBootstrap
from cayu.runtime._browser_control_channel import (
    BrowserGuestCommandOwner,
    acquire_browser_guest_control,
    bind_browser_guest_channel,
    browser_allocation_digest,
)
from cayu.runtime.browser_control import (
    BrowserControlAllocation,
    BrowserControlConflict,
    BrowserControlPrincipal,
)
from cayu.tools._browser_control_guest import (
    GuestControlChannel,
    GuestControlFailure,
    GuestControlFence,
)
from cayu.tools._browser_control_transport import CONTROL_SUBPROTOCOL, open_guest_control_channel
from cayu.tools._browser_guest import _InteractiveDaemon, _InteractivePage

control_tls = _control_tls


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("through_owner", [False, True])
@pytest.mark.parametrize(
    "stale_page,cancel_publication", [(False, False), (True, False), (False, True)]
)
def test_durable_takeover_requires_exact_native_acquisition(
    control_tls, tmp_path, backend, stale_page, cancel_publication, through_owner, monkeypatch
):
    async def scenario():
        server_tls, client_tls = control_tls
        async with publication_fixture(backend, tmp_path) as (store, bootstrap):
            owner = coordinator(store, Policy(True))
            allocation = BrowserControlAllocation.model_validate(
                bootstrap.changed_record.identity.model_dump(exclude={"worker_instance_id"})
            )
            daemon = _InteractiveDaemon(allocation.browser_session_id)
            daemon.context = object()
            daemon.control = GuestControlFence(
                worker_instance=daemon.visual_worker_instance, wall_clock=lambda: 1.0
            )
            daemon.pages["page"] = _InteractivePage(
                page=object(),
                session_id=allocation.browser_session_id,
                page_id="page",
                revision="changed" if stale_page else "revision",
                lifecycle="active",
            )
            finished = asyncio.Event()
            outcomes = []

            async def peer(connection):
                try:
                    bound = await bind_browser_guest_channel(
                        coordinator=owner, allocation=allocation, connection=connection
                    )
                    intent = intent_for(bootstrap).model_copy(
                        update={"identity": bound.record.identity}
                    )
                    pending = await owner.request_takeover(
                        principal=BrowserControlPrincipal(subject="operator"),
                        operator_session_id="operator-session",
                        intent=intent,
                    )
                    assert pending.state == "takeover_requested"
                    committed = asyncio.Event()
                    release = asyncio.Event()
                    original = store.publish_session_operation_guarded_with_store_time
                    writes = []

                    async def paused_ack(*args, **kwargs):
                        result = await original(*args, **kwargs)
                        writes.append(kwargs["idempotency_key"])
                        if len(writes) == 1:
                            committed.set()
                            await release.wait()
                        return result

                    if cancel_publication:
                        monkeypatch.setattr(
                            store, "publish_session_operation_guarded_with_store_time", paused_ack
                        )
                    commands = BrowserGuestCommandOwner(
                        coordinator=owner, connection=connection, bound=bound
                    )
                    monkeypatch.setattr(owner, "_acquisition_lease_until", lambda pending: 4000)
                    exchange = asyncio.create_task(
                        commands.step()
                        if through_owner
                        else acquire_browser_guest_control(
                            coordinator=owner,
                            connection=connection,
                            bound=bound,
                            pending=pending,
                            sequence=1,
                            lease_until_ms=4000,
                        )
                    )
                    if cancel_publication:
                        await committed.wait()
                        written = (await owner._load(pending.identity))[1]
                        assert written.acquisition_audit is not None
                        if through_owner:
                            assert commands.publication_transition == (
                                pending,
                                written,
                            )
                        assert exchange.cancel("operator channel lost during publication")
                        release.set()
                    try:
                        acquired = await exchange
                    except asyncio.CancelledError:
                        assert cancel_publication
                        assert exchange.cancelled() and exchange.cancelling() == 1
                        _, current = await owner._load(pending.identity)
                        assert current.state == "control_uncertain" and current.control_epoch == 2
                        assert current.acquisition_audit is not None
                        successor = owner._acquisition_successor(
                            pending, lease_until_ms=4000, audit=current.acquisition_audit
                        )
                        assert current == successor.model_copy(
                            update={
                                "revision": successor.revision + 1,
                                "state": "control_uncertain",
                            }
                        )
                        assert current.request == pending.request
                        if through_owner:
                            await commands.disconnect()
                            assert (await owner._load(pending.identity))[1] == current
                        assert len(writes) == 2  # Acquisition and exact teardown, no replay.
                        assert (
                            await owner._fence_guest_acquisition(
                                expected=pending,
                                lease_until_ms=4000,
                                audit=current.acquisition_audit,
                            )
                            == current
                        )
                    except Exception:
                        if not stale_page:
                            raise
                        _, current = await owner._load(pending.identity)
                        assert current.state == "control_uncertain"
                        assert current.lease_until_ms is None
                    else:
                        assert not stale_page
                        _, current = await coordinator(store, Policy(True))._load(pending.identity)
                        if through_owner:
                            assert acquired is None
                            assert current == commands.bound.record
                            assert commands.publication_transition is None
                        else:
                            assert current == acquired.record
                        assert current.acquisition_audit is not None
                        assert current == owner._acquisition_successor(
                            pending, lease_until_ms=4000, audit=current.acquisition_audit
                        )
                        assert current.state == "operator_controlled"
                        assert current.control_epoch == 2 and current.lease_until_ms == 4000
                    outcomes.append("verified")
                finally:
                    finished.set()

            async with serve(
                peer,
                "127.0.0.1",
                0,
                ssl=server_tls,
                subprotocols=[Subprotocol(CONTROL_SUBPROTOCOL)],
            ) as server:
                connection = await open_guest_control_channel(
                    endpoint=f"wss://127.0.0.1:{server.sockets[0].getsockname()[1]}/guest",
                    credential="a" * 64,
                    tls=client_tls,
                )
                channel = GuestControlChannel(
                    daemon, scope_sha256=browser_allocation_digest(allocation)
                )
                task = asyncio.create_task(channel.run(connection))
                await asyncio.wait_for(finished.wait(), 10)
                with pytest.raises(GuestControlFailure if stale_page else ConnectionClosed):
                    await task
            assert outcomes == ["verified"]
            assert await owner.drain()

    asyncio.run(scenario())


@pytest.mark.parametrize("peer_close_code", [None, 1000, 1008])
def test_private_bootstrap_and_shutdown_close_real_tls_channel(
    control_tls, monkeypatch, peer_close_code
):
    from cayu.tools import _browser_guest

    async def scenario():
        server_tls, client_tls = control_tls
        ready = asyncio.Event()
        native_release = asyncio.Event()
        if peer_close_code is None:
            native_release.set()
        daemon = _InteractiveDaemon("bs_shutdown")

        class Context:
            async def close(self):
                await native_release.wait()

        daemon.context = Context()

        async def connect(**kwargs):
            return await open_guest_control_channel(**kwargs, tls=client_tls)

        monkeypatch.setattr(_browser_guest, "open_guest_control_channel", connect)

        async def peer(connection):
            hello = json.loads(await connection.recv())
            await connection.send(
                json.dumps(
                    {
                        "kind": "bind",
                        "schema_version": 1,
                        "scope_sha256": "d" * 64,
                        "channel_id": hello["channel_id"],
                        "worker_instance": hello["worker_instance"],
                        "binding_sha256": "a" * 64,
                    }
                )
            )
            await connection.recv()
            ready.set()
            if peer_close_code is not None:
                await connection.close(code=peer_close_code)
            await connection.wait_closed()

        async with serve(
            peer, "127.0.0.1", 0, ssl=server_tls, subprotocols=[Subprotocol(CONTROL_SUBPROTOCOL)]
        ) as server:
            port = server.sockets[0].getsockname()[1]
            await daemon.bootstrap_operator_channel(
                {
                    "endpoint": f"wss://127.0.0.1:{port}/control",
                    "credential": "a" * 64,
                    "scope_sha256": "d" * 64,
                }
            )
            try:
                async with asyncio.timeout(5):
                    await ready.wait()
                if peer_close_code is not None:
                    assert daemon._operator_bootstrap_task is not None
                    errors = await asyncio.wait_for(daemon._operator_bootstrap_task, 5)
                    assert len(errors) == 1 and isinstance(errors[0], ConnectionClosed)
                    assert daemon.control.state == "control_uncertain"
                    with pytest.raises(GuestControlFailure):
                        daemon.control.check_model(None, "observe")
                    # A normal network close is not proof that native work stopped.
                    assert not await daemon.close(timeout_seconds=0.05)
                    assert daemon.context is not None
                    assert not daemon.session_cleanup_tasks["context"].done()
                    assert daemon.control.state == "control_uncertain"
                    native_release.set()
                assert await daemon.close() is (peer_close_code != 1008)
                assert daemon._operator_bootstrap_task is not None
                assert daemon._operator_bootstrap_task.done()
                if peer_close_code is None:
                    assert daemon._operator_bootstrap_task.result() == ()
                else:
                    assert daemon._operator_bootstrap_task.result() == errors
                assert daemon._operator_connection is None
                assert daemon._operator_channel_id is None
                assert daemon.control.state == (
                    "control_uncertain" if peer_close_code == 1008 else "closed"
                )
            finally:
                native_release.set()
                await daemon.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("wrong_scope", [False, True])
def test_server_handshake_binds_real_guest_to_durable_allocation(
    control_tls, backend, tmp_path, wrong_scope
):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, bootstrap):
            allocation = BrowserControlAllocation.model_validate(
                bootstrap.changed_record.identity.model_dump(exclude={"worker_instance_id"})
            )
            daemon = _InteractiveDaemon(allocation.browser_session_id)
            daemon.context = object()
            owner = coordinator(store, Policy(True))
            before = await store.load_checkpoint(allocation.session_id)
            server_tls, client_tls = control_tls
            bound = asyncio.get_running_loop().create_future()
            release = asyncio.Event()
            disconnected = asyncio.get_running_loop().create_future()
            capabilities = BrowserGuestBootstrap()
            credential = capabilities.issue(allocation)
            authenticated = {}

            async def authenticate(connection, request):
                header = request.headers.get("Authorization", "")
                try:
                    if not header.startswith("Bearer "):
                        raise BrowserControlPermissionDenied()
                    authenticated[id(connection)] = capabilities.consume(header[7:])
                except BrowserControlPermissionDenied:
                    return connection.respond(HTTPStatus.FORBIDDEN, "Browser control denied.")
                return None

            async def server_peer(connection):
                try:
                    result = await bind_browser_guest_channel(
                        coordinator=owner,
                        allocation=authenticated.pop(id(connection)),
                        connection=connection,
                    )
                    bound.set_result(result)
                    await release.wait()
                    await connection.wait_closed()
                    disconnected.set_result(
                        await owner.mark_channel_uncertain(expected=result.record)
                    )
                except Exception as error:
                    if not bound.done():
                        bound.set_exception(error)
                    else:
                        disconnected.set_exception(error)

            async with serve(
                server_peer,
                "127.0.0.1",
                0,
                ssl=server_tls,
                subprotocols=[Subprotocol(CONTROL_SUBPROTOCOL)],
                process_request=authenticate,
            ) as server:
                port = server.sockets[0].getsockname()[1]
                with pytest.raises(InvalidStatus) as rejected:
                    await open_guest_control_channel(
                        endpoint=f"wss://127.0.0.1:{port}/control",
                        credential="0" * 64,
                        tls=client_tls,
                    )
                assert rejected.value.response.status_code == 403
                assert not bound.done()
                assert await store.load_checkpoint(allocation.session_id) == before
                connection = await open_guest_control_channel(
                    endpoint=f"wss://127.0.0.1:{port}/control",
                    credential=credential,
                    tls=client_tls,
                )
                with pytest.raises(InvalidStatus) as replayed:
                    await open_guest_control_channel(
                        endpoint=f"wss://127.0.0.1:{port}/control",
                        credential=credential,
                        tls=client_tls,
                    )
                assert replayed.value.response.status_code == 403
                task = asyncio.create_task(
                    GuestControlChannel(
                        daemon,
                        scope_sha256="0" * 64
                        if wrong_scope
                        else browser_allocation_digest(allocation),
                    ).run(connection)
                )
                try:
                    if wrong_scope:
                        async with asyncio.timeout(10):
                            with pytest.raises(BrowserControlConflict):
                                await bound
                        assert await store.load_checkpoint(allocation.session_id) == before
                        assert daemon.control.binding_sha256 is None
                        return
                    async with asyncio.timeout(10):
                        result = await bound
                    assert (
                        result.record.identity.worker_instance_id == daemon.visual_worker_instance
                    )
                    assert daemon.control.binding_sha256 == result.binding_sha256
                    # Reconstruction may read the exact binding, but another worker
                    # cannot replace it even when every allocation field is identical.
                    restarted = coordinator(store, Policy(True))
                    assert (
                        await restarted.bind_guest(
                            allocation=allocation, worker_instance_id=daemon.visual_worker_instance
                        )
                        == result.record
                    )
                    with pytest.raises(BrowserControlConflict):
                        await restarted.bind_guest(
                            allocation=allocation, worker_instance_id="vw_" + "f" * 32
                        )
                finally:
                    release.set()
                    await connection.close()
                    await asyncio.gather(task, return_exceptions=True)
                async with asyncio.timeout(10):
                    fenced = await disconnected
                assert fenced.state == "control_uncertain"
                assert fenced.revision == result.record.revision + 1
                assert fenced.request is None
                assert await owner.mark_channel_uncertain(expected=result.record) == fenced
            assert daemon.control.state == "control_uncertain"
            assert await owner.drain()

    asyncio.run(scenario())


def test_guest_control_exchange_uses_exact_worker_and_sequence(control_tls):
    async def scenario():
        server_tls, client_tls = control_tls
        daemon = _InteractiveDaemon("bs_channel")
        daemon.context = object()
        observed = []

        async def server_peer(connection):
            hello = json.loads(await connection.recv())
            assert hello["worker_instance"] == daemon.visual_worker_instance
            assert hello["scope_sha256"] == "d" * 64
            await connection.send(
                json.dumps(
                    {
                        "kind": "bind",
                        "schema_version": 1,
                        "scope_sha256": "d" * 64,
                        "channel_id": hello["channel_id"],
                        "worker_instance": hello["worker_instance"],
                        "binding_sha256": "a" * 64,
                    }
                )
            )
            observed.append(json.loads(await connection.recv()))
            common = {
                "channel_id": hello["channel_id"],
                "worker_instance": hello["worker_instance"],
                "binding_sha256": "a" * 64,
            }
            material = takeover_material()
            await connection.send(
                json.dumps({**common, **material, "kind": "takeover", "sequence": 1})
            )
            observed.append(json.loads(await connection.recv()))
            await connection.send(
                json.dumps(
                    {
                        **common,
                        "kind": "handback",
                        "sequence": 2,
                        "request_id": material["request_id"],
                        "epoch": 2,
                    }
                )
            )
            observed.append(json.loads(await connection.recv()))
            # Repeating a connection sequence cannot invoke the native method.
            await connection.send(json.dumps({**common, "kind": "status", "sequence": 2}))
            await connection.wait_closed()

        async with serve(
            server_peer,
            "127.0.0.1",
            0,
            ssl=server_tls,
            subprotocols=[Subprotocol(CONTROL_SUBPROTOCOL)],
        ) as server:
            port = server.sockets[0].getsockname()[1]
            connection = await open_guest_control_channel(
                endpoint=f"wss://127.0.0.1:{port}/control", credential="a" * 64, tls=client_tls
            )
            with pytest.raises(GuestControlFailure):
                await GuestControlChannel(daemon, scope_sha256="d" * 64).run(connection)
        assert [item["state"] for item in observed] == [
            "agent_controlled",
            "operator_controlled",
            "agent_controlled",
        ]
        assert observed[-1]["fresh_observation_required"] is True
        assert daemon.control.epoch == 3
        assert daemon.control.state == "control_uncertain"
        assert daemon._operator_channel_id is None

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_guest_handshake_mismatch_or_owner_cancellation_closes_channel(control_tls, cancel):
    async def scenario():
        server_tls, client_tls = control_tls
        daemon = _InteractiveDaemon("bs_channel")
        daemon.context = object()
        ready = asyncio.Event()

        async def server_peer(connection):
            hello = json.loads(await connection.recv())
            await connection.send(
                json.dumps(
                    {
                        "kind": "bind",
                        "schema_version": 1,
                        "scope_sha256": "d" * 64,
                        "channel_id": hello["channel_id"],
                        "worker_instance": hello["worker_instance"] if cancel else "wrong-worker",
                        "binding_sha256": "a" * 64,
                    }
                )
            )
            if cancel:
                await connection.recv()
                ready.set()
            await connection.wait_closed()

        async with serve(
            server_peer,
            "127.0.0.1",
            0,
            ssl=server_tls,
            subprotocols=[Subprotocol(CONTROL_SUBPROTOCOL)],
        ) as server:
            port = server.sockets[0].getsockname()[1]
            connection = await open_guest_control_channel(
                endpoint=f"wss://127.0.0.1:{port}/control", credential="a" * 64, tls=client_tls
            )
            task = asyncio.create_task(
                GuestControlChannel(daemon, scope_sha256="d" * 64).run(connection)
            )
            if cancel:
                await ready.wait()
                assert task.cancel("runtime disconnect")
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert task.cancelling() == 1
                assert task.cancelled()
                assert daemon.control.state == "control_uncertain"
            else:
                with pytest.raises(GuestControlFailure):
                    await task
                assert daemon.control.binding_sha256 is None
            with pytest.raises(ConnectionClosed):
                await connection.recv()
            assert daemon._operator_channel_id is None

    asyncio.run(scenario())
