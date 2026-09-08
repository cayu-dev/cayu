"""Real WSS loss through the guest ASGI owner at durable transition barriers.

Native control is real; page rendering/model work are bounded test doubles.
The existing Chromium worker-loss suite separately proves external process loss.
"""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from tests.core.test_browser_control import operator_purpose
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_control_coordinator import coordinator, intent_for
from tests.core.test_browser_control_publisher import publication_fixture
from tests.core.test_browser_control_service import bound_browser_context
from tests.core.test_browser_control_transport import control_tls as _control_tls
from tests.core.test_browser_session import _interactive_request
from tests.server._browser_control_tls_server import browser_control_tls_server

from cayu.runtime._browser_control_checkpoint import (
    BrowserControlCheckpointMutation,
    browser_control_checkpoint_read_scope,
)
from cayu.runtime._browser_control_model import browser_model_control_admission
from cayu.runtime._browser_control_publication import BrowserControlPublication
from cayu.runtime._browser_control_publisher import BrowserControlPublisher
from cayu.runtime._browser_control_service import BrowserControlService
from cayu.runtime.browser_control import (
    BrowserControlAllocation,
    BrowserControlConflict,
    BrowserControlPrincipal,
    BrowserHandbackIntent,
    BrowserRenewIntent,
    BrowserSensitiveEntryIntent,
    closed_browser_control_successor,
)
from cayu.server._browser_guest_routes import _GuestSocket, create_browser_guest_router
from cayu.tools import _browser_guest
from cayu.tools._browser_control_guest import GuestControlChannel, GuestControlFence
from cayu.tools._browser_control_transport import open_guest_control_channel
from cayu.tools.browser_session import BrowserSessionTool, _RunnerBrowserSessionBackend

control_tls = _control_tls


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "phase",
    [
        "acquisition_wait",
        "takeover_queued",
        "renewal_reply",
        "renewal_queued",
        "renewal_commit",
        "handback_reply",
        "handback_queued",
        "sensitive_queued",
        "handback_commit",
        "handed_back",
        "observation_pending",
        "observation_committed",
        "closed",
        "uncertain",
    ],
)
def test_transport_loss_fences_exact_transition(tmp_path, monkeypatch, control_tls, backend, phase):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, bootstrap):
            control = coordinator(store, Policy(True))
            app = FastAPI()
            service = None
            release, reached = asyncio.Event(), asyncio.Event()
            model_release, model_started = asyncio.Event(), asyncio.Event()
            model_effects = []
            tasks = []
            connection = None
            idle_reached, idle_release = asyncio.Event(), asyncio.Event()
            idle = _GuestSocket.idle

            async def pause_idle(self):
                idle_reached.set()
                await idle_release.wait()
                return await idle(self)

            if phase in {"takeover_queued", "closed", "uncertain"}:
                monkeypatch.setattr(_GuestSocket, "idle", pause_idle)
            else:
                idle_release.set()

            async with browser_control_tls_server(app, tmp_path) as port:
                endpoint = f"wss://127.0.0.1:{port}/browser-control/guest"
                service = BrowserControlService(purpose=operator_purpose(), guest_endpoint=endpoint)
                app.include_router(
                    create_browser_guest_router(service=service, coordinator=control)
                )
                delivery = asyncio.Queue()

                async def deliver(self, ctx, **kwargs):
                    await delivery.put(kwargs)

                monkeypatch.setattr(_RunnerBrowserSessionBackend, "bootstrap_control", deliver)
                backend_tool = BrowserSessionTool()._backend
                assert isinstance(backend_tool, _RunnerBrowserSessionBackend)
                bootstrap_task = asyncio.create_task(
                    service.bootstrap(
                        bound_browser_context(bootstrap.changed_record.identity),
                        backend=backend_tool,
                        browser_session_id=bootstrap.changed_record.identity.browser_session_id,
                        arguments={"operation": "navigate"},
                    )
                )
                tasks.append(bootstrap_task)
                material = await asyncio.wait_for(delivery.get(), 5)
                daemon = _browser_guest._InteractiveDaemon(
                    bootstrap.changed_record.identity.browser_session_id
                )
                daemon.context = object()
                daemon.control = GuestControlFence(
                    worker_instance=daemon.visual_worker_instance, wall_clock=lambda: 1.0
                )
                daemon.configuration_limits = _interactive_request("observe").limits
                daemon.pages["page"] = _browser_guest._InteractivePage(
                    page=SimpleNamespace(url="https://site.test/"),
                    session_id=daemon.session_id,
                    page_id="page",
                    revision="revision",
                    lifecycle="active",
                    configured=True,
                )
                daemon.active_page_id = "page"
                connection = await open_guest_control_channel(
                    endpoint=endpoint, credential=material["credential"], tls=control_tls[1]
                )
                native = GuestControlChannel(daemon, scope_sha256=material["scope_sha256"])
                last_kind = None

                class Wire:
                    async def recv(self):
                        nonlocal last_kind
                        value = await connection.recv()
                        last_kind = json.loads(value)["kind"]
                        return value

                    async def send(self, value):
                        if (phase == "renewal_reply" and last_kind == "renew") or (
                            phase == "handback_reply" and last_kind == "handback"
                        ):
                            reached.set()
                            await release.wait()
                        await connection.send(value)

                    async def close(self):
                        await connection.close()

                guest = asyncio.create_task(native.run(Wire()))
                tasks.append(guest)
                try:
                    bound = await asyncio.wait_for(bootstrap_task, 5)
                    identity = bound.record.identity
                    principal = BrowserControlPrincipal(subject="operator")

                    async def wait_record(predicate):
                        async with asyncio.timeout(10):
                            while True:
                                record = (await control._load(identity))[1]
                                if predicate(record):
                                    return record
                                await asyncio.sleep(0.01)

                    async def wait_route():
                        # Let the actual socket-loss finalizer run. drain() must
                        # not manufacture success by cancelling a still-live route.
                        async with asyncio.timeout(10):
                            while any(not task.done() for task in service.channels._tasks):
                                await asyncio.sleep(0.01)
                        assert await service.channels.drain()

                    if phase == "acquisition_wait":

                        async def configured(request):
                            pass

                        async def operation(request):
                            model_started.set()
                            await model_release.wait()
                            model_effects.append("settled")
                            return {"kind": "success", "observation": {"revision": "revision"}}

                        monkeypatch.setattr(daemon, "_ensure_configuration", configured)
                        monkeypatch.setattr(daemon, "_execute_locked", operation)
                        model = asyncio.create_task(
                            daemon.execute(
                                replace(
                                    _interactive_request("observe"),
                                    invocation_control_epoch=1,
                                    session_id=daemon.session_id,
                                )
                            )
                        )
                        tasks.append(model)
                        await asyncio.wait_for(model_started.wait(), 5)
                        acquire = daemon.acquire_operator_control

                        async def acquiring(**kwargs):
                            reached.set()
                            return await acquire(**kwargs)

                        monkeypatch.setattr(daemon, "acquire_operator_control", acquiring)

                    if phase in {"closed", "uncertain"}:
                        await asyncio.wait_for(idle_reached.wait(), 5)
                        controls, current = await control._load(identity)
                        if phase == "closed":
                            desired = closed_browser_control_successor(current)
                            assert desired is not None
                            # Terminal production is covered by the runtime-close
                            # suite. Here its exact durable successor races actual
                            # route teardown, while that route still owns current.
                            await BrowserControlPublisher(store).publish(
                                BrowserControlPublication(
                                    BrowserControlCheckpointMutation(
                                        identity.session_id,
                                        controls,
                                        controls.replace_record(expected=current, desired=desired),
                                    )
                                )
                            )
                        else:
                            desired = await control.mark_channel_uncertain(expected=current)
                        await connection.close()
                        idle_release.set()
                        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 10)
                        await wait_route()
                        assert (await control._read_record(identity))[3] == desired
                        assert await control.mark_channel_uncertain(expected=current) == desired
                        with browser_control_checkpoint_read_scope(identity.session_id):
                            checkpoint = await store.load_checkpoint(identity.session_id)
                        allocation = BrowserControlAllocation.model_validate(
                            identity.model_dump(exclude={"worker_instance_id"})
                        )
                        with pytest.raises(BrowserControlConflict):
                            browser_model_control_admission(
                                checkpoint, allocation=allocation, operation_name="observe"
                            )
                        return
                    if phase == "takeover_queued":
                        await asyncio.wait_for(idle_reached.wait(), 5)
                    pending = await control.request_takeover(
                        principal=principal,
                        operator_session_id="operator-session",
                        intent=intent_for(bootstrap).model_copy(
                            update={"identity": identity, "maximum_until_ms": 60_000}
                        ),
                    )
                    source = pending
                    if phase == "takeover_queued":
                        reached.set()
                    elif phase != "acquisition_wait":
                        acquired = await wait_record(
                            lambda value: value.state == "operator_controlled"
                        )
                        if phase.endswith("queued"):
                            idle_release.clear()
                            idle_reached.clear()
                            monkeypatch.setattr(_GuestSocket, "idle", pause_idle)
                            await asyncio.wait_for(idle_reached.wait(), 5)
                        if phase.endswith("commit"):
                            name = (
                                "_publish_guest_renewal"
                                if phase == "renewal_commit"
                                else "_publish_guest_handback"
                            )
                            publish = getattr(control, name)

                            async def committed(**kwargs):
                                result = await publish(**kwargs)
                                reached.set()
                                await release.wait()
                                return result

                            monkeypatch.setattr(control, name, committed)
                        common = dict(
                            identity=identity,
                            expected_record_revision=acquired.revision,
                            expected_control_epoch=acquired.control_epoch,
                            request_id=pending.request.request_id,
                        )
                        if phase.startswith("renewal"):
                            source = await control.request_renewal(
                                principal=principal,
                                operator_session_id="operator-session",
                                intent=BrowserRenewIntent.model_validate(
                                    {
                                        **common,
                                        "expected_lease_until_ms": acquired.lease_until_ms,
                                        "lease_until_ms": acquired.lease_until_ms + 1,
                                    }
                                ),
                            )
                        elif phase == "sensitive_queued":
                            source = await control.request_sensitive_entry(
                                principal=principal,
                                operator_session_id="operator-session",
                                intent=BrowserSensitiveEntryIntent.model_validate(common),
                            )
                        else:
                            source = await control.request_handback(
                                principal=principal,
                                operator_session_id="operator-session",
                                intent=BrowserHandbackIntent.model_validate(common),
                            )
                            if phase in {
                                "handed_back",
                                "observation_pending",
                                "observation_committed",
                            }:
                                returned = await wait_record(
                                    lambda value: (
                                        value.state == "agent_controlled"
                                        and value.fresh_observation_required
                                    )
                                )
                                if phase.startswith("observation"):
                                    idle_release.clear()
                                    idle_reached.clear()
                                    monkeypatch.setattr(_GuestSocket, "idle", pause_idle)
                                    await asyncio.wait_for(idle_reached.wait(), 5)
                                    # Native observation completes first. Its host
                                    # terminal publication is deliberately absent;
                                    # disconnect must not erase that durable fence.
                                    daemon.control.fresh_observation_required = False
                                    if phase == "observation_committed":
                                        controls, current = await control._load(identity)
                                        assert current == returned
                                        desired = current.model_copy(
                                            update={
                                                "revision": current.revision + 1,
                                                "fresh_observation_required": False,
                                            }
                                        )
                                        await BrowserControlPublisher(store).publish(
                                            BrowserControlPublication(
                                                BrowserControlCheckpointMutation(
                                                    identity.session_id,
                                                    controls,
                                                    controls.replace_record(
                                                        expected=current, desired=desired
                                                    ),
                                                )
                                            )
                                        )
                                reached.set()
                        if phase.endswith("queued"):
                            reached.set()
                    await asyncio.wait_for(reached.wait(), 5)
                    before = (await control._load(identity))[1]
                    # Close a real TLS WebSocket, not a constructed exception or
                    # a direct invocation of the owner's disconnect helper.
                    await connection.close()
                    idle_release.set()
                    if phase == "acquisition_wait":
                        assert not model.done() and model.cancelling() == 0
                        fenced = await wait_record(lambda value: value.state == "control_uncertain")
                        assert fenced.control_epoch == source.control_epoch
                        assert fenced.lease_until_ms is None
                    release.set()
                    model_release.set()
                    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 10)
                    fenced = await wait_record(lambda value: value.state == "control_uncertain")
                    assert fenced == before.model_copy(
                        update={"revision": before.revision + 1, "state": "control_uncertain"}
                    )
                    assert fenced.identity == identity and fenced.request == pending.request
                    assert fenced.pending_input_sequence is None
                    assert fenced.control_epoch == before.control_epoch
                    if phase.startswith("handback") or phase == "handed_back":
                        assert (
                            fenced.fresh_observation_required == before.fresh_observation_required
                        )
                    with browser_control_checkpoint_read_scope(identity.session_id):
                        checkpoint = await store.load_checkpoint(identity.session_id)
                    allocation = BrowserControlAllocation.model_validate(
                        identity.model_dump(exclude={"worker_instance_id"})
                    )
                    with pytest.raises(BrowserControlConflict):
                        browser_model_control_admission(
                            checkpoint, allocation=allocation, operation_name="observe"
                        )
                    assert daemon.control.state == "control_uncertain"
                    with pytest.raises(_browser_guest._GuestFailure, match="policy_denied"):
                        await daemon.execute(
                            replace(
                                _interactive_request("observe"),
                                session_id=daemon.session_id,
                                operation_id="after-disconnect",
                                invocation_control_epoch=fenced.control_epoch,
                            )
                        )
                    if phase == "acquisition_wait":
                        assert model_effects == ["settled"]
                    await wait_route()
                    assert await control.drain()
                    assert (await control._load(identity))[1] == fenced
                finally:
                    idle_release.set()
                    release.set()
                    model_release.set()
                    if connection is not None:
                        await connection.close()
                    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 10)
                    await service.channels.drain()
                    await service.drain_bootstrap_deliveries()
                    await control.drain()

    asyncio.run(scenario())
