"""Kill a coordinator across native input and handback publication boundaries."""

import asyncio
import multiprocessing
import os
import ssl
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_control_coordinator import coordinator, intent_for
from tests.core.test_browser_control_publisher import publication_fixture
from tests.core.test_browser_control_transport import control_tls as _control_tls
from tests.core.test_browser_session import _interactive_request
from websockets.asyncio.server import serve
from websockets.typing import Subprotocol

from cayu import SQLiteSessionStore
from cayu.runtime._browser_control_channel import (
    BrowserGuestCommandOwner,
    bind_browser_guest_channel,
    browser_allocation_digest,
)
from cayu.runtime._browser_control_checkpoint import browser_control_checkpoint_read_scope
from cayu.runtime._browser_control_model import browser_model_control_admission
from cayu.runtime.browser_control import (
    BrowserControlAllocation,
    BrowserControlConflict,
    BrowserControlPageAudit,
    BrowserControlPrincipal,
    BrowserControlRecord,
    BrowserHandbackIntent,
    BrowserSensitiveEntryIntent,
    BrowserTextInputIntent,
)
from cayu.tools._browser_control_guest import GuestControlChannel, GuestControlFence
from cayu.tools._browser_control_transport import CONTROL_SUBPROTOCOL, open_guest_control_channel
from cayu.tools._browser_guest import _GuestFailure, _InteractiveDaemon, _InteractivePage

control_tls = _control_tls
pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_BROWSER_CONTROL_LIVE") != "1", reason="Opt-in local Chromium acceptance."
)


def _coordinator_worker(directory, pipe, phase="input"):
    async def scenario():
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(directory / "certificate.pem", directory / "key.pem")
        async with publication_fixture("sqlite", directory) as (store, bootstrap):
            control = coordinator(store, Policy(True))
            allocation = BrowserControlAllocation.model_validate(
                bootstrap.changed_record.identity.model_dump(exclude={"worker_instance_id"})
            )

            async def no_viewers(identity):
                # This fixture never opens a viewer or admits a frame.
                return None

            async def peer(connection):
                assert connection.request.headers["Authorization"] == "Bearer " + "c" * 64
                bound = await bind_browser_guest_channel(
                    coordinator=control, allocation=allocation, connection=connection
                )
                owner = BrowserGuestCommandOwner(
                    coordinator=control,
                    connection=connection,
                    bound=bound,
                    suspend_viewer_delivery=no_viewers,
                )
                principal = BrowserControlPrincipal(subject="operator")
                intent = intent_for(bootstrap).model_copy(
                    update={"identity": bound.record.identity}
                )
                await control.request_takeover(
                    principal=principal, operator_session_id="server-session", intent=intent
                )
                await owner.step()
                acquired = owner.bound.record
                assert acquired.request is not None
                if phase != "input":
                    await control.request_handback(
                        principal=principal,
                        operator_session_id="server-session",
                        intent=BrowserHandbackIntent(
                            identity=acquired.identity,
                            request_id=acquired.request.request_id,
                            expected_record_revision=acquired.revision,
                            expected_control_epoch=acquired.control_epoch,
                        ),
                    )
                    original = store.publish_session_operation_guarded_with_store_time

                    async def publication(*args, **kwargs):
                        assert owner.publication_transition is not None
                        source, successor = owner.publication_transition
                        assert successor.handback_audit is not None
                        if phase == "handback_after_commit":
                            await original(*args, **kwargs)
                        pipe.send(
                            (source.model_dump(mode="json"), successor.model_dump(mode="json"))
                        )
                        await asyncio.Event().wait()
                        raise AssertionError("The publication waiter should die.")

                    store.publish_session_operation_guarded_with_store_time = publication
                    await owner.step()
                    raise AssertionError("The worker should die during handback publication.")
                await control.request_sensitive_entry(
                    principal=principal,
                    operator_session_id="server-session",
                    intent=BrowserSensitiveEntryIntent(
                        identity=acquired.identity,
                        request_id=acquired.request.request_id,
                        expected_record_revision=acquired.revision,
                        expected_control_epoch=acquired.control_epoch,
                    ),
                )
                await owner.step()
                acquired = owner.bound.record
                assert acquired.request is not None
                intent = BrowserTextInputIntent(
                    identity=acquired.identity,
                    request_id=acquired.request.request_id,
                    expected_record_revision=acquired.revision,
                    expected_control_epoch=acquired.control_epoch,
                    input_sequence=1,
                    page=acquired.request.pages[0],
                )
                sending = asyncio.create_task(
                    owner.request_text_input(
                        principal=principal,
                        operator_session_id="server-session",
                        intent=intent,
                        text="worker-loss-native-canary",
                    )
                )
                while owner._input is None:
                    await asyncio.sleep(0)
                assert acquired.acquisition_audit is not None
                pipe.send(
                    (
                        intent.model_dump(mode="json"),
                        acquired.acquisition_audit.model_dump(mode="json"),
                    )
                )
                await owner.step()
                await sending
                raise AssertionError("The worker should die before native acknowledgement.")

            async with serve(
                peer, "127.0.0.1", 0, ssl=tls, subprotocols=[Subprotocol(CONTROL_SUBPROTOCOL)]
            ) as server:
                pipe.send(
                    (
                        f"wss://127.0.0.1:{server.sockets[0].getsockname()[1]}/guest",
                        allocation.model_dump(mode="json"),
                    )
                )
                await asyncio.Event().wait()

    asyncio.run(scenario())


async def _assert_handback_after_worker_loss(
    *, receiver, worker, phase, task, daemon, allocation, directory
):
    assert await asyncio.to_thread(receiver.poll, 10), "Handback publication did not start."
    raw_source, raw_successor = receiver.recv()
    source = BrowserControlRecord.model_validate(raw_source)
    successor = BrowserControlRecord.model_validate(raw_successor)
    assert daemon.control.state == "agent_controlled"
    assert daemon.control.fresh_observation_required
    assert daemon.pages["page"].revision is None
    worker.kill()
    worker.join(5)
    assert not worker.is_alive() and worker.exitcode != 0
    result = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 10)
    assert isinstance(result[0], BaseException)
    assert daemon.control.state == "control_uncertain"
    raw = SQLiteSessionStore(
        directory / "publication.sqlite",
        ownership_clock=lambda: datetime(1970, 1, 1, 0, 0, 1, tzinfo=UTC),
    )
    try:
        recovered = coordinator(raw, Policy(True))
        _, current = await recovered._load(source.identity)
        assert current == (successor if phase == "handback_after_commit" else source)
        assert current.acquisition_audit is not None
        assert current.acquisition_audit.locations[0].origin == "https://worker-loss.test"
        assert "private-path" not in current.model_dump_json()
        assert "not-an-audit-field" not in current.model_dump_json()
        with pytest.raises(BrowserControlConflict):
            await recovered.bind_guest(allocation=allocation, worker_instance_id="vw_" + "f" * 32)
        with browser_control_checkpoint_read_scope("session"):
            checkpoint = await raw.load_checkpoint("session")
        with pytest.raises(BrowserControlConflict):
            browser_model_control_admission(
                checkpoint, allocation=allocation, operation_name="click"
            )
        if phase == "handback_after_commit":
            assert current.handback_audit is not None
            assert current.handback_audit.locations[0].origin == "https://worker-loss.test"
            assert current.fresh_observation_required
            # Committed handback allows only an observation attempt at the host;
            # it cannot override the actual guest's disconnected channel fence.
            assert (
                browser_model_control_admission(
                    checkpoint, allocation=allocation, operation_name="observe"
                )
                is not None
            )
        else:
            assert current.handback_audit is None
            with pytest.raises(BrowserControlConflict):
                browser_model_control_admission(
                    checkpoint, allocation=allocation, operation_name="observe"
                )
        with pytest.raises(_GuestFailure):
            await daemon.execute(
                replace(
                    _interactive_request("observe"),
                    session_id=daemon.session_id,
                    page_id="page",
                    invocation_control_epoch=daemon.control.epoch,
                )
            )
        assert (await recovered._load(source.identity))[1] == current
    finally:
        await raw.close()


@pytest.mark.parametrize("phase", ["input", "handback_before_commit", "handback_after_commit"])
def test_killed_coordinator_cannot_replay_native_work(tmp_path, control_tls, monkeypatch, phase):
    from playwright.async_api import async_playwright

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    worker = context.Process(target=_coordinator_worker, args=(tmp_path, sender, phase))
    worker.start()
    sender.close()
    try:
        assert receiver.poll(30), "Coordinator did not start."
        endpoint, raw_allocation = receiver.recv()

        async def scenario():
            allocation = BrowserControlAllocation.model_validate(raw_allocation)
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(
                    executable_path=os.environ.get("CAYU_BROWSER_CONTROL_CHROMIUM"), headless=True
                )
                release, dispatched = asyncio.Event(), asyncio.Event()
                task = None
                try:
                    page = await browser.new_page()
                    await page.route(
                        "https://worker-loss.test/**",
                        lambda route: route.fulfill(
                            body="<input type='password'>", content_type="text/html"
                        ),
                    )
                    await page.goto(
                        "https://worker-loss.test/private-path?token=not-an-audit-field"
                    )
                    await page.locator("input").focus()
                    daemon = _InteractiveDaemon(allocation.browser_session_id)
                    daemon.context = page.context
                    daemon.control = GuestControlFence(
                        worker_instance=daemon.visual_worker_instance, wall_clock=lambda: 1.0
                    )
                    daemon.configuration_limits = _interactive_request("observe").limits
                    daemon.pages["page"] = _InteractivePage(
                        page=page,
                        session_id=daemon.session_id,
                        page_id="page",
                        lifecycle="active",
                        revision="revision",
                        configured=True,
                    )
                    daemon.pages["page"].cdp = await page.context.new_cdp_session(page)
                    daemon.active_page_id = "page"
                    insert = page.keyboard.insert_text

                    async def delayed_ack(value):
                        await insert(value)
                        dispatched.set()
                        await release.wait()

                    monkeypatch.setattr(page.keyboard, "insert_text", delayed_ack)
                    connection = await open_guest_control_channel(
                        endpoint=endpoint, credential="c" * 64, tls=control_tls[1]
                    )
                    task = asyncio.create_task(
                        GuestControlChannel(
                            daemon, scope_sha256=browser_allocation_digest(allocation)
                        ).run(connection)
                    )
                    if phase != "input":
                        await _assert_handback_after_worker_loss(
                            receiver=receiver,
                            worker=worker,
                            phase=phase,
                            task=task,
                            daemon=daemon,
                            allocation=allocation,
                            directory=tmp_path,
                        )
                        return
                    await asyncio.wait_for(dispatched.wait(), 10)
                    assert await page.locator("input").input_value() == "worker-loss-native-canary"
                    assert receiver.poll(0)
                    raw_intent, raw_audit = receiver.recv()
                    intent = BrowserTextInputIntent.model_validate(raw_intent)
                    expected_audit = BrowserControlPageAudit.model_validate(raw_audit)
                    worker.kill()
                    worker.join(5)
                    assert not worker.is_alive() and worker.exitcode != 0
                    assert daemon._operator_input_task is not None
                    assert not daemon._operator_input_task.done()
                    raw = SQLiteSessionStore(
                        tmp_path / "publication.sqlite",
                        ownership_clock=lambda: datetime(1970, 1, 1, 0, 0, 1, tzinfo=UTC),
                    )
                    try:
                        recovered = coordinator(raw, Policy(True))
                        _, pending = await recovered._load(intent.identity)
                        assert pending.pending_input_sequence == 1
                        assert pending.settled_input_sequence == 0
                        assert pending.acquisition_audit == expected_audit
                        assert tuple(item.origin for item in expected_audit.locations) == (
                            "https://worker-loss.test",
                        )
                        assert "private-path" not in pending.model_dump_json()
                        assert "worker-loss-native-canary" not in pending.model_dump_json()
                        with pytest.raises(BrowserControlConflict):
                            await recovered.bind_guest(
                                allocation=allocation, worker_instance_id="vw_" + "f" * 32
                            )
                        assert (await recovered._load(intent.identity))[1] == pending
                        with pytest.raises(BrowserControlConflict):
                            await recovered._admit_text_input(
                                principal=BrowserControlPrincipal(subject="operator"),
                                operator_session_id="server-session",
                                intent=intent,
                            )
                        with browser_control_checkpoint_read_scope("session"):
                            checkpoint = await raw.load_checkpoint("session")
                        with pytest.raises(BrowserControlConflict):
                            browser_model_control_admission(
                                checkpoint, allocation=allocation, operation_name="observe"
                            )
                        with pytest.raises(_GuestFailure):
                            await daemon.execute(
                                replace(
                                    _interactive_request("observe"),
                                    session_id=daemon.session_id,
                                    page_id="page",
                                    invocation_control_epoch=daemon.control.epoch,
                                )
                            )
                        release.set()
                        result = await asyncio.wait_for(
                            asyncio.gather(task, return_exceptions=True), 10
                        )
                        assert isinstance(result[0], BaseException)
                        assert daemon.control.state == "control_uncertain"
                        assert (
                            await page.locator("input").input_value() == "worker-loss-native-canary"
                        )
                        assert (await recovered._load(intent.identity))[1] == pending
                    finally:
                        await raw.close()
                finally:
                    release.set()
                    if task is not None:
                        if not task.done():
                            task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                    await browser.close()

        asyncio.run(scenario())
    finally:
        if worker.is_alive():
            worker.kill()
            worker.join(5)
        receiver.close()
