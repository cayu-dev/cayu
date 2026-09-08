"""A real takeover during read authorization must not destroy the command owner."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_control_coordinator import coordinator, intent_for
from tests.core.test_browser_control_publisher import publication_fixture

from cayu.runtime._browser_control_authorization import BrowserControlRevisionChanged
from cayu.runtime._browser_control_channel import BoundBrowserGuest, BrowserGuestCommandOwner
from cayu.runtime._browser_control_frames import BrowserViewUnavailable
from cayu.runtime._browser_control_publisher import BrowserControlPublisher
from cayu.runtime.browser_control import (
    BrowserControlConflict,
    BrowserControlPage,
    BrowserControlPrincipal,
)
from cayu.tools._browser_control_guest import GuestControlChannel, GuestControlFence
from cayu.tools._browser_guest import _InteractiveDaemon, _InteractivePage


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("entrance", ["view", "pages"])
@pytest.mark.parametrize("read_number", [1, 2])
@pytest.mark.parametrize("outcome", ["takeover", "missing", "store_failure"])
def test_read_authorization_race_preserves_only_positively_known_revision_changes(
    tmp_path, monkeypatch, backend, entrance, read_number, outcome
):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, bootstrap):
            record = await BrowserControlPublisher(store).publish(bootstrap)
            control = coordinator(store, Policy(True))
            calls = []

            class Connection:
                async def send(self, raw):
                    calls.append(raw)
                    raise AssertionError("Rejected read must not reach the guest.")

            owner = BrowserGuestCommandOwner(
                coordinator=control,
                connection=Connection(),
                bound=BoundBrowserGuest(record, "bc_test", "b" * 64),
            )
            entered, release = asyncio.Event(), asyncio.Event()
            original_load = control._load
            drive = None
            reads = 0
            failure = (
                BrowserControlConflict("The expected live browser allocation is unavailable.")
                if outcome == "missing"
                else RuntimeError("store unavailable")
            )

            async def blocked_read(identity):
                nonlocal reads
                if asyncio.current_task() is drive:
                    reads += 1
                    if reads == read_number:
                        entered.set()
                        await release.wait()
                        if outcome != "takeover":
                            raise failure
                return await original_load(identity)

            monkeypatch.setattr(control, "_load", blocked_read)
            principal = BrowserControlPrincipal(subject="operator")
            if entrance == "view":
                request = owner.request_view(
                    principal=principal,
                    operator_session_id="continuity",
                    page=BrowserControlPage(page_id="page", revision="revision", control_epoch=1),
                    until_ms=4000,
                )
            else:
                request = owner.request_pages(principal=principal, operator_session_id="continuity")
            caller = asyncio.create_task(request)
            await asyncio.sleep(0)
            drive = asyncio.create_task(owner.step())
            async with asyncio.timeout(2):
                await entered.wait()
            expected = record
            if outcome == "takeover":
                expected = await control.request_takeover(
                    principal=principal,
                    operator_session_id="continuity",
                    intent=intent_for(bootstrap),
                )
            release.set()
            if outcome == "takeover":
                await drive
                with pytest.raises((BrowserControlRevisionChanged, BrowserViewUnavailable)):
                    await caller
                assert not owner._closed
            else:
                with pytest.raises(type(failure)) as raised:
                    await drive
                assert raised.value is failure
                with pytest.raises(BrowserControlConflict):
                    await caller
            assert owner._pages is None and owner._view is None
            assert calls == [] and owner.sequence == 0
            assert (await original_load(record.identity))[1] == expected
            assert await control.drain()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("malformed_reply", [False, True])
def test_takeover_during_settled_page_discovery_keeps_channel_usable(
    tmp_path, backend, malformed_reply
):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, bootstrap):
            record = await BrowserControlPublisher(store).publish(bootstrap)
            control = coordinator(store, Policy(True))
            daemon = _InteractiveDaemon(record.identity.browser_session_id)
            daemon.context = object()
            daemon.visual_worker_instance = record.identity.worker_instance_id
            daemon.control = GuestControlFence(
                worker_instance=daemon.visual_worker_instance, wall_clock=lambda: 1.0
            )
            daemon.pages["page"] = _InteractivePage(
                page=SimpleNamespace(url="https://site.test/"),
                session_id=daemon.session_id,
                page_id="page",
                lifecycle="active",
                revision="revision",
                configured=True,
            )
            daemon.active_page_id = "page"
            channel = GuestControlChannel(daemon, scope_sha256="a" * 64)
            channel._binding = "b" * 64
            daemon.claim_operator_channel(channel._nonce)
            await daemon.bind_operator_control("b" * 64)
            dispatched, release = asyncio.Event(), asyncio.Event()
            replies = asyncio.Queue()
            calls = []

            class Connection:
                async def send(self, raw):
                    command = json.loads(raw)
                    calls.append(command["kind"])
                    result = await channel._command(command, self)
                    reply = {
                        "kind": "settled",
                        "channel_id": channel._nonce,
                        "sequence": channel._sequence,
                        **result,
                    }
                    if command["kind"] == "pages":
                        dispatched.set()
                        await release.wait()
                        if malformed_reply:
                            reply["sequence"] = True
                    await replies.put(json.dumps(reply))

                async def recv(self):
                    return await replies.get()

            owner = BrowserGuestCommandOwner(
                coordinator=control,
                connection=Connection(),
                bound=BoundBrowserGuest(record, channel._nonce, "b" * 64),
            )
            principal = BrowserControlPrincipal(subject="operator")
            caller = asyncio.create_task(
                owner.request_pages(principal=principal, operator_session_id="continuity")
            )
            await asyncio.sleep(0)
            drive = asyncio.create_task(owner.step())
            async with asyncio.timeout(2):
                await dispatched.wait()
            pending = await control.request_takeover(
                principal=principal, operator_session_id="continuity", intent=intent_for(bootstrap)
            )
            release.set()
            if malformed_reply:
                with pytest.raises(BrowserControlConflict) as raised:
                    await drive
                assert not isinstance(raised.value, BrowserControlRevisionChanged)
                with pytest.raises(BrowserControlConflict):
                    await caller
                assert calls == ["pages"]
            else:
                await drive
                with pytest.raises(BrowserControlRevisionChanged):
                    await caller
                assert (await control._load(record.identity))[1] == pending
                assert not owner._closed
                # Continue through real guest acquisition, not just a flag check.
                await owner.step()
                assert owner.bound.record.state == "operator_controlled"
                assert daemon.control.state == "operator_controlled"
                assert calls == ["pages", "takeover"]
            assert await control.drain()

    asyncio.run(scenario())
