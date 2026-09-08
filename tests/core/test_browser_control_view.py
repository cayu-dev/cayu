"""Policy-to-guest view exchange; pixels never enter durable publication."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_control_coordinator import coordinator, intent_for
from tests.core.test_browser_control_publisher import publication_fixture

from cayu.runtime._browser_control_authorization import BrowserControlPermissionDenied
from cayu.runtime._browser_control_channel import BoundBrowserGuest, BrowserGuestCommandOwner
from cayu.runtime._browser_control_frames import BrowserViewUnavailable
from cayu.runtime._browser_control_publisher import BrowserControlPublisher
from cayu.runtime.browser_control import (
    BrowserControlConflict,
    BrowserControlPage,
    BrowserControlPrincipal,
    BrowserHandbackIntent,
)
from cayu.tools._browser_control_guest import (
    GuestControlChannel,
    GuestControlFence,
)
from cayu.tools._browser_guest import _InteractiveDaemon, _InteractivePage


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "allowed,cancel_viewer",
    [
        (False, False),
        (True, False),
        (True, True),
        (True, "suspend"),
        (True, "profile"),
        (True, "background"),
        (True, "stale_page"),
        (True, "unpublished_observation"),
        (True, "takeover"),
    ],
)
def test_view_policy_precedes_guest_dispatch_and_pixels_stay_private(
    tmp_path, monkeypatch, backend, allowed, cancel_viewer
):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, bootstrap):
            record = await BrowserControlPublisher(store).publish(bootstrap)
            if cancel_viewer == "unpublished_observation":
                setup = coordinator(store, Policy(True))
                principal = BrowserControlPrincipal(subject="operator")
                pending = await setup.request_takeover(
                    principal=principal,
                    operator_session_id="operator-session",
                    intent=intent_for(bootstrap),
                )
                acquired = await setup._publish_guest_acquisition(
                    expected=pending, lease_until_ms=2000
                )
                returning = await setup.request_handback(
                    principal=principal,
                    operator_session_id="operator-session",
                    intent=BrowserHandbackIntent(
                        identity=acquired.identity,
                        expected_record_revision=acquired.revision,
                        expected_control_epoch=acquired.control_epoch,
                        request_id=acquired.request.request_id,
                    ),
                )
                record = await setup._publish_guest_handback(expected=returning)
                assert record.fresh_observation_required
                assert await setup.drain()
            before = await store.load_checkpoint(record.identity.session_id)
            daemon = _InteractiveDaemon(record.identity.browser_session_id)
            daemon.context = object()
            daemon.visual_worker_instance = record.identity.worker_instance_id
            daemon.control = GuestControlFence(
                worker_instance=daemon.visual_worker_instance, wall_clock=lambda: 1.0
            )
            channel = GuestControlChannel(daemon, scope_sha256="a" * 64)
            channel._binding = "b" * 64
            daemon.claim_operator_channel(channel._nonce)
            await daemon.bind_operator_control("b" * 64)
            if cancel_viewer == "unpublished_observation":
                daemon.control.epoch = record.control_epoch
            pixels = (
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
                + (16).to_bytes(4, "big") * 2
                + b"private-view-canary"
            )
            calls = []
            queue = asyncio.Queue()
            capturing = asyncio.Event()
            release_capture = asyncio.Event()

            class Sink:
                async def send(self, value):
                    await queue.put(value)

            class Connection:
                async def send(self, raw):
                    command = json.loads(raw)
                    calls.append(command["kind"])
                    result = await channel._command(command, Sink())
                    await queue.put(
                        json.dumps(
                            {
                                "kind": "settled",
                                "channel_id": channel._nonce,
                                "sequence": channel._sequence,
                                **result,
                            }
                        )
                    )

                async def recv(self):
                    return await queue.get()

                async def recv_frame(self):
                    return await queue.get()

            async def capture(*, view_id, epoch, page_id, page_epoch, send):
                assert page_id == "page" and page_epoch == 1
                daemon.control.check_view(view_id=view_id, epoch=epoch)
                capturing.set()
                if cancel_viewer and cancel_viewer != "unpublished_observation":
                    await release_capture.wait()
                await send(
                    daemon.operator_frames.generation,
                    {
                        "page_id": "page",
                        "page_epoch": 1,
                        "control_epoch": epoch,
                        "width": 16,
                        "height": 16,
                        "content_type": "image/png",
                    },
                    pixels,
                )

            screenshots = []
            if cancel_viewer in {"profile", "background", "stale_page"}:

                async def screenshot(**kwargs):
                    screenshots.append(kwargs)
                    return pixels

                # This is the private state installed by profile restoration.
                # View permission alone must not bypass its capture restriction.
                if cancel_viewer == "profile":
                    daemon.profile_output_values = ("profile-credential-canary",)
                daemon.pages["page"] = _InteractivePage(
                    page=SimpleNamespace(
                        viewport_size={"width": 16, "height": 16}, screenshot=screenshot
                    ),
                    session_id=daemon.session_id,
                    page_id="page",
                    lifecycle="active",
                    revision="revision",
                    configured=True,
                )
                daemon.active_page_id = "page"
                if cancel_viewer == "background":
                    daemon.pages["background"] = _InteractivePage(
                        page=SimpleNamespace(screenshot=screenshot),
                        session_id=daemon.session_id,
                        page_id="background",
                        lifecycle="active",
                        revision="revision",
                        configured=True,
                    )
            else:
                monkeypatch.setattr(daemon, "capture_operator_frame", capture)
            owner = BrowserGuestCommandOwner(
                coordinator=coordinator(store, Policy(allowed)),
                connection=Connection(),
                bound=BoundBrowserGuest(record, channel._nonce, "b" * 64),
            )

            async def request_view():
                return await owner.request_view(
                    principal=BrowserControlPrincipal(subject="operator"),
                    operator_session_id="operator-session",
                    page=BrowserControlPage(
                        page_id="background" if cancel_viewer == "background" else "page",
                        revision="revision",
                        control_epoch=2 if cancel_viewer == "stale_page" else 1,
                    ),
                    until_ms=4000,
                )

            operation = asyncio.create_task(request_view())
            await asyncio.sleep(0)
            with pytest.raises(BrowserControlConflict):
                await request_view()
            drive = asyncio.create_task(owner.step())
            if cancel_viewer in {"profile", "background", "stale_page"}:
                await drive
                with pytest.raises(BrowserViewUnavailable):
                    await operation
                assert calls == ["view", "frame"] and screenshots == []
                assert not owner._closed
                await owner.step()  # A normal heartbeat still succeeds.
                assert calls == ["view", "frame", "status"]
                if cancel_viewer in {"background", "stale_page"}:
                    active_view = asyncio.create_task(
                        owner.request_view(
                            principal=BrowserControlPrincipal(subject="operator"),
                            operator_session_id="operator-session",
                            page=BrowserControlPage(
                                page_id="page", revision="revision", control_epoch=1
                            ),
                            until_ms=4000,
                        )
                    )
                    await asyncio.sleep(0)
                    await owner.step()
                    assert (await active_view).png == pixels
                    assert len(screenshots) == 1
                assert await store.load_checkpoint(record.identity.session_id) == before
                assert await owner.coordinator.drain()
                return
            if cancel_viewer == "takeover":
                await capturing.wait()
                pending = await owner.coordinator.request_takeover(
                    principal=BrowserControlPrincipal(subject="operator"),
                    operator_session_id="operator-session",
                    intent=intent_for(bootstrap),
                )
                release_capture.set()
                await drive
                with pytest.raises(BrowserViewUnavailable):
                    await operation
                assert not owner._closed and owner._view is None
                assert (await owner.coordinator._load(record.identity))[1] == pending
                assert await owner.coordinator.drain()
                return
            if cancel_viewer and cancel_viewer != "unpublished_observation":
                await capturing.wait()
                suspension = None
                if cancel_viewer == "suspend":
                    suspension = asyncio.create_task(owner.suspend_views())
                    with pytest.raises(BrowserControlConflict):
                        await operation
                    assert not operation.cancelled() and operation.cancelling() == 0
                    assert not suspension.done()
                else:
                    assert operation.cancel("viewer disconnected")
                    with pytest.raises(asyncio.CancelledError):
                        await operation
                    assert operation.cancelled() and operation.cancelling() == 1
                assert not drive.done() and drive.cancelling() == 0
                with pytest.raises(BrowserControlConflict):
                    await request_view()
                release_capture.set()
                await drive
                if suspension is not None:
                    await suspension
                    with pytest.raises(BrowserControlConflict):
                        await request_view()
                assert owner._view is None
                assert calls == ["view", "frame"]
            elif allowed:
                frame = await operation
                assert frame.png == pixels and calls == ["view", "frame"]
            else:
                with pytest.raises(BrowserControlPermissionDenied):
                    await operation
                assert calls == []
            await drive
            assert await store.load_checkpoint(record.identity.session_id) == before
            assert await owner.coordinator.drain()

    asyncio.run(scenario())
