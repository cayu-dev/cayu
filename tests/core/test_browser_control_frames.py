"""Private frame framing through the real guest command encoder."""

import asyncio
import json

import pytest
from tests.core.test_browser_control import identity

from cayu.runtime._browser_control_channel import BoundBrowserGuest
from cayu.runtime._browser_control_frames import decode_browser_frame
from cayu.runtime.browser_control import (
    BrowserControlConflict,
    BrowserControlPage,
    BrowserControlRecord,
)
from cayu.tools._browser_control_guest import GuestControlChannel
from cayu.tools._browser_guest import _InteractiveDaemon


@pytest.mark.parametrize(
    "corruption",
    [None, "sequence", "page_epoch", "generation", "binding_sha256", "width", "png", "extra"],
)
def test_guest_encoded_frame_requires_exact_capture(corruption, monkeypatch):
    async def scenario():
        daemon = _InteractiveDaemon("bs_fixture")
        channel = GuestControlChannel(daemon, scope_sha256="a" * 64)
        channel._binding = "b" * 64
        daemon.control.bind(channel._binding)
        page = BrowserControlPage(page_id="page", revision="revision", control_epoch=1)
        bound = BoundBrowserGuest(
            BrowserControlRecord(
                identity=identity().model_copy(
                    update={"worker_instance_id": daemon.visual_worker_instance}
                )
            ),
            channel._nonce,
            "b" * 64,
        )
        view_id = "bv_" + "c" * 32
        pixels = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            + (16).to_bytes(4, "big") * 2
            + b"private-pixel-canary"
        )
        frames = []

        class Connection:
            async def send(self, value):
                frames.append(value)

        async def capture(*, view_id, epoch, page_id, page_epoch, send):
            assert page_id == "page" and page_epoch == 1
            await send(
                2,
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

        monkeypatch.setattr(daemon, "capture_operator_frame", capture)
        await channel._command(
            {
                "kind": "frame",
                "sequence": 1,
                "channel_id": channel._nonce,
                "worker_instance": daemon.visual_worker_instance,
                "binding_sha256": "b" * 64,
                "view_id": view_id,
                "epoch": 1,
                "page_id": "page",
                "page_epoch": 1,
            },
            Connection(),
        )
        wire = frames[0]
        if corruption is not None:
            size = int.from_bytes(wire[:4], "big")
            header = json.loads(wire[4 : 4 + size])
            body = wire[4 + size :]
            if corruption == "png":
                body = b"invalid-private-pixel-canary"
            elif corruption == "extra":
                header["title"] = "private-pixel-canary"
            elif corruption == "binding_sha256":
                header[corruption] = "c" * 64
            elif corruption == "sequence":
                header[corruption] = True
            else:
                header[corruption] += 1
            encoded = json.dumps(header).encode()
            wire = len(encoded).to_bytes(4, "big") + encoded + body
        if corruption is None:
            frame = decode_browser_frame(
                wire,
                bound=bound,
                view_id=view_id,
                sequence=1,
                expected_page=page,
                expected_generation=2,
            )
            assert frame is not None
            assert frame.png == pixels and frame.page == page
            assert "private-pixel-canary" not in repr(frame)
        else:
            with pytest.raises(BrowserControlConflict) as failure:
                decode_browser_frame(
                    wire,
                    bound=bound,
                    view_id=view_id,
                    sequence=1,
                    expected_page=page,
                    expected_generation=2,
                )
            assert "private-pixel-canary" not in str(failure.value)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "corruption",
    [
        None,
        "channel_id",
        "worker_instance",
        "binding_sha256",
        "sequence",
        "view_id",
        "control_epoch",
        "reason",
        "kind",
        "extra",
        "pixels",
    ],
)
def test_capture_denial_requires_exact_identity_and_no_pixels(corruption):
    bound = BoundBrowserGuest(BrowserControlRecord(identity=identity()), "bc_test", "b" * 64)
    header = {
        "kind": "frame_denied",
        "channel_id": bound.channel_id,
        "worker_instance": bound.record.identity.worker_instance_id,
        "binding_sha256": bound.binding_sha256,
        "sequence": 1,
        "view_id": "bv_" + "c" * 32,
        "control_epoch": bound.record.control_epoch,
        "reason": "capture_unavailable",
    }
    if corruption not in {None, "pixels"}:
        header[corruption] = True if corruption in {"sequence", "control_epoch"} else "wrong"
    encoded = json.dumps(header).encode()
    raw = len(encoded).to_bytes(4, "big") + encoded
    if corruption == "pixels":
        raw += b"private-canary"

    def decode():
        return decode_browser_frame(
            raw,
            bound=bound,
            view_id="bv_" + "c" * 32,
            sequence=1,
            expected_page=BrowserControlPage(page_id="page", revision="revision", control_epoch=1),
            expected_generation=2,
        )

    if corruption is None:
        assert decode() is None
    else:
        with pytest.raises(BrowserControlConflict):
            decode()
