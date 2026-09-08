"""Capture/send settling before native sensitive input (no actual pixels yet)."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_browser_operator_input import input_daemon

from cayu.tools._browser_control_guest import (
    GuestControlChannel,
    GuestControlFailure,
    GuestFrameOwner,
)


@pytest.mark.parametrize("phase", ["capture", "send"])
def test_sensitive_entry_waits_for_actual_capture_or_send(phase):
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        sent = []
        inputs = []

        async def capture():
            if phase == "capture":
                entered.set()
                await release.wait()
            return b"private-frame"

        async def send(generation, frame):
            if phase == "send":
                entered.set()
                await release.wait()
            sent.append((generation, frame))

        async def insert_text(text):
            inputs.append(text)

        daemon, material = await input_daemon(insert_text)
        daemon.operator_frames.resume()
        capture_task = asyncio.create_task(
            daemon.operator_frames.capture_one(
                capture=capture,
                send=send,
                guard=lambda: None,
            )
        )
        await entered.wait()
        channel = GuestControlChannel(daemon, scope_sha256="a" * 64)
        channel._nonce = "channel"
        channel._binding = "a" * 64
        entry = asyncio.create_task(
            channel._command(
                {
                    "kind": "sensitive",
                    "channel_id": "channel",
                    "worker_instance": daemon.visual_worker_instance,
                    "binding_sha256": "a" * 64,
                    "sequence": 1,
                    "request_id": material["request_id"],
                    "epoch": 2,
                },
                object(),
            )
        )
        await asyncio.sleep(0)
        assert not entry.done()
        assert not daemon.control.sensitive_entry
        assert daemon.operator_frames.paused
        assert daemon.operator_frames.task in daemon._operator_control_tasks()
        release.set()
        await capture_task
        acknowledgement = await entry
        assert acknowledgement["sensitive_entry"] is True
        assert acknowledgement["capture_restricted"] is True
        assert daemon.control.sensitive_entry
        assert daemon.operator_frames.task is None
        assert len(sent) == (0 if phase == "capture" else 1)
        await daemon.operator_text_input(
            request_id=material["request_id"],
            epoch=2,
            sequence=1,
            page_id="page",
            page_epoch=1,
            text="credential-input",
        )
        assert inputs == ["credential-input"]
        with pytest.raises(GuestControlFailure):
            await daemon.operator_frames.capture_one(capture=capture, send=send, guard=lambda: None)

    asyncio.run(scenario())


def test_capture_cancellation_does_not_cancel_or_forget_inflight_capture():
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        sent = []

        async def capture():
            entered.set()
            await release.wait()
            return b"private-frame"

        async def send(generation, frame):
            sent.append(frame)

        owner = GuestFrameOwner()
        owner.resume()
        task = asyncio.create_task(
            owner.capture_one(capture=capture, send=send, guard=lambda: None)
        )
        await entered.wait()
        assert task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled() and task.cancelling() == 1
        assert owner.paused
        assert owner.task is not None and not owner.task.done()
        with pytest.raises(GuestControlFailure):
            await owner.pause(timeout_s=0.01)
        release.set()
        await owner.pause()
        assert owner.task is None
        assert sent == []

    asyncio.run(scenario())
