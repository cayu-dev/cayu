"""Native input ownership with real task cancellation after dispatch."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import get_args

import pytest
from tests.core.test_browser_control_guest import takeover_material
from tests.core.test_browser_session import _interactive_request

from cayu.runtime.browser_control import BrowserTextInputIntent
from cayu.tools._browser_control_guest import GuestControlChannel, GuestControlFailure
from cayu.tools._browser_guest import _GuestFailure, _InteractiveDaemon, _InteractivePage


async def input_daemon(insert_text):
    daemon = _InteractiveDaemon("bs_input")
    daemon.context = object()
    daemon.configuration_limits = _interactive_request("observe").limits
    daemon.pages["page"] = _InteractivePage(
        page=SimpleNamespace(keyboard=SimpleNamespace(insert_text=insert_text, press=insert_text)),
        session_id="bs_input",
        page_id="page",
        lifecycle="active",
        revision="observed",
    )
    daemon.active_page_id = "page"
    daemon.claim_operator_channel("channel")
    await daemon.bind_operator_control("a" * 64)
    material = takeover_material(daemon)
    await daemon.acquire_operator_control(**material)
    return daemon, material


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("key_input", [False, True])
def test_private_channel_text_delivery_has_no_payload_receipt_or_replay(cancel, key_input):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        effects = []

        async def insert_text(text):
            entered.set()
            await release.wait()
            effects.append(text)

        daemon, material = await input_daemon(insert_text)
        channel = GuestControlChannel(daemon, scope_sha256="a" * 64)
        channel._nonce = "channel"
        channel._binding = "a" * 64
        common = {
            "channel_id": "channel",
            "worker_instance": daemon.visual_worker_instance,
            "binding_sha256": "a" * 64,
            "request_id": material["request_id"],
            "epoch": 2,
        }
        await channel._command({**common, "kind": "sensitive", "sequence": 1}, None)
        message = {
            **common,
            "kind": "text_input",
            "sequence": 2,
            "input_sequence": 1,
            "page_id": "page",
            "page_epoch": 1,
            "text": "private-channel-text-canary",
        }
        if key_input:
            message.pop("text")
            message.update(kind="key_input", key="tab")
        task = asyncio.create_task(channel._command(message, None))
        await entered.wait()
        assert "text" not in message
        native = daemon._operator_input_task
        assert native is not None and not native.done()
        if cancel:
            assert task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled() and task.cancelling() == 1
            assert daemon.control.state == "control_uncertain"
            assert not native.done()
        release.set()
        await native
        if not cancel:
            result = await task
            assert result["settled_sequence"] == 1 and result["pending_sequence"] is None
            assert "private-channel-text-canary" not in repr(result)
        with pytest.raises(GuestControlFailure):
            replay = {**message, "sequence": 2 if cancel else 3}
            if not key_input:
                replay["text"] = "replacement"
            await channel._command(replay, None)
        assert effects == (["Tab"] if key_input else ["private-channel-text-canary"])

    asyncio.run(scenario())


def test_operator_keys_reject_shortcuts_before_native_dispatch():
    async def scenario():
        calls = []

        async def press(key):
            calls.append(key)

        daemon, material = await input_daemon(press)
        await daemon.enter_operator_sensitive_entry(request_id=material["request_id"], epoch=2)
        for key in ("Control+L", "Meta+L", "F12", "tab\x00", "private-key-canary", True):
            with pytest.raises(GuestControlFailure):
                await daemon.operator_key_input(
                    request_id=material["request_id"],
                    epoch=2,
                    sequence=1,
                    page_id="page",
                    page_epoch=1,
                    key=key,
                )
        assert not calls
        assert daemon.control.settled_sequence == 0
        assert daemon.control.pending_sequence is None
        assert daemon.total_operations == 0
        expected = {
            "tab": "Tab",
            "backtab": "Shift+Tab",
            "enter": "Enter",
            "escape": "Escape",
            "backspace": "Backspace",
        }
        kinds = get_args(BrowserTextInputIntent.model_fields["input_kind"].annotation)
        assert set(kinds) == {"text", *expected}
        for sequence, key in enumerate(sorted(expected), 1):
            await daemon.operator_key_input(
                request_id=material["request_id"],
                epoch=2,
                sequence=sequence,
                page_id="page",
                page_epoch=1,
                key=key,
            )
        assert calls == [expected[key] for key in sorted(expected)]

    asyncio.run(scenario())


def test_temporary_manual_login_protects_observations_after_handback():
    async def scenario():
        async def insert_text(text):
            assert text == "manual-password-canary"

        async def storage_state(*, indexed_db):
            assert indexed_db is False
            return {
                "cookies": [
                    {
                        "name": "session",
                        "value": "live-cookie-canary",
                        "domain": ".example.test",
                        "secure": False,
                    }
                ],
                "origins": [
                    {
                        "origin": "http://example.test",
                        "localStorage": [{"name": "token", "value": "storage-canary"}],
                    }
                ],
            }

        daemon, material = await input_daemon(insert_text)
        daemon.context = SimpleNamespace(storage_state=storage_state)
        assert daemon.profile_output_values is None and daemon.profile_allowed_origins is None
        await daemon.enter_operator_sensitive_entry(request_id=material["request_id"], epoch=2)
        await daemon.operator_text_input(
            request_id=material["request_id"],
            epoch=2,
            sequence=1,
            page_id="page",
            page_epoch=1,
            text="manual-password-canary",
        )
        await daemon.handback_operator_control(request_id=material["request_id"], epoch=2)
        assert daemon.control.capture_restricted and not daemon.control.sensitive_entry
        for canary in ("manual-password-canary", "live-cookie-canary", "storage-canary"):
            protected, proof = await daemon._protect_profile_observation(
                {
                    "url": "https://example.test/account",
                    "title": canary,
                    "snapshot": "account",
                    "refs": [{"role": "button", "name": "Continue"}],
                }
            )
            assert proof and protected["title"] is None and protected["refs"] == []
            assert canary not in repr(protected)
        assert daemon.profile_allowed_origins is None  # No restore/egress authority was invented.
        for operation in ("screenshot", "download", "observe_visual"):
            with pytest.raises(_GuestFailure) as error:
                await daemon._execute_locked(_interactive_request(operation))
            assert error.value.code == "policy_denied"
        pages = daemon._page_set_payload()["pages"]
        assert all(page["title"] is None and page["url"] is None for page in pages)

    asyncio.run(scenario())


def test_native_text_requires_sensitive_entry_and_exact_page_before_dispatch():
    async def scenario():
        received = []

        async def insert_text(text):
            received.append(text)

        daemon, material = await input_daemon(insert_text)
        arguments = dict(
            request_id=material["request_id"],
            epoch=2,
            sequence=1,
            page_id="page",
            page_epoch=1,
            text="private-entry",
        )
        with pytest.raises(GuestControlFailure):
            await daemon.operator_text_input(**arguments)
        daemon.control.sensitive_entry = True  # Capture owner is tested separately when wired.
        daemon.control.capture_restricted = True
        with pytest.raises(GuestControlFailure):
            await daemon.operator_text_input(**{**arguments, "page_epoch": 2})
        assert received == []
        result = await daemon.operator_text_input(**arguments)
        assert result["settled_sequence"] == 1
        assert result["pending_sequence"] is None
        assert "private-entry" not in str(result)
        assert received == ["private-entry"]
        with pytest.raises(GuestControlFailure):
            await daemon.operator_text_input(**arguments)
        assert received == ["private-entry"]

    asyncio.run(scenario())


def test_cancellation_after_dispatch_retains_native_input_and_denies_competing_input():
    async def scenario():
        dispatched = asyncio.Event()
        release = asyncio.Event()
        effects = []

        async def insert_text(text):
            dispatched.set()
            await release.wait()
            effects.append(text)

        daemon, material = await input_daemon(insert_text)
        daemon.control.sensitive_entry = True
        daemon.control.capture_restricted = True
        arguments = dict(
            request_id=material["request_id"],
            epoch=2,
            sequence=1,
            page_id="page",
            page_epoch=1,
            text="private-entry",
        )
        task = asyncio.create_task(daemon.operator_text_input(**arguments))
        await dispatched.wait()
        assert task.cancel("operator disconnected")
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelling() == 1
        assert task.cancelled()
        assert daemon.control.state == "control_uncertain"
        assert daemon.control.pending_sequence == 1
        native = daemon._operator_input_task
        assert native is not None and not native.done()
        assert native in daemon._operator_control_tasks()
        with pytest.raises(GuestControlFailure):
            await daemon.operator_text_input(**{**arguments, "sequence": 2})
        assert effects == []
        release.set()
        assert await native == ()
        assert effects == ["private-entry"]
        assert daemon.control.state == "control_uncertain"
        assert daemon.control.pending_sequence == 1  # Completion is not an acknowledged input.

    asyncio.run(scenario())
