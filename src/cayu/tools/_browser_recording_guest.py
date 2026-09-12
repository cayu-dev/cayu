"""Standalone, opt-in sampled recording owner for the admitted Docker guest.

Only admitted pixels cross the private recording socket. Nothing is written to
local disk, and transport loss drops samples rather than buffering without bound.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import re
import time
from typing import Any
from urllib.parse import urlsplit

try:
    from ._browser_control_transport import open_guest_control_channel, validate_control_endpoint
except ImportError:
    from _browser_control_transport import (  # ty: ignore[unresolved-import]
        open_guest_control_channel,
        validate_control_endpoint,
    )

RECORDING_SUBPROTOCOL = "cayu.browser-recording.v1"


class RecordingCaptureDenied(Exception):
    pass


def recording_configuration(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    try:
        if type(raw) is not dict or set(raw) != {"policy", "endpoint", "credential", "identity"}:
            raise ValueError
        validate_control_endpoint(raw["endpoint"])
        if (
            type(raw["credential"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", raw["credential"]) is None
        ):
            raise ValueError
        identity = raw["identity"]
        if type(identity) is not dict or set(identity) != {
            "session_id",
            "session_instance_id",
            "allocation_fingerprint",
        }:
            raise ValueError
        if any(type(v) is not str or not 1 <= len(v) <= 128 for v in identity.values()):
            raise ValueError
        if re.fullmatch(r"[0-9a-f]{64}", identity["allocation_fingerprint"]) is None:
            raise ValueError
        policy = raw["policy"]
        bounds = {
            "max_duration_seconds": (1, 3600),
            "max_bytes": (4096, 256 * 1024 * 1024),
            "max_width": (16, 1920),
            "max_height": (16, 1080),
            "frames_per_second": (1, 5),
            "retention_seconds": (60, 30 * 86400),
        }
        fixed = {
            "schema_version": 1,
            "allowed_profile_contexts": ["fresh_temporary"],
            "allow_authenticated_pages": False,
            "capture_during_sensitive_entry": False,
            "capture": "active_page_viewport",
        }
        if type(policy) is not dict or set(policy) != set(bounds) | set(fixed) | {
            "scope",
            "allowed_origins",
        }:
            raise ValueError
        if any(type(policy[k]) is not type(v) or policy[k] != v for k, v in fixed.items()):
            raise ValueError
        if any(
            type(policy[k]) is not int or not lo <= policy[k] <= hi
            for k, (lo, hi) in bounds.items()
        ):
            raise ValueError
        if (
            type(policy["scope"]) is not str
            or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", policy["scope"]) is None
        ):
            raise ValueError
        origins = policy["allowed_origins"]
        if type(origins) is not list or not 1 <= len(origins) <= 64:
            raise ValueError
        for origin in origins:
            parsed = urlsplit(origin)
            if (
                type(origin) is not str
                or len(origin) > 2048
                or parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.port not in {None, 443}
                or parsed.path
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError
        return raw
    except Exception:
        raise RecordingCaptureDenied() from None


async def capture_recording_frame(daemon: Any, policy: dict[str, Any]) -> tuple[str, bytes]:
    """The caller owns the daemon lock; admission and post-capture checks precede encoding."""
    state = daemon.pages.get(daemon.active_page_id or "")
    if (
        daemon.closing
        or state is None
        or state.lifecycle != "active"
        or not state.configured
        or state.limit_exceeded
        or state.denied_code is not None
        or state.access_evidence is not None
        or daemon.profile_output_values is not None
        or daemon.control.capture_restricted
        or daemon.control.sensitive_entry
        or state.cdp is None
    ):
        raise RecordingCaptureDenied()
    viewport = state.page.viewport_size
    if (
        type(viewport) is not dict
        or type(viewport.get("width")) is not int
        or type(viewport.get("height")) is not int
        or not 1 <= viewport["width"] <= policy["max_width"]
        or not 1 <= viewport["height"] <= policy["max_height"]
    ):
        raise RecordingCaptureDenied()
    epoch = state.navigation_epoch
    cdp = state.cdp
    # Playwright's storage snapshot awaits page promises and cannot settle
    # under a debugger pause. Reject existing state before pausing; native
    # cookie and isolated-world storage checks fence the captured document
    # again on both sides of the screenshot.
    storage = await daemon.context.storage_state()
    if storage.get("cookies") or storage.get("origins"):
        raise RecordingCaptureDenied()

    async def document() -> tuple[str, str, str]:
        tree = (await cdp.send("Page.getFrameTree"))["frameTree"]
        frame = tree["frame"]
        parsed = urlsplit(frame["url"])
        origin = (
            f"https://{parsed.hostname}"
            if parsed.scheme == "https" and parsed.port in {None, 443}
            else None
        )
        if (
            tree.get("childFrames")
            or origin not in policy["allowed_origins"]
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise RecordingCaptureDenied()
        # Any profile state may authenticate a page; do not infer which cookies
        # or storage entries are harmless. No state values leave this process.
        if await daemon.context.cookies():
            raise RecordingCaptureDenied()
        world = await cdp.send(
            "Page.createIsolatedWorld",
            {
                "frameId": frame["id"],
                "worldName": "cayu-recording-admission-v1",
                "grantUniveralAccess": False,
            },
        )
        storage = await cdp.send(
            "Runtime.evaluate",
            {
                "contextId": world["executionContextId"],
                # Script suspension does not suspend HTML parsing. Do not
                # admit a document whose parser/resources can still introduce
                # sensitive controls while pixels are being captured.
                "expression": (
                    'document.readyState === "complete"'
                    " && localStorage.length === 0 && sessionStorage.length === 0"
                ),
                "returnByValue": True,
            },
        )
        if storage.get("exceptionDetails") or storage.get("result", {}).get("value") is not True:
            raise RecordingCaptureDenied()
        return frame["id"], frame["loaderId"], frame["url"]

    async def inspect_dom() -> None:
        # Inspect native DOM, including closed shadow roots. Refuse opaque
        # surfaces and sensitive controls instead of trusting page JavaScript.
        tree = await cdp.send("DOM.getDocument", {"depth": -1, "pierce": True})
        pending = [tree["root"]]
        count = 0
        while pending:
            node = pending.pop()
            count += 1
            if count > 4096 or node.get("contentDocument"):
                raise RecordingCaptureDenied()
            name = node.get("nodeName", "").lower()
            attrs = node.get("attributes", [])
            if len(attrs) > 256 or len(attrs) % 2:
                raise RecordingCaptureDenied()
            attributes = dict(zip(attrs[::2], attrs[1::2], strict=True))
            if (
                name in {"iframe", "frame", "object", "embed"}
                or (
                    name == "input"
                    and attributes.get("type", "text").lower() in {"password", "file"}
                )
                or any(
                    word in attributes.get("autocomplete", "").lower()
                    for word in ("password", "one-time-code", "cc-")
                )
            ):
                raise RecordingCaptureDenied()
            # Chromium supplies native shadow trees for ordinary form controls.
            # They are browser-owned, unlike page-authored open/closed roots.
            # Inspect their descendants with the same bounds and sensitive-field
            # checks; never treat a page-authored or unknown root as transparent.
            for root in node.get("shadowRoots", []):
                if root.get("shadowRootType") != "user-agent":
                    raise RecordingCaptureDenied()
                pending.append(root)
            pending.extend(node.get("children", []))

    debugger_enabled = False
    pause_task: asyncio.Task[Any] | None = None
    paused = asyncio.Event()

    def on_paused(_event: Any) -> None:
        paused.set()

    def on_resumed(_event: Any) -> None:
        paused.clear()

    cdp.on("Debugger.paused", on_paused)
    cdp.on("Debugger.resumed", on_resumed)
    animations_frozen = False
    try:
        frame = (await cdp.send("Page.getFrameTree"))["frameTree"]["frame"]
        world = await cdp.send(
            "Page.createIsolatedWorld",
            {"frameId": frame["id"], "worldName": "cayu-recording-pause-v1"},
        )
        # ScriptExecutionDisabled drops pending callbacks. A debugger pause
        # instead retains application work until this capture releases it.
        debugger_enabled = True
        await cdp.send("Debugger.enable", {"maxScriptsCacheSize": 0})
        # Pause an owned isolated-world statement even on an idle page. The
        # evaluation completes only after disable resumes it in finally.
        pause_task = asyncio.create_task(
            cdp.send(
                "Runtime.evaluate",
                {"contextId": world["executionContextId"], "expression": "debugger;"},
            )
        )
        await asyncio.wait_for(paused.wait(), timeout=1)
        animations_frozen = True
        await cdp.send("Animation.setPlaybackRate", {"playbackRate": 0})
        before = await document()
        await inspect_dom()
        if not paused.is_set():
            raise RecordingCaptureDenied()
        # Playwright's screenshot helper waits on page animation callbacks;
        # capture the viewport directly while application execution is paused.
        async with asyncio.timeout(2):
            screenshot = await cdp.send(
                "Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False}
            )
        encoded = screenshot.get("data")
        if type(encoded) is not str or len(encoded) > 3 * 1024 * 1024:
            raise RecordingCaptureDenied()
        png = base64.b64decode(encoded, validate=True)
        if (
            type(png) is not bytes
            or not 24 <= len(png) <= 2 * 1024 * 1024
            or png[:8] != b"\x89PNG\r\n\x1a\n"
            or int.from_bytes(png[16:20], "big") != viewport["width"]
            or int.from_bytes(png[20:24], "big") != viewport["height"]
        ):
            raise RecordingCaptureDenied()
        await inspect_dom()
        if (
            not paused.is_set()
            or await document() != before
            or state.navigation_epoch != epoch
            or daemon.active_page_id != state.page_id
            or daemon.closing
            or daemon.profile_output_values is not None
            or daemon.control.capture_restricted
            or daemon.control.sensitive_entry
            or state.lifecycle != "active"
        ):
            raise RecordingCaptureDenied()
        return state.page_id, png
    finally:
        # Restoration belongs to this capture, not a model action. A failure
        # cannot silently leave a page frozen while business effects proceed.
        try:
            if animations_frozen:
                await cdp.send("Animation.setPlaybackRate", {"playbackRate": 1})
            if debugger_enabled:
                # Disabling this owned debugger resumes any pending pause, even
                # if cancellation arrived before its acknowledgement/event.
                await cdp.send("Debugger.disable")
            if pause_task is not None:
                await asyncio.wait_for(pause_task, timeout=1)
        except BaseException:
            daemon.closing = True
            daemon.close_requested.set()
            raise
        finally:
            if pause_task is not None and not pause_task.done():
                pause_task.cancel()
                await asyncio.gather(pause_task, return_exceptions=True)
            cdp.remove_listener("Debugger.paused", on_paused)
            cdp.remove_listener("Debugger.resumed", on_resumed)


class GuestRecording:
    def __init__(self, daemon: Any, configuration: dict[str, Any]) -> None:
        self.daemon = daemon
        self.configuration = configuration
        self.policy = configuration["policy"]
        self.started = time.monotonic()
        self.started_at_ms = int(time.time() * 1000)
        self.sequence = 0
        self.gap = False
        self.suspended = False
        self.connection: Any = None
        self.stop = asyncio.Event()
        self.reason = "normal_close"
        self.task = asyncio.create_task(self.run(), name="cayu-private-browser-recording")

    async def _connect(self) -> None:
        self.connection = await open_guest_control_channel(
            endpoint=self.configuration["endpoint"],
            credential=self.configuration["credential"],
            subprotocol=RECORDING_SUBPROTOCOL,
        )
        identity = {
            **self.configuration["identity"],
            "browser_id": self.daemon.session_id,
            "worker_instance": self.daemon.visual_worker_instance,
        }
        await self.connection.send(
            json.dumps({"identity": identity, "started_at_ms": self.started_at_ms})
        )
        response = json.loads(await self.connection.recv())
        if response.get("ready") is not True:
            raise RecordingCaptureDenied()

    async def run(self) -> None:
        interval = 1 / self.policy["frames_per_second"]
        try:
            while not self.stop.is_set():
                elapsed = time.monotonic() - self.started
                if elapsed >= self.policy["max_duration_seconds"]:
                    self.reason = "limit_exhausted"
                    break
                sequence = int(elapsed / interval)
                self.gap |= sequence != self.sequence
                self.sequence = sequence + 1
                try:
                    async with asyncio.timeout(5):
                        if self.connection is None:
                            await self._connect()
                        if self.suspended or self.daemon.lock.locked():
                            raise RecordingCaptureDenied()
                        async with self.daemon.lock:
                            if self.suspended:
                                raise RecordingCaptureDenied()
                            page_id, png = await capture_recording_frame(self.daemon, self.policy)
                        header = json.dumps(
                            {
                                "sequence": sequence,
                                "page_id": page_id,
                                "elapsed_ms": int(elapsed * 1000),
                            }
                        ).encode()
                        assert self.connection is not None
                        await self.connection.send(len(header).to_bytes(4, "big") + header + png)
                        png = b""
                        response = json.loads(await self.connection.recv())
                        if response.get("accepted") != sequence:
                            self.reason = "limit_exhausted"
                            break
                except RecordingCaptureDenied:
                    self.gap = True
                    if self.connection is not None:
                        try:
                            async with asyncio.timeout(2):
                                await self.connection.send(
                                    json.dumps({"heartbeat": int(elapsed * 1000)})
                                )
                                await self.connection.recv()
                        except Exception:
                            await self.connection.close()
                            self.connection = None
                except Exception:
                    self.gap = True
                    if self.connection is not None:
                        await self.connection.close()
                        self.connection = None
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        self.stop.wait(),
                        timeout=max(
                            0.001, (sequence + 1) * interval - (time.monotonic() - self.started)
                        ),
                    )
        finally:
            if self.connection is not None:
                try:
                    async with asyncio.timeout(2):
                        await self.connection.send(
                            json.dumps(
                                {
                                    "finish": self.reason,
                                    "gap": self.gap,
                                    "elapsed_ms": min(
                                        3600000, int((time.monotonic() - self.started) * 1000)
                                    ),
                                }
                            )
                        )
                        await self.connection.recv()
                except Exception:
                    pass
                await self.connection.close()
                self.connection = None

    async def close(self, *, normal: bool) -> None:
        self.suspended = True
        if not normal:
            self.reason = "worker_lost"
        self.stop.set()
        try:
            async with asyncio.timeout(3):
                await asyncio.shield(self.task)
        except TimeoutError:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
