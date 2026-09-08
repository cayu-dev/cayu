"""Bounded transient guest frames. Never serialize these as tool/event payloads."""

from __future__ import annotations

import json
from dataclasses import dataclass

from cayu.runtime._browser_control_channel import BoundBrowserGuest
from cayu.runtime.browser_control import BrowserControlConflict, BrowserControlPage

MAX_BROWSER_FRAME_BYTES = 2 * 1024 * 1024
MAX_BROWSER_FRAME_HEADER_BYTES = 4096
MAX_BROWSER_FRAME_MESSAGE_BYTES = 4 + MAX_BROWSER_FRAME_HEADER_BYTES + MAX_BROWSER_FRAME_BYTES


class BrowserViewUnavailable(BrowserControlConflict):
    """A denied or invalidated view whose guest exchange is positively settled."""


@dataclass(frozen=True, slots=True, repr=False)
class PrivateBrowserFrame:
    """One transport-owned frame; no URL, title, credential or replay identity."""

    page: BrowserControlPage
    generation: int
    width: int
    height: int
    png: bytes


def decode_browser_frame(
    raw: bytes,
    *,
    bound: BoundBrowserGuest,
    view_id: str,
    sequence: int,
    expected_page: BrowserControlPage,
    expected_generation: int,
) -> PrivateBrowserFrame | None:
    """Authenticate framing against the current serialized capture command.

    Transport allocation must enforce MAX_BROWSER_FRAME_MESSAGE_BYTES as well;
    checking it here cannot undo an oversized allocation by an upstream server.
    """
    failure = False
    result = None
    try:
        expected_page = BrowserControlPage.model_validate(expected_page)
        if type(raw) is not bytes or not 4 < len(raw) <= MAX_BROWSER_FRAME_MESSAGE_BYTES:
            raise ValueError
        size = int.from_bytes(raw[:4], "big")
        if not 1 <= size <= MAX_BROWSER_FRAME_HEADER_BYTES or len(raw) < 4 + size:
            raise ValueError

        def unique(pairs):
            values = {}
            for key, value in pairs:
                if key in values:
                    raise ValueError
                values[key] = value
            return values

        header = json.loads(raw[4 : 4 + size], object_pairs_hook=unique)
        denied = {
            "kind": "frame_denied",
            "channel_id": bound.channel_id,
            "worker_instance": bound.record.identity.worker_instance_id,
            "binding_sha256": bound.binding_sha256,
            "sequence": sequence,
            "view_id": view_id,
            "control_epoch": bound.record.control_epoch,
            "reason": "capture_unavailable",
        }
        if (
            type(header) is dict
            and set(header) == set(denied)
            and all(
                type(header[key]) is type(value) and header[key] == value
                for key, value in denied.items()
            )
            and len(raw) == 4 + size
        ):
            return None
        fixed = {
            "kind": "frame",
            "channel_id": bound.channel_id,
            "worker_instance": bound.record.identity.worker_instance_id,
            "binding_sha256": bound.binding_sha256,
            "sequence": sequence,
            "view_id": view_id,
            "generation": expected_generation,
            "page_id": expected_page.page_id,
            "page_epoch": expected_page.control_epoch,
            "control_epoch": bound.record.control_epoch,
            "content_type": "image/png",
        }
        if type(header) is not dict or set(header) != set(fixed) | {"width", "height"}:
            raise ValueError
        if any(
            type(header[key]) is not type(value) or header[key] != value
            for key, value in fixed.items()
        ):
            raise ValueError
        width, height = header["width"], header["height"]
        if (
            type(width) is not int
            or type(height) is not int
            or not 1 <= width <= 1920
            or not 1 <= height <= 1080
        ):
            raise ValueError
        png = raw[4 + size :]
        if (
            not 24 <= len(png) <= MAX_BROWSER_FRAME_BYTES
            or png[:8] != b"\x89PNG\r\n\x1a\n"
            or png[8:16] != b"\x00\x00\x00\rIHDR"
            or int.from_bytes(png[16:20], "big") != width
            or int.from_bytes(png[20:24], "big") != height
        ):
            raise ValueError
        result = PrivateBrowserFrame(expected_page, expected_generation, width, height, png)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        failure = True
    finally:
        # Fixed diagnostics and no raw frame retained by this boundary's traceback.
        raw = b""
        header = None
        png = b""
    if failure or result is None:
        raise BrowserControlConflict("Browser frame differs from its authorized capture.")
    return result
