from __future__ import annotations

import contextlib
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import pytest
from PIL import Image

import cayu.artifacts._images as image_validation


def _animated_gif_bytes(
    *,
    frame_count: int = 3,
    size: tuple[int, int] = (10, 10),
) -> bytes:
    frames = [
        Image.new("RGB", size, (index % 256, (index * 3) % 256, (index * 7) % 256))
        for index in range(frame_count)
    ]
    try:
        buffer = BytesIO()
        frames[0].save(
            buffer,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
        )
        return buffer.getvalue()
    finally:
        for frame in frames:
            frame.close()


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _oversized_png_header_bytes() -> bytes:
    ihdr = struct.pack(">IIBBBBB", 10_000, 10_000, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IEND", b"")


def test_decode_verified_image_format_decodes_every_animated_gif_frame() -> None:
    content = _animated_gif_bytes()

    assert image_validation.decode_verified_image_format(Image, content) == "GIF"


def test_decode_verified_image_format_accepts_hundreds_of_small_frames() -> None:
    content = _animated_gif_bytes(frame_count=450, size=(1, 1))

    assert image_validation.decode_verified_image_format(Image, content) == "GIF"


def test_decode_verified_image_format_rejects_truncated_later_gif_frame() -> None:
    content = _animated_gif_bytes()[:-5]

    with Image.open(BytesIO(content)) as image:
        image.verify()
    with Image.open(BytesIO(content)) as image:
        image.load()
    with pytest.raises(OSError), Image.open(BytesIO(content)) as image:
        image.seek(2)
        image.load()

    with pytest.raises(OSError):
        image_validation.decode_verified_image_format(Image, content)


def test_decode_verified_image_format_applies_decoded_limit_per_frame(monkeypatch) -> None:
    content = _animated_gif_bytes()
    monkeypatch.setattr(image_validation, "MAX_IMAGE_DECODED_BYTES", 400)

    assert image_validation.decode_verified_image_format(Image, content) == "GIF"

    monkeypatch.setattr(image_validation, "MAX_IMAGE_DECODED_BYTES", 399)

    with pytest.raises(ValueError, match=r"400 > 399 bytes"):
        image_validation.decode_verified_image_format(Image, content)


def test_decode_verified_image_format_applies_aggregate_decoded_limit(monkeypatch) -> None:
    content = _animated_gif_bytes()
    monkeypatch.setattr(
        image_validation,
        "MAX_IMAGE_TOTAL_DECODED_BYTES",
        1_200,
    )

    assert image_validation.decode_verified_image_format(Image, content) == "GIF"

    monkeypatch.setattr(
        image_validation,
        "MAX_IMAGE_TOTAL_DECODED_BYTES",
        1_199,
    )

    with pytest.raises(ValueError, match=r"1,200 > 1,199 bytes"):
        image_validation.decode_verified_image_format(Image, content)


def test_decode_verified_image_format_rejects_too_many_frames(monkeypatch) -> None:
    content = _animated_gif_bytes()
    monkeypatch.setattr(image_validation, "MAX_IMAGE_FRAMES", 2)

    with pytest.raises(ValueError, match=r"more than 2 frames"):
        image_validation.decode_verified_image_format(Image, content)


def test_decode_verified_image_format_rejects_real_decompression_bomb_header() -> None:
    content = _oversized_png_header_bytes()

    assert len(content) == 45

    with pytest.raises(Image.DecompressionBombWarning):
        image_validation.decode_verified_image_format(Image, content)


def test_decode_verified_image_format_serializes_warning_filter_contexts(monkeypatch) -> None:
    content = _animated_gif_bytes(frame_count=1)
    first_entered = threading.Event()
    second_started = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    state_lock = threading.Lock()
    active_contexts = 0
    max_active_contexts = 0
    entry_count = 0

    @contextlib.contextmanager
    def tracked_warning_context():
        nonlocal active_contexts, entry_count, max_active_contexts
        with state_lock:
            active_contexts += 1
            entry_count += 1
            current_entry = entry_count
            max_active_contexts = max(max_active_contexts, active_contexts)
        if current_entry == 1:
            first_entered.set()
            assert release_first.wait(timeout=2)
        else:
            second_entered.set()
        try:
            yield
        finally:
            with state_lock:
                active_contexts -= 1

    monkeypatch.setattr(image_validation.warnings, "catch_warnings", tracked_warning_context)
    monkeypatch.setattr(image_validation.warnings, "simplefilter", lambda *_args, **_kwargs: None)

    def second_decode() -> str | None:
        second_started.set()
        return image_validation.decode_verified_image_format(Image, content)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(image_validation.decode_verified_image_format, Image, content)
        assert first_entered.wait(timeout=2)
        second = executor.submit(second_decode)
        assert second_started.wait(timeout=2)
        try:
            assert not second_entered.wait(timeout=0.1)
        finally:
            release_first.set()
        assert first.result(timeout=2) == "GIF"
        assert second.result(timeout=2) == "GIF"

    assert second_entered.is_set()
    assert max_active_contexts == 1
