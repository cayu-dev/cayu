"""Shared object-retention assertions for provider exception boundaries."""

from __future__ import annotations

import traceback


def is_cayu_source_filename(filename: str) -> bool:
    """Return whether a traceback filename is below ``src/cayu`` on any OS."""

    parts = tuple(part for part in filename.replace("\\", "/").split("/") if part)
    return any(parts[index : index + 2] == ("src", "cayu") for index in range(len(parts) - 1))


def assert_cayu_traceback_does_not_retain(
    exc: BaseException,
    retained_object: object,
) -> None:
    """Assert that Cayu traceback frame locals do not retain an object by identity."""

    retained_frames = [
        frame.f_code.co_name
        for frame, _line_number in traceback.walk_tb(exc.__traceback__)
        if is_cayu_source_filename(frame.f_code.co_filename)
        and any(value is retained_object for value in frame.f_locals.values())
    ]
    assert retained_frames == []


__all__ = ["assert_cayu_traceback_does_not_retain", "is_cayu_source_filename"]
