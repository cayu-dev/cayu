"""Shared object-retention assertions for provider exception boundaries."""

from __future__ import annotations

import traceback
from pathlib import Path

import cayu

_CAYU_PACKAGE_ROOT = str(Path(cayu.__file__).resolve().parent).replace("\\", "/")


def is_cayu_source_filename(filename: str) -> bool:
    """Recognize the active installation and source-checkout frames on any OS."""

    normalized = filename.replace("\\", "/")
    if normalized.startswith(_CAYU_PACKAGE_ROOT + "/"):
        return True
    parts = tuple(part for part in normalized.split("/") if part)
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
