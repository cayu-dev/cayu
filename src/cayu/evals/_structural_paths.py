"""Canonical cross-platform paths for portable structural eval assertions."""

from __future__ import annotations

import re
import unicodedata

from cayu.workspaces.base import _validate_workspace_relative_path

_WINDOWS_FORBIDDEN_PATH_CHARACTERS = frozenset('<>:"\\|?*')
_WINDOWS_RESERVED_COMPONENT = re.compile(
    r"(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?",
    re.IGNORECASE,
)


def _validate_portable_structural_workspace_path(path: str) -> str:
    """Require one canonical relative path with stable POSIX/Windows spelling."""

    normalized = _validate_workspace_relative_path(path)
    if normalized != path:
        raise ValueError("Workspace assertion paths must use canonical POSIX spelling.")
    if unicodedata.normalize("NFC", path) != path:
        raise ValueError("Workspace assertion paths must use canonical Unicode spelling.")
    if any(
        ord(character) < 0x20 or character in _WINDOWS_FORBIDDEN_PATH_CHARACTERS
        for character in path
    ):
        raise ValueError("Workspace assertion paths must use portable POSIX characters.")
    for component in path.split("/"):
        if component.endswith((" ", ".")) or _WINDOWS_RESERVED_COMPONENT.fullmatch(component):
            raise ValueError("Workspace assertion paths must not use platform-reserved names.")
    return path
