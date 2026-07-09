from __future__ import annotations

import posixpath
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache

from cayu._validation import require_nonblank


@dataclass(frozen=True)
class WorkspaceReadResult:
    content: bytes
    total_bytes: int
    truncated: bool = False

    def __post_init__(self) -> None:
        if type(self.content) is not bytes:
            raise TypeError("WorkspaceReadResult content must be bytes.")
        if type(self.total_bytes) is not int:
            raise TypeError("WorkspaceReadResult total_bytes must be an integer.")
        if self.total_bytes < 0:
            raise ValueError("WorkspaceReadResult total_bytes must be non-negative.")
        if type(self.truncated) is not bool:
            raise TypeError("WorkspaceReadResult truncated must be a bool.")
        if self.total_bytes < len(self.content):
            raise ValueError("WorkspaceReadResult total_bytes cannot be smaller than content.")
        expected_truncated = len(self.content) < self.total_bytes
        if self.truncated != expected_truncated:
            raise ValueError("WorkspaceReadResult truncated must match content and total_bytes.")


@dataclass(frozen=True)
class WorkspaceListResult:
    paths: tuple[str, ...]
    total_count: int | None
    truncated: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.paths, str | bytes):
            raise TypeError("WorkspaceListResult paths must be an iterable of strings.")
        try:
            paths = tuple(self.paths)
        except TypeError as exc:
            raise TypeError("WorkspaceListResult paths must be an iterable of strings.") from exc
        for path in paths:
            if type(path) is not str:
                raise TypeError("WorkspaceListResult paths entries must be strings.")
        if self.total_count is not None:
            if type(self.total_count) is not int:
                raise TypeError("WorkspaceListResult total_count must be an integer.")
            if self.total_count < 0:
                raise ValueError("WorkspaceListResult total_count must be non-negative.")
        if type(self.truncated) is not bool:
            raise TypeError("WorkspaceListResult truncated must be a bool.")
        if not self.truncated and self.total_count is None:
            raise ValueError("WorkspaceListResult total_count is required when not truncated.")
        if not self.truncated and self.total_count is not None and self.total_count != len(paths):
            raise ValueError("WorkspaceListResult total_count must equal paths when not truncated.")
        if self.total_count is not None and self.total_count < len(paths):
            raise ValueError("WorkspaceListResult total_count cannot be smaller than paths.")
        object.__setattr__(self, "paths", paths)


def validate_list_pattern(pattern: str) -> str:
    """Validate a Workspace ``list()`` pattern shared by every backend."""
    value = require_nonblank(pattern, "pattern")
    if posixpath.isabs(value):
        raise ValueError("Workspace list pattern must stay inside the workspace.")
    parts = tuple(part for part in value.split("/") if part)
    if ".." in parts:
        raise ValueError("Workspace list pattern must stay inside the workspace.")
    return value


def translate_list_pattern(pattern: str) -> str:
    """Translate a Workspace ``list()`` pattern into an anchored regular expression.

    This defines the one normative matching semantics shared by every
    Workspace backend, applied to a file's full workspace-relative POSIX path:

    - The pattern is anchored at both ends (``*.txt`` does NOT match
      ``nested/a.txt``).
    - ``*`` matches any run of characters within one path segment, ``?``
      matches one character within a segment, and ``[...]``/``[!...]``
      character classes match one character within a segment; none of them
      cross ``/``.
    - ``**`` as a whole segment matches zero or more directories when more
      pattern follows (``**/*.txt`` matches ``a.txt`` and ``d/a.txt``), and
      matches any remaining path when it is the final segment.
    - Empty segments and ``.`` segments in the pattern are ignored.
    """
    segments = [segment for segment in pattern.split("/") if segment not in {"", "."}]
    if not segments:
        return r"(?!)"
    parts: list[str] = []
    last_index = len(segments) - 1
    for index, segment in enumerate(segments):
        is_last = index == last_index
        if segment == "**":
            if is_last:
                parts.append(r"[^/]+(?:/[^/]+)*")
            else:
                parts.append(r"(?:[^/]+/)*")
            continue
        parts.append(_segment_regex(segment) + ("" if is_last else "/"))
    return "".join(parts)


def matches_list_pattern(path: str, pattern: str) -> bool:
    """Report whether a workspace-relative POSIX file path matches a list pattern."""
    return _compiled_list_pattern(pattern).fullmatch(path) is not None


@lru_cache(maxsize=256)
def _compiled_list_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(translate_list_pattern(pattern))


def _segment_regex(segment: str) -> str:
    parts: list[str] = []
    index = 0
    length = len(segment)
    while index < length:
        char = segment[index]
        if char == "*":
            parts.append(r"[^/]*")
        elif char == "?":
            parts.append(r"[^/]")
        elif char == "[":
            closing = index + 1
            if closing < length and segment[closing] in "!^":
                closing += 1
            if closing < length and segment[closing] == "]":
                closing += 1
            while closing < length and segment[closing] != "]":
                closing += 1
            if closing >= length:
                parts.append(r"\[")
            else:
                inner = segment[index + 1 : closing]
                inner = (
                    "^" + _literal_regex_class(inner[1:])
                    if inner.startswith("!")
                    else _literal_regex_class(inner)
                )
                parts.append(f"[{inner}]")
                index = closing
        else:
            parts.append(re.escape(char))
        index += 1
    return "".join(parts)


def _literal_regex_class(value: str) -> str:
    """Escape glob class content where every character is a literal option."""
    escaped = value.replace("\\", "\\\\")
    if escaped.startswith("^"):
        escaped = "\\" + escaped
    return escaped


class Workspace(ABC):
    """Filesystem/artifact area an agent can work in."""

    id: str

    @abstractmethod
    async def read_bytes(
        self,
        path: str,
        *,
        max_bytes: int | None = None,
    ) -> WorkspaceReadResult:
        """Read a file from the workspace."""

    @abstractmethod
    async def write_bytes(self, path: str, content: bytes) -> None:
        """Write a file into the workspace."""

    @abstractmethod
    async def delete(self, path: str) -> None:
        """Delete a file from the workspace if it exists."""

    @abstractmethod
    async def list(
        self,
        pattern: str = "**/*",
        *,
        limit: int | None = None,
    ) -> WorkspaceListResult:
        """List files in the workspace.

        Every backend must match ``pattern`` against workspace-relative POSIX
        file paths with the normative semantics of ``matches_list_pattern``.
        """

    @property
    def resource_key(self) -> tuple[object, ...] | None:
        """Stable, hashable token identifying the underlying resource (filesystem/sandbox area)
        this workspace reads and writes, so callers can tell whether two ``Workspace`` objects
        point at the SAME place.

        Returns ``None`` when identity cannot be determined. ``SyncBinding`` then refuses to bind
        rather than risk clearing a target that is actually the source; override this in a custom
        ``Workspace`` to return a stable identity token and enable that safety check.
        """
        return None


def _local_resource_key(path: object) -> tuple[object, ...]:
    """Canonical identity for a host-filesystem directory, shared by every host-backed workspace view."""
    return ("local", str(path))


def _runner_resource_key(runner: object) -> tuple[object, ...] | None:
    """Stable identity for a runner, or ``None`` when the runner exposes no stable identifier.

    Returning ``None`` for an indeterminate runner lets runner-backed workspaces fail closed rather
    than treating Python object identity as proof that two runners are distinct resources.
    """
    if runner is None:
        return None
    for attr in ("sandbox_id", "name", "container_name", "sandbox_name", "root"):
        value = getattr(runner, attr, None)
        if value is not None:
            return (type(runner), attr, str(value))
    return None


def _runner_workspace_resource_key(runner: object, path: str) -> tuple[object, ...] | None:
    """Compose a runner-backed workspace key, or ``None`` when the runner identity is indeterminate."""
    runner_key = _runner_resource_key(runner)
    if runner_key is None:
        return None
    return ("runner", runner_key, path)
