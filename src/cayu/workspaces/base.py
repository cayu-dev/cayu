from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


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
            raise ValueError(
                "WorkspaceReadResult total_bytes cannot be smaller than content."
            )
        expected_truncated = len(self.content) < self.total_bytes
        if self.truncated != expected_truncated:
            raise ValueError(
                "WorkspaceReadResult truncated must match content and total_bytes."
            )


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
            raise TypeError(
                "WorkspaceListResult paths must be an iterable of strings."
            ) from exc
        for path in paths:
            if type(path) is not str:
                raise TypeError("WorkspaceListResult paths entries must be strings.")
        if self.total_count is not None:
            if type(self.total_count) is not int:
                raise TypeError("WorkspaceListResult total_count must be an integer.")
            if self.total_count < 0:
                raise ValueError(
                    "WorkspaceListResult total_count must be non-negative."
                )
        if type(self.truncated) is not bool:
            raise TypeError("WorkspaceListResult truncated must be a bool.")
        if not self.truncated and self.total_count is None:
            raise ValueError(
                "WorkspaceListResult total_count is required when not truncated."
            )
        if (
            not self.truncated
            and self.total_count is not None
            and self.total_count != len(paths)
        ):
            raise ValueError(
                "WorkspaceListResult total_count must equal paths when not truncated."
            )
        if self.total_count is not None and self.total_count < len(paths):
            raise ValueError(
                "WorkspaceListResult total_count cannot be smaller than paths."
            )
        object.__setattr__(self, "paths", paths)


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
    async def list(
        self,
        pattern: str = "**/*",
        *,
        limit: int | None = None,
    ) -> WorkspaceListResult:
        """List files in the workspace."""
