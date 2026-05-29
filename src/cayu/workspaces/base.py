from __future__ import annotations

from abc import ABC, abstractmethod


class Workspace(ABC):
    """Filesystem/artifact area an agent can work in."""

    id: str

    @abstractmethod
    async def read_bytes(self, path: str) -> bytes:
        """Read a file from the workspace."""

    @abstractmethod
    async def write_bytes(self, path: str, content: bytes) -> None:
        """Write a file into the workspace."""

    @abstractmethod
    async def list(self, pattern: str = "**/*") -> list[str]:
        """List files in the workspace."""
