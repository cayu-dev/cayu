"""Executable-specific command-selector recipe rendered into the authoring guide."""

from __future__ import annotations

# cayu-guide-include:pytest-selector:start
import re
from pathlib import Path, PurePosixPath

_NODE_ID = re.compile(r"[A-Za-z0-9_.\[\]-]+")


def pytest_selector(raw: str, *, workspace: Path) -> str:
    if not raw or "\0" in raw or raw.startswith("-") or "\\" in raw:
        raise ValueError("unsupported test selector")
    path_text, *node_ids = raw.split("::")
    raw_path_parts = path_text.split("/")
    path = PurePosixPath(path_text)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} or part.startswith("-") for part in raw_path_parts)
        or path.parts[0] != "tests"
        or path.suffix != ".py"
        or any(not _NODE_ID.fullmatch(node_id) for node_id in node_ids)
    ):
        raise ValueError("unsupported test selector")
    root = workspace.resolve()
    candidate = (root / Path(*path.parts)).resolve()
    candidate.relative_to(root)
    if not candidate.is_file():
        raise ValueError("unsupported test selector")
    return raw


# cayu-guide-include:pytest-selector:end
