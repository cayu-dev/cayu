"""A target enclosing the source must be rejected before any destructive clean."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from cayu.environments.bindings import SyncBinding
from cayu.workspaces import LocalWorkspace


def test_bind_rejects_target_containing_source_without_deleting_source(tmp_path: Path) -> None:
    target_root = tmp_path / "scratch"
    source_root = target_root / "source"
    source_root.mkdir(parents=True)
    source_file = source_root / "important.txt"
    source_file.write_bytes(b"durable user work")
    source = LocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="target")
    binding = SyncBinding(
        target_workspace=target,
        source_conflict_policy="require_revision",
        preserve_git_modes=True,
        max_file_bytes=1024,
    )

    async def run() -> None:
        with pytest.raises(ValueError):
            await binding.bind(source, None, session_id="overlapping-roots")

    try:
        asyncio.run(run())
    finally:
        # Even a failed bind must never remove the durable source it was given.
        assert source_file.is_file(), "target cleaning deleted the durable source file"
        assert source_file.read_bytes() == b"durable user work"
