from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from cayu.workspaces import LocalWorkspace, WorkspaceGitEntryObservationUnsupportedError


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX symlinks")
def test_git_listing_does_not_walk_directory_swapped_for_symlink(tmp_path, monkeypatch) -> None:
    root = tmp_path / "workspace"
    pivot = root / "pivot"
    pivot.mkdir(parents=True)
    (pivot / "owned.txt").write_text("inside")
    (root / "keep.txt").write_text("inside")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "outside-name.txt").write_text("outside")
    workspace = LocalWorkspace(root)
    original_scandir = os.scandir

    def swap_at_descent(path):
        if not isinstance(path, int) and Path(os.fsdecode(path)) == pivot:
            pivot.rename(tmp_path / "held")
            pivot.symlink_to(outside, target_is_directory=True)
        return original_scandir(path)

    monkeypatch.setattr(os, "scandir", swap_at_descent)
    try:
        result = asyncio.run(workspace.list_git_entries(limit=10))
    except (OSError, ValueError, WorkspaceGitEntryObservationUnsupportedError):
        # Refusing to return inventory when the traversal races a swap is safe.
        return
    paths = {entry.path for entry in result.entries}
    assert "keep.txt" in paths
    assert "pivot/outside-name.txt" not in paths, "Git listing followed a symlink outside the root"
