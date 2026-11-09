"""Git observations must distinguish mode changes on already-dirty files."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from tests.environments.test_bindings import _create_bare_origin

from cayu.environments.bindings import GitRepositoryBinding
from cayu.workspaces import (
    LocalWorkspace,
    WorkspaceRevisionDeltaStatus,
    WorkspaceRevisionObservationStatus,
    compare_workspace_revisions,
)


@pytest.mark.skipif(sys.platform == "win32", reason="Requires POSIX executable bits")
def test_git_observation_detects_chmod_of_content_modified_file(tmp_path: Path) -> None:
    origin, _commit = _create_bare_origin(tmp_path)
    root = tmp_path / "checkout"
    root.mkdir()
    workspace = LocalWorkspace(root, workspace_id="checkout")
    binding = GitRepositoryBinding(repo_url=str(origin), ref="main")

    async def run() -> None:
        bound = await binding.bind(workspace, None, session_id="observe-mode")
        path = root / "README.md"
        # First endpoint is dirty, as is normal after an earlier editing tool.
        path.write_bytes(b"changed content\n")
        path.chmod(0o644)
        before = await binding.observe_revision(bound)
        path.chmod(0o755)
        after = await binding.observe_revision(bound)
        assert before.status is WorkspaceRevisionObservationStatus.SUPPORTED
        assert after.status is WorkspaceRevisionObservationStatus.SUPPORTED
        delta = compare_workspace_revisions(before, after)
        assert delta.status is WorkspaceRevisionDeltaStatus.CHANGED
        assert [(item.path, item.change) for item in delta.paths] == [("README.md", "modified")]

    asyncio.run(run())
