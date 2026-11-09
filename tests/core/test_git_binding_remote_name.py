"""The configured Git remote must survive clone and subsequent binds."""

from __future__ import annotations

import asyncio
from pathlib import Path

from tests.environments.test_bindings import _create_bare_origin

from cayu.environments.bindings import GitRepositoryBinding
from cayu.workspaces import LocalWorkspace


def test_clone_with_nondefault_remote_can_bind_again(tmp_path: Path) -> None:
    origin, _commit = _create_bare_origin(tmp_path)
    target_root = tmp_path / "target"
    target_root.mkdir()
    workspace = LocalWorkspace(target_root, workspace_id="checkout")
    binding = GitRepositoryBinding(repo_url=str(origin), remote_name="upstream", ref="main")

    async def run() -> None:
        first = await binding.bind(workspace, None, session_id="first")
        await binding.finalize(first, outcome="completed")
        # No worktree edits or remote changes: rebinding the checkout must work.
        second = await binding.bind(workspace, None, session_id="second")
        await binding.finalize(second, outcome="completed")

    asyncio.run(run())
