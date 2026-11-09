"""SyncBinding recovery must accept file paths that bind itself accepted."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from cayu.environments.bindings import SyncBinding
from cayu.workspaces import LocalWorkspace


@pytest.mark.parametrize("path", [" report.txt", "report.txt "])
def test_completion_recovery_round_trips_whitespace_file_names(tmp_path: Path, path: str) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / path).write_bytes(b"before")
    source = LocalWorkspace(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="target")

    def make_binding() -> SyncBinding:
        return SyncBinding(
            target_workspace=target,
            source_conflict_policy="require_revision",
            preserve_git_modes=True,
            max_file_bytes=1024,
            max_total_bytes=4096,
        )

    async def run() -> None:
        original = make_binding()
        bound = await original.bind(source, None, session_id="space-path")
        assert (target_root / path).read_bytes() == b"before"
        await target.write_bytes(path, b"after")
        saved = original._completion_finalization_recovery_state(bound)
        assert saved is not None
        original.abandon(bound)  # Drop process-local ownership, as on process loss.

        binding = make_binding()
        recovered = await binding._recover_completion_finalization(
            source,
            None,
            session_id="space-path",
            agent_name=None,
            environment_name=None,
            recovery_state=saved,
        )
        try:
            snapshot = await binding.finalize(recovered, outcome="completed")
            assert snapshot is not None
            assert (source_root / path).read_bytes() == b"after"
        finally:
            binding.abandon(recovered)

    asyncio.run(run())
