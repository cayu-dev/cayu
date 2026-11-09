"""Partial deletion must not make a source path become target-only on retry."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from cayu.environments.bindings import SyncBinding, SyncBindingSourceConflictError
from cayu.workspaces import LocalWorkspace


@pytest.mark.parametrize("clean_policy", ["never", "always"])
def test_retry_republishes_recreated_source_path_in_initial_target_baseline(
    tmp_path: Path, clean_policy: str
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "a.txt").write_bytes(b"source-a")
    (source_root / "b.txt").write_bytes(b"source-b")
    (target_root / "a.txt").write_bytes(b"stale target copy")

    class FailOnce(LocalWorkspace):
        fail_replace = True

        async def replace_bytes(self, path, content, *, expected_revision):
            if path == "b.txt" and self.fail_replace:
                self.fail_replace = False
                raise OSError("injected publication failure")
            return await super().replace_bytes(path, content, expected_revision=expected_revision)

    source = FailOnce(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="target")
    binding = SyncBinding(
        target_workspace=target,
        clean_target=clean_policy,
        source_conflict_policy="require_revision",
        max_file_bytes=1024,
    )

    async def run() -> None:
        bound = await binding.bind(source, None, session_id="recreate-retry")
        try:
            await target.delete("a.txt")
            await target.write_bytes("b.txt", b"edited-b")
            with pytest.raises(SyncBindingSourceConflictError) as conflict:
                await binding.finalize(bound, outcome="completed")
            assert conflict.value.applied_paths == ("a.txt",)
            assert not (source_root / "a.txt").exists()
            await target.write_bytes("a.txt", b"recreated-a")
            snapshot = await binding.finalize(bound, outcome="completed")
            assert snapshot is not None
            assert (source_root / "a.txt").is_file(), (
                "retry excluded the recreated source file as pre-existing target-only content"
            )
            assert (source_root / "a.txt").read_bytes() == b"recreated-a"
            assert (source_root / "b.txt").read_bytes() == b"edited-b"
        finally:
            binding.abandon(bound)

    asyncio.run(run())
