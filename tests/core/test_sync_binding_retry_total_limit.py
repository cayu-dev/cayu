"""A partial publication can transiently grow the source beyond transfer limits."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from cayu.environments.bindings import SyncBinding, SyncBindingSourceConflictError
from cayu.runners import LocalRunner
from cayu.workspaces import LocalWorkspace, RunnerWorkspace


def test_retry_accepts_its_own_partly_published_tree_over_total_limit(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "a.txt").write_bytes(b"a")
    (source_root / "b.txt").write_bytes(b"b" * 8)

    class FailOnce(LocalWorkspace):
        fail_next = True

        async def replace_bytes_with_git_mode(self, path, content, **options):
            if path == "b.txt" and self.fail_next:
                self.fail_next = False
                raise OSError("injected transient write failure")
            return await super().replace_bytes_with_git_mode(path, content, **options)

    source = FailOnce(source_root, workspace_id="source")
    target = RunnerWorkspace(LocalRunner(target_root), workspace_id="target")
    binding = SyncBinding(
        target_workspace=target,
        source_conflict_policy="require_revision",
        preserve_git_modes=True,
        max_file_bytes=8,
        max_total_bytes=10,
    )

    async def run() -> None:
        bound = await binding.bind(source, None, session_id="retry-total")
        try:
            await target.write_bytes("a.txt", b"A" * 8)
            await target.write_bytes("b.txt", b"B")
            # Both the original and desired trees contain nine bytes.
            with pytest.raises(SyncBindingSourceConflictError) as conflict:
                await binding.finalize(bound, outcome="completed")
            assert conflict.value.applied_paths == ("a.txt",)
            assert (source_root / "a.txt").read_bytes() == b"A" * 8
            assert (source_root / "b.txt").read_bytes() == b"b" * 8
            # The transient error is gone; every source revision is exactly the
            # revision retained by the binding after its own first pass.
            snapshot = await binding.finalize(bound, outcome="completed")
            assert snapshot is not None
            assert (source_root / "a.txt").read_bytes() == b"A" * 8
            assert (source_root / "b.txt").read_bytes() == b"B"
        finally:
            binding.abandon(bound)

    asyncio.run(run())
