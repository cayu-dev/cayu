"""A retried overwrite pass must remove files it published before failure."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from cayu.environments.bindings import SyncBinding, SyncBindingSourceConflictError
from cayu.workspaces import LocalWorkspace


@pytest.mark.parametrize("policy", ["overwrite", "require_revision"])
def test_overwrite_retry_removes_a_partially_published_create(tmp_path: Path, policy: str) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "b.txt").write_bytes(b"before")

    class FailOnce(LocalWorkspace):
        fail_write = False

        def maybe_fail(self, path: str) -> None:
            if self.fail_write and path == "b.txt":
                self.fail_write = False
                raise OSError("injected publication failure")

        async def write_bytes(self, path: str, content: bytes) -> None:
            self.maybe_fail(path)
            await super().write_bytes(path, content)

        async def replace_bytes(self, path, content, *, expected_revision):
            self.maybe_fail(path)
            return await super().replace_bytes(path, content, expected_revision=expected_revision)

    source = FailOnce(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="target")
    binding = SyncBinding(
        target_workspace=target, max_file_bytes=1024, source_conflict_policy=policy
    )

    async def run() -> None:
        bound = await binding.bind(source, None, session_id="overwrite-retry")
        try:
            await target.write_bytes("a-created.txt", b"temporary output")
            await target.write_bytes("b.txt", b"after")
            source.fail_write = True
            error_type = OSError if policy == "overwrite" else SyncBindingSourceConflictError
            with pytest.raises(error_type):
                await binding.finalize(bound, outcome="completed")
            assert (source_root / "a-created.txt").read_bytes() == b"temporary output"
            # Undo the create before retrying the failed publication.
            await target.delete("a-created.txt")
            snapshot = await binding.finalize(bound, outcome="completed")
            assert snapshot is not None
            assert not (source_root / "a-created.txt").exists(), (
                "retry reported success but kept the earlier, now-reverted create"
            )
            assert (source_root / "b.txt").read_bytes() == b"after"
        finally:
            binding.abandon(bound)

    asyncio.run(run())


def test_failed_create_can_be_replaced_with_directory(tmp_path: Path) -> None:
    async def scenario() -> None:
        source_root = tmp_path / "source"
        target_root = tmp_path / "target"
        source_root.mkdir()
        target_root.mkdir()
        source = LocalWorkspace(source_root, workspace_id="source")
        target = LocalWorkspace(target_root, workspace_id="target")
        binding = SyncBinding(
            target_workspace=target, max_file_bytes=8, source_conflict_policy="overwrite"
        )
        bound = await binding.bind(source, None, session_id="retry-directory")
        try:
            await target.write_bytes("out", b"x" * 9)
            with pytest.raises(RuntimeError, match="exceeds max_file_bytes"):
                await binding.finalize(bound, outcome="completed")
            await target.delete("out")
            await target.write_bytes("out/part.txt", b"valid")
            await binding.finalize(bound, outcome="completed")
            assert (source_root / "out/part.txt").read_bytes() == b"valid"
        finally:
            binding.abandon(bound)

    asyncio.run(scenario())


def test_reverted_failed_create_preserves_source_directory(tmp_path: Path) -> None:
    async def scenario() -> None:
        source_root = tmp_path / "source"
        target_root = tmp_path / "target"
        source_root.mkdir()
        target_root.mkdir()
        (source_root / "out").mkdir()
        source = LocalWorkspace(source_root, workspace_id="source")
        target = LocalWorkspace(target_root, workspace_id="target")
        binding = SyncBinding(target_workspace=target, source_conflict_policy="overwrite")
        bound = await binding.bind(source, None, session_id="retry-source-directory")
        try:
            await target.write_bytes("out", b"create")
            with pytest.raises(IsADirectoryError):
                await binding.finalize(bound, outcome="completed")
            await target.delete("out")
            snapshot = await binding.finalize(bound, outcome="completed")
            assert snapshot is not None
            assert (source_root / "out").is_dir()
        finally:
            binding.abandon(bound)

    asyncio.run(scenario())
