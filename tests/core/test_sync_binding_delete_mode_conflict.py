"""Conditional copy-back deletion must protect Git executable modes too."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from cayu.environments.bindings import SyncBinding, SyncBindingSourceConflictError
from cayu.workspaces import LocalWorkspace, WorkspaceReadResult


class _ModeChangesAfterPreflightRead(LocalWorkspace):
    change_mode_after_read = False

    async def read_bytes(
        self,
        path: str,
        *,
        offset: int = 0,
        max_bytes: int | None = None,
    ) -> WorkspaceReadResult:
        result = await super().read_bytes(path, offset=offset, max_bytes=max_bytes)
        if self.change_mode_after_read:
            self.change_mode_after_read = False
            # The preflight read saw the original mode; model another source
            # owner changing it immediately after that read completed.
            self.resolve_no_symlinks(path).chmod(0o755)
        return result


@pytest.mark.skipif(sys.platform == "win32", reason="Requires POSIX executable bits")
def test_copy_back_delete_rejects_concurrent_git_mode_change(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    source_path = source_root / "script.sh"
    source_path.write_bytes(b"echo important\n")
    source_path.chmod(0o644)
    source = _ModeChangesAfterPreflightRead(source_root, workspace_id="source")
    target = LocalWorkspace(target_root, workspace_id="target")
    binding = SyncBinding(
        target_workspace=target,
        source_conflict_policy="require_revision",
        preserve_git_modes=True,
        max_file_bytes=1024,
        max_total_bytes=4096,
    )

    async def run() -> None:
        bound = await binding.bind(source, None, session_id="delete-mode-conflict")
        try:
            await target.delete("script.sh")
            source.change_mode_after_read = True
            with pytest.raises(SyncBindingSourceConflictError):
                await binding.finalize(bound, outcome="completed")
        finally:
            binding.abandon(bound)

    asyncio.run(run())
    assert source_path.is_file()
    assert source_path.read_bytes() == b"echo important\n"
    assert source_path.stat().st_mode & 0o111


@pytest.mark.skipif(sys.platform == "win32", reason="Requires POSIX executable bits")
def test_recovered_delete_rejects_source_mode_drift(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    source_path = source_root / "script.sh"
    source_path.write_bytes(b"echo important\n")
    source_path.chmod(0o644)
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
        bound = await original.bind(source, None, session_id="deleted-script")
        await target.delete("script.sh")
        saved = original._completion_finalization_recovery_state(bound)
        assert saved is not None
        original.abandon(bound)  # Discard process-local ownership as on process loss.
        # The source file still holds its original bytes, but its owner made it
        # executable before recovery; no scheduling hooks or mocks are involved.
        source_path.chmod(0o755)
        binding = make_binding()
        recovered = await binding._recover_completion_finalization(
            source,
            None,
            session_id="deleted-script",
            agent_name=None,
            environment_name=None,
            recovery_state=saved,
        )
        try:
            with pytest.raises(SyncBindingSourceConflictError):
                await binding.finalize(recovered, outcome="completed")
        finally:
            binding.abandon(recovered)

    asyncio.run(run())
    assert source_path.is_file()
    assert source_path.read_bytes() == b"echo important\n"
    assert source_path.stat().st_mode & 0o111
