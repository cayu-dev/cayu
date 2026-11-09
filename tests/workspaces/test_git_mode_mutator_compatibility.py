"""Existing Git-mode workspace implementations remain constructible."""

import asyncio

import pytest

from cayu.workspaces import WorkspaceGitModeMutator


def test_existing_git_mode_mutator_without_delete_remains_constructible():
    class ExistingMutator(WorkspaceGitModeMutator):
        async def write_bytes_with_git_mode(self, *args, **kwargs):
            raise AssertionError("not used")

        async def create_bytes_with_git_mode(self, *args, **kwargs):
            raise AssertionError("not used")

        async def replace_bytes_with_git_mode(self, *args, **kwargs):
            raise AssertionError("not used")

    mutator = ExistingMutator()
    with pytest.raises(NotImplementedError, match="Git-mode conditional deletion"):
        asyncio.run(
            mutator.delete_if_revision_with_git_mode(
                "file", expected_revision="recorded", expected_git_mode="100644"
            )
        )


@pytest.mark.parametrize("sync_back", ["always", "never"])
def test_sync_binding_checks_legacy_mode_deletion_only_for_copy_back(tmp_path, sync_back):
    from cayu.environments.bindings import SyncBinding
    from cayu.workspaces import LocalWorkspace

    class LegacySource(LocalWorkspace):
        delete_if_revision_with_git_mode = WorkspaceGitModeMutator.delete_if_revision_with_git_mode

    async def scenario():
        source_root = tmp_path / "source"
        target_root = tmp_path / "target"
        source_root.mkdir()
        target_root.mkdir()
        (source_root / "input.txt").write_text("source")
        (target_root / "existing.txt").write_text("target")
        source = LegacySource(source_root, workspace_id="source")
        target = LocalWorkspace(target_root, workspace_id="target")
        binding = SyncBinding(
            target_workspace=target,
            sync_back=sync_back,
            source_conflict_policy="require_revision",
            preserve_git_modes=True,
            max_file_bytes=1024,
            max_total_bytes=4096,
        )
        if sync_back == "never":
            bound = await binding.bind(source, None, session_id="copy-in-only")
            try:
                assert (target_root / "input.txt").read_text() == "source"
            finally:
                binding.abandon(bound)
        else:
            with pytest.raises(TypeError, match="delete_if_revision_with_git_mode"):
                await binding.bind(source, None, session_id="unsupported-delete")
            assert (target_root / "existing.txt").read_text() == "target"
            assert not (target_root / "input.txt").exists()

    asyncio.run(scenario())
