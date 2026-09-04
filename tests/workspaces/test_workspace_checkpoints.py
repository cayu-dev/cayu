from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cayu._validation import canonical_durable_json_bytes
from cayu.artifacts import LocalArtifactStore
from cayu.workspaces import LocalWorkspace
from cayu.workspaces.checkpoints import (
    WorkspaceCheckpointError,
    WorkspaceCheckpointPolicy,
    capture_workspace_checkpoint,
    load_workspace_checkpoint,
    pin_workspace_checkpoint,
    release_workspace_checkpoint,
    restore_workspace_checkpoint,
    workspace_checkpoint_revision,
)
from cayu.workspaces.revisions import WorkspaceWriterIsolationEvidence


def isolation():
    return WorkspaceWriterIsolationEvidence(
        status="exclusive", mechanism="test-lease", generation="1", detail_code=None
    )


def workspace(path):
    path.mkdir(exist_ok=True)
    return LocalWorkspace(path)


def test_capture_restore_multistep_deletes_modes_and_large_files(tmp_path):
    async def run():
        source = workspace(tmp_path / "source")
        store = LocalArtifactStore(tmp_path / "artifacts")
        policy = WorkspaceCheckpointPolicy()
        await source.write_bytes("old", b"delete me")
        await source.write_bytes("renamed", b"first")
        await source.write_bytes("renamed", b"second")
        old = await source.read_bytes("old")
        await source.delete_if_revision("old", expected_revision=old.revision)
        await source.write_bytes_with_git_mode("bin/exec", b"#!/bin/sh\n", git_mode="100755")
        large = b"bytes\x00" * 400000
        await source.write_bytes("large", large)
        aid, manifest = await capture_workspace_checkpoint(
            source,
            store,
            policy=policy,
            environment_name="test",
            owner="session:revision1",
            isolation=isolation,
        )
        digest = hashlib.sha256(
            canonical_durable_json_bytes(manifest.model_dump(mode="json"), "manifest")
        ).hexdigest()
        loaded = await load_workspace_checkpoint(
            store, aid, policy=policy, expected_manifest_sha256=digest
        )
        dest = workspace(tmp_path / "dest")
        await dest.write_bytes("unwanted", b"seed")
        await dest.write_bytes("renamed", b"stale")
        await restore_workspace_checkpoint(dest, store, loaded, policy=policy, isolation=isolation)
        assert await workspace_checkpoint_revision(dest, policy=policy) == manifest.revision
        assert (await dest.read_bytes("large")).content == large
        assert (await dest.read_bytes("bin/exec")).git_mode == "100755"
        assert not (tmp_path / "dest" / "unwanted").exists()
        # A partial/acknowledgement-lost restore is safely repeatable.
        await restore_workspace_checkpoint(dest, store, loaded, policy=policy, isolation=isolation)

    asyncio.run(run())


@pytest.mark.parametrize(
    "limit", ["max_paths", "max_file_bytes", "max_total_bytes", "max_manifest_bytes"]
)
def test_capture_fails_closed_at_bounds(tmp_path, limit):
    async def run():
        source = workspace(tmp_path / "source")
        for index in range(10):
            await source.write_bytes(f"file{index}", b"123456")
        policy = WorkspaceCheckpointPolicy(**{limit: 1024 if limit == "max_manifest_bytes" else 1})
        with pytest.raises(WorkspaceCheckpointError):
            await capture_workspace_checkpoint(
                source,
                LocalArtifactStore(tmp_path / "a"),
                policy=policy,
                environment_name="e",
                owner="s",
                isolation=isolation,
            )

    asyncio.run(run())


def test_symlink_and_unknown_isolation_fail_closed(tmp_path):
    async def run():
        source = workspace(tmp_path / "source")
        store = LocalArtifactStore(tmp_path / "a")
        with pytest.raises(WorkspaceCheckpointError, match="exclusive"):
            await capture_workspace_checkpoint(
                source,
                store,
                policy=WorkspaceCheckpointPolicy(),
                environment_name="e",
                owner="s",
                isolation=WorkspaceWriterIsolationEvidence,
            )
        (tmp_path / "source" / "link").symlink_to("outside")
        with pytest.raises(WorkspaceCheckpointError, match="symbolic"):
            await capture_workspace_checkpoint(
                source,
                store,
                policy=WorkspaceCheckpointPolicy(),
                environment_name="e",
                owner="s",
                isolation=isolation,
            )

    asyncio.run(run())


def test_descendant_pin_blocks_gc_across_store_instances(tmp_path):
    async def run():
        source = workspace(tmp_path / "source")
        await source.write_bytes("a", b"preserved")
        store = LocalArtifactStore(tmp_path / "a")
        other = LocalArtifactStore(tmp_path / "a")
        policy = WorkspaceCheckpointPolicy()
        aid, manifest = await capture_workspace_checkpoint(
            source,
            store,
            policy=policy,
            environment_name="e",
            owner="session:r1",
            isolation=isolation,
        )
        digest = hashlib.sha256(
            canonical_durable_json_bytes(manifest.model_dump(mode="json"), "manifest")
        ).hexdigest()
        await pin_workspace_checkpoint(
            other, aid, owner="snapshot:r1", policy=policy, expected_manifest_sha256=digest
        )
        await release_workspace_checkpoint(
            store, aid, owner="session:r1", policy=policy, expected_manifest_sha256=digest
        )
        for artifact_id in [aid, *(entry.artifact_id for entry in manifest.files)]:
            with pytest.raises(ValueError, match="pin"):
                await store.delete(artifact_id)
        await release_workspace_checkpoint(
            other, aid, owner="snapshot:r1", policy=policy, expected_manifest_sha256=digest
        )
        for artifact_id in [aid, *(entry.artifact_id for entry in manifest.files)]:
            await store.delete(artifact_id)

    asyncio.run(run())


def test_capture_rejects_mutation_during_upload(tmp_path):
    async def run():
        source = workspace(tmp_path / "source")
        await source.write_bytes("a", b"before")

        class ChangingStore(LocalArtifactStore):
            async def pin(self, artifact_id, *, owner):
                await super().pin(artifact_id, owner=owner)
                await source.write_bytes("a", b"after")

        with pytest.raises(WorkspaceCheckpointError, match="changed"):
            await capture_workspace_checkpoint(
                source,
                ChangingStore(tmp_path / "a"),
                policy=WorkspaceCheckpointPolicy(),
                environment_name="e",
                owner="s",
                isolation=isolation,
            )

    asyncio.run(run())


def test_fresh_process_restores_after_source_is_deleted(tmp_path):
    async def capture():
        source = workspace(tmp_path / "source")
        await source.write_bytes("survives", b"exact bytes")
        aid, manifest = await capture_workspace_checkpoint(
            source,
            LocalArtifactStore(tmp_path / "artifacts"),
            policy=WorkspaceCheckpointPolicy(),
            environment_name="e",
            owner="s:r1",
            isolation=isolation,
        )
        (tmp_path / "manifest.json").write_text(manifest.model_dump_json())
        return aid

    asyncio.run(capture())
    import shutil

    shutil.rmtree(tmp_path / "source")
    (tmp_path / "dest").mkdir()
    script = """
import asyncio, sys
from pathlib import Path
from cayu.artifacts import LocalArtifactStore
from cayu.workspaces import LocalWorkspace
from cayu.workspaces.checkpoints import WorkspaceCheckpointManifest, WorkspaceCheckpointPolicy, restore_workspace_checkpoint
from cayu.workspaces.revisions import WorkspaceWriterIsolationEvidence
root=Path(sys.argv[1])
manifest=WorkspaceCheckpointManifest.model_validate_json((root/'manifest.json').read_text())
asyncio.run(restore_workspace_checkpoint(LocalWorkspace(root/'dest'), LocalArtifactStore(root/'artifacts'), manifest,
policy=WorkspaceCheckpointPolicy(), isolation=lambda: WorkspaceWriterIsolationEvidence(status='exclusive', mechanism='test', generation='2', detail_code=None)))
assert (root/'dest'/'survives').read_bytes()==b'exact bytes'
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
    )
    assert result.returncode == 0, result.stderr


def test_pin_survives_pinning_process_exit_and_serializes_with_delete(tmp_path):
    async def put():
        return await LocalArtifactStore(tmp_path / "store").put_bytes(
            b"keep", filename="x", session_id="s"
        )

    artifact = asyncio.run(put())
    script = """
import asyncio, os, sys
from cayu.artifacts import LocalArtifactStore
asyncio.run(LocalArtifactStore(sys.argv[1]).pin(sys.argv[2], owner='session:alive'))
os._exit(0)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "store"), artifact.id],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
    )
    assert result.returncode == 0, result.stderr

    async def check():
        reopened = LocalArtifactStore(tmp_path / "store")
        with pytest.raises(ValueError, match="pin"):
            await reopened.delete(artifact.id)
        await reopened.release_pin(artifact.id, owner="wrong-owner")
        with pytest.raises(ValueError, match="pin"):
            await reopened.delete(artifact.id)
        await reopened.release_pin(artifact.id, owner="session:alive")
        await reopened.delete(artifact.id)
        with pytest.raises(FileNotFoundError):
            await reopened.pin(artifact.id, owner="too-late")

    asyncio.run(check())


def test_corrupt_artifact_never_exposes_a_restored_revision(tmp_path):
    async def run():
        source = workspace(tmp_path / "source")
        await source.write_bytes("a", b"original")
        store = LocalArtifactStore(tmp_path / "store")
        _, manifest = await capture_workspace_checkpoint(
            source,
            store,
            policy=WorkspaceCheckpointPolicy(),
            environment_name="e",
            owner="s:r1",
            isolation=isolation,
        )
        # Simulate storage corruption, bypassing the immutable application API.
        (tmp_path / "store" / manifest.files[0].artifact_id / "content").write_bytes(b"tampered")
        with pytest.raises(WorkspaceCheckpointError, match="digest"):
            await restore_workspace_checkpoint(
                workspace(tmp_path / "dest"),
                store,
                manifest,
                policy=WorkspaceCheckpointPolicy(),
                isolation=isolation,
            )
        assert not (tmp_path / "dest" / "a").exists()

    asyncio.run(run())


def checkpoint_workspace(root, adapter, **kwargs):
    from cayu.runners import LocalRunner
    from cayu.workspaces import RunnerWorkspace

    root.mkdir(exist_ok=True)
    if adapter == "local":
        return LocalWorkspace(root, **kwargs)
    return RunnerWorkspace(
        LocalRunner(root, inherit_env=False), python_executable=sys.executable, **kwargs
    )


@pytest.mark.parametrize("adapter", ["local", "runner"])
def test_restore_directory_file_transitions_and_retry(tmp_path, adapter):
    async def run():
        source = workspace(tmp_path / "source")
        await source.write_bytes_with_git_mode("package", b"#!/bin/sh\n", git_mode="100755")
        await source.write_bytes("nested/new.py", b"new")
        store = LocalArtifactStore(tmp_path / "artifacts")
        policy = WorkspaceCheckpointPolicy()
        _, manifest = await capture_workspace_checkpoint(
            source, store, policy=policy, environment_name="e", owner="s", isolation=isolation
        )
        dest = checkpoint_workspace(tmp_path / "dest", adapter)
        await dest.write_bytes("package/old/deep.py", b"old")
        (tmp_path / "dest/package/empty/nested").mkdir(parents=True)
        await dest.write_bytes("nested", b"old file")
        for _ in range(2):
            await restore_workspace_checkpoint(
                dest, store, manifest, policy=policy, isolation=isolation
            )
            assert await workspace_checkpoint_revision(dest, policy=policy) == manifest.revision
            assert (await dest.read_bytes("package")).git_mode == "100755"
            assert (await dest.read_bytes("nested/new.py")).content == b"new"

    asyncio.run(run())


@pytest.mark.parametrize("adapter", ["local", "runner"])
@pytest.mark.parametrize(
    "obstacle", ["file", "symlink", "excluded_name", "excluded_pattern", "limit"]
)
def test_directory_pruning_preserves_unsafe_or_excluded_descendants(tmp_path, adapter, obstacle):
    async def run():
        kwargs = {}
        if obstacle == "excluded_name":
            kwargs["excluded_directory_names"] = ["keep"]
        if obstacle == "excluded_pattern":
            kwargs["excluded_path_patterns"] = ["package/keep"]
        dest = checkpoint_workspace(tmp_path / "dest", adapter, **kwargs)
        package = tmp_path / "dest/package"
        package.mkdir()
        (package / "empty").mkdir()
        keep = package / "keep"
        if obstacle == "file":
            keep.write_bytes(b"preserve")
        elif obstacle == "symlink":
            outside = tmp_path / "outside"
            outside.mkdir()
            keep.symlink_to(outside, target_is_directory=True)
        else:
            keep.mkdir()
        with pytest.raises((ValueError, RuntimeError)):
            await dest.prune_empty_directories(
                "package", max_directories=1 if obstacle == "limit" else 10
            )
        assert package.is_dir()
        assert (package / "empty").is_dir()
        assert keep.exists()
        if obstacle == "file":
            assert keep.read_bytes() == b"preserve"
        if obstacle == "symlink":
            assert keep.is_symlink() and (tmp_path / "outside").is_dir()

    asyncio.run(run())
