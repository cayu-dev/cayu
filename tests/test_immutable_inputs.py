from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from cayu import (
    DockerImmutableInputMount,
    ImmutableInputAttachment,
    ImmutableInputAttachmentStateError,
    ImmutableInputMutationError,
    ImmutableInputProjectionCapability,
    ImmutableInputProjectionUnsupportedError,
    ImmutableInputStore,
    LocalWorkspace,
    NativeBinding,
    NoWorkspaceBinding,
    SyncBinding,
    docker_immutable_input_capability,
    inspect_local_immutable_input,
    require_immutable_input_projection,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _source(root: Path, *, target_path: str = "/opt/cayu/inputs/runtime"):
    return inspect_local_immutable_input(
        root,
        target_path=target_path,
        policy_fingerprint=_digest("a"),
        runtime_compatibility_fingerprint=_digest("b"),
        authorization_scope_fingerprint=_digest("c"),
        max_files=100,
        max_file_bytes=1024 * 1024,
        max_total_bytes=4 * 1024 * 1024,
    )


def test_projection_identity_binds_target_policy_runtime_and_authority(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "runtime.py").write_text("VERSION = 1\n", encoding="utf-8")
    source = _source(source_root)
    baseline = source.projection

    identities = {
        baseline.fingerprint,
        baseline.model_copy(update={"target_path": "/opt/cayu/inputs/support"}).fingerprint,
        baseline.model_copy(update={"policy_fingerprint": _digest("d")}).fingerprint,
        baseline.model_copy(update={"runtime_compatibility_fingerprint": _digest("e")}).fingerprint,
        baseline.model_copy(update={"authorization_scope_fingerprint": _digest("f")}).fingerprint,
    }

    assert len(identities) == 5
    assert baseline.content_root.startswith("sha256:")
    assert baseline.logical_bytes == len("VERSION = 1\n")
    assert baseline.file_count == 1


@pytest.mark.parametrize(
    "target_path",
    ["relative", "/", "/workspace/runtime", "/proc/version", "/tmp/runtime"],
)
def test_projection_rejects_unsafe_or_mutable_workspace_targets(
    tmp_path: Path,
    target_path: str,
) -> None:
    with pytest.raises(ValueError):
        _source(tmp_path, target_path=target_path)


def test_binding_capability_distinguishes_projection_sync_and_materialization(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    sync = SyncBinding(target_workspace=LocalWorkspace(target_root))

    assert (
        docker_immutable_input_capability().capability
        is ImmutableInputProjectionCapability.SHARED_READ_ONLY
    )
    assert (
        sync.input_capability().capability
        is ImmutableInputProjectionCapability.MUTABLE_SYNC_BINDING
    )
    assert (
        NativeBinding().input_capability().capability
        is ImmutableInputProjectionCapability.WORKSPACE_MATERIALIZATION
    )
    assert (
        NoWorkspaceBinding().input_capability().capability
        is ImmutableInputProjectionCapability.UNSUPPORTED
    )
    with pytest.raises(ImmutableInputProjectionUnsupportedError):
        require_immutable_input_projection(sync.input_capability())
    assert (
        require_immutable_input_projection(
            sync.input_capability(),
            allow_bounded_copy_fallback=True,
        ).capability
        is ImmutableInputProjectionCapability.MUTABLE_SYNC_BINDING
    )


def test_store_100_way_fanout_creates_one_physical_materialization(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "runtime.bin").write_bytes(b"runtime" * 4096)
    source = _source(source_root)
    store_root = tmp_path / "managed"
    store = ImmutableInputStore(store_root)

    async def run():
        attachments = await asyncio.gather(
            *(
                store.attach(
                    source,
                    attachment_id=f"environment:{index}",
                    owner_id=f"session:{index}",
                )
                for index in range(100)
            )
        )
        return attachments

    attachments = asyncio.run(run())
    diagnostic = store.inspect()[0]

    assert len({attachment.materialization_path for attachment in attachments}) == 1
    assert sum(not attachment.reused for attachment in attachments) == 1
    assert diagnostic.logical_bytes == len(b"runtime" * 4096)
    assert diagnostic.physical_bytes == diagnostic.logical_bytes
    assert diagnostic.reference_count == 100
    assert diagnostic.attachment_count == 100
    assert diagnostic.reuse_count == 99
    assert diagnostic.cleanup_state == "retained"
    assert len(tuple((store_root / "objects").iterdir())) == 1

    async def release_all() -> None:
        await asyncio.gather(
            *(store.release(attachment.attachment_id) for attachment in attachments)
        )

    asyncio.run(release_all())
    recovered = ImmutableInputStore(store_root)
    recovered_diagnostic = recovered.inspect()[0]
    assert recovered_diagnostic.reference_count == 0
    assert recovered_diagnostic.cleanup_state == "eligible"
    assert recovered.collect(source.projection.fingerprint) is True
    assert recovered.inspect() == ()
    assert tuple((store_root / "objects").iterdir()) == ()


def test_attachment_acknowledgement_replay_is_idempotent_and_release_is_final(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "input.txt").write_text("same", encoding="utf-8")
    source = _source(source_root)
    store = ImmutableInputStore(tmp_path / "managed")

    first = store.attach_sync(source, attachment_id="attachment:1", owner_id="session:1")
    replay = store.attach_sync(source, attachment_id="attachment:1", owner_id="session:1")
    assert replay.materialization_path == first.materialization_path
    assert store.inspect()[0].reference_count == 1

    store.release_sync("attachment:1")
    store.release_sync("attachment:1")
    assert store.inspect()[0].reference_count == 0
    with pytest.raises(ImmutableInputAttachmentStateError) as caught:
        store.attach_sync(source, attachment_id="attachment:1", owner_id="session:1")
    assert caught.value.state == "released"


def test_attachment_and_mount_authority_cannot_be_publicly_minted_or_rewritten(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "input.txt").write_text("trusted", encoding="utf-8")
    source = _source(source_root)
    store = ImmutableInputStore(tmp_path / "managed")
    attachment = store.attach_sync(
        source,
        attachment_id="attachment:issued",
        owner_id="session:issued",
    )

    with pytest.raises(TypeError, match="issued by ImmutableInputStore"):
        ImmutableInputAttachment(
            attachment_id="attachment:forged",
            owner_id="session:forged",
            projection=source.projection,
            materialization_path=tmp_path / "attacker",
            reused=False,
        )
    with pytest.raises(TypeError, match="issued by ImmutableInputStore"):
        replace(attachment, materialization_path=tmp_path / "attacker")

    mount = attachment.docker_mount()
    with pytest.raises(TypeError, match="issued by ImmutableInputStore"):
        DockerImmutableInputMount(
            source_path=str(tmp_path / "attacker"),
            target_path=source.projection.target_path,
            projection_fingerprint=source.projection.fingerprint,
            attachment_id=attachment.attachment_id,
        )
    with pytest.raises(TypeError, match="issued by ImmutableInputStore"):
        replace(mount, source_path=str(tmp_path / "attacker"))


def test_fresh_store_reclaims_interrupted_publication_staging(tmp_path: Path) -> None:
    store_root = tmp_path / "managed"
    store = ImmutableInputStore(store_root)
    orphan = store.root / "objects" / ".publishing-crashed"
    orphan.mkdir()
    (orphan / "partial.bin").write_bytes(b"partial")

    assert ImmutableInputStore(store_root).inspect() == ()
    assert orphan.exists() is False


def test_fresh_store_reconciles_durable_container_closing_state(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "input.txt").write_text("trusted", encoding="utf-8")
    source = _source(source_root)
    store_root = tmp_path / "managed"
    store = ImmutableInputStore(store_root)
    attachment = store.attach_sync(
        source,
        attachment_id="attachment:closing",
        owner_id="session:closing",
    )
    container_id = "d" * 64
    store.mark_container_closing_sync((attachment,), container_id=container_id)

    recovered = ImmutableInputStore(store_root)
    assert (
        recovered.interrupted_cleanup_container_id_sync((attachment.attachment_id,)) == container_id
    )
    assert (
        recovered.reconcile_interrupted_container_cleanup_sync(
            (attachment.attachment_id,),
            container_id=container_id,
            container_exists=True,
        )
        is True
    )
    replay = recovered.attach_sync(
        source,
        attachment_id=attachment.attachment_id,
        owner_id=attachment.owner_id,
    )
    recovered.mark_container_closing_sync((replay,), container_id=container_id)
    assert (
        recovered.reconcile_interrupted_container_cleanup_sync(
            (attachment.attachment_id,),
            container_id=container_id,
            container_exists=False,
        )
        is False
    )
    assert recovered.inspect()[0].reference_count == 0


def test_import_cayu_does_not_require_posix_fcntl() -> None:
    script = """
import builtins

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "fcntl":
        raise ModuleNotFoundError("simulated Windows without fcntl")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import cayu
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_source_and_materialization_drift_fail_with_typed_evidence(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_file = source_root / "input.txt"
    source_file.write_text("original", encoding="utf-8")
    source = _source(source_root)
    source_file.write_text("changed", encoding="utf-8")
    store = ImmutableInputStore(tmp_path / "managed")

    with pytest.raises(ImmutableInputMutationError) as source_error:
        store.attach_sync(source, attachment_id="attachment:source", owner_id="session")
    assert source_error.value.reason_code == "source_identity_drift"

    source_file.write_text("original", encoding="utf-8")
    attached = store.attach_sync(
        source,
        attachment_id="attachment:materialization",
        owner_id="session",
    )
    materialized = attached.materialization_path / "input.txt"
    os.chmod(materialized, 0o644)
    materialized.write_text("tampered", encoding="utf-8")

    with pytest.raises(ImmutableInputMutationError) as materialization_error:
        store.attach_sync(source, attachment_id="attachment:second", owner_id="session:2")
    assert materialization_error.value.reason_code == "materialization_drift"


def test_fresh_store_adopts_exact_orphaned_publication_after_ack_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "input.txt").write_text("durable", encoding="utf-8")
    source = _source(source_root)
    store_root = tmp_path / "managed"
    interrupted = ImmutableInputStore(store_root)

    def lose_registry_ack(_registry) -> None:
        raise OSError("simulated registry acknowledgement loss")

    monkeypatch.setattr(interrupted, "_write_registry", lose_registry_ack)
    with pytest.raises(OSError, match="acknowledgement loss"):
        interrupted.attach_sync(
            source,
            attachment_id="attachment:lost",
            owner_id="session:lost",
        )

    recovered = ImmutableInputStore(store_root)
    attachment = recovered.attach_sync(
        source,
        attachment_id="attachment:recovered",
        owner_id="session:recovered",
    )
    diagnostic = recovered.inspect()[0]
    assert attachment.materialization_path.is_dir()
    assert diagnostic.reference_count == 1
    assert diagnostic.physical_bytes == len("durable")


def test_symbolic_links_and_special_entries_are_rejected(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "real.txt").write_text("real", encoding="utf-8")
    (source_root / "link.txt").symlink_to("real.txt")

    with pytest.raises(ValueError, match="regular files"):
        _source(source_root)


def test_store_rejects_source_overlap_and_never_collects_a_registry_path(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "input.txt").write_text("content", encoding="utf-8")
    source = _source(source_root)
    overlapping = ImmutableInputStore(source_root / "managed")

    with pytest.raises(ValueError, match="must not overlap"):
        overlapping.attach_sync(
            source,
            attachment_id="attachment:overlap",
            owner_id="session",
        )

    store = ImmutableInputStore(tmp_path / "managed")
    attachment = store.attach_sync(
        source,
        attachment_id="attachment:safe",
        owner_id="session",
    )
    store.release_sync(attachment.attachment_id)
    registry_path = store.root / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    victim = tmp_path / "victim"
    victim.mkdir()
    registry["materializations"][source.projection.fingerprint]["path"] = str(victim)
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ImmutableInputMutationError) as caught:
        store.collect(source.projection.fingerprint)

    assert caught.value.reason_code == "registry_path_conflict"
    assert victim.is_dir()
