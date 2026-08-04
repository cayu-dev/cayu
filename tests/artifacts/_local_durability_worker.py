from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

from cayu.artifacts import LocalArtifactStore
from cayu.artifacts import local as local_module


def _crash() -> None:
    os._exit(86)


def main() -> None:
    root = Path(sys.argv[1])
    phase = sys.argv[2]
    artifact_id = sys.argv[3]

    if phase.startswith("root-"):
        real_ancestry = local_module._create_durable_directory_ancestry
        real_marker = local_module._create_pending_root_marker
        real_mkdir = local_module.os.mkdir
        real_sync = local_module._sync_open_directory
        real_unlink = local_module.os.unlink
        real_marker_write = local_module.os.write
        root_was_synced = False

        def create_racing_root(path) -> None:
            real_ancestry(path)
            root.mkdir()

        def crash_during_marker_write(descriptor, content):
            if phase.endswith("during-marker-write"):
                written = real_marker_write(
                    descriptor,
                    content[: max(1, len(content) // 2)],
                )
                _crash()
                return written
            return real_marker_write(descriptor, content)

        def crash_after_marker(*args, **kwargs) -> None:
            real_marker(*args, **kwargs)
            if phase.endswith("after-marker-sync"):
                _crash()

        def crash_after_root_mkdir(path, mode=0o777, *, dir_fd=None):
            result = real_mkdir(path, mode, dir_fd=dir_fd)
            if phase == "root-after-root-create" and Path(path).name == root.name:
                _crash()
            return result

        def crash_after_root_sync(*args, **kwargs) -> None:
            nonlocal root_was_synced
            real_sync(*args, **kwargs)
            path = args[1]
            if path == root:
                root_was_synced = True
                if phase.endswith("after-root-sync"):
                    _crash()
            if (
                phase.endswith("after-parent-sync")
                and path == root.parent
                and root_was_synced
                and local_module._root_pending_marker(root).exists()
            ):
                _crash()

        def crash_after_marker_unlink(path, *, dir_fd=None):
            result = real_unlink(path, dir_fd=dir_fd)
            if (
                phase.endswith("after-marker-remove")
                and Path(path).name == local_module._root_pending_marker(root).name
            ):
                _crash()
            return result

        if phase.startswith("root-raced-"):
            local_module._create_durable_directory_ancestry = create_racing_root  # ty: ignore[invalid-assignment]
        local_module.os.write = crash_during_marker_write  # ty: ignore[invalid-assignment]
        local_module._create_pending_root_marker = crash_after_marker  # ty: ignore[invalid-assignment]
        local_module.os.mkdir = crash_after_root_mkdir  # ty: ignore[invalid-assignment]
        local_module._sync_open_directory = crash_after_root_sync  # ty: ignore[invalid-assignment]
        local_module.os.unlink = crash_after_marker_unlink  # ty: ignore[invalid-assignment]
        LocalArtifactStore(root)
        raise RuntimeError(f"root phase did not trigger: {phase}")

    real_artifact_write = local_module._write_artifact_file
    real_sync = local_module._sync_open_directory
    real_rename = local_module._rename_directory_no_replace
    root_syncs = 0

    def crash_after_file(*args, **kwargs) -> None:
        real_artifact_write(*args, **kwargs)
        filename = args[3]
        if phase == "after-content-sync" and filename == "content":
            _crash()
        if phase == "after-metadata-sync" and filename == "metadata.json":
            _crash()

    def crash_after_directory(*args, **kwargs) -> None:
        nonlocal root_syncs
        real_sync(*args, **kwargs)
        path = args[1]
        if phase == "after-staging-sync" and ".staging-" in path.name:
            _crash()
        if path == root:
            root_syncs += 1
            if phase == "after-root-sync" and root_syncs == 1:
                _crash()

    def crash_after_publish(*args, **kwargs) -> None:
        real_rename(*args, **kwargs)
        if phase == "after-publish":
            _crash()

    local_module._write_artifact_file = crash_after_file  # ty: ignore[invalid-assignment]
    local_module._sync_open_directory = crash_after_directory  # ty: ignore[invalid-assignment]
    local_module._rename_directory_no_replace = crash_after_publish  # ty: ignore[invalid-assignment]

    store = LocalArtifactStore(root)
    asyncio.run(
        store.put_bytes(
            b"durable-content",
            artifact_id=artifact_id,
            filename="durable.txt",
            content_type="text/plain",
            session_id="sess_durable",
        )
    )
    if phase == "acknowledged":
        print("acknowledged", flush=True)
        time.sleep(60)
    raise RuntimeError(f"phase did not trigger: {phase}")


if __name__ == "__main__":
    main()
