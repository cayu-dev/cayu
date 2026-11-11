from __future__ import annotations

import asyncio
from contextlib import contextmanager

from cayu.workspaces import LocalWorkspace
from cayu.workspaces import local as local_module


def test_bounded_read_does_not_certify_a_prefix_as_the_whole_file(tmp_path, monkeypatch) -> None:
    target = tmp_path / "changing.txt"
    before = b"old!"
    after = b"new! with more bytes"
    target.write_bytes(before)
    workspace = LocalWorkspace(tmp_path)
    original_open = local_module.open_regular_for_read

    @contextmanager
    def grow_after_open(root, path):
        with original_open(root, path) as opened:
            # Change the same inode after its size was sampled but before read().
            with target.open("r+b", buffering=0) as writer:
                assert writer.write(after) == len(after)
            yield opened

    monkeypatch.setattr(local_module, "open_regular_for_read", grow_after_open)
    try:
        result = asyncio.run(workspace.read_bytes(target.name, max_bytes=len(before)))
    except (OSError, ValueError):
        # Rejecting the racing read is preferable to publishing a false revision.
        return
    assert result.truncated or result.content == before, (
        "a prefix of the grown file was reported as a complete snapshot"
    )
    if result.content != before:
        assert result.revision is None
