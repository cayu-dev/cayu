from __future__ import annotations

import asyncio
import errno
import os
from hashlib import sha256

import pytest

from cayu.workspaces import LocalWorkspace
from cayu.workspaces import local as local_workspace

pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires POSIX descriptor traversal")


@pytest.mark.parametrize("swap_at", ["open", "scandir", "readlink"])
def test_git_listing_keeps_traversal_and_metadata_with_pinned_directories(
    tmp_path, monkeypatch, swap_at
) -> None:
    root = tmp_path / "workspace"
    pivot = root / "pivot"
    pivot.mkdir(parents=True)
    (root / "keep.txt").write_text("inside")
    (pivot / "owned.txt").write_text("inside")
    (pivot / "link").symlink_to("inside-target")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "outside-name.txt").write_text("outside")
    (outside / "owned.txt").write_text("outside")
    (outside / "owned.txt").chmod(0o755)
    (outside / "link").symlink_to("outside-target")
    workspace = LocalWorkspace(root)
    pivot_info = pivot.stat()
    original_open = os.open
    original_scandir = os.scandir
    original_readlink = os.readlink
    swapped = False

    def swap() -> None:
        nonlocal swapped
        pivot.rename(tmp_path / "held")
        pivot.symlink_to(outside, target_is_directory=True)
        swapped = True

    def swapping_open(path, flags, *args, **kwargs):
        if swap_at == "open" and not swapped and os.fsdecode(path) == "pivot":
            swap()
        return original_open(path, flags, *args, **kwargs)

    def swapping_scandir(path):
        if swap_at == "scandir" and not swapped and isinstance(path, int):
            info = os.fstat(path)
            if (info.st_dev, info.st_ino) == (pivot_info.st_dev, pivot_info.st_ino):
                swap()
        return original_scandir(path)

    def swapping_readlink(path, *args, **kwargs):
        if swap_at == "readlink" and not swapped:
            swap()
        return original_readlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", swapping_open)
    monkeypatch.setattr(os, "scandir", swapping_scandir)
    monkeypatch.setattr(os, "readlink", swapping_readlink)

    result = asyncio.run(workspace.list_git_entries(limit=10))

    assert swapped
    entries = {entry.path: entry for entry in result.entries}
    assert "keep.txt" in entries
    assert "pivot/outside-name.txt" not in entries
    if swap_at != "open":
        assert entries["pivot/owned.txt"].git_mode == "100644"
        link = entries["pivot/link"]
        assert link.git_mode == "120000"
        assert link.symlink_target_sha256 == sha256(b"inside-target").hexdigest()
        assert link.symlink_target_bytes == len(b"inside-target")


@pytest.mark.parametrize("fail_at", [None, "scandir", "stat", "readlink"])
def test_git_listing_bounds_and_closes_descriptors_on_deep_trees(
    tmp_path, monkeypatch, fail_at
) -> None:
    deep = "/".join(["d"] * 100)
    paths = ("a.txt", f"{deep}/file.txt", "d/z.txt", "z.txt")
    for path in paths:
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("inside")
    (tmp_path / "d/link").symlink_to("inside-target")
    (tmp_path / deep / "link").symlink_to("inside-target")
    expected = tuple(sorted((*paths, "d/link", f"{deep}/link")))
    workspace = LocalWorkspace(tmp_path)
    deep_info = (tmp_path / deep).stat()
    monkeypatch.setattr(local_workspace, "_LIST_OPEN_DIRECTORY_LIMIT", 3)
    live: set[int] = set()
    peak = 0
    original_open = os.open
    original_close = os.close

    def tracking_open(path, flags, *args, **kwargs):
        nonlocal peak
        descriptor = original_open(path, flags, *args, **kwargs)
        if flags & os.O_DIRECTORY:
            live.add(descriptor)
            peak = max(peak, len(live))
        return descriptor

    def tracking_close(descriptor):
        live.discard(descriptor)
        return original_close(descriptor)

    monkeypatch.setattr(os, "open", tracking_open)
    monkeypatch.setattr(os, "close", tracking_close)
    if fail_at is not None:
        original = getattr(os, fail_at)

        def failing_operation(path, *args, **kwargs):
            parent_fd = path if fail_at == "scandir" else kwargs.get("dir_fd")
            if isinstance(parent_fd, int):
                info = os.fstat(parent_fd)
                if (info.st_dev, info.st_ino) == (deep_info.st_dev, deep_info.st_ino):
                    raise OSError(errno.EIO, "Input/output error")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(os, fail_at, failing_operation)
        with pytest.raises(OSError) as raised:
            asyncio.run(workspace.list_git_entries(limit=10))
        assert raised.value.errno == errno.EIO
    else:
        for limit in (1, len(expected)):
            result = asyncio.run(workspace.list_git_entries(limit=limit))
            assert tuple(entry.path for entry in result.entries) == expected[:limit]
            assert result.total_count == len(expected)
            assert result.truncated is (limit < len(expected))
            for entry in result.entries:
                if entry.git_mode == "120000":
                    assert entry.symlink_target_sha256 == sha256(b"inside-target").hexdigest()
    assert not live
    # Root plus three descendants, with one extra fd during open/reopen.
    assert 0 < peak <= 5


def test_git_listing_does_not_follow_a_symlink_when_reopening_an_ancestor(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "workspace"
    pivot = root / "pivot"
    deep = pivot / "a/b/c/d"
    deep.mkdir(parents=True)
    (deep / "link").symlink_to("inside-target")
    (pivot / "z.txt").write_text("inside")
    (root / "keep.txt").write_text("inside")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "z.txt").write_text("outside")
    (outside / "z.txt").chmod(0o755)
    workspace = LocalWorkspace(root)
    monkeypatch.setattr(local_workspace, "_LIST_OPEN_DIRECTORY_LIMIT", 2)
    original_readlink = os.readlink
    swapped = False

    def swapping_readlink(path, *args, **kwargs):
        nonlocal swapped
        if not swapped:
            pivot.rename(tmp_path / "held")
            pivot.symlink_to(outside, target_is_directory=True)
            swapped = True
        return original_readlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "readlink", swapping_readlink)

    result = asyncio.run(workspace.list_git_entries(limit=10))

    assert swapped
    assert tuple(entry.path for entry in result.entries) == ("keep.txt", "pivot/a/b/c/d/link")
    assert result.entries[1].symlink_target_sha256 == sha256(b"inside-target").hexdigest()


@pytest.mark.parametrize("replaced", ["root", "ancestor"])
def test_git_listing_rejects_replaced_root_components(tmp_path, replaced) -> None:
    root = tmp_path / "parent" / "workspace"
    root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    workspace = LocalWorkspace(root)
    pivot = root if replaced == "root" else root.parent
    pivot.rename(tmp_path / "held")
    pivot.symlink_to(outside, target_is_directory=True)

    with pytest.raises((OSError, ValueError)):
        asyncio.run(workspace.list_git_entries(limit=10))
