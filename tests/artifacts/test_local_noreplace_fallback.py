"""Publication on filesystems that refuse renameat2 RENAME_NOREPLACE."""

import asyncio
import errno
import os
import sys

import pytest

from cayu.artifacts import ArtifactIdentityConflictError, LocalArtifactStore
from cayu.artifacts import local as local_module

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="renameat2 is Linux-only"
)


@pytest.fixture(params=[errno.EINVAL, errno.ENOTSUP, errno.ENOSYS])
def refused_noreplace(request, monkeypatch):
    real = local_module._call_native_rename
    calls = []

    def refuse(function_name, arguments):
        if function_name == "renameat2":
            calls.append(arguments)
            raise OSError(request.param, os.strerror(request.param))
        return real(function_name, arguments)

    monkeypatch.setattr(local_module, "_call_native_rename", refuse)
    return calls


def test_put_and_read_when_noreplace_is_refused(tmp_path, refused_noreplace):
    async def run():
        store = LocalArtifactStore(tmp_path / "artifacts")
        artifact = await store.put_bytes(b"hello", filename="a.txt", session_id="s1")
        assert (await store.read_bytes(artifact.id)).content == b"hello"
        same = await store.put_bytes(
            b"same", artifact_id="art_" + "1" * 32, filename="b.txt", session_id="s1"
        )
        again = await store.put_bytes(
            b"same", artifact_id=same.id, filename="b.txt", session_id="s1"
        )
        assert again.id == same.id
        with pytest.raises(ArtifactIdentityConflictError):
            await store.put_bytes(
                b"different", artifact_id=same.id, filename="b.txt", session_id="s1"
            )
        assert (await store.read_bytes(same.id)).content == b"same"
        assert not [name for name in os.listdir(store.root) if ".staging-" in name]

    asyncio.run(run())
    assert refused_noreplace


def test_fallback_never_replaces_existing_targets(tmp_path, refused_noreplace):
    root = tmp_path / "root"
    root.mkdir()
    existing_dir = root / "existing-dir"
    existing_dir.mkdir()
    (existing_dir / "content").write_bytes(b"kept")
    existing_file = root / "existing-file"
    existing_file.write_bytes(b"kept")

    staged_dir = root / "staged-dir"
    staged_dir.mkdir()
    (staged_dir / "content").write_bytes(b"new")
    staged_file = root / "staged-file"
    staged_file.write_bytes(b"new")

    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for parent_fd in (None, fd):
            for source in (staged_dir, staged_file):
                for target in (existing_dir, existing_file):
                    with pytest.raises(OSError) as raised:
                        local_module._rename_directory_no_replace(
                            source, target, parent_fd=parent_fd
                        )
                    assert raised.value.errno == errno.EEXIST
        assert (existing_dir / "content").read_bytes() == b"kept"
        assert existing_file.read_bytes() == b"kept"

        local_module._rename_directory_no_replace(staged_file, root / "file", parent_fd=fd)
        assert (root / "file").read_bytes() == b"new"
        assert not staged_file.exists()
        local_module._rename_directory_no_replace(staged_dir, root / "dir", parent_fd=None)
        assert (root / "dir" / "content").read_bytes() == b"new"
        assert not staged_dir.exists()
    finally:
        os.close(fd)


def test_other_rename_errors_are_not_retried(tmp_path, monkeypatch):
    def deny(function_name, arguments):
        raise OSError(errno.EACCES, os.strerror(errno.EACCES))

    monkeypatch.setattr(local_module, "_call_native_rename", deny)
    source = tmp_path / "source"
    source.write_bytes(b"x")
    with pytest.raises(PermissionError):
        local_module._rename_directory_no_replace(source, tmp_path / "target", parent_fd=None)
    assert source.exists()
    assert not (tmp_path / "target").exists()


def test_session_closure_claim_when_noreplace_is_refused(tmp_path, refused_noreplace):
    async def run():
        store = LocalArtifactStore(tmp_path / "artifacts")
        artifact = await store.put_bytes(b"owned", filename="owned.txt", session_id="s1")
        await store.claim_session_closure("s1", "a" * 64, max_records=10, max_bytes=10000)
        claim = await store.load_session_closure_claim("s1")
        assert [item.artifact_id for item in claim.artifacts] == [artifact.id]

    asyncio.run(run())
