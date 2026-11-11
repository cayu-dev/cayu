from __future__ import annotations

import asyncio
import errno
import os
from pathlib import Path

import pytest

import cayu.workspaces._local_guard as guard_module
from cayu.workspaces import LocalWorkspace


@pytest.mark.parametrize(
    ("name_max", "name"),
    [(limit, "a" * limit) for limit in (1, 6, 7, 14, 30, 255)] + [(40, "é" * 7)],
)
def test_conditional_replace_bounds_staging_name_and_retries_collisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name_max: int, name: str
) -> None:
    target = tmp_path / name
    target.write_bytes(b"before")
    target.chmod(0o640)
    workspace = LocalWorkspace(tmp_path)
    original_open = os.open
    attempts: list[str] = []
    queried_directories: list[int] = []
    tokens = iter(("0" * 24, "1" * 24))

    def limited_fpathconf(descriptor: int, key: str) -> int:
        assert key == "PC_NAME_MAX"
        queried_directories.append(os.fstat(descriptor).st_ino)
        return name_max

    def limited_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if dir_fd is not None:
            if len(os.fsencode(path)) > name_max:
                raise OSError(errno.ENAMETOOLONG, "File name too long", path)
            if flags & os.O_CREAT:
                assert flags & os.O_EXCL
                candidate = os.fsdecode(path)
                attempts.append(candidate)
                if len(attempts) == 1:
                    # Leave a real colliding file that exclusive creation must preserve.
                    descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
                    with os.fdopen(descriptor, "wb") as collision:
                        collision.write(b"unrelated")
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "fpathconf", limited_fpathconf)
    monkeypatch.setattr(os, "open", limited_open)
    monkeypatch.setattr(guard_module.secrets, "token_hex", lambda _size: next(tokens))

    async def replace() -> None:
        before = await workspace.read_bytes(name)
        assert before.revision is not None
        result = await workspace.replace_bytes(name, b"after", expected_revision=before.revision)
        assert result.before_revision == before.revision
        assert result.after_revision == (await workspace.read_bytes(name)).revision

    asyncio.run(replace())
    assert queried_directories == [tmp_path.stat().st_ino]
    assert len(attempts) == 2
    assert attempts[0] != attempts[1]
    assert (tmp_path / attempts[0]).read_bytes() == b"unrelated"
    assert not (tmp_path / attempts[1]).exists()
    assert target.read_bytes() == b"after"
    assert target.stat().st_mode & 0o777 == 0o640
    assert {entry.name for entry in tmp_path.iterdir()} == {name, attempts[0]}
