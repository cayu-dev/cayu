from __future__ import annotations

import asyncio
import os

import pytest

from cayu.workspaces import LocalWorkspace


@pytest.mark.skipif(not hasattr(os, "pathconf"), reason="requires POSIX NAME_MAX")
def test_conditional_replace_accepts_existing_file_at_name_max(tmp_path) -> None:
    name_max = os.pathconf(tmp_path, "PC_NAME_MAX")
    if name_max <= 0:
        pytest.skip("filesystem does not report NAME_MAX")
    name = "a" * name_max
    target = tmp_path / name
    target.write_bytes(b"before")
    workspace = LocalWorkspace(tmp_path)

    async def replace() -> None:
        read = await workspace.read_bytes(name)
        assert read.revision is not None
        result = await workspace.replace_bytes(name, b"after", expected_revision=read.revision)
        assert result.before_revision == read.revision
        assert result.after_revision == (await workspace.read_bytes(name)).revision

    asyncio.run(replace())
    assert target.read_bytes() == b"after"
