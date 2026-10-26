"""Docker coding refuses Git-ignored source paths at bind time, with a count and fingerprint."""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from cayu.environments.docker_coding import (
    DockerCodingWorkspaceBinding,
    DockerWorkspaceTransferLimits,
)
from cayu.runners.docker import DockerRunner
from cayu.runners.local import LocalRunner
from cayu.workspaces.local import LocalWorkspace
from cayu.workspaces.runner import RunnerWorkspace

_CONTAINER_ID = "c" * 64
_PROTECTED = (".cayu", ".git", ".runtime")


class _LocalDockerRunner(DockerRunner):
    """A Docker-shaped runner over one local directory, for binding tests."""

    def __init__(self, root: Path) -> None:
        super().__init__(
            "local-docker-test",
            default_cwd="/workspace",
            docker_path="/usr/bin/docker",
            _container_id=_CONTAINER_ID,
        )
        self.local = LocalRunner(root, inherit_env=False)
        self._test_root = root

    def resolve_cwd(self, cwd: str | None = None) -> str:
        del cwd
        return str(self._test_root)

    async def exec(self, command, **kwargs: Any):
        kwargs["cwd"] = None
        return await self.local.exec(command, **kwargs)

    async def exec_stream(self, command, **kwargs: Any):
        kwargs["cwd"] = None
        return await self.local.exec_stream(command, **kwargs)

    async def close(self) -> None:
        await self.local.close()
        self._closed = True


def _bind(tmp_path: Path, *, excluded: tuple[str, ...], sync_back: str) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir(parents=True)
    target_root.mkdir(parents=True)
    (source_root / ".gitignore").write_text("data/\n", encoding="utf-8")
    (source_root / "code.py").write_text("value = 1\n", encoding="utf-8")
    (source_root / "data").mkdir()
    (source_root / "data" / "note.txt").write_text("run file\n", encoding="utf-8")
    runner = _LocalDockerRunner(target_root)
    names = (*_PROTECTED, *excluded)
    source = LocalWorkspace(source_root, workspace_id="source", excluded_directory_names=names)
    target = RunnerWorkspace(
        runner,
        workspace_id="target",
        python_executable=sys.executable,
        excluded_directory_names=names,
    )
    binding = DockerCodingWorkspaceBinding(
        target_workspace=target,
        limits=DockerWorkspaceTransferLimits(
            max_file_bytes=1024, max_total_bytes=4096, max_archive_bytes=32 * 1024
        ),
    )
    binding.sync_back = sync_back

    async def run() -> None:
        bound = await binding.bind(source, runner, session_id="ignored-at-bind")
        assert binding.abandon(bound) is True

    asyncio.run(run())


def test_an_ignored_source_path_fails_at_bind_before_any_model_call(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError) as refused:
        _bind(tmp_path, excluded=(), sync_back="on_success")

    message = str(refused.value)
    assert message.startswith("Docker coding refused publication because Git would omit")
    assert "1 Git-ignored path is outside" in message
    assert re.search(r"paths fingerprint sha256:[0-9a-f]{64}", message)
    assert "note.txt" not in message and "data/" not in message


def test_excluding_the_ignored_directory_or_not_publishing_binds(tmp_path: Path) -> None:
    _bind(tmp_path / "excluded", excluded=("data",), sync_back="on_success")
    _bind(tmp_path / "never", excluded=(), sync_back="never")
