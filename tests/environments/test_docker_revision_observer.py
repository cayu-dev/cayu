from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from tests.environments.test_docker_coding import _LocalDockerRunner

from cayu import BoundWorkspace, DockerCodingWorkspaceBinding, DockerWorkspaceTransferLimits
from cayu.workspaces import RunnerWorkspace
from cayu.workspaces.revisions import WorkspaceRevisionObservationLimits


def test_docker_revision_tracks_guest_files_excludes_protected_paths_and_bounds_reads(
    tmp_path: Path,
):
    runner = _LocalDockerRunner(tmp_path)
    target = RunnerWorkspace(
        runner,
        workspace_id="guest",
        python_executable=sys.executable,
        excluded_directory_names=(".git", ".cayu", ".runtime"),
    )
    binding = DockerCodingWorkspaceBinding(
        target_workspace=target, limits=DockerWorkspaceTransferLimits()
    )
    bound = BoundWorkspace(workspace=target, runner=runner)
    (tmp_path / "main.py").write_text("print('ok')\n")

    async def run():
        try:
            initial = await binding.observe_revision(bound)
            assert initial.status == "supported"
            assert initial.total_paths == 1
            for directory in (".git", ".cayu", ".runtime"):
                (tmp_path / directory).mkdir()
                (tmp_path / directory / "private").write_text("protected-state")
            protected = await binding.observe_revision(bound)
            assert protected == initial
            (tmp_path / "main.py").write_text("print('changed')\n")
            changed = await binding.observe_revision(bound)
            assert changed.status == "supported"
            assert changed.revision != initial.revision
            with (tmp_path / "large.bin").open("wb") as output:
                output.truncate(WorkspaceRevisionObservationLimits().max_file_bytes + 1)
            limited = await binding.observe_revision(bound)
            assert limited.status == "truncated"
            assert limited.revision is None
        finally:
            await runner.close()

    asyncio.run(run())
