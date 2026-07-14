from __future__ import annotations

import asyncio

import pytest

from cayu import (
    BoundWorkspace,
    EFSAccessPointBinding,
    ExecCommand,
    ExecResult,
    S3FilesAccessPointBinding,
    WorkspaceMountError,
)
from cayu.runners import Runner
from cayu.workspaces import RunnerWorkspace


class _MountRunner(Runner):
    isolation = "lambda-microvm"
    default_cwd = "/workspace"

    def __init__(self, *, fail_mount: bool = False) -> None:
        self.calls: list[tuple[ExecCommand, str | None, int | None]] = []
        self.mountpoint_checks = 0
        self.fail_mount = fail_mount

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = None,
    ) -> ExecResult:
        self.calls.append((command, cwd, timeout_s))
        argv = command.argv or []
        if argv[:2] == ["mountpoint", "-q"]:
            self.mountpoint_checks += 1
            # Not mounted before mount; mounted for verification and finalize.
            return ExecResult(exit_code=1 if self.mountpoint_checks == 1 else 0)
        if argv and argv[0] == "mount" and self.fail_mount:
            return ExecResult(exit_code=32, stderr="access point denied")
        return ExecResult()


def _argv_calls(runner: _MountRunner) -> list[list[str]]:
    return [call[0].argv or [] for call in runner.calls]


def test_efs_access_point_binding_mounts_and_returns_runner_workspace() -> None:
    runner = _MountRunner()
    binding = EFSAccessPointBinding(
        file_system_id="fs-0123456789abcdef0",
        access_point_id="fsap-0123456789abcdef0",
        mount_target_ip="10.0.1.25",
        path="/workspace/repo",
    )

    bound = asyncio.run(
        binding.bind(
            None,
            runner,
            session_id="sess_1",
            agent_name="agent",
            environment_name="aws",
            metadata={"request": "kept"},
        )
    )

    assert isinstance(bound.workspace, RunnerWorkspace)
    assert bound.workspace.cwd == "repo"
    assert bound.workspace.id == ("efs:fs-0123456789abcdef0:fsap-0123456789abcdef0:/workspace/repo")
    assert bound.source_workspace is None
    assert bound.runner is runner
    assert bound.path == "/workspace/repo"
    assert bound.metadata == {
        "request": "kept",
        "aws_filesystem_binding": {
            "backend": "efs",
            "file_system_id": "fs-0123456789abcdef0",
            "access_point_id": "fsap-0123456789abcdef0",
            "mount_target_ip": "10.0.1.25",
            "path": "/workspace/repo",
        },
    }
    assert _argv_calls(runner) == [
        ["mkdir", "-p", "--", "/workspace/repo"],
        ["mountpoint", "-q", "--", "/workspace/repo"],
        [
            "mount",
            "-t",
            "efs",
            "-o",
            "tls,iam,accesspoint=fsap-0123456789abcdef0,mounttargetip=10.0.1.25",
            "fs-0123456789abcdef0:/",
            "/workspace/repo",
        ],
        ["mountpoint", "-q", "--", "/workspace/repo"],
    ]


def test_s3_files_access_point_binding_uses_explicit_az_and_region() -> None:
    runner = _MountRunner()
    binding = S3FilesAccessPointBinding(
        file_system_id="fs-0s3files",
        access_point_id="fsap-0s3files",
        mount_target_ip="10.0.2.50",
        availability_zone_id="use1-az2",
        region_name="us-east-1",
        path="/workspace/s3files",
    )

    bound = asyncio.run(binding.bind(None, runner, session_id="sess_s3"))

    mount_call = next(call for call in _argv_calls(runner) if call[0] == "mount")
    assert mount_call == [
        "mount",
        "-t",
        "s3files",
        "-o",
        ("accesspoint=fsap-0s3files,mounttargetip=10.0.2.50,azid=use1-az2,region=us-east-1"),
        "fs-0s3files:/",
        "/workspace/s3files",
    ]
    assert bound.metadata["aws_filesystem_binding"]["backend"] == "s3files"


def test_access_point_binding_syncs_and_unmounts_on_finalize() -> None:
    runner = _MountRunner()
    binding = EFSAccessPointBinding(
        file_system_id="fs-1",
        access_point_id="fsap-1",
        mount_target_ip="10.0.0.10",
    )

    async def run() -> None:
        bound = await binding.bind(None, runner, session_id="sess_1")
        snapshot = await binding.finalize(bound, outcome="completed")
        assert snapshot is None

    asyncio.run(run())

    assert _argv_calls(runner)[-3:] == [
        ["sync", "-f", "/workspace"],
        ["mountpoint", "-q", "--", "/workspace"],
        ["umount", "--", "/workspace"],
    ]


def test_access_point_binding_fails_closed_when_mount_fails() -> None:
    runner = _MountRunner(fail_mount=True)
    binding = EFSAccessPointBinding(
        file_system_id="fs-1",
        access_point_id="fsap-1",
        mount_target_ip="10.0.0.10",
    )

    with pytest.raises(WorkspaceMountError, match="access point denied"):
        asyncio.run(binding.bind(None, runner, session_id="sess_1"))


def test_access_point_binding_rejects_invalid_or_foreign_finalize_input() -> None:
    with pytest.raises(ValueError, match="absolute"):
        EFSAccessPointBinding(
            file_system_id="fs-1",
            access_point_id="fsap-1",
            mount_target_ip="10.0.0.10",
            path="relative",
        )
    with pytest.raises(ValueError, match="IPv4"):
        EFSAccessPointBinding(
            file_system_id="fs-1",
            access_point_id="fsap-1",
            mount_target_ip="not-an-ip",
        )

    runner = _MountRunner()
    binding = EFSAccessPointBinding(
        file_system_id="fs-1",
        access_point_id="fsap-1",
        mount_target_ip="10.0.0.10",
    )
    foreign = BoundWorkspace(runner=runner, path="/tmp", metadata={})

    with pytest.raises(ValueError, match="does not belong"):
        asyncio.run(binding.finalize(foreign, outcome="completed"))
