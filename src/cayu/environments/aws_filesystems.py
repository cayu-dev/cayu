from __future__ import annotations

import ipaddress
import posixpath
from typing import Any

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.environments.bindings import (
    BoundWorkspace,
    WorkspaceBinding,
    WorkspaceSnapshot,
    _validate_bind_request,
    _validate_finalize_request,
)
from cayu.runners import ExecCommand, ExecResult, Runner
from cayu.workspaces import RunnerWorkspace, Workspace

_COMMAND_OUTPUT_LIMIT_BYTES = 64 * 1024


class WorkspaceMountError(RuntimeError):
    """A mounted workspace could not be established or released safely."""


class _AccessPointBinding(WorkspaceBinding):
    backend: str
    filesystem_type: str

    def __init__(
        self,
        *,
        file_system_id: str,
        access_point_id: str,
        mount_target_ip: str,
        path: str,
        workspace_id: str | None,
        mount_timeout_s: int,
        unmount_timeout_s: int,
    ) -> None:
        self.file_system_id = _mount_option_value(file_system_id, "file_system_id")
        self.access_point_id = _mount_option_value(access_point_id, "access_point_id")
        self.mount_target_ip = _ipv4_address(mount_target_ip)
        self.path = _absolute_guest_path(path)
        self.workspace_id = (
            require_clean_nonblank(workspace_id, "workspace_id")
            if workspace_id is not None
            else f"{self.backend}:{self.file_system_id}:{self.access_point_id}:{self.path}"
        )
        self.mount_timeout_s = _positive_int(mount_timeout_s, "mount_timeout_s")
        self.unmount_timeout_s = _positive_int(unmount_timeout_s, "unmount_timeout_s")

    async def bind(
        self,
        workspace: Workspace | None,
        runner: Runner | None,
        *,
        session_id: str,
        agent_name: str | None = None,
        environment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BoundWorkspace:
        copied_metadata = _validate_bind_request(
            workspace,
            runner,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata=metadata,
        )
        if runner is None:
            raise ValueError(f"{type(self).__name__} requires a runner.")

        await self._required(
            runner,
            ExecCommand.process("mkdir", "-p", "--", self.path),
            action="create mount path",
            timeout_s=self.mount_timeout_s,
        )
        if await self._is_mounted(runner):
            await self._required(
                runner,
                ExecCommand.process("umount", "--", self.path),
                action="replace existing mount",
                timeout_s=self.unmount_timeout_s,
            )
        await self._required(
            runner,
            self._mount_command(),
            action="mount workspace",
            timeout_s=self.mount_timeout_s,
        )
        if not await self._is_mounted(runner):
            await self._best_effort_unmount(runner)
            raise WorkspaceMountError(
                f"{type(self).__name__} mount command succeeded but {self.path} is not mounted."
            )

        binding_metadata = self._binding_metadata()
        copied_metadata["aws_filesystem_binding"] = binding_metadata
        runner_root = runner.resolve_cwd()
        if self.path == runner_root:
            workspace_cwd = None
        elif self.path.startswith(f"{runner_root.rstrip('/')}/"):
            workspace_cwd = posixpath.relpath(self.path, runner_root)
        else:
            await self._best_effort_unmount(runner)
            raise WorkspaceMountError(
                f"Mounted path {self.path} is outside runner root {runner_root}."
            )
        mounted_workspace = RunnerWorkspace(
            runner,
            cwd=workspace_cwd,
            workspace_id=self.workspace_id,
        )
        return BoundWorkspace(
            workspace=mounted_workspace,
            source_workspace=workspace,
            runner=runner,
            path=self.path,
            metadata=copied_metadata,
        )

    async def finalize(
        self,
        bound: BoundWorkspace,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot | None:
        _validate_finalize_request(bound, outcome=outcome, metadata=metadata)
        if (
            bound.runner is None
            or bound.path != self.path
            or bound.metadata.get("aws_filesystem_binding") != self._binding_metadata()
        ):
            raise ValueError(f"Bound workspace does not belong to {type(self).__name__}.")

        sync_error: BaseException | None = None
        try:
            await self._required(
                bound.runner,
                ExecCommand.process("sync", "-f", self.path),
                action="flush workspace",
                timeout_s=self.unmount_timeout_s,
            )
        except BaseException as exc:
            sync_error = exc
        if await self._is_mounted(bound.runner):
            await self._required(
                bound.runner,
                ExecCommand.process("umount", "--", self.path),
                action="unmount workspace",
                timeout_s=self.unmount_timeout_s,
            )
        if sync_error is not None:
            raise sync_error
        return None

    async def _is_mounted(self, runner: Runner) -> bool:
        result = await runner.exec_system(
            ExecCommand.process("mountpoint", "-q", "--", self.path),
            timeout_s=self.mount_timeout_s,
            output_limit_bytes=_COMMAND_OUTPUT_LIMIT_BYTES,
        )
        return result.exit_code == 0 and not result.timed_out

    async def _required(
        self,
        runner: Runner,
        command: ExecCommand,
        *,
        action: str,
        timeout_s: int,
    ) -> ExecResult:
        result = await runner.exec_system(
            command,
            timeout_s=timeout_s,
            output_limit_bytes=_COMMAND_OUTPUT_LIMIT_BYTES,
        )
        if result.exit_code == 0 and not result.timed_out and not result.cancelled:
            return result
        detail = "\n".join(part.strip() for part in (result.stderr, result.stdout) if part.strip())[
            :1000
        ]
        suffix = f": {detail}" if detail else ""
        raise WorkspaceMountError(
            f"{type(self).__name__} could not {action} "
            f"(exit {result.exit_code}, timed_out={result.timed_out}){suffix}"
        )

    async def _best_effort_unmount(self, runner: Runner) -> None:
        try:
            await runner.exec_system(
                ExecCommand.process("umount", "--", self.path),
                timeout_s=self.unmount_timeout_s,
                output_limit_bytes=_COMMAND_OUTPUT_LIMIT_BYTES,
            )
        except Exception:
            return

    def _binding_metadata(self) -> dict[str, str]:
        return {
            "backend": self.backend,
            "file_system_id": self.file_system_id,
            "access_point_id": self.access_point_id,
            "mount_target_ip": self.mount_target_ip,
            "path": self.path,
        }

    def _mount_command(self) -> ExecCommand:
        raise NotImplementedError


class EFSAccessPointBinding(_AccessPointBinding):
    """Mount an IAM/access-point-scoped EFS workspace into a runner."""

    backend = "efs"
    filesystem_type = "efs"

    def __init__(
        self,
        *,
        file_system_id: str,
        access_point_id: str,
        mount_target_ip: str,
        path: str = "/workspace",
        workspace_id: str | None = None,
        mount_timeout_s: int = 45,
        unmount_timeout_s: int = 30,
    ) -> None:
        super().__init__(
            file_system_id=file_system_id,
            access_point_id=access_point_id,
            mount_target_ip=mount_target_ip,
            path=path,
            workspace_id=workspace_id,
            mount_timeout_s=mount_timeout_s,
            unmount_timeout_s=unmount_timeout_s,
        )

    def _mount_command(self) -> ExecCommand:
        options = f"tls,iam,accesspoint={self.access_point_id},mounttargetip={self.mount_target_ip}"
        return ExecCommand.process(
            "mount",
            "-t",
            self.filesystem_type,
            "-o",
            options,
            f"{self.file_system_id}:/",
            self.path,
        )


class S3FilesAccessPointBinding(_AccessPointBinding):
    """Mount an access-point-scoped Amazon S3 File System workspace."""

    backend = "s3files"
    filesystem_type = "s3files"

    def __init__(
        self,
        *,
        file_system_id: str,
        access_point_id: str,
        mount_target_ip: str,
        availability_zone_id: str,
        region_name: str,
        path: str = "/workspace",
        workspace_id: str | None = None,
        mount_timeout_s: int = 45,
        unmount_timeout_s: int = 30,
    ) -> None:
        super().__init__(
            file_system_id=file_system_id,
            access_point_id=access_point_id,
            mount_target_ip=mount_target_ip,
            path=path,
            workspace_id=workspace_id,
            mount_timeout_s=mount_timeout_s,
            unmount_timeout_s=unmount_timeout_s,
        )
        self.availability_zone_id = _mount_option_value(
            availability_zone_id, "availability_zone_id"
        )
        self.region_name = _mount_option_value(region_name, "region_name")

    def _binding_metadata(self) -> dict[str, str]:
        metadata = super()._binding_metadata()
        metadata["availability_zone_id"] = self.availability_zone_id
        metadata["region_name"] = self.region_name
        return copy_json_value(metadata, "metadata")

    def _mount_command(self) -> ExecCommand:
        options = (
            f"accesspoint={self.access_point_id},mounttargetip={self.mount_target_ip},"
            f"azid={self.availability_zone_id},region={self.region_name}"
        )
        return ExecCommand.process(
            "mount",
            "-t",
            self.filesystem_type,
            "-o",
            options,
            f"{self.file_system_id}:/",
            self.path,
        )


def _mount_option_value(value: str, field_name: str) -> str:
    cleaned = require_clean_nonblank(value, field_name)
    if any(character in cleaned for character in (",", "=", "\x00", "\n", "\r")):
        raise ValueError(f"{field_name} contains characters invalid in a mount option.")
    return cleaned


def _ipv4_address(value: str) -> str:
    try:
        address = ipaddress.ip_address(require_clean_nonblank(value, "mount_target_ip"))
    except ValueError as exc:
        raise ValueError("mount_target_ip must be an IPv4 address.") from exc
    if not isinstance(address, ipaddress.IPv4Address):
        raise ValueError("mount_target_ip must be an IPv4 address.")
    return str(address)


def _absolute_guest_path(value: str) -> str:
    cleaned = require_clean_nonblank(value, "path")
    normalized = posixpath.normpath(cleaned)
    if not posixpath.isabs(cleaned) or normalized != cleaned or cleaned == "/":
        raise ValueError("path must be a normalized absolute guest path below root.")
    return cleaned


def _positive_int(value: int, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return value
