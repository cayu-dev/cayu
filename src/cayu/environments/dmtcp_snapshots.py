"""Selected-process snapshots on unprivileged Docker and Lambda MicroVM runners."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import posixpath
import tarfile
import time
from contextlib import suppress
from pathlib import Path, PurePosixPath

from cayu._validation import canonical_durable_json_bytes, require_durable_clean_nonblank
from cayu.environments.snapshots import (
    CapturedExecutionSnapshot,
    ExecutionSnapshotAdapter,
    ExecutionSnapshotCapability,
    ExecutionSnapshotError,
    ExecutionSnapshotFidelity,
    ExecutionSnapshotPolicy,
    SnapshotFence,
)
from cayu.runners.base import ExecCommand, Runner

_CHUNK = 1024 * 1024
_SOURCE = Path(__file__).with_name("_snapshot_guest.py").read_text()


def _path(value: str) -> str:
    require_durable_clean_nonblank(value, "snapshot guest path")
    if (
        not value.startswith("/")
        or value != posixpath.normpath(value)
        or value == "/"
        or len(value) > 1024
    ):
        raise ValueError("Snapshot paths must be bounded, normalized absolute guest paths.")
    return value


def validate_snapshot_archive(
    content: bytes, policy: ExecutionSnapshotPolicy, *, process: bool
) -> None:
    """Reject unsafe/incomplete material before dispatching any target mutation."""
    if not 0 < len(content) <= policy.max_artifact_bytes:
        raise ExecutionSnapshotError("Execution snapshot archive exceeds policy.")
    seen = set()
    total = 0
    images = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as archive:
            for entry in archive:
                path = PurePosixPath(entry.name)
                if (
                    not (entry.isfile() or entry.isdir())
                    or path.is_absolute()
                    or not path.parts
                    or path.as_posix() != entry.name
                    or any(part in (".", "..") for part in path.parts)
                    or entry.name in seen
                    or entry.mode & ~0o777
                    or entry.size < 0
                    or (entry.isdir() and entry.size != 0)
                ):
                    raise ExecutionSnapshotError("Unsafe execution snapshot archive.")
                seen.add(entry.name)
                total += entry.size
                if len(seen) > policy.max_files or total > policy.max_artifact_bytes:
                    raise ExecutionSnapshotError(
                        "Execution snapshot archive inventory exceeds policy."
                    )
                if entry.name.endswith(".dmtcp") and len(path.parts) == 1:
                    images += 1
                if process and entry.name.endswith(".temp"):
                    raise ExecutionSnapshotError("Incomplete process checkpoint.")
                if entry.isfile():
                    stream = archive.extractfile(entry)
                    if stream is None or len(stream.read(entry.size + 1)) != entry.size:
                        raise ExecutionSnapshotError("Truncated execution snapshot archive.")
        if process and images != 1:
            raise ExecutionSnapshotError("Selected-process snapshot requires one complete image.")
    except (tarfile.TarError, OSError, ValueError) as error:
        raise ExecutionSnapshotError("Invalid execution snapshot archive.") from error


class DmtcpExecutionSnapshotAdapter(ExecutionSnapshotAdapter):
    """One explicitly launched process (including threads), its files and local sockets.

    The immutable image must install DMTCP 4.2.0 and Cayu's barrier plugin at
    prefix/lib/dmtcp/libcayu_snapshot.so. No package installation or privilege
    escalation occurs. All workload and transfer commands use Runner.exec's
    agent profile. Fork trees, PTYs, browsers and live external TCP connections
    are excluded from this initial fidelity. The workspace must be exclusively
    owned and no unmanaged/background writer may run while snapshotting.
    """

    def __init__(self, runner, request, capability, allocation_sha256):
        self.runner = runner
        self._request = request
        self._capability = capability
        self._allocation_sha256 = allocation_sha256
        self._observed = None

    @property
    def capability(self) -> ExecutionSnapshotCapability:
        return self._capability

    @property
    def allocation_sha256(self) -> str:
        return self._allocation_sha256

    @classmethod
    async def create(
        cls,
        runner: Runner,
        *,
        workspace_path: str = "/workspace",
        workload_id: str,
        prefix: str = "/opt/cayu-dmtcp",
        coordinator_port: int = 7779,
    ) -> DmtcpExecutionSnapshotAdapter:
        from cayu.runners.aws_lambda_microvm import LambdaMicroVMRunner
        from cayu.runners.docker import DockerRunner

        if not isinstance(runner, (DockerRunner, LambdaMicroVMRunner)):
            raise ExecutionSnapshotError(
                "DMTCP snapshots require a maintained Docker or AWS runner."
            )
        workload_id = require_durable_clean_nonblank(workload_id, "workload_id")
        if (
            len(workload_id.encode()) > 128
            or type(coordinator_port) is not int
            or not 1024 <= coordinator_port <= 65535
        ):
            raise ValueError("Invalid snapshot workload identity or coordinator port.")
        request = {
            "workspace": _path(workspace_path),
            "prefix": _path(prefix),
            "port": coordinator_port,
            "root": "/tmp/cayu-snapshot-" + hashlib.sha256(workload_id.encode()).hexdigest()[:32],
        }
        allocation = hashlib.sha256(
            canonical_durable_json_bytes(list(runner.resource_key), "snapshot_allocation")
        ).hexdigest()
        adapter = cls(runner, request, ExecutionSnapshotCapability(), allocation)
        observed = await adapter._guest("inspect")
        adapter._observed = observed
        if isinstance(runner, LambdaMicroVMRunner):
            if runner.image_identifier is None or runner.image_version is None:
                raise ExecutionSnapshotError(
                    "AWS snapshots require exact image and version identity."
                )
            image = [runner.image_identifier, runner.image_version, runner.region_name]
        else:
            # Inspect the exact container; a mutable image tag cannot establish
            # compatibility with a replacement allocation.
            evidence = await runner.collect_execution_admission_candidate()
            image = evidence.evidence.image_fingerprint
            if image is None:
                raise ExecutionSnapshotError(
                    "Docker snapshots require observed immutable image identity."
                )
        identity = hashlib.sha256(
            canonical_durable_json_bytes([request, observed, image], "snapshot_compatibility")
        ).hexdigest()
        adapter._capability = ExecutionSnapshotCapability(
            fidelity=ExecutionSnapshotFidelity.SELECTED_PROCESSES,
            adapter="dmtcp",
            adapter_version="1",
            snapshot_format="dmtcp-4.2.0+cayu-barrier-1",
            compatibility_sha256=identity,
            requires_managed_launch=True,
        )
        return adapter

    async def _guest(self, operation: str, *, fence: SnapshotFence | None = None, **payload):
        if fence is not None:
            await fence()
        result = await self.runner.exec(
            ExecCommand.process("python3", "-I", "-c", _SOURCE),
            stdin=json.dumps({**self._request, "operation": operation, **payload}),
            timeout_s=120,
            output_limit_bytes=2 * _CHUNK,
        )
        if (
            result.exit_code
            or result.timed_out
            or result.stdout_truncated
            or result.stderr_truncated
        ):
            errno = None
            with suppress(ValueError, AttributeError):
                errno = json.loads(result.stdout).get("errno")
            code = f" (errno {errno})" if type(errno) is int and 0 <= errno <= 4096 else ""
            raise ExecutionSnapshotError(
                f"Execution snapshot guest operation failed during {operation}{code}."
            )
        try:
            response = json.loads(result.stdout)
            if type(response) is not dict or "error" in response:
                raise ValueError
            return response
        except (ValueError, TypeError) as error:
            raise ExecutionSnapshotError(
                "Invalid execution snapshot guest acknowledgement."
            ) from error

    async def launch(self, command: ExecCommand) -> None:
        """Launch an application-owned checkpointable workload on the agent lane.

        Invoke before normal tool use. This does not wrap/replay arbitrary
        already-running commands or resolve their external effects.
        """
        if command.argv is None:
            raise ValueError("Snapshot-managed launch requires an explicit argument vector.")
        await self._guest("launch", argv=list(command.argv))

    async def _ready(self, operation_id: str, kind: str, policy, fence):
        # Polling is read-only; fence once rather than on every poll.
        await fence()
        deadline = time.monotonic() + policy.timeout_seconds
        while time.monotonic() < deadline:
            if (await self._guest("ready", id=operation_id, kind=kind)).get("ready") is True:
                return
            await asyncio.sleep(0.1)
        raise ExecutionSnapshotError("Execution snapshot barrier did not become ready.")

    async def preflight_capture(self, policy) -> None:
        await self._verify_profile(None)
        await self._guest("preflight", **policy.model_dump())

    async def capture(self, operation_id, policy, fence) -> CapturedExecutionSnapshot:
        await self._verify_profile(fence)
        await self._guest("checkpoint", fence=fence, id=operation_id, **policy.model_dump())
        return await self.recover_capture(operation_id, policy, fence)

    async def recover_capture(self, operation_id, policy, fence) -> CapturedExecutionSnapshot:
        await self._verify_profile(fence)
        await self._ready(operation_id, "capture", policy, fence)
        inventory = await self._guest("pack", fence=fence, id=operation_id, **policy.model_dump())
        contents = {}
        for role in ("process", "workspace"):
            size = inventory[role]["size"]
            if type(size) is not int or not 0 < size <= policy.max_artifact_bytes:
                raise ExecutionSnapshotError("Execution snapshot size exceeds policy.")
            blocks = bytearray()
            # Downloads do not mutate the guest; pack was the last fenced step.
            for offset in range(0, size, _CHUNK):
                response = await self._guest(
                    "read", role=role, offset=offset, length=min(_CHUNK, size - offset)
                )
                block = base64.b64decode(response["bytes"], validate=True)
                if len(block) != min(_CHUNK, size - offset):
                    raise ExecutionSnapshotError("Incomplete execution snapshot download.")
                blocks.extend(block)
            content = bytes(blocks)
            if hashlib.sha256(content).hexdigest() != inventory[role]["sha256"]:
                raise ExecutionSnapshotError("Execution snapshot download integrity mismatch.")
            validate_snapshot_archive(content, policy, process=role == "process")
            contents[role] = content
        if sum(len(value) for value in contents.values()) > policy.max_total_bytes:
            raise ExecutionSnapshotError("Execution snapshot exceeds aggregate bound.")
        return CapturedExecutionSnapshot(**contents)

    async def restore(self, operation_id, snapshot, policy, fence) -> None:
        await self._verify_profile(fence)
        validate_snapshot_archive(snapshot.process, policy, process=True)
        validate_snapshot_archive(snapshot.workspace, policy, process=False)
        await self._guest("prepare_restore", fence=fence, id=operation_id)
        hashes = {}
        for role, content in (("process", snapshot.process), ("workspace", snapshot.workspace)):
            hashes[role] = hashlib.sha256(content).hexdigest()
            for offset in range(0, len(content), _CHUNK):
                block = content[offset : offset + _CHUNK]
                result = await self._guest(
                    "write",
                    fence=fence,
                    role=role,
                    offset=offset,
                    bytes=base64.b64encode(block).decode(),
                )
                if result.get("written") != len(block):
                    raise ExecutionSnapshotError("Incomplete execution snapshot upload.")
        await self._guest(
            "restart", fence=fence, id=operation_id, hashes=hashes, **policy.model_dump()
        )
        await self._ready(operation_id, "restore", policy, fence)

    async def activate(self, operation_id, fence) -> None:
        await self._guest("release", fence=fence, id=operation_id, kind="restore")

    async def verify_restore(self, operation_id, policy, fence) -> None:
        await self._verify_profile(fence)
        await self._ready(operation_id, "restore", policy, fence)

    async def resume_source(self, operation_id, fence) -> None:
        await self._guest("release", fence=fence, id=operation_id, kind="capture")

    async def _verify_profile(self, fence):
        if await self._guest("inspect", fence=fence) != self._observed:
            raise ExecutionSnapshotError("Snapshot execution profile or tool bytes changed.")
