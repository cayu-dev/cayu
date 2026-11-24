"""Run one fixed command on some files inside a runner and read back its outputs.

:func:`run_isolated` copies ``inputs`` into the runner's working directory,
runs ``command`` there and reads the declared ``outputs`` back as bytes. It
works on any :class:`~cayu.runners.base.Runner`; which runner you give it
decides the isolation (a network-disabled ``DockerRunner`` locally, a Cayu
Cloud brokered sandbox when deployed). The caller creates and closes the
runner.

Files move as one uncompressed tar in each direction, by the best transport
the runner offers: staged copies when the runner has them (Cayu Cloud's
brokered sandboxes), binary streams (Docker and local runners), and otherwise
the runner's bounded text output. The guest needs ``python3`` for the last two.

:func:`isolated_run_key` is a stable digest of the command and its inputs.
Use it as an idempotency key or a task step name, so a rerun of the same
command on the same files can reuse a saved result.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tarfile
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cayu.runners.base import ExecCommand, ExecResult, Runner

#: Default bound on the stdout and stderr text a run keeps.
ISOLATED_OUTPUT_LIMIT_BYTES = 64 * 1024


class IsolatedLimits(BaseModel):
    """Bounds for one :func:`run_isolated` call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timeout_s: int = Field(default=60, ge=1, le=24 * 60 * 60)
    #: Largest total size of the input files.
    max_input_bytes: int = Field(default=64 * 1024 * 1024, ge=0)
    #: Largest single output file.
    max_output_file_bytes: int = Field(default=64 * 1024 * 1024, ge=1)
    #: Largest total size of the output files.
    max_output_bytes: int = Field(default=64 * 1024 * 1024, ge=1)
    #: Bound on the stdout and stderr text kept in the result.
    output_limit_bytes: int = Field(default=ISOLATED_OUTPUT_LIMIT_BYTES, ge=1)


class IsolatedRun(BaseModel):
    """What one :func:`run_isolated` call produced.

    ``outputs`` holds the declared outputs the command wrote; ``missing`` names
    the ones it did not. Outputs are read only when the command exited 0 and
    did not time out or get cancelled; otherwise both are empty.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    key: str
    result: ExecResult
    outputs: dict[str, bytes] = Field(default_factory=dict)
    missing: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        """True when the command exited 0 and wrote every declared output."""

        return _finished(self.result) and not self.missing


def isolated_run_key(command: ExecCommand, inputs: Mapping[str, bytes]) -> str:
    """Return a stable SHA-256 hex digest of ``command`` and ``inputs``.

    Equal commands on equal files give equal keys, whatever order the inputs
    are given in.
    """

    if not isinstance(command, ExecCommand):
        raise TypeError("command must be an ExecCommand.")
    files = _validated_inputs(inputs, max_bytes=None)
    material = {
        "command": command.model_dump(mode="json"),
        "inputs": {path: hashlib.sha256(data).hexdigest() for path, data in files.items()},
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


async def run_isolated(
    runner: Runner,
    command: ExecCommand,
    *,
    inputs: Mapping[str, bytes],
    outputs: Sequence[str],
    limits: IsolatedLimits | None = None,
    env: Mapping[str, str] | None = None,
) -> IsolatedRun:
    """Copy ``inputs`` into ``runner``, run ``command`` and read ``outputs`` back.

    Paths are relative to the runner's working directory, use ``/`` and never
    ``..``. ``inputs`` maps paths to file contents; ``outputs`` lists the paths
    to read once the command exits 0. A nonzero exit, a timeout or a
    cancellation returns the result without outputs, so check
    :attr:`IsolatedRun.succeeded` or ``result``.
    """

    if not isinstance(runner, Runner):
        raise TypeError("runner must be a Runner.")
    if not isinstance(command, ExecCommand):
        raise TypeError("command must be an ExecCommand.")
    limits = IsolatedLimits() if limits is None else limits
    if type(limits) is not IsolatedLimits:
        raise TypeError("limits must be IsolatedLimits.")
    files = _validated_inputs(inputs, max_bytes=limits.max_input_bytes)
    wanted = _validated_outputs(outputs)
    key = isolated_run_key(command, files)
    transfer = _transfer_for(runner)
    # A reused runner may still hold an earlier run's outputs; never return them.
    await _clear_outputs(runner, [path for path in wanted if path not in files])
    if files:
        await transfer.put(_tar(files))
    result = await runner.exec(
        command,
        env=None if env is None else dict(env),
        timeout_s=limits.timeout_s,
        output_limit_bytes=limits.output_limit_bytes,
    )
    if not _finished(result) or not wanted:
        return IsolatedRun(key=key, result=result)
    archive = await transfer.get(wanted, limits)
    found = _untar(archive, wanted, limits)
    missing = tuple(path for path in wanted if path not in found)
    return IsolatedRun(key=key, result=result, outputs=found, missing=missing)


def _finished(result: ExecResult) -> bool:
    return result.exit_code == 0 and not result.timed_out and not result.cancelled


def _validated_path(path: object, field_name: str) -> str:
    if type(path) is not str or not path or path != path.strip():
        raise ValueError(f"{field_name} paths must be nonblank strings.")
    parts = path.split("/")
    if path.startswith("/") or "\\" in path or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"{field_name} path {path!r} must be relative, without '.' or '..'.")
    if any(ord(ch) < 32 for ch in path):
        raise ValueError(f"{field_name} path {path!r} must not contain control characters.")
    return path


def _validated_inputs(inputs: object, *, max_bytes: int | None) -> dict[str, bytes]:
    if not isinstance(inputs, Mapping):
        raise TypeError("inputs must map paths to bytes.")
    files: dict[str, bytes] = {}
    for path, data in inputs.items():
        path = _validated_path(path, "inputs")
        if type(data) is not bytes:
            raise TypeError(f"inputs[{path!r}] must be bytes.")
        files[path] = data
    if max_bytes is not None and sum(len(data) for data in files.values()) > max_bytes:
        raise ValueError(f"inputs exceed max_input_bytes={max_bytes}.")
    return dict(sorted(files.items()))


def _validated_outputs(outputs: object) -> tuple[str, ...]:
    if isinstance(outputs, (str, bytes)) or not isinstance(outputs, Sequence):
        raise TypeError("outputs must be a sequence of paths.")
    wanted = tuple(_validated_path(path, "outputs") for path in outputs)
    if len(set(wanted)) != len(wanted):
        raise ValueError("outputs must not repeat a path.")
    return wanted


def _tar(files: Mapping[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path, data in files.items():
            member = tarfile.TarInfo(path)
            member.size, member.mode = len(data), 0o644
            archive.addfile(member, io.BytesIO(data))
    return buffer.getvalue()


def _untar(data: bytes, wanted: Sequence[str], limits: IsolatedLimits) -> dict[str, bytes]:
    """Read only the requested regular files; nothing is extracted on the host."""

    found: dict[str, bytes] = {}
    total = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if member.isdir() and any(path.startswith(name.rstrip("/") + "/") for path in wanted):
                continue
            if name not in wanted or name in found or not member.isfile():
                raise ValueError(f"The runner returned an unexpected archive entry {name!r}.")
            if member.size > limits.max_output_file_bytes:
                raise ValueError(f"Output {name!r} exceeds max_output_file_bytes.")
            total += member.size
            if total > limits.max_output_bytes:
                raise ValueError("Outputs exceed max_output_bytes.")
            extracted = archive.extractfile(member)
            if extracted is None:  # pragma: no cover - isfile() members always extract
                raise ValueError(f"Output {name!r} could not be read.")
            found[name] = extracted.read()
    return found


class _Transfer:
    async def put(self, archive: bytes) -> None:
        raise NotImplementedError

    async def get(self, paths: Sequence[str], limits: IsolatedLimits) -> bytes:
        raise NotImplementedError


class _StagedTransfer(_Transfer):
    """Staged copies through the runner's own storage, for sandboxes that offer them."""

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    async def put(self, archive: bytes) -> None:
        digest = hashlib.sha256(archive).hexdigest()
        await self._runner.copy_in(io.BytesIO(archive), size=len(archive), sha256=digest)

    async def get(self, paths: Sequence[str], limits: IsolatedLimits) -> bytes:
        sink = io.BytesIO()
        await self._runner.copy_out(sink, max_bytes=_archive_bound(paths, limits), paths=paths)
        return sink.getvalue()


class _WorkspaceTransfer(_Transfer):
    """Guarded workspace tar operations, streamed when the runner supports it."""

    def __init__(self, runner: Runner) -> None:
        from cayu.workspaces.runner import RunnerWorkspace

        self._runner = runner
        self._workspace = RunnerWorkspace(runner)

    async def put(self, archive: bytes) -> None:
        if self._runner.binary_stream_capability() is not None:
            await self._workspace.write_tar_stream(io.BytesIO(archive), archive_bytes=len(archive))
        else:
            await self._workspace.write_tar_bytes(archive)

    async def get(self, paths: Sequence[str], limits: IsolatedLimits) -> bytes:
        present = [path for path in paths if await self._exists(path)]
        if not present:
            return _tar({})
        bounds = {
            "max_file_bytes": limits.max_output_file_bytes,
            "max_total_bytes": limits.max_output_bytes,
            "max_archive_bytes": _archive_bound(present, limits),
        }
        if self._runner.binary_stream_capability() is not None:
            sink = io.BytesIO()
            await self._workspace.read_tar_stream(present, sink, **bounds)
            return sink.getvalue()
        return await self._workspace.read_tar_bytes(present, **bounds)

    async def _exists(self, path: str) -> bool:
        try:
            await self._workspace.read_bytes(path, max_bytes=1)
        except (FileNotFoundError, IsADirectoryError):
            return False
        return True


def _archive_bound(paths: Sequence[str], limits: IsolatedLimits) -> int:
    from cayu.workspaces._tar import tar_archive_size_bound

    return tar_archive_size_bound(limits.max_output_bytes, paths)


async def _clear_outputs(runner: Runner, paths: Sequence[str]) -> None:
    if not paths:
        return
    from cayu.workspaces.runner import RunnerWorkspace

    workspace = RunnerWorkspace(runner)
    for path in paths:
        with contextlib.suppress(FileNotFoundError):
            await workspace.delete(path)


def _transfer_for(runner: Runner) -> _Transfer:
    staged = getattr(runner, "staged_copies", None)
    if (
        staged is not None
        and callable(getattr(runner, "copy_in", None))
        and callable(getattr(runner, "copy_out", None))
    ):
        return _StagedTransfer(runner)
    return _WorkspaceTransfer(runner)


__all__ = [
    "ISOLATED_OUTPUT_LIMIT_BYTES",
    "IsolatedLimits",
    "IsolatedRun",
    "isolated_run_key",
    "run_isolated",
]
