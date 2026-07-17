"""Explicit, bounded deployment-readiness checks for Cayu tools."""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from enum import StrEnum
from math import isfinite
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, StrictBool, field_validator, model_validator

from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank
from cayu.core.tools import ToolContext, ToolEffect, ToolResult
from cayu.runtime.app import CayuApp
from cayu.workspaces import LocalWorkspace

_DEFAULT_MAX_FILES = 1_000
_DEFAULT_MAX_ENTRIES = 2_000
_DEFAULT_MAX_FILE_BYTES = 16 * 1024 * 1024
_DEFAULT_MAX_TOTAL_BYTES = 64 * 1024 * 1024
_DEFAULT_TIMEOUT_SECONDS = 30.0
_HASH_CHUNK_BYTES = 64 * 1024

_BOUNDARY_NAME = "isolated_workspace"
_BASE_UNOBSERVED_SYSTEMS = (
    "artifact_stores",
    "databases_outside_workspace",
    "host_filesystem_outside_workspace",
    "network_and_external_services",
    "process_and_tool_instance_state",
    "runner_execution",
)
_LIMITATIONS = (
    "Evidence covers only files visible through the isolated workspace supplied to the tool.",
    "Empty directories, symlinks, non-regular entries, permissions, timestamps, and other filesystem metadata are not observed as mutations.",
    "The tool runs in the current Python process; this is an observation boundary, not a security sandbox.",
    "Execution deadlines use cooperative asyncio cancellation; a hard stop requires a killable process boundary.",
    "Tool policy, approvals, hooks, events, and the model loop are not evaluated.",
    "Work scheduled by the tool after run() returns is outside the before/after snapshot.",
    "No observed mutation is scoped evidence, not universal proof of purity.",
)


class ToolEffectVerificationStatus(StrEnum):
    """Outcome of one scoped tool-effect verification run."""

    CONSISTENT = "consistent"
    MISMATCH = "mismatch"
    OBSERVED = "observed"
    EXECUTION_FAILED = "execution_failed"


class ToolEffectVerification(BaseModel):
    """Content-free evidence from one isolated-workspace tool invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    status: ToolEffectVerificationStatus
    agent_name: str
    tool_name: str
    declared_effect: ToolEffect
    observation_boundary: Literal["isolated_workspace"] = _BOUNDARY_NAME
    created_paths: tuple[str, ...] = ()
    updated_paths: tuple[str, ...] = ()
    deleted_paths: tuple[str, ...] = ()
    observed_mutation: StrictBool
    execution_succeeded: StrictBool
    result_is_error: StrictBool | None = None
    exception_type: str | None = None
    timeout_seconds: float
    workspace_max_entries: int
    workspace_max_files: int
    workspace_max_file_bytes: int
    workspace_max_total_bytes: int
    unobserved_systems: tuple[str, ...]
    limitations: tuple[str, ...] = _LIMITATIONS

    @field_validator("agent_name", "tool_name")
    @classmethod
    def validate_name(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("created_paths", "updated_paths", "deleted_paths")
    @classmethod
    def normalize_paths(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        paths = tuple(require_nonblank(path, info.field_name) for path in value)
        if len(paths) != len(set(paths)):
            raise ValueError(f"{info.field_name} entries must be unique")
        return tuple(sorted(paths))

    @field_validator("unobserved_systems")
    @classmethod
    def normalize_unobserved_systems(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        systems = tuple(require_clean_nonblank(item, "unobserved_systems") for item in value)
        if len(systems) != len(set(systems)):
            raise ValueError("unobserved_systems entries must be unique")
        return tuple(sorted(systems))

    @field_validator("exception_type")
    @classmethod
    def validate_exception_type(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, "exception_type")

    @model_validator(mode="after")
    def validate_evidence(self) -> ToolEffectVerification:
        changed = bool(self.created_paths or self.updated_paths or self.deleted_paths)
        if self.observed_mutation is not changed:
            raise ValueError("observed_mutation must match the reported workspace changes")
        if self.execution_succeeded:
            if self.result_is_error is not False or self.exception_type is not None:
                raise ValueError("successful execution requires a non-error ToolResult")
        else:
            has_error_result = self.result_is_error is True
            has_exception = self.exception_type is not None
            if has_error_result == has_exception:
                raise ValueError(
                    "failed execution requires either an error ToolResult or an exception type"
                )
        _positive_float(self.timeout_seconds, "timeout_seconds")
        _positive_int(self.workspace_max_entries, "workspace_max_entries")
        _positive_int(self.workspace_max_files, "workspace_max_files")
        _positive_int(self.workspace_max_file_bytes, "workspace_max_file_bytes")
        _positive_int(self.workspace_max_total_bytes, "workspace_max_total_bytes")

        if self.status is ToolEffectVerificationStatus.CONSISTENT:
            valid = (
                self.declared_effect is ToolEffect.NONE
                and self.execution_succeeded
                and not self.observed_mutation
            )
        elif self.status is ToolEffectVerificationStatus.MISMATCH:
            valid = self.declared_effect is ToolEffect.NONE and self.observed_mutation
        elif self.status is ToolEffectVerificationStatus.OBSERVED:
            valid = self.declared_effect is not ToolEffect.NONE and self.execution_succeeded
        else:
            valid = not self.execution_succeeded and not (
                self.declared_effect is ToolEffect.NONE and self.observed_mutation
            )
        if not valid:
            raise ValueError("status does not match the verification evidence")
        return self


async def verify_tool_effect(
    app: CayuApp,
    *,
    agent_name: str,
    tool_name: str,
    arguments: Mapping[str, Any],
    workspace_files: Mapping[str, bytes] | None = None,
    unobserved_systems: Iterable[str] = (),
    session_id: str = "tool-effect-verification",
    idempotency_key: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    allow_effectful_execution: bool = False,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    max_entries: int = _DEFAULT_MAX_ENTRIES,
    max_files: int = _DEFAULT_MAX_FILES,
    max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
    max_total_bytes: int = _DEFAULT_MAX_TOTAL_BYTES,
) -> ToolEffectVerification:
    """Invoke one registered tool against a bounded, isolated temporary workspace.

    ``NONE`` receives a scoped consistency verdict. ``IDEMPOTENT`` and
    ``EXTERNAL`` require ``allow_effectful_execution=True`` and are observed
    once without a replay-safety verdict. The tool runs directly: runtime
    policy, hooks, events, and the model loop are intentionally outside this
    deployment-readiness seam. One cooperative asyncio deadline covers seeding,
    both workspace snapshots, tool execution, and cleanup. Expiration raises
    ``TimeoutError`` without returning a verdict. A blocking tool or filesystem
    operation can delay that failure; a hard stop requires a process boundary.
    """

    verification_task = asyncio.current_task()
    if verification_task is None:
        raise RuntimeError("verify_tool_effect must run in an asyncio task")
    initial_cancellation_requests = verification_task.cancelling()
    if not isinstance(app, CayuApp):
        raise TypeError("app must be a CayuApp")
    agent_name = require_clean_nonblank(agent_name, "agent_name")
    tool_name = require_clean_nonblank(tool_name, "tool_name")
    if not isinstance(arguments, Mapping):
        raise TypeError("arguments must be a mapping")
    copied_arguments = copy_json_value(dict(arguments), "arguments")
    copied_metadata = _copy_metadata(metadata)
    seeded_files = _copy_workspace_files(workspace_files)
    declared_unobserved = _copy_names(unobserved_systems, "unobserved_systems")
    session_id = require_clean_nonblank(session_id, "session_id")
    if idempotency_key is not None:
        idempotency_key = require_clean_nonblank(idempotency_key, "idempotency_key")
    if type(allow_effectful_execution) is not bool:
        raise TypeError("allow_effectful_execution must be a bool")
    timeout_seconds = _positive_float(timeout_seconds, "timeout_seconds")
    max_entries = _positive_int(max_entries, "max_entries")
    max_files = _positive_int(max_files, "max_files")
    max_file_bytes = _positive_int(max_file_bytes, "max_file_bytes")
    max_total_bytes = _positive_int(max_total_bytes, "max_total_bytes")
    _validate_seed_bounds(
        seeded_files,
        max_files=max_files,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
    )

    registered_agent = app.get_agent(agent_name)
    try:
        registered_tool = registered_agent.tools[tool_name]
    except KeyError as exc:
        raise KeyError(f"Tool not registered for agent {agent_name}: {tool_name}") from exc
    effect = registered_tool.effect
    if effect is not ToolEffect.NONE and not allow_effectful_execution:
        raise ValueError(
            "IDEMPOTENT and EXTERNAL tools require allow_effectful_execution=True; "
            "the verifier executes them once and does not claim replay safety"
        )

    effective_metadata = {
        **copied_metadata,
        "tool_call_id": "tool-effect-verification",
        "tool_effect": effect.value,
    }
    if idempotency_key is not None:
        effective_metadata["idempotency_key"] = idempotency_key

    result_is_error: bool | None = None
    exception_type: str | None = None
    result: object | None = None
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    async with asyncio.timeout_at(deadline):
        with tempfile.TemporaryDirectory(prefix="cayu-tool-effect-") as directory:
            workspace = LocalWorkspace(directory, workspace_id=_BOUNDARY_NAME)
            await _seed_workspace(
                workspace,
                seeded_files,
                clock=loop.time,
                deadline=deadline,
            )
            before = _capture_workspace(
                workspace,
                max_entries=max_entries,
                max_files=max_files,
                max_file_bytes=max_file_bytes,
                max_total_bytes=max_total_bytes,
                clock=loop.time,
                deadline=deadline,
            )
            context = ToolContext(
                session_id=session_id,
                agent_name=agent_name,
                environment_name=_BOUNDARY_NAME,
                workspace_id=workspace.id,
                idempotency_key=idempotency_key,
                workspace=workspace,
                metadata=effective_metadata,
            )
            tool_task = asyncio.create_task(
                registered_tool.tool.run(context, copied_arguments),
                name=f"cayu-tool-effect-verification:{agent_name}:{tool_name}",
            )
            try:
                result = await tool_task
            except Exception as exc:
                exception_type = type(exc).__name__
            if verification_task.cancelling() > initial_cancellation_requests:
                raise asyncio.CancelledError
            _raise_if_deadline_exceeded(loop.time, deadline)
            if exception_type is None:
                if type(result) is ToolResult:
                    result_is_error = result.is_error
                else:
                    exception_type = "InvalidToolResult"
            after = _capture_workspace(
                workspace,
                max_entries=max_entries,
                max_files=max_files,
                max_file_bytes=max_file_bytes,
                max_total_bytes=max_total_bytes,
                clock=loop.time,
                deadline=deadline,
            )
        _raise_if_deadline_exceeded(loop.time, deadline)

    created, updated, deleted = _compare_snapshots(before, after)
    observed_mutation = bool(created or updated or deleted)
    execution_succeeded = exception_type is None and result_is_error is False
    if effect is ToolEffect.NONE and observed_mutation:
        status = ToolEffectVerificationStatus.MISMATCH
    elif not execution_succeeded:
        status = ToolEffectVerificationStatus.EXECUTION_FAILED
    elif effect is ToolEffect.NONE:
        status = ToolEffectVerificationStatus.CONSISTENT
    else:
        status = ToolEffectVerificationStatus.OBSERVED

    return ToolEffectVerification(
        status=status,
        agent_name=agent_name,
        tool_name=tool_name,
        declared_effect=effect,
        created_paths=created,
        updated_paths=updated,
        deleted_paths=deleted,
        observed_mutation=observed_mutation,
        execution_succeeded=execution_succeeded,
        result_is_error=result_is_error,
        exception_type=exception_type,
        timeout_seconds=timeout_seconds,
        workspace_max_entries=max_entries,
        workspace_max_files=max_files,
        workspace_max_file_bytes=max_file_bytes,
        workspace_max_total_bytes=max_total_bytes,
        unobserved_systems=tuple(sorted(set(_BASE_UNOBSERVED_SYSTEMS) | set(declared_unobserved))),
    )


def _copy_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping")
    return copy_json_value(dict(value), "metadata")


def _copy_workspace_files(value: Mapping[str, bytes] | None) -> tuple[tuple[str, bytes], ...]:
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise TypeError("workspace_files must be a mapping")
    files: list[tuple[str, bytes]] = []
    for path, content in value.items():
        workspace_path = require_nonblank(path, "workspace_files path")
        if type(content) is not bytes:
            raise TypeError("workspace_files values must be bytes")
        files.append((workspace_path, bytes(content)))
    return tuple(sorted(files))


def _copy_names(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, str | bytes):
        raise TypeError(f"{field_name} must be an iterable of strings")
    try:
        names = tuple(require_clean_nonblank(value, field_name) for value in values)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of strings") from exc
    if len(names) != len(set(names)):
        raise ValueError(f"{field_name} entries must be unique")
    return tuple(sorted(names))


def _positive_int(value: int, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return value


def _positive_float(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{field_name} must be a number")
    normalized = float(value)
    if not isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{field_name} must be finite and greater than zero")
    return normalized


def _validate_seed_bounds(
    files: tuple[tuple[str, bytes], ...],
    *,
    max_files: int,
    max_file_bytes: int,
    max_total_bytes: int,
) -> None:
    if len(files) > max_files:
        raise ValueError("isolated workspace observation exceeds max_files")
    total_bytes = 0
    for path, content in files:
        if len(content) > max_file_bytes:
            raise ValueError(f"isolated workspace file exceeds max_file_bytes: {path}")
        total_bytes += len(content)
        if total_bytes > max_total_bytes:
            raise ValueError("isolated workspace observation exceeds max_total_bytes")


async def _seed_workspace(
    workspace: LocalWorkspace,
    files: tuple[tuple[str, bytes], ...],
    *,
    clock: Callable[[], float],
    deadline: float,
) -> None:
    for path, content in files:
        _raise_if_deadline_exceeded(clock, deadline)
        write_task = asyncio.create_task(
            workspace.write_bytes(path, content),
            name="cayu-tool-effect-verification:seed-workspace",
        )
        try:
            await asyncio.shield(write_task)
        except asyncio.CancelledError:
            try:
                await write_task
            finally:
                raise
        _raise_if_deadline_exceeded(clock, deadline)


def _capture_workspace(
    workspace: LocalWorkspace,
    *,
    max_entries: int,
    max_files: int,
    max_file_bytes: int,
    max_total_bytes: int,
    clock: Callable[[], float],
    deadline: float,
) -> dict[str, str]:
    paths = _bounded_workspace_files(
        workspace.root,
        max_entries=max_entries,
        max_files=max_files,
        clock=clock,
        deadline=deadline,
    )
    snapshot: dict[str, str] = {}
    total_bytes = 0
    for path in paths:
        _raise_if_deadline_exceeded(clock, deadline)
        remaining_total_bytes = max_total_bytes - total_bytes
        observed_bytes, digest = _hash_workspace_file(
            workspace.resolve(path),
            relative_path=path,
            max_file_bytes=max_file_bytes,
            remaining_total_bytes=remaining_total_bytes,
            clock=clock,
            deadline=deadline,
        )
        total_bytes += observed_bytes
        snapshot[path] = digest
    return snapshot


def _bounded_workspace_files(
    root: str | os.PathLike[str],
    *,
    max_entries: int,
    max_files: int,
    clock: Callable[[], float],
    deadline: float,
) -> tuple[str, ...]:
    directories: list[tuple[str | os.PathLike[str], str]] = [(root, "")]
    files: list[str] = []
    entry_count = 0
    while directories:
        directory, prefix = directories.pop()
        _raise_if_deadline_exceeded(clock, deadline)
        with os.scandir(directory) as entries:
            for entry in entries:
                entry_count += 1
                if entry_count > max_entries:
                    raise ValueError("isolated workspace observation exceeds max_entries")
                relative_path = f"{prefix}/{entry.name}" if prefix else entry.name
                if entry.is_symlink():
                    _raise_if_deadline_exceeded(clock, deadline)
                    continue
                if entry.is_dir(follow_symlinks=False):
                    directories.append((entry.path, relative_path))
                elif entry.is_file(follow_symlinks=False):
                    files.append(relative_path)
                    if len(files) > max_files:
                        raise ValueError("isolated workspace observation exceeds max_files")
                _raise_if_deadline_exceeded(clock, deadline)
        _raise_if_deadline_exceeded(clock, deadline)
    return tuple(sorted(files))


def _hash_workspace_file(
    path: os.PathLike[str],
    *,
    relative_path: str,
    max_file_bytes: int,
    remaining_total_bytes: int,
    clock: Callable[[], float],
    deadline: float,
) -> tuple[int, str]:
    digest = hashlib.sha256(b"file\0")
    observed_bytes = 0
    with open(path, "rb") as file:
        initial_size = os.fstat(file.fileno()).st_size
        if initial_size > max_file_bytes:
            raise ValueError(f"isolated workspace file exceeds max_file_bytes: {relative_path}")
        if initial_size > remaining_total_bytes:
            raise ValueError("isolated workspace observation exceeds max_total_bytes")
        while True:
            _raise_if_deadline_exceeded(clock, deadline)
            chunk = file.read(
                min(
                    _HASH_CHUNK_BYTES,
                    max_file_bytes - observed_bytes + 1,
                    remaining_total_bytes - observed_bytes + 1,
                )
            )
            if not chunk:
                break
            observed_bytes += len(chunk)
            if observed_bytes > max_file_bytes:
                raise ValueError(f"isolated workspace file exceeds max_file_bytes: {relative_path}")
            if observed_bytes > remaining_total_bytes:
                raise ValueError("isolated workspace observation exceeds max_total_bytes")
            digest.update(chunk)
    _raise_if_deadline_exceeded(clock, deadline)
    return observed_bytes, digest.hexdigest()


def _raise_if_deadline_exceeded(clock: Callable[[], float], deadline: float) -> None:
    if clock() >= deadline:
        raise TimeoutError


def _compare_snapshots(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    created = tuple(sorted(after.keys() - before.keys()))
    deleted = tuple(sorted(before.keys() - after.keys()))
    updated = tuple(
        sorted(path for path in before.keys() & after.keys() if before[path] != after[path])
    )
    return created, updated, deleted


__all__ = [
    "ToolEffectVerification",
    "ToolEffectVerificationStatus",
    "verify_tool_effect",
]
