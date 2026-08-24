"""Cross-process binding authority used only by the gated remote proof."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

from cayu.workspaces import (
    WorkspaceBranchAuthority,
    WorkspaceBranchBindingAuthority,
    WorkspaceBranchBindingAuthorityClaimScope,
    WorkspaceBranchOperationConflict,
)


class _FileAuthorityClaim:
    def __init__(self, descriptor: int) -> None:
        self._descriptor = descriptor

    def release(self) -> None:
        descriptor = self._descriptor
        if descriptor < 0:
            return
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        self._descriptor = -1


class FileWorkspaceBranchAuthorityProvider:
    """Small POSIX proof provider whose shared claims survive process boundaries."""

    def __init__(
        self,
        path: Path,
        *,
        initial: WorkspaceBranchBindingAuthority | None = None,
    ) -> None:
        self._path = path
        if initial is not None:
            descriptor = self._open()
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                self._write(
                    descriptor,
                    binding=initial,
                    operations={},
                )
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    @property
    def claim_scope(self) -> WorkspaceBranchBindingAuthorityClaimScope:
        return WorkspaceBranchBindingAuthorityClaimScope.DURABLE

    def __call__(self) -> WorkspaceBranchBindingAuthority:
        descriptor = self._open()
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            binding, _ = self._read(descriptor)
            return binding
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def claim(self, expected: WorkspaceBranchBindingAuthority) -> _FileAuthorityClaim:
        descriptor = self._open()
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            binding, _ = self._read(descriptor)
            if binding != expected:
                raise WorkspaceBranchOperationConflict(
                    "Workspace branch binding authority is no longer current."
                )
            return _FileAuthorityClaim(descriptor)
        except BaseException:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
            raise

    def authorize_operation(self, authority: WorkspaceBranchAuthority) -> None:
        descriptor = self._open()
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            binding, operations = self._read(descriptor)
            if binding != WorkspaceBranchBindingAuthority(
                environment_name=authority.environment_name,
                binding_generation=authority.binding_generation,
                binding_identity=authority.binding_identity,
            ):
                raise WorkspaceBranchOperationConflict(
                    "Workspace branch operation authority uses a different binding."
                )
            operations[authority.session_id] = authority
            self._write(descriptor, binding=binding, operations=operations)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def current_operation_authority(self, session_id: str) -> WorkspaceBranchAuthority:
        descriptor = self._open()
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            _, operations = self._read(descriptor)
            try:
                return operations[session_id]
            except KeyError:
                raise WorkspaceBranchOperationConflict(
                    "Workspace branch operation authority is unavailable."
                ) from None
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def claim_operation(self, expected: WorkspaceBranchAuthority) -> _FileAuthorityClaim:
        descriptor = self._open()
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            binding, operations = self._read(descriptor)
            expected_binding = WorkspaceBranchBindingAuthority(
                environment_name=expected.environment_name,
                binding_generation=expected.binding_generation,
                binding_identity=expected.binding_identity,
            )
            if binding != expected_binding or operations.get(expected.session_id) != expected:
                raise WorkspaceBranchOperationConflict(
                    "Workspace branch invocation authority is no longer current."
                )
            return _FileAuthorityClaim(descriptor)
        except BaseException:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
            raise

    def replace(self, authority: WorkspaceBranchBindingAuthority) -> None:
        descriptor = self._open()
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            self._write(descriptor, binding=authority, operations={})
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _open(self) -> int:
        return os.open(
            self._path,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )

    @staticmethod
    def _read(
        descriptor: int,
    ) -> tuple[WorkspaceBranchBindingAuthority, dict[str, WorkspaceBranchAuthority]]:
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 4096)
            if not chunk:
                break
            chunks.append(chunk)
            if sum(map(len, chunks)) > 4096:
                raise RuntimeError("Workspace branch authority record is oversized.")
        value = json.loads(b"".join(chunks))
        if type(value) is not dict or type(value.get("operations")) is not dict:
            raise RuntimeError("Workspace branch authority record is invalid.")
        binding = WorkspaceBranchBindingAuthority.model_validate(value.get("binding"))
        operations = {
            session_id: WorkspaceBranchAuthority.model_validate(authority)
            for session_id, authority in value["operations"].items()
        }
        if any(type(session_id) is not str for session_id in operations):
            raise RuntimeError("Workspace branch authority record is invalid.")
        return binding, operations

    @staticmethod
    def _write(
        descriptor: int,
        *,
        binding: WorkspaceBranchBindingAuthority,
        operations: dict[str, WorkspaceBranchAuthority],
    ) -> None:
        payload = json.dumps(
            {
                "binding": binding.model_dump(mode="json", warnings=False),
                "operations": {
                    session_id: authority.model_dump(mode="json", warnings=False)
                    for session_id, authority in sorted(operations.items())
                },
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.ftruncate(descriptor, 0)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Workspace branch authority write made no progress.")
            view = view[written:]
        os.fsync(descriptor)


__all__ = ["FileWorkspaceBranchAuthorityProvider"]
