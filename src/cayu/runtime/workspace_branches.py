"""Runtime adapter for durable workspace-branch records."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cayu.runtime.service_manifest import RuntimeStoreDurability
from cayu.runtime.sessions import (
    SessionOperationPublication,
    _OwnedOffThreadSessionCommitGuard,
)
from cayu.workspaces.branches import WorkspaceBranchStoreDurability

if TYPE_CHECKING:
    from collections.abc import Callable

    from cayu.runtime.sessions import SessionStore
    from cayu.workspaces.branches import WorkspaceBranchRecordTransform


class SessionWorkspaceBranchStore:
    """Expose a SessionStore through the workspace-owned branch journal seam."""

    def __init__(self, store: SessionStore) -> None:
        required = (
            "load_session_operation",
            "publish_session_operation",
            "publish_session_operation_guarded",
        )
        if any(not callable(getattr(store, name, None)) for name in required):
            raise TypeError("Workspace branch storage requires an atomic SessionStore.")
        capability_check = getattr(
            store,
            "_supports_owned_off_thread_session_commit_guard_protocol",
            None,
        )
        if not callable(capability_check) or capability_check() is not True:
            raise TypeError(
                "Workspace branch storage requires owned off-thread session commit guards."
            )
        try:
            self._durability = WorkspaceBranchStoreDurability(
                RuntimeStoreDurability(store.service_durability).value
            )
        except (AttributeError, TypeError, ValueError):
            raise TypeError(
                "Workspace branch storage requires a declared SessionStore durability."
            ) from None
        self._store = store

    @property
    def durability(self) -> WorkspaceBranchStoreDurability:
        """Return the exact persistence evidence declared by the wrapped store."""

        return self._durability

    async def load_workspace_branch_record(
        self,
        session_id: str,
        storage_key: str,
    ) -> dict[str, Any] | None:
        return await self._store.load_session_operation(session_id, storage_key)

    async def publish_workspace_branch_record(
        self,
        session_id: str,
        storage_key: str,
        *,
        record_transform: WorkspaceBranchRecordTransform,
        expected_run_epoch: int,
        commit_guard: Callable[[], None] | None = None,
    ) -> None:
        if not callable(record_transform):
            raise TypeError("Workspace branch record transform must be callable.")

        def operation_transform(_session, checkpoint, current_record):
            updated_record = record_transform(current_record)
            if type(updated_record) is not dict:
                raise TypeError("Workspace branch record transform must return an object.")
            return SessionOperationPublication(
                checkpoint={} if checkpoint is None else checkpoint,
                operation_records={storage_key: updated_record},
            )

        if commit_guard is None:
            await self._store.publish_session_operation(
                session_id,
                idempotency_key=storage_key,
                operation_transform=operation_transform,
                events=[],
                expected_run_epoch=expected_run_epoch,
            )
            return
        await self._store.publish_session_operation_guarded(
            session_id,
            idempotency_key=storage_key,
            operation_transform=operation_transform,
            commit_guard=_OwnedOffThreadSessionCommitGuard(commit_guard),
            events=[],
            expected_run_epoch=expected_run_epoch,
        )


__all__ = ["SessionWorkspaceBranchStore"]
