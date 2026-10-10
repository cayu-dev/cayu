"""Shared recovery claim records and process-local worker ownership."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.sessions._terminal_evidence import _SessionRunOperation
from cayu.sessions.base import (
    _SessionRunFenceOwnership,
)
from cayu.sessions.records import Session


class _RecoveryWorkerSettlement:
    """Process-local work lifetime, never authority to acquire an invocation."""

    __slots__ = ("_workers",)

    def __init__(self) -> None:
        self._workers: tuple[asyncio.Task[Any], ...] = ()

    @property
    def settled(self) -> bool:
        return all(task.done() for task in self._workers)

    def owns_current_worker(self) -> bool:
        return bool(self._workers) and self._workers[0] is asyncio.current_task()

    def own_workers(self, *workers: asyncio.Task[Any]) -> None:
        if not self.settled:
            raise RuntimeError("Recovery claim already owns unsettled workers.")
        self._workers = workers

    async def await_worker_settlement(self) -> None:
        """Quiescence dependency; the worker supervisor owns failure propagation."""
        if any(task is asyncio.current_task() for task in self._workers):
            raise RuntimeError("A recovery worker cannot release its own live claim.")
        pending = {task for task in self._workers if not task.done()}
        while pending:
            try:
                _, pending = await asyncio.wait(pending)
            except asyncio.CancelledError:
                # Only supervised cleanup waits here. Its cancellation probe
                # cannot establish worker quiescence or authorize claim release.
                continue


class _IncompleteRecoveryClaimAuthority(_RecoveryWorkerSettlement):
    """Exact durable claim and transferable process-local fence authority."""

    __slots__ = (
        "_finalization_lock",
        "_finalized",
        "claim_id",
        "run_fence",
        "session_id",
    )

    def __init__(
        self,
        *,
        session_id: str,
        claim_id: str,
        run_fence: _SessionRunFenceOwnership,
    ) -> None:
        super().__init__()
        if run_fence.session_id != session_id:
            raise ValueError("Recovery claim and run-fence session identities differ.")
        self.session_id = session_id
        self.claim_id = claim_id
        self.run_fence = run_fence
        self._finalization_lock = asyncio.Lock()
        self._finalized = False

    def own_workers(self, *workers: asyncio.Task[Any]) -> None:
        """Keep the exact recovery work and heartbeat ahead of claim release."""
        if self._finalized:
            raise RuntimeError("Recovery claim already owns unsettled workers.")
        super().own_workers(*workers)

    @property
    def run_epoch(self) -> int:
        return self.run_fence.run_epoch

    def retire(self) -> bool:
        """Idempotently invalidate this exact process-local owner in all tasks."""

        return self.run_fence.retire()

    async def begin_finalization(self) -> bool:
        """Elect one finalizer; waiters retry an abort and ignore a finished owner."""

        await self._finalization_lock.acquire()
        if self._finalized:
            self._finalization_lock.release()
            return False
        return True

    def finish_finalization(self) -> None:
        """Publish finalization and release every waiter after local retirement."""

        if not self._finalization_lock.locked():
            raise RuntimeError("Recovery claim finalization was not acquired.")
        self._finalized = True
        self.retire()
        self._finalization_lock.release()

    def abort_finalization(self) -> None:
        """Release the finalizer election while retaining retryable authority."""

        if not self._finalization_lock.locked():
            raise RuntimeError("Recovery claim finalization was not acquired.")
        self._finalization_lock.release()


@dataclass(frozen=True)
class _IncompleteRecoveryClaim:
    claim_id: str
    claim_expires_at: datetime
    local_lease_deadline: float
    session_before_fence: Session
    session: Session
    run_operation: _SessionRunOperation | None = None
    invocation_context: InvocationContext | None = None
    authority: _IncompleteRecoveryClaimAuthority | None = None

    def __post_init__(self) -> None:
        if self.authority is None:
            return
        if (
            self.authority.claim_id != self.claim_id
            or self.authority.session_id != self.session.id
            or self.authority.run_epoch != self.session.run_epoch
        ):
            raise ValueError("Recovery claim authority does not match the claimed session.")

    def require_authority(self) -> _IncompleteRecoveryClaimAuthority:
        if self.authority is None:
            raise RuntimeError("Recovery claim has no run-fence authority.")
        return self.authority


class _IncompleteRecoveryClaimLost(RuntimeError):
    """The durable incomplete-session recovery lease is no longer owned."""


def _require_live_incomplete_recovery_claim_acknowledgement(
    *,
    session_id: str,
    local_lease_deadline: float,
) -> None:
    """Reject an acknowledgement that consumed its complete local lease budget."""

    if time.monotonic() >= local_lease_deadline:
        raise _IncompleteRecoveryClaimLost(
            "Incomplete-session recovery claim acknowledgement consumed its lease "
            f"before work could start for session {session_id}."
        )
