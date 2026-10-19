"""Keep one brokered sandbox's lease alive while its user needs it."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime

from cayu.sandbox_broker.client import (
    SandboxBrokerClient,
    SandboxBrokerRefusedError,
    SandboxLeaseLostError,
    SandboxRecord,
    validate_lease_seconds,
)

__all__ = ["SandboxLeaseKeeper"]

_LOGGER = logging.getLogger(__name__)
_RETRY_AFTER_FAILURE_SECONDS = 5.0


class SandboxLeaseKeeper:
    """Renew a sandbox's lease at a third of its length until stopped.

    The first renewal comes at a third of the time the lease has left, so a
    lease partly used before the keeper started is renewed in time.

    The keeper never destroys anything: stopping it only ends renewal, so the
    broker destroys the sandbox when the lease lapses. A refused renewal (the
    sandbox is gone or no longer ``ready``) or a lease that runs out while the
    broker is unreachable marks the lease lost; ``check`` then raises
    ``SandboxLeaseLostError`` and ``on_lost`` is called once.
    """

    def __init__(
        self,
        client: SandboxBrokerClient,
        sandbox: SandboxRecord,
        *,
        lease_seconds: int,
        renew_every_s: float | None = None,
        on_lost: Callable[[SandboxLeaseLostError], None] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(client, SandboxBrokerClient):
            raise TypeError("SandboxLeaseKeeper requires a SandboxBrokerClient.")
        if not isinstance(sandbox, SandboxRecord):
            raise TypeError("SandboxLeaseKeeper requires the sandbox's SandboxRecord.")
        self._client = client
        self.sandbox_id = sandbox.id
        self.lease_seconds = validate_lease_seconds(lease_seconds)
        interval = self.lease_seconds / 3 if renew_every_s is None else renew_every_s
        if type(interval) not in {int, float} or not 0 < interval < self.lease_seconds:
            raise ValueError("renew_every_s must be positive and shorter than the lease.")
        self.renew_every_s = float(interval)
        self.lease_expires_at = sandbox.lease_expires_at
        self.renewals = 0
        self._on_lost = on_lost
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lost: SandboxLeaseLostError | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def lost(self) -> SandboxLeaseLostError | None:
        return self._lost

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def check(self) -> None:
        if self._lost is not None:
            raise SandboxLeaseLostError(str(self._lost), code=self._lost.code)

    def start(self) -> SandboxLeaseKeeper:
        if self._task is not None:
            raise RuntimeError("SandboxLeaseKeeper is already started.")
        self._task = asyncio.create_task(self._run(), name=f"sandbox-lease:{self.sandbox_id}")
        return self

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def renew_now(self) -> SandboxRecord:
        """Renew once, outside the schedule. Raises ``SandboxLeaseLostError`` on refusal."""

        self.check()
        try:
            record = await self._client.renew(self.sandbox_id, lease_seconds=self.lease_seconds)
        except SandboxBrokerRefusedError as exc:
            self._mark_lost(
                SandboxLeaseLostError(
                    f"Sandbox lease renewal was refused: {exc.code or exc.status}.",
                    status=exc.status,
                    code=exc.code,
                )
            )
            self.check()
            raise AssertionError("unreachable") from None
        self.lease_expires_at = record.lease_expires_at
        self.renewals += 1
        return record

    async def _run(self) -> None:
        delay = self._first_delay()
        while self._lost is None:
            await asyncio.sleep(delay)
            try:
                await self.renew_now()
                delay = self.renew_every_s
            except SandboxLeaseLostError:
                return
            except Exception as exc:  # broker unreachable: retry until the lease runs out
                expires_at = self.lease_expires_at
                if expires_at is not None and self._clock() >= expires_at:
                    self._mark_lost(
                        SandboxLeaseLostError(
                            "Sandbox lease expired while the broker was unreachable."
                        )
                    )
                    return
                _LOGGER.warning("Sandbox lease renewal failed; retrying (%s).", type(exc).__name__)
                delay = min(_RETRY_AFTER_FAILURE_SECONDS, self.renew_every_s)

    def _first_delay(self) -> float:
        # Schedule against what is left of the lease, not its full length.
        expires_at = self.lease_expires_at
        if expires_at is None:
            return self.renew_every_s
        remaining = (expires_at - self._clock()).total_seconds()
        return max(0.0, min(self.renew_every_s, remaining / 3))

    def _mark_lost(self, error: SandboxLeaseLostError) -> None:
        if self._lost is not None:
            return
        self._lost = error
        if self._on_lost is not None:
            try:
                self._on_lost(error)
            except Exception:
                _LOGGER.exception("Sandbox lease-lost callback failed.")
