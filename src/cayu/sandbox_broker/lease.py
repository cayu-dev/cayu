"""Keep one brokered sandbox's lease alive while its user needs it."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

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
    ``SandboxLeaseLostError`` and ``on_lost`` is called once. A scheduled
    renewal still unanswered when the lease runs out by the Agent's clock is
    abandoned and the lease marked lost, so a slow broker can't keep the
    runner working past its lease.

    One renewal call that fails outright takes up to four times the client's
    request timeout plus its retry pauses (about three minutes with the
    default 45 s). Renewing at a third of the lease leaves two thirds for
    that and the keeper's own retries, so a lease of five minutes or more
    rides out one failed call; a shorter one can lapse inside it.
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
        # The furthest a renewal sent so far could have pushed the lease, whether
        # or not its answer arrived: a renewal that was cancelled or lost may
        # still have reached the broker.
        self.latest_possible_expiry = sandbox.lease_expires_at
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
        self._extend_possible_expiry()
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
        finally:
            # The client may have retried, so any request it sent went out by now.
            self._extend_possible_expiry()
        self.lease_expires_at = record.lease_expires_at
        self.renewals += 1
        return record

    def _extend_possible_expiry(self) -> None:
        possible = self._clock() + timedelta(seconds=self.lease_seconds)
        if self.latest_possible_expiry is None or possible > self.latest_possible_expiry:
            self.latest_possible_expiry = possible

    async def _run(self) -> None:
        delay = self._first_delay()
        while self._lost is None:
            await asyncio.sleep(delay)
            try:
                await self._renew_within_lease()
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

    async def _renew_within_lease(self) -> None:
        while True:
            expires_at = self.lease_expires_at
            remaining = (
                None
                if expires_at is None
                else max(0.0, (expires_at - self._clock()).total_seconds())
            )
            bound = asyncio.timeout(remaining)
            try:
                async with bound:
                    await self.renew_now()
                return
            except TimeoutError:
                if not bound.expired():
                    raise
            # An explicit renew_now() may have extended the lease meanwhile;
            # then renew again against the new expiry instead of losing it.
            current = self.lease_expires_at
            if current is None or current == expires_at or current <= self._clock():
                self._mark_lost(
                    SandboxLeaseLostError("Sandbox lease expired before its renewal was answered.")
                )
                self.check()

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
