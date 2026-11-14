"""Bounded lock waits and retries for PostgreSQL migration transactions.

A revision runs in one transaction next to live readers and writers. It can
wait behind a long transaction, and if it takes locks in a different order than
a writer it can deadlock with it. Each migration transaction therefore sets
``lock_timeout`` and, when Postgres aborts it with a deadlock (SQLSTATE 40P01)
or a lock timeout (55P03), rolls back, waits and runs again from the recorded
revision. Nothing in a rolled-back attempt is kept, so a retry is the same as a
first run. Other errors propagate at once.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from psycopg.errors import DeadlockDetected, LockNotAvailable

from cayu.storage.targets import DATABASE_MIGRATION_LOCK_TIMEOUT_ENV

logger = logging.getLogger("cayu.storage.postgres")

#: Attempts per migration transaction, the first included.
MIGRATION_LOCK_ATTEMPTS = 5
#: Wait before the first retry; each later retry waits twice as long.
MIGRATION_LOCK_FIRST_RETRY_SECONDS = 1.0


async def set_migration_lock_timeout(cur: Any, seconds: int) -> None:
    """Bound every lock wait in the cursor's current transaction (0 means no bound)."""

    await cur.execute("SELECT set_config('lock_timeout', %s, true)", (f"{seconds}s",))


class MigrationLockRetry:
    """Run migration transactions again after a deadlock or a lock timeout.

    Wrap each transaction in :meth:`transaction` and set ``step`` inside it to
    name what the transaction applies. When it fails with SQLSTATE 40P01 or
    55P03, the context suppresses the error after the connection is released,
    sleeps, and sets ``retrying``; the caller then starts the same step again.
    After ``attempts`` failures in a row the error propagates with a note on how
    to proceed. A transaction that commits resets the count.
    """

    def __init__(
        self,
        *,
        attempts: int = MIGRATION_LOCK_ATTEMPTS,
        first_retry_seconds: float = MIGRATION_LOCK_FIRST_RETRY_SECONDS,
        sleep: Callable[[float], Awaitable[object]] = asyncio.sleep,
    ) -> None:
        if attempts < 1:
            raise ValueError("attempts must be at least 1.")
        self._attempts = attempts
        self._first_retry_seconds = first_retry_seconds
        self._sleep = sleep
        self._failures = 0
        self.retrying = False
        self.step = "schema transaction"

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Run one migration transaction, retrying it as the class describes."""

        self.retrying = False
        self.step = "schema transaction"
        try:
            yield
        except (DeadlockDetected, LockNotAvailable) as exc:
            self._failures += 1
            reason = "deadlocked" if isinstance(exc, DeadlockDetected) else "timed out on a lock"
            if self._failures >= self._attempts:
                exc.add_note(
                    f"Postgres migration {self.step} {reason}; gave up after "
                    f"{self._attempts} attempts. Stop the application's writers (workers, servers and "
                    "anything else that opens the database) and run `cayu storage "
                    f"migrate` again, or raise {DATABASE_MIGRATION_LOCK_TIMEOUT_ENV} "
                    "(0 waits indefinitely)."
                )
                raise
            delay = self._first_retry_seconds * 2 ** (self._failures - 1)
            logger.warning(
                "Postgres migration %s %s (SQLSTATE %s); rolled back, retrying in %.1fs "
                "(attempt %d of %d).",
                self.step,
                reason,
                exc.sqlstate,
                delay,
                self._failures + 1,
                self._attempts,
            )
            await self._sleep(delay)
            self.retrying = True
            return
        self._failures = 0
