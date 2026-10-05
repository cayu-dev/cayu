"""Explicit, bounded external-wait servicing; no worker starts on construction."""

import asyncio
import logging
import math
import time
from contextlib import suppress
from typing import Literal

from pydantic import StrictInt

from cayu.external_waits import ExternalWaitContext, ExternalWaitSnapshot, _snapshot
from cayu.runtime._external_wait_observation import external_wait_entrance
from cayu.session_external_waits import SessionExternalWaitAdapter
from cayu.sessions._session_continuation import ContinuationConflict, ContinuationUnavailable
from cayu.sessions.base import SessionRunFenced
from cayu.sessions.external_waits import (
    EXTERNAL_WAIT_PAGE_SIZE,
    ExternalWaitCapacityExceeded,
    ExternalWaitConflict,
    ExternalWaitScope,
    ExternalWaitUnavailable,
    Identifier,
    _Value,
)

# Matches recover_to_wait()'s default inactivity grace.
_ORPHANED_PREPARATION_AGE_S = 60


class ExternalWaitHostFailure(_Value):
    """Safe per-row diagnostics; no private exception text or receiving authority."""

    correlation_key: Identifier
    kind: Literal["conflict", "fenced", "capacity", "orphaned"]


class ExternalWaitHostPage(_Value):
    inspected: StrictInt
    settled: tuple[Identifier, ...]
    pending: tuple[Identifier, ...]
    next_cursor: Identifier | None
    failures: tuple[ExternalWaitHostFailure, ...] = ()


class ExternalWaitHost:
    """Discover durable responsibility; native receiving admission arbitrates workers.

    The caller owns scheduling and shutdown. A page does not claim work: a process
    that dies leaves the same native handoff available to another configured host.
    Permission failures propagate. Row conflicts remain explicit failures in the
    returned page while independent rows continue; they are never exclusion evidence.
    """

    def __init__(self, adapter: SessionExternalWaitAdapter, *, context: ExternalWaitContext):
        self.adapter = adapter
        self.app = adapter.app
        self.context = _snapshot(context, ExternalWaitContext)
        self._admission = adapter._admission

    @external_wait_entrance
    async def service_once(
        self,
        *,
        scope: ExternalWaitScope,
        source: str,
        after: str = "",
        limit: int = EXTERNAL_WAIT_PAGE_SIZE,
    ) -> ExternalWaitHostPage:
        rows = await self.adapter.waits.list(
            scope=scope, source=source, after=after, limit=limit, context=self.context
        )
        settled: list[str] = []
        pending: list[str] = []
        failures: list[ExternalWaitHostFailure] = []
        for row in rows:
            key = row.correlation.request.correlation_key
            try:
                disposition = await self._service_row(row)
            except (ExternalWaitUnavailable, ContinuationUnavailable):
                disposition = "pending"
            except (
                ExternalWaitConflict,
                ContinuationConflict,
                SessionRunFenced,
                ExternalWaitCapacityExceeded,
            ) as error:
                kind: Literal["conflict", "fenced", "capacity"] = (
                    "fenced"
                    if isinstance(error, SessionRunFenced)
                    else "capacity"
                    if isinstance(error, ExternalWaitCapacityExceeded)
                    else "conflict"
                )
                failures.append(ExternalWaitHostFailure(correlation_key=key, kind=kind))
                disposition = "pending"
            if disposition == "orphaned":
                failures.append(ExternalWaitHostFailure(correlation_key=key, kind="orphaned"))
                disposition = "pending"
            if disposition == "settled":
                settled.append(key)
            elif disposition == "pending":
                pending.append(key)
        return ExternalWaitHostPage(
            inspected=len(rows),
            settled=tuple(settled),
            pending=tuple(pending),
            failures=tuple(failures),
            next_cursor=rows[-1].correlation.request.correlation_key
            if len(rows) == limit
            else None,
        )

    async def _service_row(
        self, row: ExternalWaitSnapshot
    ) -> Literal["settled", "pending", "orphaned"] | None:
        # Observation elects using store time, never this worker's wall clock.
        current = await self.adapter.waits.observe(row.correlation, context=self.context)
        if (
            current.handoff == "unbound"
            and current.pending_handoff
            and current.registration is not None
        ):
            with suppress(ExternalWaitUnavailable):
                current = await self.adapter.reconcile_binding(
                    current.registration, context=self.context
                )
        if current.handoff == "unbound":
            if current.execution_excluded:
                return None
            if (
                current.pending_handoff
                and current.registration is not None
                and current.outcome is not None
                and current.outcome.kind in {"cancelled", "unavailable"}
            ):
                await self.adapter.exclude_prepared_execution(
                    current.registration, context=self.context
                )
                return "settled"
            if current.pending_handoff and await self._preparation_orphaned(current):
                return "orphaned"
            return "pending"
        if not current.pending_handoff:
            return None
        if (
            not current.retirement_requested
            and current.registration is not None
            and (
                current.outcome is None or current.outcome.kind not in {"cancelled", "unavailable"}
            )
        ):
            recovered = await self.adapter.recover_to_wait(
                current.registration, context=self.context
            )
            current = recovered.wait
        if current.registration is None or current.outcome is None:
            return "pending"
        result = await self.adapter.service_wait(current.registration, context=self.context)
        return "pending" if result.wait.pending_handoff else "settled"

    async def _preparation_orphaned(self, current: ExternalWaitSnapshot) -> bool:
        """Diagnose a preparation whose writer stopped before its native wait was bound.

        Only the report depends on this; it grants no exclusion or takeover authority.
        Its writer cannot resume the preparation, so it stays pending until an operator
        recovers the session and calls ``exclude_prepared_execution()``.
        """
        store = self.adapter.waits.store
        request = current.correlation.request
        record = await store._read_external_wait(request.scope, request.correlation_key)
        if record is None or record.execution is None or record.continuation is not None:
            return False
        try:
            execution = await store.inspect_session_execution(record.execution.intent.session_id)
        except KeyError:
            execution = None
        if execution is not None:
            if execution.state == "owner_lost":
                return True
            if execution.state not in {"idle", "waiting", "terminal"}:
                return False
        # Initial runs prepare before the session exists, and resumes before they claim
        # execution, so without a lost lease only an aged preparation is reported.
        prepared_age_s = time.time() - record.execution.prepared_at_ms / 1000
        return prepared_age_s >= _ORPHANED_PREPARATION_AGE_S

    async def run(
        self,
        *,
        scope: ExternalWaitScope,
        source: str,
        stop: asyncio.Event,
        interval_s: float = 1.0,
    ) -> None:
        """Explicit polling loop. Cancellation propagates; it never cancels the external job."""
        if (
            type(interval_s) not in {int, float}
            or not math.isfinite(interval_s)
            or not 0 < interval_s <= 60
        ):
            raise ValueError("External host interval must be finite and within (0, 60] seconds.")
        cursor = ""
        # Log each failing row once per failure kind, not on every sweep; a row that
        # recovers and fails again is logged again.
        reported: dict[str, str] = {}
        failing: dict[str, str] = {}
        while not stop.is_set():
            page = await self.service_once(scope=scope, source=source, after=cursor)
            for failure in page.failures:
                failing[failure.correlation_key] = failure.kind
                if reported.get(failure.correlation_key) != failure.kind:
                    logging.getLogger(__name__).warning(
                        "External wait %s requires reconciliation (%s).",
                        failure.correlation_key,
                        failure.kind,
                    )
            cursor = page.next_cursor or ""
            if page.next_cursor is not None:
                continue
            reported, failing = failing, {}
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=interval_s)
