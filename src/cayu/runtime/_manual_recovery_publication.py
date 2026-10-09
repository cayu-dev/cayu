"""Persistence and acknowledgement recovery for operator-supplied tool results."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from cayu._task_wait import await_shielded_task_outcome
from cayu.events import Event
from cayu.runtime._diagnostics import exception_diagnostic
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.vaults.redaction import SecretRedactor


@dataclass(frozen=True)
class ManualRecoveryPersistenceReconciliation:
    persisted: bool | None
    error: Exception | None = None
    cancellation: asyncio.CancelledError | None = None

    def failure_payload(self, *, redactor: SecretRedactor) -> dict[str, Any] | None:
        """Project persistence evidence after the caller handles cancellation."""
        if self.persisted is True:
            return {"manual_recovery_persisted": True}
        if self.error is not None:
            return {
                "manual_recovery_persistence_unknown": True,
                "persistence_reconciliation_error_type": exception_diagnostic(
                    self.error, redactor=redactor
                ).error_type,
            }
        return None


async def reconcile_manual_recovery_persistence(
    event_writer: RuntimeEventWriter,
    event: Event,
) -> ManualRecoveryPersistenceReconciliation:
    """Classify an append failure using the preassigned durable event id."""
    outcome = await await_shielded_task_outcome(
        asyncio.create_task(event_writer.is_persisted(event))
    )
    if outcome.error is None:
        return ManualRecoveryPersistenceReconciliation(
            persisted=bool(outcome.result),
            cancellation=outcome.cancellation,
        )
    if isinstance(outcome.error, asyncio.CancelledError):
        return ManualRecoveryPersistenceReconciliation(
            persisted=None,
            cancellation=outcome.cancellation or outcome.error,
        )
    if not isinstance(outcome.error, Exception):
        raise outcome.error
    return ManualRecoveryPersistenceReconciliation(
        persisted=None,
        error=outcome.error,
        cancellation=outcome.cancellation,
    )


@dataclass(slots=True)
class ManualRecoveryPublication:
    """Retain append evidence before fanout, hooks or continuation can fail.

    Callers supply their typed, redacted events and retain authority validation,
    interruption and cleanup. Receipt recovery can adopt already-persisted
    evidence without appending another operator result.
    """

    event_writer: RuntimeEventWriter
    event: Event | None = field(default=None, init=False)
    persisted: bool = field(default=False, init=False)

    async def persist(self, session_id: str, events: list[Event]) -> list[Event]:
        emitted = await self.event_writer.persist_many(session_id, events)
        self.persisted = True
        return emitted

    async def reconcile(self) -> ManualRecoveryPersistenceReconciliation:
        """Read uncertain append evidence without accepting it before cancellation.

        Callers retain cancellation precedence and update their persisted state
        only after handling the read's cancellation evidence.
        """
        if self.persisted:
            return ManualRecoveryPersistenceReconciliation(persisted=True)
        if self.event is None:
            return ManualRecoveryPersistenceReconciliation(persisted=False)
        return await reconcile_manual_recovery_persistence(self.event_writer, self.event)
