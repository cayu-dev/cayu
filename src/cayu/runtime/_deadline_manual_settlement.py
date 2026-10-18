"""Durable hook and deadline settlement phases for manual user-input recovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from cayu.events import Event
from cayu.sessions._checkpoint_mutations import SessionRuntimePublicationConflict
from cayu.sessions._runtime_publication import SessionOperationPublication
from cayu.sessions.records import Session, SessionStatus
from cayu.sessions.store import SessionStore

_Phase = Literal["tool-hooks", "settlement"]


@dataclass(frozen=True)
class ManualUserInputDeadlineSettlement:
    store: SessionStore
    session: Session
    event: Event

    def _completion(self, phase: _Phase) -> tuple[str, dict[str, Any]]:
        digest = self.event.payload.get("resolution_request_digest")
        if (
            (phase == "settlement" and not self.session.execution_deadline.expired)
            or self.event.session_id != self.session.id
            or self.event.payload.get("manual_recovery") is not True
            or not isinstance(digest, str)
        ):
            raise SessionRuntimePublicationConflict("Invalid manual deadline settlement evidence.")
        return f"manual-input-deadline:{self.event.id}:{phase}", {
            "schema_version": 1,
            "session_instance_id": self.session.instance_id,
            "terminal_event_id": self.event.id,
            "request_digest": digest,
            "phase": phase,
        }

    async def completed(self, phase: _Phase) -> bool:
        key, expected = self._completion(phase)
        observed = await self.store.load_session_operation(self.session.id, key)
        if observed is not None and observed != expected:
            raise SessionRuntimePublicationConflict("Conflicting manual deadline settlement.")
        return observed is not None

    async def complete(self, phase: _Phase) -> None:
        if phase == "settlement" and not await self.completed("tool-hooks"):
            raise SessionRuntimePublicationConflict("Manual recovery tool hooks have not settled.")
        key, completion = self._completion(phase)

        def transform(current: Session, checkpoint: dict[str, Any] | None, previous: dict | None):
            if current.instance_id != self.session.instance_id:
                raise SessionRuntimePublicationConflict("Manual settlement changed incarnation.")
            if previous is not None and previous != completion:
                raise SessionRuntimePublicationConflict("Conflicting manual deadline settlement.")
            return SessionOperationPublication(
                checkpoint={} if checkpoint is None else checkpoint,
                operation_records={key: completion},
            )

        await self.store.publish_session_operation(
            self.session.id,
            idempotency_key=key,
            operation_transform=transform,
            events=[],
            expected_statuses=(
                {SessionStatus.INTERRUPTED}
                if phase == "settlement"
                else {SessionStatus.RUNNING, SessionStatus.INTERRUPTED}
            ),
            expected_run_epoch=self.session.run_epoch,
        )
