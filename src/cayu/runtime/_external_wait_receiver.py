"""Registered native continuation receiver for one exact external wait."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.external_waits import ExternalEventWaits, ExternalWaitContext, _snapshot
from cayu.runtime._external_wait_binding import load_prepared_continuation
from cayu.sessions._external_wait_records import elected_external_latch
from cayu.sessions._session_continuation import (
    ContinuationConflict,
    ContinuationLatch,
    RetainedContinuationLatchReceiver,
    continuation_digest,
    require_latch_identity,
)
from cayu.sessions.external_waits import (
    ExternalWaitRecord,
    ExternalWaitRegistration,
    ExternalWaitUnavailable,
    external_wait_digest,
)

if TYPE_CHECKING:
    from cayu.runtime._session_continuation_owner import SessionContinuationOwner


class ExternalWaitLatchReceiver(RetainedContinuationLatchReceiver):
    """Configured owner, not a receipt or an origin accepted from caller input."""

    def __init__(
        self,
        waits: ExternalEventWaits,
        registration: ExternalWaitRegistration,
        context: ExternalWaitContext,
    ) -> None:
        self._waits = waits
        self._registration = _snapshot(registration, ExternalWaitRegistration)
        self._context = _snapshot(context, ExternalWaitContext)

    async def reconcile_binding(self) -> ExternalWaitRecord:
        """Recover a committed native preparation, never infer one from absence."""
        from cayu.runtime._external_wait_settlement import settlement_scope
        from cayu.sessions._external_wait_records import external_continuation_intent

        correlation = self._registration.correlation
        self._waits._authorize(correlation.request, self._context, "service")
        record = await self._waits.store._read_external_wait(
            correlation.request.scope, correlation.request.correlation_key
        )
        if record is None or record.registration != self._registration or record.execution is None:
            raise ExternalWaitUnavailable("External binding has no retained execution preparation.")
        if record.continuation is not None:
            self._waits._authorize(correlation.request, self._context, "service")
            return record
        try:
            native = await self._waits.store.load_continuation_ticket(
                record.execution.intent.session_id,
                registration_key=external_continuation_intent(self._registration).registration_key,
                session_instance_id=record.execution.session_instance_id,
            )
        except KeyError:
            # Initial preparation commits before the session exists.
            native = None
        if native is None:
            raise ExternalWaitUnavailable("External binding has no native preparation receipt.")
        command = self._waits._command(
            "reconcile_binding",
            correlation,
            registration=self._registration,
            continuation=native.preparation,
        )
        self._waits._authorize(correlation.request, self._context, "service")
        with settlement_scope(command):
            return await self._waits._mutate(command)

    async def retire_released(self, owner: SessionContinuationOwner) -> ExternalWaitRecord:
        """Reconstruct exact cleanup; never cancel a live external job or invocation."""
        from cayu.runtime._continuation_wait_settlement import acknowledge_retirement
        from cayu.sessions._session_continuation import (
            ContinuationReleasedRetirement,
            ContinuationRetirement,
        )

        correlation = self._registration.correlation
        self._waits._authorize(correlation.request, self._context, "service")
        record = await self._waits.store._read_external_wait(
            correlation.request.scope, correlation.request.correlation_key
        )
        if (
            record is None
            or record.registration != self._registration
            or record.continuation is None
            or record.outcome is None
            or (
                record.execution_retirement is None
                and record.outcome.kind not in {"cancelled", "unavailable"}
            )
            or owner.store is not self._waits.store
            or owner.receiver is not self
        ):
            raise ExternalWaitUnavailable(
                "External wait lacks this owner's terminal cleanup evidence."
            )
        if record.retirement_complete:
            self._waits._authorize(correlation.request, self._context, "service")
            return record
        native = await load_prepared_continuation(self._waits.store, record.continuation)
        if native is None or native.preparation != record.continuation:
            raise ExternalWaitUnavailable("External retirement receiving evidence is unavailable.")
        if native.ticket.state == "CONSUMED" and record.execution_retirement is not None:
            # Retirement intent is not proof of exclusion. A previously admitted
            # continuation may win the native race; preserve that exact result.
            return await self.reconcile_handoff()
        control = record.execution_retirement
        retirement = ContinuationRetirement(
            ticket=native.ticket,
            control_id=(
                "external-retire:" + external_wait_digest(self._registration)
                if control is None
                else "external-retire-control:" + external_wait_digest(control)
            ),
            reason="cancelled"
            if control is None and record.outcome.kind == "cancelled"
            else "unavailable",
            retired_at=datetime.fromtimestamp(
                (record.outcome.selected_at_ms if control is None else control.prepared_at_ms)
                / 1000,
                tz=UTC,
            ).isoformat(),
        )
        self._waits._authorize(correlation.request, self._context, "service")
        # An automatic cancellation retirement may have won before an explicit
        # request reached the native owner. Reconcile its authentic terminal
        # receipt; do not try to rewrite that decision under the new control.
        retired = native
        if native.ticket.state != "RETIRED":
            retired = await owner.retire_released(
                ContinuationReleasedRetirement(
                    retirement=retirement,
                    permit_operation=None,
                    permit_commitment=None,
                )
            )
        await self.reconcile_handoff()
        if retired.released_retirement is not None:
            await acknowledge_retirement(owner, self._registration, retired)
        return await self.reconcile_handoff()

    async def authenticate_continuation_retirement(self, registration, candidate) -> str:
        from cayu.runtime._continuation_wait_settlement import retirement_receipt
        from cayu.sessions._session_continuation import ContinuationRecord

        registration = _snapshot(registration, ExternalWaitRegistration)
        native = prepare_contract(ContinuationRecord, candidate, redactor=self._waits.redactor)
        if registration != self._registration:
            raise PermissionError("External retirement registration conflicts.")
        correlation = registration.correlation
        self._waits._authorize(correlation.request, self._context, "service")
        retained = await self._waits.store._read_external_wait(
            correlation.request.scope, correlation.request.correlation_key
        )
        expected = continuation_digest(retirement_receipt(native))
        if (
            retained is None
            or retained.registration != registration
            or retained.continuation != native.preparation
            or retained.handoff != "excluded"
            or retained.handoff_receipt_sha256 != expected
        ):
            # Row state that moved after discovery, not a receiving-authority refusal.
            raise ContinuationConflict("External wait has not acknowledged the exact retirement.")
        self._waits._authorize(correlation.request, self._context, "service")
        return expected

    async def reconcile_handoff(self) -> ExternalWaitRecord:
        """Release only the external fence, after exact native terminal readback."""
        from cayu.runtime._external_wait_settlement import settlement_scope

        correlation = self._registration.correlation
        self._waits._authorize(correlation.request, self._context, "service")
        record = await self._waits.store._read_external_wait(
            correlation.request.scope, correlation.request.correlation_key
        )
        if (
            record is None
            or record.registration != self._registration
            or record.continuation is None
        ):
            raise ExternalWaitUnavailable("External continuation binding is unavailable.")
        if record.handoff in {"settled", "excluded"}:
            disposition = record.handoff
            receipt_digest = record.handoff_receipt_sha256
        else:
            native = await load_prepared_continuation(self._waits.store, record.continuation)
            if native is None or native.ticket.state not in {"CONSUMED", "RETIRED"}:
                raise ExternalWaitUnavailable(
                    "External continuation receiving outcome is unresolved."
                )
            disposition = "settled" if native.ticket.state == "CONSUMED" else "excluded"
            receipt_digest = continuation_digest(native)
        self._waits._authorize(correlation.request, self._context, "service")
        command = self._waits._command(
            "settle",
            correlation,
            registration=self._registration,
            continuation=record.continuation,
            handoff_disposition=disposition,
            handoff_receipt_sha256=receipt_digest,
        )
        with settlement_scope(command):
            settled = await self._waits._mutate(command)
        if settled.handoff != "excluded" or settled.retirement_complete:
            return settled
        assert settled.continuation is not None
        native = await load_prepared_continuation(self._waits.store, settled.continuation)
        if native is None or (
            native.released_retirement is not None and not native.retirement_acknowledged
        ):
            return settled
        self._waits._authorize(correlation.request, self._context, "service")
        completion = command.model_copy(update={"kind": "complete_retirement"})
        with settlement_scope(completion):
            return await self._waits._mutate(completion)

    async def authenticate_continuation_latch(self, latch: ContinuationLatch) -> ContinuationLatch:
        return await self._receive(latch, wait_for_settlement=False)

    async def _authenticate_latch_owned(self, latch: ContinuationLatch) -> ContinuationLatch:
        return await self._receive(latch, wait_for_settlement=True)

    async def _receive(
        self, latch: ContinuationLatch, *, wait_for_settlement: bool
    ) -> ContinuationLatch:
        from cayu.runtime._external_wait_observation import external_wait_tracker

        latch = prepare_contract(ContinuationLatch, latch, redactor=self._waits.redactor)
        return await self._waits._owners.run(
            lambda: self._authenticate(latch),
            key=("external-latch", external_wait_digest(self._registration)),
            expectation=contract_bytes(latch, redactor=self._waits.redactor),
            redactor=self._waits.redactor,
            track=external_wait_tracker(),
            wait_for_settlement=wait_for_settlement,
        )

    async def _authenticate(self, latch: ContinuationLatch) -> ContinuationLatch:
        correlation = self._registration.correlation
        self._waits._authorize(correlation.request, self._context, "service")
        record = await self._waits.store._read_external_wait(
            correlation.request.scope, correlation.request.correlation_key
        )
        if record is None or record.registration != self._registration:
            raise ContinuationConflict(
                "External continuation registration is unavailable or changed."
            )
        elected = elected_external_latch(record)
        require_latch_identity(elected, latch)
        # A historical outcome cannot renew revoked service permission after a
        # potentially suspended native read. The projection itself is immutable.
        self._waits._authorize(correlation.request, self._context, "service")
        return elected
