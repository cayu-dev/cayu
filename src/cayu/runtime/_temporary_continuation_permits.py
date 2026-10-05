"""Configured foreign permit readback for the existing continuation receiver.

Registration delegates to the collaboration admission transaction; receiving reads
its exact durable result before consuming it. Caller receipts never grant authority.
No cross-store transaction is held.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from time import monotonic

from cayu.collaboration._clarification_services import ClarificationServiceRecord
from cayu.collaboration._contracts import ExactMatch, ExactUnavailable, OwnerRef
from cayu.collaboration._ownership import _MutationOwners, _MutationScope
from cayu.collaboration._permits import (
    PermitCommand,
    PermitSettlementReader,
    ReceivingSettlementReceipt,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.base import CollaborationStore
from cayu.collaboration.participants import CollaborationInitialization, CollaborationUnavailable
from cayu.deadlines import ExecutionDeadline
from cayu.sessions._session_continuation import (
    ContinuationConflict,
    continuation_digest,
)
from cayu.sessions._temporary_continuation import (
    TemporaryServiceAdmission,
    TemporaryServiceDispatch,
    TemporaryServiceIntent,
    TemporaryServicePreparation,
)
from cayu.sessions.base import SessionStore
from cayu.vaults.redaction import SecretRedactor


class TemporaryServicePermitAuthority:
    """Runtime registration dependency, never accepted in a service request."""

    def __init__(
        self,
        store: CollaborationStore,
        initialized: CollaborationInitialization,
        *,
        redactor: SecretRedactor,
        admission_guard: Callable[[TemporaryServiceDispatch], AbstractAsyncContextManager[int]]
        | None = None,
        owners: _MutationOwners | _MutationScope | None = None,
    ) -> None:
        self.store = store
        # The owning application's mutation view, so its shutdown waits for these;
        # used on its own, the authority runs on the store's owners.
        self.owners = store._owners if owners is None else owners
        self.initialized = prepare_contract(
            CollaborationInitialization, initialized, redactor=redactor
        )
        self.redactor = redactor
        self.admission_guard = admission_guard

    async def execution_deadline(self, candidate: TemporaryServiceIntent) -> ExecutionDeadline:
        """Translate owner UTC into a conservative local execution duration.

        Charge the complete clock-read round trip, including acknowledgement
        latency. No worker wall-clock offset may extend the question/wait bound.
        This is a timer, not registration, exclusion, or renewed authority.
        """
        intent = prepare_contract(TemporaryServiceIntent, candidate, redactor=self.redactor)
        if intent.operation.application_scope != self.initialized.binding.application_scope:
            raise PermissionError("Temporary service belongs to another application scope.")
        started = monotonic()
        async with self.store._transaction(
            self.initialized.binding.application_scope, write=False
        ) as tx:
            await self.store._anchor(tx, self.initialized, self.redactor)
            now_ms = await tx.now_ms()
        remaining = min(
            intent.question.policy.service_timeout_ms / 1000,
            (intent.question.deadline_at_ms - now_ms) / 1000,
            float("inf")
            if intent.ticket.deadline is None
            else datetime.fromisoformat(intent.ticket.deadline).timestamp() - now_ms / 1000,
        ) - (monotonic() - started)
        return ExecutionDeadline.after(
            max(0.0, remaining), source="clarification_service", scope="temporary_service"
        )

    async def lookup(self, candidate: TemporaryServiceIntent) -> ClarificationServiceRecord | None:
        """Exact durable service discovery before any runtime re-entry."""
        from cayu.collaboration._permit_store import registered_receipt

        intent = prepare_contract(TemporaryServiceIntent, candidate, redactor=self.redactor)
        if intent.operation.application_scope != self.initialized.binding.application_scope:
            raise PermissionError("Temporary service belongs to another application scope.")
        async with self.store._transaction(
            self.initialized.binding.application_scope, write=False
        ) as tx:
            await self.store._anchor(tx, self.initialized, self.redactor)
            raw = await tx.get("clarification_services", operation_key(intent.operation))
            if raw is None:
                return None
            record = prepare_contract(ClarificationServiceRecord, raw, redactor=self.redactor)
            if intent != record.dispatch.intent:
                raise ContinuationConflict("Temporary service operation has different intent.")
            if await registered_receipt(tx, record.permit.expected, self.redactor) != record.permit:
                raise PermissionError("Temporary service registration evidence is unavailable.")
            return record

    async def prepare(self, candidate: TemporaryServiceDispatch) -> TemporaryServicePreparation:
        from cayu.collaboration._clarification_service_store import (
            prepare_runtime_service_in_transaction,
        )

        dispatch = prepare_contract(TemporaryServiceDispatch, candidate, redactor=self.redactor)
        async with self.store._transaction(
            self.initialized.binding.application_scope, write=False
        ) as tx:
            return await prepare_runtime_service_in_transaction(
                self.store, tx, self.initialized, dispatch, redactor=self.redactor
            )

    async def register(
        self, candidate: TemporaryServiceDispatch | TemporaryServicePreparation
    ) -> ClarificationServiceRecord:
        """Called after runtime preparation; retain disclosure inside the owned task.

        A registered public coordinator supplies admission_guard. The private
        no-guard composition remains for owners already holding authorization;
        it must not be used as the public service authorization boundary.
        """
        from cayu.collaboration._clarification_service_store import (
            register_runtime_service_in_transaction,
            register_service_in_transaction,
        )

        preparation = (
            prepare_contract(TemporaryServicePreparation, candidate, redactor=self.redactor)
            if isinstance(candidate, TemporaryServicePreparation)
            else None
        )
        dispatch = (
            preparation.dispatch
            if preparation is not None
            else prepare_contract(TemporaryServiceDispatch, candidate, redactor=self.redactor)
        )
        operation = dispatch.intent.operation

        async def commit(authority_expires_at_ms: int | None):
            async with self.store._transaction(
                self.initialized.binding.application_scope, write=True
            ) as tx:
                if preparation is not None:
                    return await register_service_in_transaction(
                        self.store,
                        tx,
                        self.initialized,
                        dispatch,
                        preparation.permit,
                        redactor=self.redactor,
                        authority_expires_at_ms=authority_expires_at_ms,
                    )
                return await register_runtime_service_in_transaction(
                    self.store,
                    tx,
                    self.initialized,
                    dispatch,
                    redactor=self.redactor,
                    authority_expires_at_ms=authority_expires_at_ms,
                )

        async def mutation():
            if self.admission_guard is None:
                return await commit(None)
            # An existing permit is historical authority for exactly its work.
            # Do not renew it or replace its lifecycle tuple after disablement.
            prior = await self.lookup(dispatch.intent)
            if prior is not None:
                if prior.dispatch != dispatch or (
                    preparation is not None and prior.permit.expected != preparation.permit
                ):
                    raise ContinuationConflict("Registered service preparation conflicts.")
                return prior
            result = None
            async with self.admission_guard(dispatch) as expires_at_ms:
                if type(expires_at_ms) is not int or not 0 < expires_at_ms <= 2**53 - 1:
                    raise PermissionError("Service authorization requires bounded expiry evidence.")
                result = await commit(expires_at_ms)
            if result is None:
                raise CollaborationUnavailable("Service guard produced no registered decision.")
            return result

        return await self.owners.run(
            mutation,
            key=("clarification-service", operation.application_scope, *operation_key(operation)),
            expectation=contract_bytes(preparation or dispatch, redactor=self.redactor),
            redactor=self.redactor,
        )

    async def authenticate(self, candidate: TemporaryServiceAdmission) -> TemporaryServiceAdmission:
        admission = prepare_contract(TemporaryServiceAdmission, candidate, redactor=self.redactor)
        if admission.permit.source != self.initialized.owner:
            raise PermissionError("Temporary service belongs to another permit owner.")
        retained = await self.store._lookup_registered_permit(
            self.initialized, admission.permit.operation, redactor=self.redactor
        )
        if (
            retained is None
            or retained.expected != admission.permit
            or continuation_digest(retained) != admission.permit_receipt_sha256
        ):
            raise PermissionError("Temporary service lacks its exact registered permit.")
        async with self.store._transaction(
            self.initialized.binding.application_scope, write=False
        ) as tx:
            await self.store._anchor(tx, self.initialized, self.redactor)
            raw = await tx.get(
                "clarification_services", operation_key(admission.dispatch.intent.operation)
            )
        if raw is None:
            raise PermissionError("Temporary service lacks its charged durable responsibility.")
        service = prepare_contract(ClarificationServiceRecord, raw, redactor=self.redactor)
        if service.dispatch != admission.dispatch or service.permit != retained:
            raise PermissionError("Temporary service responsibility conflicts with its permit.")
        # Deliberately do not replace this with a new participant lifecycle read:
        # a permit registered before disablement remains valid for its exact work.
        return admission

    async def settle(
        self, admission: TemporaryServiceAdmission, *, reader: PermitSettlementReader
    ) -> None:
        from cayu.collaboration._clarification_service_store import settle_service_in_transaction

        if await self._retired_settlement(admission, reader=reader):
            return
        authenticated = await self.authenticate(admission)
        await self.store._settle_permit(
            self.initialized, authenticated.permit, reader=reader, redactor=self.redactor
        )
        # A lost acknowledgement between these commits leaves a discoverable
        # pending service. Reconciliation consumes the same durable settlement.
        async with self.store._transaction(
            self.initialized.binding.application_scope, write=True
        ) as tx:
            await settle_service_in_transaction(
                self.store,
                tx,
                self.initialized,
                authenticated.dispatch,
                authenticated.permit,
                redactor=self.redactor,
            )

    async def _retired_settlement(
        self,
        candidate: TemporaryServiceAdmission | TemporaryServicePreparation,
        *,
        reader: PermitSettlementReader,
        require_exclusion: bool = False,
    ) -> bool:
        """Recover an ACK after pruning, never authorize execution from retirement.

        The receiving owner must still hold the exact positive terminal receipt.
        Namespace retirement independently certifies that all registered service
        and permit obligations settled before their records could be pruned.
        Missing records alone prove neither settlement nor exclusion.
        """
        from cayu.collaboration._namespace_store import inspect_retirement
        from cayu.collaboration.lifecycle import NamespaceRef

        expected = prepare_contract(type(candidate), candidate, redactor=self.redactor)
        permit = expected.permit
        operation = permit.operation
        if (
            permit.source != self.initialized.owner
            or reader.owner != permit.intent.request.target.owner
        ):
            raise PermissionError("Temporary settlement belongs to another receiving owner.")
        retired = await inspect_retirement(
            self.store,
            self.initialized,
            NamespaceRef(
                owner=self.initialized.owner,
                namespace_incarnation=operation.namespace_incarnation,
                generation=operation.generation,
            ),
            self.redactor,
        )
        if retired is None:
            return False
        observed = await reader.lookup(permit)
        if not isinstance(observed, ExactMatch):
            raise CollaborationUnavailable("Retirement requires exact native terminal evidence.")
        receipt = prepare_contract(
            ReceivingSettlementReceipt, observed.receipt, redactor=self.redactor
        )
        if (
            receipt.expected != permit
            or receipt.receiving_owner != reader.owner
            or (require_exclusion and not receipt.proves_exclusion)
        ):
            raise PermissionError("Retired settlement does not match the exact native operation.")
        return True

    async def exclude(
        self, expected: TemporaryServicePreparation, *, reader: PermitSettlementReader
    ) -> None:
        """Discharge only a receiving-owner fence, including lost registration ACKs."""
        from cayu.collaboration._clarification_service_store import settle_service_in_transaction

        expected = prepare_contract(TemporaryServicePreparation, expected, redactor=self.redactor)
        if await self._retired_settlement(expected, reader=reader, require_exclusion=True):
            return
        await self.store._exclude_permit(
            self.initialized, expected.permit, reader=reader, redactor=self.redactor
        )
        async with self.store._transaction(
            self.initialized.binding.application_scope, write=True
        ) as tx:
            if (
                await tx.get(
                    "clarification_services", operation_key(expected.dispatch.intent.operation)
                )
                is not None
            ):
                await settle_service_in_transaction(
                    self.store,
                    tx,
                    self.initialized,
                    expected.dispatch,
                    expected.permit,
                    redactor=self.redactor,
                )


class TemporaryServiceSettlementReader(PermitSettlementReader):
    """Positive native release/exclusion evidence; absence never means exclusion."""

    def __init__(
        self,
        store: SessionStore,
        admission: TemporaryServiceAdmission | TemporaryServicePreparation,
        *,
        redactor: SecretRedactor,
    ) -> None:
        self.store = store
        self.admission = (
            prepare_contract(TemporaryServiceAdmission, admission, redactor=redactor)
            if isinstance(admission, TemporaryServiceAdmission)
            else prepare_contract(TemporaryServicePreparation, admission, redactor=redactor)
        )
        self.redactor = redactor

    @property
    def owner(self) -> OwnerRef:
        return self.admission.dispatch.intent.target.owner

    async def lookup(self, expected: PermitCommand):
        expected = prepare_contract(PermitCommand, expected, redactor=self.redactor)
        if expected != self.admission.permit:
            raise ContinuationConflict("Temporary settlement expected another permit.")
        retained = await self.store._load_temporary_continuation_service(self.admission)
        if retained is None:
            return ExactUnavailable()
        # Preserve exact already-returned evidence after native receipt compaction.
        if retained.settlement is not None:
            return ExactMatch[ReceivingSettlementReceipt](receipt=retained.settlement)
        if not isinstance(self.admission, TemporaryServiceAdmission):
            return ExactUnavailable()
        native = await self.store._read_temporary_continuation_outcome(self.admission)
        if native is None or native.settlement is None:
            return ExactUnavailable()
        return ExactMatch[ReceivingSettlementReceipt](receipt=native.settlement)
