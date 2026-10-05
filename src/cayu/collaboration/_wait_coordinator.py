"""Public bounded wait orchestration over the existing request owner."""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256

from cayu.collaboration._contracts import (
    CollaborationConflict,
    ExactConflict,
    ExactLookup,
    ExactMatch,
    ExactNotFound,
    ExactUnavailable,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration._request_coordinator import RequestCoordinator, _initiator
from cayu.collaboration.access import CollaborationAccessDenied
from cayu.collaboration.mandates import MandateAccessContext
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.requests import RequestObservation
from cayu.collaboration.waits import (
    CollaborationWait,
    WaitEvidence,
    WaitRegistration,
    WaitSnapshot,
    request_object_ref,
    source_key,
    wait_operation_key,
)
from cayu.sessions._session_continuation import RetainedContinuationLatchReceiver


class _WaitObservationNotStarted(Exception):
    """Native pre-mutation read failed; no observation effect was entered."""

    def __init__(self, error: Exception) -> None:
        super().__init__("Wait observation did not enter mutation.")
        self.error = error


class WaitCoordinator:
    """Authenticate request sources, then delegate durable state to the store."""

    def __init__(self, *, participants, requests: RequestCoordinator, redactor) -> None:
        self._participants = participants
        self._requests = requests
        self._redactor = redactor

    def latch_receiver(self):
        """Return the registered durable receiver for session-bound delivery."""
        store, initialized = self._participants._ready()
        return CollaborationWaitLatchReceiver(
            store=store,
            initialized=initialized,
            redactor=self._redactor,
            owners=self._requests.owners,
        )

    async def register(
        self, wait: CollaborationWait, *, context: MandateAccessContext
    ) -> WaitSnapshot:
        wait = prepare_contract(CollaborationWait, wait, redactor=self._redactor)
        await self.authorize_registration(wait, context=context)
        store, initialized = self._participants._ready()
        snapshot = await store.register_wait(initialized, wait, redactor=self._redactor)
        for target in wait.targets:
            await self._register_source_observation(wait, target, context=context)
        return snapshot

    async def authorize_registration(
        self, wait: CollaborationWait, *, context: MandateAccessContext
    ) -> None:
        """Read-only preflight; registration always repeats current authorization."""
        wait = prepare_contract(CollaborationWait, wait, redactor=self._redactor)
        _, initialized = self._participants._ready()
        if wait.source_owner != initialized.owner:
            raise PermissionError("Wait source belongs to another collaboration owner.")
        if wait.initiator != _initiator(context):
            raise CollaborationAccessDenied(
                "Wait initiator does not match authenticated authority."
            )
        # Authenticate every target before accepting the aggregate.  The store
        # still revalidates exact registration identity during its transaction.
        for target in wait.targets:
            snapshot = await self._requests.inspect(target, context=context)
            if snapshot is None:
                raise CollaborationUnavailable("Wait target is unavailable.")

    async def observe(
        self, wait: CollaborationWait, *, context: MandateAccessContext
    ) -> WaitSnapshot:
        return await self._observe(wait, context=context, retained_observer=False)

    async def _observe_owned(self, wait, *, context):
        """Retain each nested native observation beneath the host's bounded waiter."""
        return await self._observe(wait, context=context, retained_observer=True)

    async def _observe(self, wait, *, context, retained_observer):
        wait = prepare_contract(CollaborationWait, wait, redactor=self._redactor)
        store, initialized = self._participants._ready()
        # Authorize every frozen target without requiring its source history
        # to remain retained.  Pending waits still reacquire source evidence
        # below; terminal waits use their retained evidence.
        try:
            for target in wait.targets:
                await self._requests._authorize_retained_source(
                    target, context=context, wait_for_settlement=retained_observer
                )
            current = await store.load_wait(
                initialized, wait, redactor=self._redactor, wait_for_settlement=retained_observer
            )
        except Exception as error:
            if retained_observer:
                # The retained authorization/read has returned. No source registration,
                # evidence write, release or latch dispatch has been entered.
                # Do not extend this handoff across the mutations below, or catch
                # caller cancellation while a nested operation remains owned.
                raise _WaitObservationNotStarted(error) from None
            raise
        if current is None:
            raise CollaborationUnavailable("Wait registration is unavailable.")
        if current.state != "pending":
            if current.state == "elected" and current.delivery == "none" and current.source_pins:
                return await store.release_wait_sources(
                    initialized,
                    wait,
                    redactor=self._redactor,
                    wait_for_settlement=retained_observer,
                )
            return current
        for target in wait.targets:
            try:
                observation = await self._register_source_observation(
                    wait, target, context=context, wait_for_settlement=retained_observer
                )
                source_read = await self._requests.read_observation_source(
                    target, observation, context=context, wait_for_settlement=retained_observer
                )
                page = source_read.page
                if not page.complete:
                    raise CollaborationUnavailable("Wait source coverage is incomplete.")
                evidence = self._evidence_for(source_read.snapshot, target)
            except CollaborationUnavailable:
                # A source read can be unavailable without proving that the
                # request failed.  Record that bounded observation fact so a
                # later retry can replace it with authenticated terminal
                # evidence; never manufacture a failure outcome.
                evidence = WaitEvidence(
                    target=target.intent.selection.reference,
                    status="unavailable",
                    source_sequence=1,
                    accepted_at_ms=1,
                    observed_at_ms=1,
                    receipt_digest=sha256(source_key(target).encode()).hexdigest(),
                )
            current = await store.record_wait_evidence(
                initialized,
                wait,
                evidence,
                redactor=self._redactor,
                wait_for_settlement=retained_observer,
            )
            if current.state != "pending":
                break
        if current.state == "elected" and current.delivery == "none":
            current = await store.release_wait_sources(
                initialized, wait, redactor=self._redactor, wait_for_settlement=retained_observer
            )
        return current

    async def inspect(
        self, wait: CollaborationWait, *, context: MandateAccessContext, wait_for_settlement=False
    ) -> WaitSnapshot | None:
        wait = prepare_contract(CollaborationWait, wait, redactor=self._redactor)
        store, initialized = self._participants._ready()
        for target in wait.targets:
            await self._requests._authorize_retained_source(
                target, context=context, wait_for_settlement=wait_for_settlement
            )
        return await store.load_wait(
            initialized, wait, redactor=self._redactor, wait_for_settlement=wait_for_settlement
        )

    async def lookup(
        self, wait: CollaborationWait, *, context: MandateAccessContext
    ) -> ExactLookup[WaitRegistration]:
        """Return the shared four-way exact registration result.

        The compact registration receipt is the identity lookup.  Callers use
        ``inspect`` for the mutable election/delivery snapshot; returning the
        full snapshot here would make the exact wrapper consume the same
        envelope reserved for terminal evidence.
        """
        wait = prepare_contract(CollaborationWait, wait, redactor=self._redactor)
        store, initialized = self._participants._ready()
        for target in wait.targets:
            await self._requests._authorize_retained_source(target, context=context)
        try:
            current = await store.load_wait(initialized, wait, redactor=self._redactor)
        except CollaborationConflict:
            return ExactConflict()
        except (CollaborationUnavailable, ValueError):
            return ExactUnavailable()
        if current is None:
            return ExactNotFound()
        return ExactMatch[WaitRegistration](receipt=current.registration)

    async def _register_source_observation(
        self, wait, target, *, context, wait_for_settlement=False
    ):
        key = self._observation_key(wait, target)
        observation = RequestObservation(
            key=key,
            filter_commitment=sha256(b"request-outcome").hexdigest(),
            projection_commitment=sha256(
                contract_bytes(target, redactor=self._redactor)
            ).hexdigest(),
            after_sequence=0,
            coverage_sequence=0,
            revision=1,
            retention_until_ms=int(datetime.fromisoformat(wait.deadline).timestamp() * 1000),
        )
        await self._requests._observe_retained_source(
            target, observation, context=context, wait_for_settlement=wait_for_settlement
        )
        # The request API keeps the caller's immutable registration intent
        # separate from its advanced coverage frontier. Reads must replay the
        # former while validating the latter transactionally.
        return observation

    @staticmethod
    def _observation_key(wait, target) -> str:
        return (
            "wait:"
            + sha256(
                (wait_operation_key(wait).__repr__() + source_key(target)).encode()
            ).hexdigest()
        )

    def _evidence_for(self, snapshot, target):
        from cayu.collaboration._preparation import contract_bytes

        if snapshot.outcome is not None:
            outcome = snapshot.outcome
            status = "success" if outcome.command.outcome == "answered" else "failure"
            sequence = outcome.event.sequence
            accepted_at = outcome.elected_at_ms
            digest = sha256(contract_bytes(outcome, redactor=self._redactor)).hexdigest()
            commitment = outcome.command.commitment
        elif snapshot.terminal is not None:
            terminal = snapshot.terminal
            status = "settled"
            sequence = terminal.event.sequence
            accepted_at = terminal.elected_at_ms
            digest = sha256(contract_bytes(terminal, redactor=self._redactor)).hexdigest()
            commitment = None
        else:
            status = "ambiguous"
            sequence = snapshot.receipt.event.sequence
            # There is no source-owned completion timestamp for an open
            # request.  The store assigns the authoritative observation time;
            # this lower bound is deliberately not a worker-clock claim.
            accepted_at = 1
            digest = sha256(contract_bytes(snapshot.receipt, redactor=self._redactor)).hexdigest()
            commitment = None
        return WaitEvidence(
            target=target.intent.selection.reference,
            status=status,
            source_sequence=sequence,
            accepted_at_ms=max(1, accepted_at),
            observed_at_ms=max(1, accepted_at),
            receipt_digest=digest,
            commitment=commitment,
        )

    async def cancel(
        self, wait: CollaborationWait, *, context: MandateAccessContext, expired: bool = False
    ) -> WaitSnapshot:
        wait = prepare_contract(CollaborationWait, wait, redactor=self._redactor)
        store, initialized = self._participants._ready()
        # Authorization is deliberately performed before the exact state result.
        for target in wait.targets:
            await self._requests._authorize_retained_source(target, context=context)
        return await store.cancel_wait(initialized, wait, expired=expired, redactor=self._redactor)

    async def cancel_prepared_registration(
        self, wait: CollaborationWait, *, context: MandateAccessContext
    ) -> WaitSnapshot:
        """Fence a native preparation whose foreign registration may be missing.

        The existing operation key serializes this decision with late registration.
        This is only cancellation of the wait; native exclusion still requires
        the separate released-invocation proof and acknowledgement handshake.
        """
        wait = prepare_contract(CollaborationWait, wait, redactor=self._redactor)
        for target in wait.targets:
            await self._requests._authorize_retained_source(target, context=context)
        store, initialized = self._participants._ready()
        snapshot = await store._owned_wait(
            "cancel_prepared_registration",
            store._register_wait,
            initialized,
            wait,
            self._redactor,
            cancelled_preparation=True,
        )
        if snapshot.state == "pending":
            return await store.cancel_wait(
                initialized, wait, expired=False, redactor=self._redactor
            )
        return snapshot

    async def deliver(self, wait: CollaborationWait, *, context, continuation_owner):
        """Deliver an elected result through the existing authenticated latch owner."""
        return await self._deliver(
            wait, context=context, continuation_owner=continuation_owner, retained_observer=False
        )

    async def _deliver_owned(self, wait, *, context, continuation_owner):
        """The host keeps receiving work attached beyond its foreground bound."""
        return await self._deliver(
            wait, context=context, continuation_owner=continuation_owner, retained_observer=True
        )

    async def _reconcile_delivery_owned(self, wait, *, context, continuation_owner):
        """Repair only the source ACK of a positively retained native latch.

        Safe during host close: no new latch, election or invocation is created.
        A missing native receipt leaves the existing ownership unresolved.
        """
        from cayu.runtime._session_continuation_owner import SessionContinuationOwner

        if type(continuation_owner) is not SessionContinuationOwner:
            raise PermissionError("Wait recovery requires its native continuation owner.")
        wait = prepare_contract(CollaborationWait, wait, redactor=self._redactor)
        for target in wait.targets:
            await self._requests._authorize_retained_source(
                target, context=context, wait_for_settlement=True
            )
        store, initialized = self._participants._ready()
        current = await store.load_wait(
            initialized, wait, redactor=self._redactor, wait_for_settlement=True
        )
        if current is None or current.state != "elected" or current.election is None:
            return None
        if wait.delivery_ticket is None:
            raise CollaborationUnavailable("Wait has no session delivery binding.")
        latch = _elected_latch(current, self._redactor)
        retained = await continuation_owner._inspect_latch_owned(latch)
        if retained is None:
            return None
        # Identity permits ticket lifecycle revisions; receipt bytes must still
        # match the native owner's retained representation, as in normal delivery.
        return await store.record_wait_delivery(
            initialized,
            wait,
            receipt_digest=sha256(
                contract_bytes(retained.latch, redactor=self._redactor)
            ).hexdigest(),
            redactor=self._redactor,
            wait_for_settlement=True,
        )

    async def _deliver(self, wait, *, context, continuation_owner, retained_observer):

        from cayu.runtime._session_continuation_owner import SessionContinuationOwner

        if not isinstance(continuation_owner, SessionContinuationOwner):
            raise PermissionError("Wait delivery requires a registered session continuation owner.")
        if retained_observer and type(continuation_owner) is not SessionContinuationOwner:
            raise PermissionError("Retained wait delivery requires its native receiving owner.")
        wait = prepare_contract(CollaborationWait, wait, redactor=self._redactor)
        store, initialized = self._participants._ready()
        for target in wait.targets:
            await self._requests._authorize_retained_source(
                target, context=context, wait_for_settlement=retained_observer
            )
        current = await store.load_wait(
            initialized, wait, redactor=self._redactor, wait_for_settlement=retained_observer
        )
        if current is None or current.state != "elected" or current.election is None:
            raise CollaborationUnavailable("Wait has no elected result.")
        if wait.delivery_ticket is None:
            raise CollaborationUnavailable("Wait has no session delivery binding.")
        from cayu.sessions._session_continuation import require_latch_identity

        latch = _elected_latch(current, self._redactor)
        record = await (
            continuation_owner._latch_owned(latch)
            if retained_observer
            else continuation_owner.latch(latch)
        )
        if record.latch is None:
            raise CollaborationUnavailable("Session did not acknowledge the exact latch.")
        require_latch_identity(latch, record.latch)
        digest = sha256(contract_bytes(record.latch, redactor=self._redactor)).hexdigest()
        return await store.record_wait_delivery(
            initialized,
            wait,
            receipt_digest=digest,
            redactor=self._redactor,
            wait_for_settlement=retained_observer,
        )

    async def exclude(
        self,
        wait: CollaborationWait,
        *,
        context,
        continuation_owner,
        invocation,
        _released_permit: tuple[str, str] | None = None,
    ) -> WaitSnapshot:
        """Settle cancellation/expiry through a positive session exclusion receipt."""

        from cayu.runtime._session_continuation_owner import SessionContinuationOwner

        if not isinstance(continuation_owner, SessionContinuationOwner):
            raise PermissionError(
                "Wait exclusion requires a registered session continuation owner."
            )
        wait = prepare_contract(CollaborationWait, wait, redactor=self._redactor)
        for target in wait.targets:
            await self._requests._authorize_retained_source(target, context=context)
        return await self._exclude_owned(
            wait,
            continuation_owner=continuation_owner,
            invocation=invocation,
            _released_permit=_released_permit,
        )

    async def _exclude_owned(self, wait, *, continuation_owner, invocation, _released_permit=None):
        """Shared exclusion mutation after the entrance authenticates its authority."""
        store, initialized = self._participants._ready()
        current = await store.load_wait(initialized, wait, redactor=self._redactor)
        if current is None:
            raise CollaborationUnavailable("Wait registration is unavailable.")
        if current.delivery == "excluded" and _released_permit is None:
            return current
        if current.state not in {"cancelled", "expired", "unavailable"}:
            raise CollaborationConflict("Wait is not eligible for exclusion.")
        if wait.delivery_ticket is None:
            raise CollaborationUnavailable("Wait has no session delivery binding.")
        from cayu.sessions._session_continuation import ContinuationRetirement

        reason = (
            "expired"
            if current.state == "expired"
            else "unavailable"
            if current.state == "unavailable"
            else "cancelled"
        )
        if current.terminal_at_ms is None:
            raise CollaborationUnavailable("Wait exclusion timestamp is unavailable.")
        retirement = ContinuationRetirement(
            ticket=wait.delivery_ticket,
            control_id="wait-exclusion:" + current.registration.registration_digest,
            reason=reason,
            # Replay must use the timestamp elected by the collaboration
            # store, not a new worker-clock value.  The retirement command is
            # content-bound and may be acknowledged after restart.
            retired_at=datetime.fromtimestamp(current.terminal_at_ms / 1000, UTC).isoformat(),
        )
        if _released_permit is None:
            record = await continuation_owner.exclude(retirement, invocation=invocation)
        else:
            from cayu.sessions._session_continuation import (
                ContinuationReleasedRetirement,
                require_ticket_identity,
            )

            if invocation is not None:
                raise PermissionError("Released cleanup cannot substitute a live invocation.")
            ticket = wait.delivery_ticket
            receiving = await continuation_owner.store.load_continuation_ticket(
                ticket.session_id,
                session_instance_id=ticket.session_instance_id,
                registration_key=ticket.registration_key,
            )
            if receiving is None:
                raise CollaborationUnavailable("Wait receiving responsibility is unavailable.")
            require_ticket_identity(ticket, receiving.ticket)
            record = await continuation_owner.retire_released(
                ContinuationReleasedRetirement(
                    retirement=retirement.model_copy(update={"ticket": receiving.ticket}),
                    permit_operation=_released_permit[0],
                    permit_commitment=_released_permit[1],
                )
            )
        from cayu.runtime._continuation_wait_settlement import (
            acknowledge_retirement,
            retirement_receipt,
        )

        digest = sha256(
            contract_bytes(retirement_receipt(record), redactor=self._redactor)
        ).hexdigest()
        settled = await store.record_wait_delivery(
            initialized,
            wait,
            receipt_digest=digest,
            delivery="excluded",
            redactor=self._redactor,
        )
        if _released_permit is not None:
            await acknowledge_retirement(continuation_owner, wait, record)
        return settled

    async def _exclude_released_administrative(
        self, wait, *, context, continuation_owner, permit_operation, permit_commitment
    ):
        """Discharge native-verified debt without borrowing disclosure permission."""
        from cayu.collaboration.access import CollaborationAccessContext

        context = prepare_contract(CollaborationAccessContext, context, redactor=self._redactor)
        _, grant = self._participants._authorize(context, "request_control")
        self._participants._require_refs(grant, (), create=True)
        wait = prepare_contract(CollaborationWait, wait, redactor=self._redactor)
        store, initialized = self._participants._ready()
        current = await store.load_wait(initialized, wait, redactor=self._redactor)
        if current is None:
            current = await store._owned_wait(
                "cancel_prepared_registration",
                store._register_wait,
                initialized,
                wait,
                self._redactor,
                cancelled_preparation=True,
            )
        if current.state == "pending":
            await store.cancel_wait(initialized, wait, expired=False, redactor=self._redactor)
        return await self._exclude_owned(
            wait,
            continuation_owner=continuation_owner,
            invocation=None,
            _released_permit=(permit_operation, permit_commitment),
        )

    async def exclude_released(
        self,
        wait: CollaborationWait,
        *,
        context: MandateAccessContext,
        continuation_owner,
        permit_operation: str,
        permit_commitment: str,
    ) -> WaitSnapshot:
        """Settle an interrupted preparation through native released-writer proof."""
        return await self.exclude(
            wait,
            context=context,
            continuation_owner=continuation_owner,
            invocation=None,
            _released_permit=(permit_operation, permit_commitment),
        )


def _elected_latch(snapshot: WaitSnapshot, redactor):
    """Derive the complete receiving command solely from retained election state."""
    from cayu.sessions._session_continuation import ContinuationLatch

    election = snapshot.election
    ticket = snapshot.registration.wait.delivery_ticket
    if snapshot.state != "elected" or election is None or ticket is None:
        raise CollaborationUnavailable("Wait has no session-bound election.")
    return prepare_contract(
        ContinuationLatch,
        ContinuationLatch(
            ticket=ticket,
            wait_receipt_digest=snapshot.registration.registration_digest,
            outcome_kind=election.result,
            selected_manifest=tuple(request_object_ref(ref) for ref in election.selected),
            disclosure_digest=sha256(
                contract_bytes(snapshot.registration, redactor=redactor)
            ).hexdigest(),
            latch_key="wait-latch:" + snapshot.registration.registration_digest,
            outcome_digest=election.outcome_digest,
            accepted_at=datetime.fromtimestamp(election.elected_at_ms / 1000, UTC).isoformat(),
            wait_operation=snapshot.registration.wait.operation,
        ),
        redactor=redactor,
    )


class CollaborationWaitLatchReceiver(RetainedContinuationLatchReceiver):
    """Session continuation receiver backed by durable wait election state."""

    def __init__(self, *, store, initialized, redactor, owners=None) -> None:
        self._store = store
        self._initialized = initialized
        self._redactor = redactor
        # The owning application's mutation view, so its shutdown waits for these;
        # used on its own, the receiver runs on the store's owners.
        self._owners = store._owners if owners is None else owners

    async def authenticate_continuation_retirement(self, wait, record):
        """Authenticate settlement from the exact durable wait, never caller receipts."""
        from cayu.sessions._session_continuation import (
            ContinuationRecord,
            continuation_digest,
            require_ticket_identity,
        )

        wait = prepare_contract(CollaborationWait, wait, redactor=self._redactor)
        record = prepare_contract(ContinuationRecord, record, redactor=self._redactor)
        if wait.delivery_ticket is None or record.released_retirement is None:
            raise PermissionError("Wait has no released receiving responsibility.")
        require_ticket_identity(wait.delivery_ticket, record.ticket)
        if record.ticket.collaboration_wait_sha256 is not None and (
            continuation_digest(wait.model_copy(update={"delivery_ticket": None}))
            != record.ticket.collaboration_wait_sha256
        ):
            raise PermissionError("Retirement belongs to a different complete wait.")
        retained = await self._store.load_wait(self._initialized, wait, redactor=self._redactor)
        expected = continuation_digest(record)
        if retained is None:
            from cayu.collaboration._namespace_store import inspect_retirement
            from cayu.collaboration.lifecycle import NamespaceRef

            evidence = await inspect_retirement(
                self._store,
                self._initialized,
                NamespaceRef(
                    owner=wait.source_owner,
                    namespace_incarnation=wait.operation.namespace_incarnation,
                    generation=wait.operation.generation,
                ),
                self._redactor,
            )
            if evidence is not None and record.ticket.execution_admission_sha256 is not None:
                # Native released-retirement evidence proves this exact operation
                # is fenced; retirement proves no foreign obligation can reopen.
                return expected
        if (
            retained is None
            or retained.registration.wait != wait
            or retained.delivery != "excluded"
            or retained.delivery_receipt_digest != expected
        ):
            raise PermissionError("Wait exclusion has not acknowledged this retirement.")
        return expected

    async def authenticate_continuation_latch(self, latch):
        return await self._receive_latch(latch, wait_for_settlement=False)

    async def _authenticate_latch_owned(self, latch):
        return await self._receive_latch(latch, wait_for_settlement=True)

    async def _receive_latch(self, latch, *, wait_for_settlement):
        from cayu.sessions._session_continuation import ContinuationLatch

        latch = prepare_contract(ContinuationLatch, latch, redactor=self._redactor)
        if latch.wait_operation is None:
            raise PermissionError("Collaboration latch has no source operation binding.")
        if (
            latch.wait_operation.application_scope != self._initialized.binding.application_scope
            or latch.wait_operation.namespace_incarnation != self._initialized.namespace_incarnation
        ):
            raise PermissionError("Collaboration latch belongs to another owner namespace.")
        from functools import partial

        return await self._owners.run(
            partial(self._authenticate_latch, latch),
            key=("wait-latch", self._initialized.binding.application_scope, latch.latch_key),
            expectation=contract_bytes(latch, redactor=self._redactor),
            redactor=self._redactor,
            wait_for_settlement=wait_for_settlement,
        )

    async def _authenticate_latch(self, latch):
        from cayu.collaboration.waits import WaitSnapshot, wait_operation_key
        from cayu.sessions._session_continuation import require_latch_identity

        async with self._store._transaction(
            self._initialized.binding.application_scope, write=False
        ) as tx:
            await self._store._anchor(tx, self._initialized, self._redactor)
            raw = await tx.get("operations", wait_operation_key(latch.wait_operation))
            if not isinstance(raw, dict) or raw.get("mode") != "collaboration_wait":
                raise PermissionError("Collaboration wait election is unavailable.")
            snapshot = prepare_contract(WaitSnapshot, raw, redactor=self._redactor)
            if snapshot.delivery not in {"pending", "accepted"}:
                raise PermissionError("Collaboration latch conflicts with the elected wait.")
            require_latch_identity(_elected_latch(snapshot, self._redactor), latch)
            return latch
