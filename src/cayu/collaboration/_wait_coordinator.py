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
        )

    async def register(
        self, wait: CollaborationWait, *, context: MandateAccessContext
    ) -> WaitSnapshot:
        wait = prepare_contract(CollaborationWait, wait, redactor=self._redactor)
        store, initialized = self._participants._ready()
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
        snapshot = await store.register_wait(initialized, wait, redactor=self._redactor)
        for target in wait.targets:
            await self._register_source_observation(wait, target, context=context)
        return snapshot

    async def observe(
        self, wait: CollaborationWait, *, context: MandateAccessContext
    ) -> WaitSnapshot:
        wait = prepare_contract(CollaborationWait, wait, redactor=self._redactor)
        store, initialized = self._participants._ready()
        # Authorize every frozen target without requiring its source history
        # to remain retained.  Pending waits still reacquire source evidence
        # below; terminal waits use their retained evidence.
        for target in wait.targets:
            await self._requests._authorize_retained_source(target, context=context)
        current = await store.load_wait(initialized, wait, redactor=self._redactor)
        if current is None:
            raise CollaborationUnavailable("Wait registration is unavailable.")
        if current.state != "pending":
            if current.state == "elected" and current.delivery == "none" and current.source_pins:
                return await store.release_wait_sources(initialized, wait, redactor=self._redactor)
            return current
        for target in wait.targets:
            try:
                observation = await self._register_source_observation(wait, target, context=context)
                source_read = await self._requests.read_observation_source(
                    target, observation, context=context
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
            )
            if current.state != "pending":
                break
        if current.state == "elected" and current.delivery == "none":
            current = await store.release_wait_sources(initialized, wait, redactor=self._redactor)
        return current

    async def inspect(
        self, wait: CollaborationWait, *, context: MandateAccessContext
    ) -> WaitSnapshot | None:
        wait = prepare_contract(CollaborationWait, wait, redactor=self._redactor)
        store, initialized = self._participants._ready()
        for target in wait.targets:
            await self._requests._authorize_retained_source(target, context=context)
        return await store.load_wait(initialized, wait, redactor=self._redactor)

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

    async def _register_source_observation(self, wait, target, *, context):
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
        await self._requests._observe_retained_source(target, observation, context=context)
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

    async def deliver(self, wait: CollaborationWait, *, context, continuation_owner):
        """Deliver an elected result through the existing authenticated latch owner."""

        from cayu.runtime._session_continuation_owner import SessionContinuationOwner

        if not isinstance(continuation_owner, SessionContinuationOwner):
            raise PermissionError("Wait delivery requires a registered session continuation owner.")
        wait = prepare_contract(CollaborationWait, wait, redactor=self._redactor)
        store, initialized = self._participants._ready()
        for target in wait.targets:
            await self._requests._authorize_retained_source(target, context=context)
        current = await store.load_wait(initialized, wait, redactor=self._redactor)
        if current is None or current.state != "elected" or current.election is None:
            raise CollaborationUnavailable("Wait has no elected result.")
        if wait.delivery_ticket is None:
            raise CollaborationUnavailable("Wait has no session delivery binding.")
        from cayu.runtime._session_continuation import require_latch_identity

        latch = _elected_latch(current, self._redactor)
        record = await continuation_owner.latch(latch)
        if record.latch is None:
            raise CollaborationUnavailable("Session did not acknowledge the exact latch.")
        require_latch_identity(latch, record.latch)
        digest = sha256(contract_bytes(record.latch, redactor=self._redactor)).hexdigest()
        return await store.record_wait_delivery(
            initialized,
            wait,
            receipt_digest=digest,
            redactor=self._redactor,
        )

    async def exclude(
        self, wait: CollaborationWait, *, context, continuation_owner, invocation
    ) -> WaitSnapshot:
        """Settle cancellation/expiry through a positive session exclusion receipt."""

        from cayu.runtime._session_continuation_owner import SessionContinuationOwner

        if not isinstance(continuation_owner, SessionContinuationOwner):
            raise PermissionError(
                "Wait exclusion requires a registered session continuation owner."
            )
        wait = prepare_contract(CollaborationWait, wait, redactor=self._redactor)
        store, initialized = self._participants._ready()
        for target in wait.targets:
            await self._requests._authorize_retained_source(target, context=context)
        current = await store.load_wait(initialized, wait, redactor=self._redactor)
        if current is None:
            raise CollaborationUnavailable("Wait registration is unavailable.")
        if current.delivery == "excluded":
            return current
        if current.state not in {"cancelled", "expired", "unavailable"}:
            raise CollaborationConflict("Wait is not eligible for exclusion.")
        if wait.delivery_ticket is None:
            raise CollaborationUnavailable("Wait has no session delivery binding.")
        from cayu.runtime._session_continuation import ContinuationRetirement

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
        record = await continuation_owner.exclude(retirement, invocation=invocation)
        digest = sha256(contract_bytes(record, redactor=self._redactor)).hexdigest()
        return await store.record_wait_delivery(
            initialized,
            wait,
            receipt_digest=digest,
            delivery="excluded",
            redactor=self._redactor,
        )


def _elected_latch(snapshot: WaitSnapshot, redactor):
    """Derive the complete receiving command solely from retained election state."""
    from cayu.runtime._session_continuation import ContinuationLatch

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


class CollaborationWaitLatchReceiver:
    """Session continuation receiver backed by durable wait election state."""

    def __init__(self, *, store, initialized, redactor) -> None:
        self._store = store
        self._initialized = initialized
        self._redactor = redactor

    async def authenticate_continuation_latch(self, latch):
        from cayu.runtime._session_continuation import ContinuationLatch

        latch = prepare_contract(ContinuationLatch, latch, redactor=self._redactor)
        if latch.wait_operation is None:
            raise PermissionError("Collaboration latch has no source operation binding.")
        if (
            latch.wait_operation.application_scope != self._initialized.binding.application_scope
            or latch.wait_operation.namespace_incarnation != self._initialized.namespace_incarnation
        ):
            raise PermissionError("Collaboration latch belongs to another owner namespace.")
        from functools import partial

        return await self._store._owners.run(
            partial(self._authenticate_latch, latch),
            key=("wait-latch", self._initialized.binding.application_scope, latch.latch_key),
            expectation=contract_bytes(latch, redactor=self._redactor),
            redactor=self._redactor,
        )

    async def _authenticate_latch(self, latch):
        from cayu.collaboration.waits import WaitSnapshot, wait_operation_key
        from cayu.runtime._session_continuation import require_latch_identity

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
