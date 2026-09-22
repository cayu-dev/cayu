"""Typed identity owner and backend-neutral transaction rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from functools import partial
from hashlib import sha256
from typing import ClassVar, Literal, Protocol, cast
from uuid import uuid4

from cayu.collaboration._capabilities import CapabilityDescriptor, FamilyVersion
from cayu.collaboration._capacity import require_capacity
from cayu.collaboration._contracts import (
    CollaborationConflict,
    CollaborationContractError,
    ContractValue,
    ExactConflict,
    ExactLookup,
    ExactMatch,
    ExactNotFound,
    ExactUnavailable,
    ExpectedOperation,
    Generation,
    OperationRef,
    OwnerRef,
)
from cayu.collaboration._history_references import HistoryKey
from cayu.collaboration._ownership import _MutationOwners
from cayu.collaboration._participant_state import ParticipantPermitState
from cayu.collaboration._permits import (
    PermitCommand,
    PermitExclusion,
    PermitReceipt,
    PermitSettlement,
    PermitSettlementReader,
    PermitSnapshot,
    RetiredPermitExclusion,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.collaboration._request_receipts import request_receipt_metadata
from cayu.collaboration.lifecycle import (
    CollaborationHistoryUnavailable,
    LifecycleCommand,
    LifecycleReceipt,
    NamespaceInspection,
    NamespaceRef,
    NamespaceRetirementEvidence,
    NamespaceSnapshot,
)
from cayu.collaboration.obligations import ParticipantObligation
from cayu.collaboration.participants import (
    CollaborationBootstrap,
    CollaborationCapacityExceeded,
    CollaborationInitialization,
    CollaborationNotInitialized,
    CollaborationUnavailable,
    Counter,
    ParticipantAlias,
    ParticipantAliasChange,
    ParticipantCommand,
    ParticipantConfigurationEvidence,
    ParticipantConfigure,
    ParticipantCreate,
    ParticipantEvent,
    ParticipantInspection,
    ParticipantIntent,
    ParticipantLifecycleEvidence,
    ParticipantReceipt,
    ParticipantRef,
    ParticipantSnapshot,
)
from cayu.collaboration.requests import (
    RequestCommand,
    RequestControlCommand,
    RequestControlReceipt,
    RequestReceipt,
)
from cayu.collaboration.waits import wait_identity_digest
from cayu.vaults.redaction import SecretRedactor

Table = Literal[
    "anchors",
    "participants",
    "configurations",
    "aliases",
    "operations",
    "events",
    "namespaces",
    "lifecycle_history",
    "participant_permits",
    "permits",
    "history_uses",
    "requests",
    "request_events",
]
Key = tuple[str | int, ...]
IDENTITY_FAMILY = FamilyVersion(family="participant.identity", version=1)
LIFECYCLE_FAMILY = FamilyVersion(family="collaboration.lifecycle", version=1)
REQUEST_FAMILY = FamilyVersion(family="collaboration.request.acceptance", version=1)
# Reserve the maximum bounded anchor envelope once. Its counters can grow
# without changing the admission decision that those same counters describe.
_ANCHOR_BYTES = 64 * 1024


class _Repository(Protocol):
    async def now_ms(self) -> int:
        """Physical owner time sampled after acquiring transaction ownership."""
        ...

    async def get(self, table: Table, key: Key) -> object | None: ...
    async def put(self, table: Table, key: Key, value: ContractValue, *, insert: bool) -> None: ...
    async def delete(self, table: Table, key: Key) -> None: ...
    async def scan(
        self,
        table: Literal["participants", "events"],
        *,
        after: str | int,
        limit: int,
        allowed: tuple[str, ...] | None,
    ) -> list[object]: ...

    async def scan_permits(
        self, participant_id: str, *, after: int, limit: int, pending_only: bool
    ) -> list[object]: ...

    async def history_in_use(self, family: str, participant_id: str, revision: int) -> bool: ...

    async def scan_operations(
        self, namespace: str, generation: int, *, limit: int
    ) -> list[object]: ...

    async def scan_due_requests(self, *, after: int, now_ms: int, limit: int) -> list[object]: ...

    async def scan_request_events(self, *, after: int, limit: int) -> list[object]: ...


class _Anchor(ContractValue):
    initialization: CollaborationInitialization
    participant_count: Counter
    alias_count: Counter
    operation_count: Counter
    event_count: Counter
    event_sequence: Counter
    current_generation: Generation
    retained_generations: Generation
    retired_through: Counter
    pruned_through: Counter
    retention_revision: Generation
    permit_count: Counter
    reserved_bytes: Counter
    reserved_events: Counter
    retained_bytes: Counter
    alias_revision: Counter
    reserved_operations: Counter = 0


def _consume_wait_reservation(snapshot, updated, *, redactor):
    """Charge materialized wait growth against its terminal reserve."""
    from cayu.collaboration._preparation import contract_bytes

    old_bytes = len(contract_bytes(snapshot, redactor=redactor))
    new_bytes = len(contract_bytes(updated, redactor=redactor))
    byte_growth = max(0, new_bytes - old_bytes)
    event_growth = max(0, len(updated.events) - len(snapshot.events))
    return updated.model_copy(
        update={
            "reserved_bytes": min(
                updated.reserved_bytes,
                max(0, snapshot.reserved_bytes - byte_growth),
            ),
            "reserved_events": min(
                updated.reserved_events,
                max(0, snapshot.reserved_events - event_growth),
            ),
        }
    )


def _wait_event_growth(snapshot, updated) -> int:
    """Return newly retained wait events for aggregate capacity accounting."""
    return len(updated.events) - len(snapshot.events)


def _key(
    expected: ExpectedOperation[ParticipantIntent]
    | LifecycleCommand
    | PermitCommand
    | RequestCommand
    | RequestControlCommand,
) -> Key:
    op = expected.operation
    return op.namespace_incarnation, op.generation, op.caller_key


def _stored_mode(raw: object) -> object:
    """Select a schema only; the selected record still requires full validation."""
    if not isinstance(raw, dict):
        return None
    expected = cast("dict[object, object]", raw).get("expected")
    direct = cast("dict[object, object]", raw).get("mode")
    if isinstance(direct, str):
        return direct
    command = cast("dict[object, object]", raw).get("command")
    if isinstance(command, dict):
        command_map = cast("dict[str, object]", command)
        if isinstance(command_map.get("mode"), str):
            return command_map["mode"]
    if not isinstance(expected, dict):
        return None
    return cast("dict[object, object]", expected).get("mode")


class CollaborationStore(ABC):
    """Trusted owner API. Public authentication belongs to the application boundary.

    Implementations must provide atomic native transactions, exact immutable
    receipts and cancellation settlement. No raw record write API is public.
    """

    identity_contract_version: ClassVar[int] = 1
    request_contract_version: ClassVar[int] = 0
    _owners: _MutationOwners

    @abstractmethod
    def _transaction(
        self, scope: str, *, write: bool
    ) -> AbstractAsyncContextManager[_Repository]: ...

    @abstractmethod
    async def close(self) -> None: ...

    def capabilities(self, owner: OwnerRef) -> CapabilityDescriptor:
        families = (IDENTITY_FAMILY, LIFECYCLE_FAMILY)
        if type(self.request_contract_version) is int and self.request_contract_version == 1:
            families += (REQUEST_FAMILY,)
        return CapabilityDescriptor(
            owner=owner,
            mutations=families,
            readbacks=families,
        )

    async def initialize(
        self,
        binding: CollaborationBootstrap,
        *,
        redactor: SecretRedactor,
    ) -> CollaborationInitialization:
        binding = prepare_contract(CollaborationBootstrap, binding, redactor=redactor)
        return await self._owners.run(
            partial(self._initialize, binding, redactor=redactor),
            key=("bootstrap", binding.application_scope),
            expectation=contract_bytes(binding, redactor=redactor),
            redactor=redactor,
        )

    async def _initialize(
        self, binding: CollaborationBootstrap, *, redactor: SecretRedactor
    ) -> CollaborationInitialization:
        async with self._transaction(binding.application_scope, write=True) as tx:
            old = await tx.get("anchors", ())
            if old is not None:
                anchor = prepare_contract(_Anchor, old, redactor=redactor)
                require_exact_contract(binding, anchor.initialization.binding, redactor=redactor)
                return anchor.initialization
            initialization = prepare_contract(
                CollaborationInitialization,
                CollaborationInitialization(
                    binding=binding,
                    owner=OwnerRef(
                        application_scope=binding.application_scope,
                        owner_id=binding.owner_name,
                        incarnation=uuid4().hex,
                    ),
                    namespace_incarnation=uuid4().hex,
                ),
                redactor=redactor,
            )
            event = prepare_contract(
                ParticipantEvent,
                ParticipantEvent(
                    id=uuid4().hex,
                    sequence=1,
                    operation=None,
                    type="initialized",
                    participants=(),
                ),
                redactor=redactor,
            )
            namespace = NamespaceSnapshot(
                reference=NamespaceRef(
                    owner=initialization.owner,
                    namespace_incarnation=initialization.namespace_incarnation,
                    generation=1,
                ),
                revision=1,
                state="open",
                outstanding_obligations=0,
            )
            size = (
                _ANCHOR_BYTES
                + len(contract_bytes(event, redactor=redactor))
                + len(contract_bytes(namespace, redactor=redactor))
            )
            limits = binding.limits
            if (
                size > limits.retained_bytes - limits.control_bytes
                or limits.events - limits.control_events < 1
            ):
                raise CollaborationCapacityExceeded("Bootstrap exceeds ordinary evidence capacity.")
            await tx.put(
                "anchors",
                (),
                _Anchor(
                    initialization=initialization,
                    retained_bytes=size,
                    participant_count=0,
                    alias_count=0,
                    operation_count=0,
                    event_count=1,
                    event_sequence=1,
                    current_generation=1,
                    retained_generations=1,
                    retired_through=0,
                    pruned_through=0,
                    retention_revision=1,
                    permit_count=0,
                    reserved_bytes=0,
                    reserved_events=0,
                    alias_revision=0,
                ),
                insert=True,
            )
            await tx.put("events", (1,), event, insert=True)
            await tx.put(
                "namespaces", (initialization.namespace_incarnation, 1), namespace, insert=True
            )
            return initialization

    async def _anchor(
        self,
        tx: _Repository,
        expected: CollaborationInitialization,
        redactor: SecretRedactor,
    ) -> _Anchor:
        raw = await tx.get("anchors", ())
        if raw is None:
            raise CollaborationNotInitialized("Collaboration owner is not initialized.")
        anchor = prepare_contract(_Anchor, raw, redactor=redactor)
        require_exact_contract(expected, anchor.initialization, redactor=redactor)
        return anchor

    async def register_wait(self, initialized, wait, *, redactor: SecretRedactor):
        return await self._owned_wait("register", self._register_wait, initialized, wait, redactor)

    async def load_wait(self, initialized, wait, *, redactor: SecretRedactor):
        return await self._owned_wait("load", self._load_wait, initialized, wait, redactor)

    async def record_wait_evidence(self, initialized, wait, evidence, *, redactor: SecretRedactor):
        from cayu.collaboration.waits import WaitEvidence

        evidence = prepare_contract(WaitEvidence, evidence, redactor=redactor)
        return await self._owned_wait(
            "evidence", self._record_wait_evidence, initialized, wait, redactor, evidence=evidence
        )

    async def release_wait_sources(self, initialized, wait, *, redactor: SecretRedactor):
        return await self._owned_wait(
            "release", self._release_wait_sources, initialized, wait, redactor
        )

    async def record_wait_delivery(
        self,
        initialized,
        wait,
        *,
        receipt_digest: str,
        delivery: Literal["accepted", "excluded"] = "accepted",
        redactor: SecretRedactor,
    ):
        return await self._owned_wait(
            "delivery",
            self._record_wait_delivery,
            initialized,
            wait,
            redactor,
            receipt_digest=receipt_digest,
            delivery=delivery,
        )

    async def cancel_wait(self, initialized, wait, *, expired: bool, redactor: SecretRedactor):
        return await self._owned_wait(
            "cancel", self._cancel_wait, initialized, wait, redactor, expired=expired
        )

    async def _owned_wait(self, action, operation, initialized, wait, redactor, **kwargs):
        from cayu._validation import canonical_durable_json_bytes
        from cayu.collaboration._contracts import snapshot_input
        from cayu.collaboration.waits import CollaborationWait, wait_operation_key

        initialized = prepare_contract(CollaborationInitialization, initialized, redactor=redactor)
        wait = prepare_contract(CollaborationWait, wait, redactor=redactor)
        extra = canonical_durable_json_bytes(snapshot_input(kwargs), "wait operation")
        expectation = sha256(
            contract_bytes(initialized, redactor=redactor)
            + b"\x00"
            + contract_bytes(wait, redactor=redactor)
            + b"\x00"
            + extra
        ).digest()
        return await self._owners.run(
            partial(operation, initialized, wait, redactor=redactor, **kwargs),
            key=("wait", action, initialized.binding.application_scope, *wait_operation_key(wait)),
            expectation=expectation,
            redactor=redactor,
        )

    async def _register_wait(
        self,
        initialized: CollaborationInitialization,
        wait,
        *,
        redactor: SecretRedactor,
    ):
        """Persist one exact wait registration without dispatching source work."""

        from cayu.collaboration._namespace_store import require_open_namespace
        from cayu.collaboration.waits import (
            CollaborationWait,
            WaitEvent,
            WaitRegistration,
            WaitSnapshot,
            deadline_ms,
            wait_identity_digest,
            wait_operation_key,
        )

        wait = prepare_contract(CollaborationWait, wait, redactor=redactor)
        if (
            wait.source_owner != initialized.owner
            or wait.operation.application_scope != initialized.binding.application_scope
            or wait.operation.namespace_incarnation != initialized.namespace_incarnation
            or wait.initiator.issuer != initialized.owner
        ):
            raise CollaborationConflict("Wait registration conflicts with initialized authority.")
        key = wait_operation_key(wait)
        async with self._transaction(initialized.binding.application_scope, write=True) as tx:
            anchor = await self._anchor(tx, initialized, redactor)
            raw = await tx.get("operations", key)
            if raw is not None:
                if _stored_mode(raw) != "collaboration_wait":
                    raise CollaborationConflict("Wait key already belongs to another operation.")
                existing = prepare_contract(WaitSnapshot, raw, redactor=redactor)
                if wait_identity_digest(existing.registration.wait) != wait_identity_digest(wait):
                    raise CollaborationConflict("Wait registration intent changed.")
                return existing
            await require_open_namespace(tx, anchor, wait.operation, redactor)
            if await tx.now_ms() >= deadline_ms(wait.deadline):
                raise CollaborationConflict("Wait deadline has already passed.")
            registration = WaitRegistration(
                wait=wait,
                registration_digest=wait_identity_digest(wait),
                registered_sequence=1,
            )
            event = WaitEvent(sequence=1, kind="registered")
            snapshot = WaitSnapshot(
                registration=registration,
                state="pending",
                revision=1,
                source_pins=wait.source_keys,
                events=(event,),
            )
            updated = prepare_contract(
                _Anchor,
                anchor.model_copy(
                    update={
                        "operation_count": anchor.operation_count + 1,
                        "event_count": anchor.event_count + len(snapshot.events),
                        "reserved_events": anchor.reserved_events + snapshot.reserved_events,
                        "reserved_bytes": anchor.reserved_bytes + snapshot.reserved_bytes,
                        "retained_bytes": anchor.retained_bytes
                        + len(contract_bytes(snapshot, redactor=redactor)),
                    }
                ),
                redactor=redactor,
            )
            require_capacity(updated, ordinary=True)
            await tx.put("operations", key, snapshot, insert=True)
            await tx.put("anchors", (), updated, insert=False)
            return snapshot

    async def _load_wait(
        self,
        initialized: CollaborationInitialization,
        wait,
        *,
        redactor: SecretRedactor,
    ):
        from cayu.collaboration.waits import (
            CollaborationWait,
            WaitSnapshot,
            wait_operation_key,
        )

        wait = prepare_contract(CollaborationWait, wait, redactor=redactor)
        async with self._transaction(initialized.binding.application_scope, write=False) as tx:
            raw = await tx.get("operations", wait_operation_key(wait))
            if raw is None:
                return None
            if _stored_mode(raw) != "collaboration_wait":
                raise CollaborationConflict("Wait key belongs to another operation.")
            snapshot = prepare_contract(WaitSnapshot, raw, redactor=redactor)
            if wait_identity_digest(snapshot.registration.wait) != wait_identity_digest(wait):
                raise CollaborationConflict("Wait registration intent changed.")
            return snapshot

    async def _record_wait_evidence(
        self,
        initialized: CollaborationInitialization,
        wait,
        evidence,
        *,
        redactor: SecretRedactor,
    ):
        """Atomically append exact evidence and elect a result when possible."""

        from cayu.collaboration.waits import (
            CollaborationWait,
            WaitElection,
            WaitEvent,
            WaitEvidence,
            WaitSnapshot,
            deadline_ms,
            election_digest,
            evaluate_wait,
            ref_key,
            source_key,
            wait_operation_key,
        )

        wait = prepare_contract(CollaborationWait, wait, redactor=redactor)
        evidence = prepare_contract(WaitEvidence, evidence, redactor=redactor)
        key = wait_operation_key(wait)
        async with self._transaction(initialized.binding.application_scope, write=True) as tx:
            raw = await tx.get("operations", key)
            if raw is None:
                raise CollaborationUnavailable("Wait registration is unavailable.")
            snapshot = prepare_contract(WaitSnapshot, raw, redactor=redactor)
            if wait_identity_digest(snapshot.registration.wait) != wait_identity_digest(wait):
                raise CollaborationConflict("Wait registration intent changed.")
            if ref_key(evidence.target) not in wait.target_keys:
                raise CollaborationConflict("Evidence belongs to another wait target.")
            now = await tx.now_ms()
            prior = next(
                (
                    item
                    for item in snapshot.evidence
                    if ref_key(item.target) == ref_key(evidence.target)
                ),
                None,
            )
            if prior is not None:
                prior_without_ingestion = prior.model_copy(update={"ingestion_sequence": 0})
                if prior_without_ingestion == evidence and not (
                    snapshot.state == "pending" and now >= deadline_ms(wait.deadline)
                ):
                    return snapshot
                if (
                    prior.receipt_digest == evidence.receipt_digest
                    and prior.status == evidence.status
                    and prior.commitment == evidence.commitment
                    and not (snapshot.state == "pending" and now >= deadline_ms(wait.deadline))
                ):
                    # The store owns ingestion and observation timestamps, so
                    # the replayed source evidence will not compare equal
                    # byte-for-byte.  Its authenticated receipt digest and
                    # classification are the stable idempotency identity.
                    return snapshot
                if prior.receipt_digest == evidence.receipt_digest and not (
                    snapshot.state == "pending" and now >= deadline_ms(wait.deadline)
                ):
                    raise CollaborationConflict("Evidence receipt changed for this target.")
            if prior is not None and now < deadline_ms(wait.deadline):
                if (
                    prior.model_copy(
                        update={
                            "accepted_at_ms": evidence.accepted_at_ms,
                            "observed_at_ms": evidence.observed_at_ms,
                            "ingestion_sequence": evidence.ingestion_sequence,
                        }
                    )
                    == evidence
                ):
                    return snapshot
                if prior.status in {"success", "failure", "settled"}:
                    if evidence.status in {"ambiguous", "unavailable"}:
                        return snapshot
                    raise CollaborationConflict("Wait target evidence changed.")
                if evidence.status in {"ambiguous", "unavailable"}:
                    return snapshot
            if snapshot.state != "pending":
                raise CollaborationConflict("A terminal wait cannot accept new evidence.")
            if now >= deadline_ms(wait.deadline):
                # A deadline is an election boundary, not proof that every
                # unresolved target failed.  Close the wait with explicit
                # unavailable evidence so consumers can distinguish an
                # unknown source from a settled failure.
                evidence_items = list(snapshot.evidence)
                events = list(snapshot.events)
                known = {ref_key(item.target) for item in evidence_items}
                for target in wait.targets:
                    target_key = ref_key(target.intent.selection.reference)
                    if target_key in known:
                        continue
                    unavailable = WaitEvidence(
                        target=target.intent.selection.reference,
                        status="unavailable",
                        ingestion_sequence=len(events) + 1,
                        source_sequence=1,
                        accepted_at_ms=now,
                        observed_at_ms=now,
                        receipt_digest=sha256(source_key(target).encode()).hexdigest(),
                    )
                    evidence_items.append(unavailable)
                    events.append(
                        WaitEvent(
                            sequence=len(events) + 1,
                            kind="evidence",
                            target=unavailable.target,
                            evidence_digest=unavailable.receipt_digest,
                        )
                    )
                evidence_tuple = tuple(evidence_items)
                election_sequence = len(events) + 1
                election = WaitElection(
                    result="unavailable",
                    selected=(),
                    evidence=evidence_tuple,
                    sequence=election_sequence,
                    elected_at_ms=now,
                    outcome_digest=election_digest(
                        "unavailable", (), evidence_tuple, election_sequence, now
                    ),
                )
                events.append(WaitEvent(sequence=election_sequence, kind="elected"))
                pending_delivery = wait.delivery_ticket is not None
                expired = snapshot.model_copy(
                    update={
                        "state": "elected",
                        "election": election,
                        "evidence": evidence_tuple,
                        "delivery": "pending" if pending_delivery else "none",
                        "source_pins": snapshot.source_pins if pending_delivery else (),
                        "reserved_bytes": snapshot.reserved_bytes if pending_delivery else 0,
                        "reserved_events": snapshot.reserved_events if pending_delivery else 0,
                        "revision": snapshot.revision + 1,
                        "events": (
                            *events,
                            *(
                                ()
                                if pending_delivery
                                else (WaitEvent(sequence=len(events) + 1, kind="released"),)
                            ),
                        ),
                    }
                )
                expired = _consume_wait_reservation(snapshot, expired, redactor=redactor)
                anchor = await self._anchor(tx, initialized, redactor)
                updated_anchor = prepare_contract(
                    _Anchor,
                    anchor.model_copy(
                        update={
                            "event_count": anchor.event_count
                            + _wait_event_growth(snapshot, expired),
                            "retained_bytes": anchor.retained_bytes
                            + len(contract_bytes(expired, redactor=redactor))
                            - len(contract_bytes(snapshot, redactor=redactor)),
                            "reserved_bytes": anchor.reserved_bytes
                            + expired.reserved_bytes
                            - snapshot.reserved_bytes,
                            "reserved_events": anchor.reserved_events
                            + expired.reserved_events
                            - snapshot.reserved_events,
                        }
                    ),
                    redactor=redactor,
                )
                require_capacity(updated_anchor, ordinary=True)
                await tx.put("operations", key, expired, insert=False)
                await tx.put("anchors", (), updated_anchor, insert=False)
                return expired
            # The source may carry a completion timestamp, but it cannot
            # backdate admission into this wait.  Both admission and
            # observation are owned by this store's physical clock.
            evidence = evidence.model_copy(
                update={
                    "accepted_at_ms": now,
                    "observed_at_ms": now,
                    "ingestion_sequence": len(snapshot.events) + 1,
                }
            )
            evidence_items = (
                *(item for item in snapshot.evidence if item.target != evidence.target),
                evidence,
            )
            events = (
                *snapshot.events,
                WaitEvent(
                    sequence=len(snapshot.events) + 1,
                    kind="evidence",
                    target=evidence.target,
                    evidence_digest=evidence.receipt_digest,
                ),
            )
            result = evaluate_wait(wait, evidence_items)
            election = None
            state = "pending"
            if result is not None:
                selected = tuple(
                    item.target
                    for item in sorted(evidence_items, key=lambda item: item.ingestion_sequence)
                    if item.status == "success"
                )
                sequence = len(events) + 1
                elected_at_ms = evidence.observed_at_ms
                digest = election_digest(result, selected, evidence_items, sequence, elected_at_ms)
                election = WaitElection(
                    result=result,
                    selected=selected,
                    evidence=evidence_items,
                    sequence=sequence,
                    elected_at_ms=elected_at_ms,
                    outcome_digest=digest,
                )
                events = (*events, WaitEvent(sequence=sequence, kind="elected"))
                state = "elected"
            delivery = (
                "pending" if state == "elected" and wait.delivery_ticket is not None else "none"
            )
            terminal_no_delivery = result is not None and wait.delivery_ticket is None
            updated = snapshot.model_copy(
                update={
                    "state": state,
                    "delivery": delivery,
                    "evidence": evidence_items,
                    "election": election,
                    "source_pins": () if terminal_no_delivery else snapshot.source_pins,
                    "reserved_bytes": 0 if terminal_no_delivery else snapshot.reserved_bytes,
                    "reserved_events": 0 if terminal_no_delivery else snapshot.reserved_events,
                    "revision": snapshot.revision + 1,
                    "events": (
                        (*events, WaitEvent(sequence=len(events) + 1, kind="released"))
                        if terminal_no_delivery
                        else events
                    ),
                }
            )
            updated = _consume_wait_reservation(snapshot, updated, redactor=redactor)
            anchor = await self._anchor(tx, initialized, redactor)
            updated = prepare_contract(updated.__class__, updated, redactor=redactor)
            updated_anchor = prepare_contract(
                _Anchor,
                anchor.model_copy(
                    update={
                        "event_count": anchor.event_count + _wait_event_growth(snapshot, updated),
                        "retained_bytes": anchor.retained_bytes
                        + len(contract_bytes(updated, redactor=redactor))
                        - len(contract_bytes(snapshot, redactor=redactor)),
                        "reserved_bytes": anchor.reserved_bytes
                        + updated.reserved_bytes
                        - snapshot.reserved_bytes,
                        "reserved_events": anchor.reserved_events
                        + updated.reserved_events
                        - snapshot.reserved_events,
                    }
                ),
                redactor=redactor,
            )
            require_capacity(updated_anchor, ordinary=True)
            await tx.put("operations", key, updated, insert=False)
            await tx.put("anchors", (), updated_anchor, insert=False)
            return updated

    async def _release_wait_sources(
        self,
        initialized: CollaborationInitialization,
        wait,
        *,
        redactor: SecretRedactor,
    ):
        """Release source-retention pins after an external election is readable."""

        from cayu.collaboration.waits import (
            CollaborationWait,
            WaitEvent,
            WaitSnapshot,
            wait_operation_key,
        )

        wait = prepare_contract(CollaborationWait, wait, redactor=redactor)
        key = wait_operation_key(wait)
        async with self._transaction(initialized.binding.application_scope, write=True) as tx:
            raw = await tx.get("operations", key)
            if raw is None:
                raise CollaborationUnavailable("Wait registration is unavailable.")
            snapshot = prepare_contract(WaitSnapshot, raw, redactor=redactor)
            if wait_identity_digest(snapshot.registration.wait) != wait_identity_digest(wait):
                raise CollaborationConflict("Wait registration intent changed.")
            if snapshot.state != "elected" or not snapshot.source_pins:
                return snapshot
            if snapshot.registration.wait.delivery_ticket is not None:
                raise CollaborationConflict(
                    "Session-bound source retention requires an acknowledged delivery receipt."
                )
            released = snapshot.model_copy(
                update={
                    "source_pins": (),
                    "revision": snapshot.revision + 1,
                    "events": (
                        *snapshot.events,
                        WaitEvent(sequence=len(snapshot.events) + 1, kind="released"),
                    ),
                }
            )
            released = _consume_wait_reservation(snapshot, released, redactor=redactor)
            old_bytes = len(contract_bytes(snapshot, redactor=redactor))
            new_bytes = len(contract_bytes(released, redactor=redactor))
            anchor = await self._anchor(tx, initialized, redactor)
            updated_anchor = prepare_contract(
                _Anchor,
                anchor.model_copy(
                    update={
                        "event_count": anchor.event_count + _wait_event_growth(snapshot, released),
                        "retained_bytes": anchor.retained_bytes + new_bytes - old_bytes,
                        "reserved_bytes": anchor.reserved_bytes
                        + released.reserved_bytes
                        - snapshot.reserved_bytes,
                        "reserved_events": anchor.reserved_events
                        + released.reserved_events
                        - snapshot.reserved_events,
                    }
                ),
                redactor=redactor,
            )
            require_capacity(updated_anchor, ordinary=True)
            await tx.put("operations", key, released, insert=False)
            await tx.put("anchors", (), updated_anchor, insert=False)
            return released

    async def _record_wait_delivery(
        self,
        initialized: CollaborationInitialization,
        wait,
        *,
        receipt_digest: str,
        delivery: Literal["accepted", "excluded"] = "accepted",
        redactor: SecretRedactor,
    ):
        from cayu.collaboration.waits import (
            CollaborationWait,
            WaitEvent,
            WaitSnapshot,
            wait_operation_key,
        )

        wait = prepare_contract(CollaborationWait, wait, redactor=redactor)
        key = wait_operation_key(wait)
        async with self._transaction(initialized.binding.application_scope, write=True) as tx:
            raw = await tx.get("operations", key)
            if raw is None:
                raise CollaborationUnavailable("Wait registration is unavailable.")
            snapshot = prepare_contract(WaitSnapshot, raw, redactor=redactor)
            if wait_identity_digest(snapshot.registration.wait) != wait_identity_digest(wait):
                raise CollaborationConflict("Wait registration intent changed.")
            if delivery == "accepted" and (
                snapshot.state != "elected" or snapshot.election is None
            ):
                raise CollaborationConflict("Wait has no elected delivery.")
            if delivery == "excluded" and snapshot.state not in {
                "cancelled",
                "expired",
                "unavailable",
            }:
                raise CollaborationConflict("Wait is not eligible for exclusion.")
            if snapshot.delivery in {"accepted", "excluded"}:
                if (
                    snapshot.delivery != delivery
                    or snapshot.delivery_receipt_digest != receipt_digest
                ):
                    raise CollaborationConflict("Wait delivery receipt changed.")
                return snapshot
            if (
                snapshot.registration.wait.delivery_ticket is not None
                and snapshot.delivery != "pending"
            ):
                raise CollaborationConflict("Wait delivery has no pending handoff responsibility.")
            if snapshot.delivery not in {"none", "pending"}:
                raise CollaborationConflict("Wait delivery is already terminal.")
            updated = snapshot.model_copy(
                update={
                    "delivery": delivery,
                    "delivery_receipt_digest": receipt_digest,
                    "source_pins": (),
                    "reserved_bytes": 0,
                    "reserved_events": 0,
                    "revision": snapshot.revision + 1,
                    "events": (
                        *snapshot.events,
                        WaitEvent(sequence=len(snapshot.events) + 1, kind="released"),
                    ),
                }
            )
            updated = _consume_wait_reservation(snapshot, updated, redactor=redactor)
            anchor = await self._anchor(tx, initialized, redactor)
            updated_anchor = prepare_contract(
                _Anchor,
                anchor.model_copy(
                    update={
                        "event_count": anchor.event_count + _wait_event_growth(snapshot, updated),
                        "retained_bytes": anchor.retained_bytes
                        + len(contract_bytes(updated, redactor=redactor))
                        - len(contract_bytes(snapshot, redactor=redactor)),
                        "reserved_bytes": anchor.reserved_bytes - snapshot.reserved_bytes,
                        "reserved_events": anchor.reserved_events - snapshot.reserved_events,
                    }
                ),
                redactor=redactor,
            )
            require_capacity(updated_anchor, ordinary=True)
            await tx.put("operations", key, updated, insert=False)
            await tx.put("anchors", (), updated_anchor, insert=False)
            return updated

    async def _cancel_wait(
        self,
        initialized: CollaborationInitialization,
        wait,
        *,
        expired: bool,
        redactor: SecretRedactor,
    ):
        from cayu.collaboration.waits import (
            CollaborationWait,
            WaitEvent,
            WaitSnapshot,
            deadline_ms,
            wait_identity_digest,
            wait_operation_key,
        )

        wait = prepare_contract(CollaborationWait, wait, redactor=redactor)
        key = wait_operation_key(wait)
        async with self._transaction(initialized.binding.application_scope, write=True) as tx:
            raw = await tx.get("operations", key)
            if raw is None:
                raise CollaborationUnavailable("Wait registration is unavailable.")
            snapshot = prepare_contract(WaitSnapshot, raw, redactor=redactor)
            if wait_identity_digest(snapshot.registration.wait) != wait_identity_digest(wait):
                raise CollaborationConflict("Wait registration intent changed.")
            if snapshot.state != "pending":
                return snapshot
            now = await tx.now_ms()
            deadline_passed = now >= deadline_ms(wait.deadline)
            if expired and not deadline_passed:
                raise CollaborationConflict("Wait deadline has not passed.")
            # Expiry is the authoritative classification at and after the
            # owner deadline, even when a caller submits an ordinary cancel
            # at the same boundary.  This keeps cancellation from racing the
            # store-owned deadline into a different terminal result.
            state = "expired" if expired or deadline_passed else "cancelled"
            pending_delivery = wait.delivery_ticket is not None
            updated_events = (
                *snapshot.events,
                WaitEvent(sequence=len(snapshot.events) + 1, kind=state),
            )
            if snapshot.source_pins and not pending_delivery:
                updated_events = (
                    *updated_events,
                    WaitEvent(sequence=len(updated_events) + 1, kind="released"),
                )
            updated = snapshot.model_copy(
                update={
                    "state": state,
                    "terminal_at_ms": now,
                    "revision": snapshot.revision + 1,
                    "delivery": "pending" if pending_delivery else "none",
                    "source_pins": snapshot.source_pins if pending_delivery else (),
                    "reserved_bytes": snapshot.reserved_bytes if pending_delivery else 0,
                    "reserved_events": snapshot.reserved_events if pending_delivery else 0,
                    "events": updated_events,
                }
            )
            updated = _consume_wait_reservation(snapshot, updated, redactor=redactor)
            anchor = await self._anchor(tx, initialized, redactor)
            updated_anchor = prepare_contract(
                _Anchor,
                anchor.model_copy(
                    update={
                        "event_count": anchor.event_count + _wait_event_growth(snapshot, updated),
                        "retained_bytes": anchor.retained_bytes
                        + len(contract_bytes(updated, redactor=redactor))
                        - len(contract_bytes(snapshot, redactor=redactor)),
                        "reserved_bytes": anchor.reserved_bytes
                        + updated.reserved_bytes
                        - snapshot.reserved_bytes,
                        "reserved_events": anchor.reserved_events
                        + updated.reserved_events
                        - snapshot.reserved_events,
                    }
                ),
                redactor=redactor,
            )
            require_capacity(updated_anchor, ordinary=True)
            await tx.put("operations", key, updated, insert=False)
            await tx.put("anchors", (), updated_anchor, insert=False)
            return updated

    def _command(
        self,
        initialized: CollaborationInitialization,
        expected: ExpectedOperation[ParticipantIntent],
        redactor: SecretRedactor,
    ) -> ParticipantCommand:
        expected = prepare_contract(ParticipantCommand, expected, redactor=redactor)
        request = expected.intent.request
        if (
            expected.source != initialized.owner
            or expected.destination != initialized.owner
            or expected.initiator.issuer != initialized.owner
            or any(
                v is not None
                for v in (
                    expected.initiator.participant,
                    expected.initiator.mandate,
                    expected.initiator.invocation_id,
                    expected.initiator.interaction_id,
                )
            )
            or expected.operation != request.operation
            or expected.operation.application_scope != initialized.binding.application_scope
            or expected.operation.namespace_incarnation != initialized.namespace_incarnation
            or expected.kind != request.kind
            or expected.schema_version != 1
            or expected.mode != "identity"
            or expected.receipt_stage != "committed"
            or expected.intent.limits != initialized.binding.limits
        ):
            raise CollaborationConflict("Participant command conflicts with initialized authority.")
        return expected

    async def _receipt(
        self,
        tx: _Repository,
        expected: ExpectedOperation[ParticipantIntent],
        redactor: SecretRedactor,
    ) -> ExactLookup[ParticipantReceipt]:
        raw = await tx.get("operations", _key(expected))
        if raw is None:
            return ExactNotFound()
        if request_receipt_metadata(raw, redactor=redactor) is not None:
            return ExactConflict()
        if _stored_mode(raw) == "lifecycle":
            prepare_contract(LifecycleReceipt, raw, redactor=redactor)
            return ExactConflict()
        if _stored_mode(raw) == "permit":
            from cayu.collaboration._permit_store import prepare_permit_record

            prepare_permit_record(raw, redactor)
            return ExactConflict()
        if _stored_mode(raw) == "request":
            prepare_contract(RequestReceipt, raw, redactor=redactor)
            return ExactConflict()
        if _stored_mode(raw) == "request_control":
            prepare_contract(RequestControlReceipt, raw, redactor=redactor)
            return ExactConflict()
        receipt = prepare_contract(ParticipantReceipt, raw, redactor=redactor)
        try:
            require_exact_contract(expected, receipt.expected, redactor=redactor)
        except CollaborationConflict:
            return ExactConflict()
        event = await tx.get("events", (receipt.event.sequence,))
        if event is None:
            raise CollaborationUnavailable("Participant receipt event is unavailable.")
        require_exact_contract(
            receipt.event,
            prepare_contract(ParticipantEvent, event, redactor=redactor),
            redactor=redactor,
        )
        for snapshot in receipt.participants:
            await self._require_snapshot_history(tx, snapshot, redactor)
        return ExactMatch[ParticipantReceipt](receipt=receipt)

    async def _require_snapshot_history(
        self, tx: _Repository, snapshot: ParticipantSnapshot, redactor: SecretRedactor
    ) -> None:
        stored = await tx.get(
            "configurations",
            (
                snapshot.reference.participant_id,
                snapshot.configuration_revision,
            ),
        )
        if stored is None:
            raise CollaborationUnavailable("Participant configuration history is unavailable.")
        configuration = prepare_contract(
            ParticipantConfigurationEvidence, stored, redactor=redactor
        )
        if (
            configuration.reference != snapshot.reference
            or configuration.configuration != snapshot.configuration
            or configuration.configuration_revision != snapshot.configuration_revision
        ):
            raise CollaborationUnavailable("Participant configuration history conflicts.")
        raw = await tx.get(
            "lifecycle_history",
            (
                snapshot.reference.participant_id,
                snapshot.lifecycle_revision,
            ),
        )
        if raw is None:
            raise CollaborationUnavailable("Participant lifecycle history is unavailable.")
        require_exact_contract(
            ParticipantLifecycleEvidence.from_snapshot(snapshot),
            prepare_contract(ParticipantLifecycleEvidence, raw, redactor=redactor),
            redactor=redactor,
        )

    async def _permit_state(
        self, tx: _Repository, ref: ParticipantRef, redactor: SecretRedactor
    ) -> ParticipantPermitState:
        raw = await tx.get("participant_permits", (ref.participant_id,))
        if raw is None:
            raise CollaborationUnavailable("Participant permit frontier is unavailable.")
        state = prepare_contract(ParticipantPermitState, raw, redactor=redactor)
        if state.reference != ref:
            raise CollaborationUnavailable("Participant permit frontier authority conflicts.")
        return state

    async def lookup(
        self,
        initialized: CollaborationInitialization,
        expected: ExpectedOperation[ParticipantIntent],
        *,
        redactor: SecretRedactor,
    ) -> ExactLookup[ParticipantReceipt]:
        initialized = prepare_contract(CollaborationInitialization, initialized, redactor=redactor)
        try:
            expected = self._command(initialized, expected, redactor)
        except CollaborationConflict:
            return ExactConflict()
        material = contract_bytes(expected, redactor=redactor)
        return await self._owners.run(
            partial(self._lookup, initialized, expected, redactor=redactor),
            # Reads do not elect intent. Distinct expectations must each be
            # compared with durable evidence, not with another reader.
            key=("readback", initialized.binding.application_scope, *_key(expected), material),
            expectation=material,
            redactor=redactor,
        )

    async def _lookup(
        self,
        initialized: CollaborationInitialization,
        expected: ParticipantCommand,
        *,
        redactor: SecretRedactor,
    ) -> ExactLookup[ParticipantReceipt]:
        try:
            async with self._transaction(initialized.binding.application_scope, write=False) as tx:
                anchor = await self._anchor(tx, initialized, redactor)
                result = await self._receipt(tx, expected, redactor)
                if isinstance(result, ExactNotFound):
                    return await self._missing_operation(
                        tx,
                        anchor,
                        expected.operation.generation,
                        redactor,
                    )
                return result
        except CollaborationConflict:
            return ExactConflict()
        except (CollaborationUnavailable, CollaborationContractError):
            return ExactUnavailable()

    async def _missing_operation(
        self,
        tx: _Repository,
        anchor: _Anchor,
        generation: int,
        redactor: SecretRedactor,
    ) -> ExactNotFound | ExactConflict | ExactUnavailable:
        from cayu.collaboration._namespace_store import load_namespace

        if generation <= anchor.pruned_through:
            return ExactUnavailable()
        if generation > anchor.current_generation:
            return ExactConflict()
        namespace = await load_namespace(tx, anchor, generation, redactor)
        if namespace.content == "partial":
            return ExactUnavailable()
        return ExactNotFound()

    async def _participant(
        self, tx: _Repository, ref: ParticipantRef, owner: OwnerRef, redactor: SecretRedactor
    ) -> ParticipantSnapshot:
        if ref.owner != owner:
            raise CollaborationConflict("Participant belongs to a different owner.")
        raw = await tx.get("participants", (ref.participant_id,))
        if raw is None:
            raise CollaborationUnavailable("Participant is unavailable.")
        snapshot = prepare_contract(ParticipantSnapshot, raw, redactor=redactor)
        if snapshot.reference != ref:
            raise CollaborationConflict("Participant incarnation conflicts.")
        await self._require_snapshot_history(tx, snapshot, redactor)
        return snapshot

    async def apply(
        self,
        initialized: CollaborationInitialization,
        expected: ExpectedOperation[ParticipantIntent],
        *,
        redactor: SecretRedactor,
    ) -> ParticipantReceipt:
        initialized = prepare_contract(CollaborationInitialization, initialized, redactor=redactor)
        expected = self._command(initialized, expected, redactor)
        return await self._owners.run(
            partial(self._apply, initialized, expected, redactor=redactor),
            key=("mutation", initialized.binding.application_scope, *_key(expected)),
            expectation=contract_bytes(expected, redactor=redactor),
            redactor=redactor,
        )

    async def _apply(
        self,
        initialized: CollaborationInitialization,
        expected: ParticipantCommand,
        *,
        redactor: SecretRedactor,
    ) -> ParticipantReceipt:
        async with self._transaction(initialized.binding.application_scope, write=True) as tx:
            anchor = await self._anchor(tx, initialized, redactor)
            replay = await self._receipt(tx, expected, redactor)
            if isinstance(replay, ExactMatch):
                return replay.receipt
            if isinstance(replay, ExactConflict):
                raise CollaborationConflict("Operation key already has different intent.")
            from cayu.collaboration._namespace_store import require_open_namespace

            await require_open_namespace(tx, anchor, expected.operation, redactor)
            request = expected.intent.request
            participants: tuple[ParticipantSnapshot, ...]
            alias: ParticipantAlias | None = None
            added_participant = 0
            superseded_histories: tuple[HistoryKey, ...] = ()
            alias_delta = 0
            revision = anchor.alias_revision
            if isinstance(request, ParticipantCreate):
                added_participant = 1
                participants = (
                    ParticipantSnapshot(
                        reference=ParticipantRef(
                            owner=initialized.owner,
                            participant_id=uuid4().hex,
                            incarnation=uuid4().hex,
                        ),
                        configuration=request.configuration,
                        configuration_revision=1,
                    ),
                )
                event_type = "created"
                if request.alias is not None:
                    if (
                        request.expected_alias_revision != revision
                        or await tx.get("aliases", (request.alias,)) is not None
                    ):
                        raise CollaborationConflict("Alias binding changed or is already occupied.")
                    revision += 1
                    alias_delta = 1
                    alias = ParticipantAlias(
                        alias=request.alias, target=participants[0].reference, revision=revision
                    )
            elif isinstance(request, ParticipantConfigure):
                current = await self._participant(
                    tx, request.participant, initialized.owner, redactor
                )
                if current.configuration_revision != request.expected_configuration_revision:
                    raise CollaborationConflict("Participant configuration revision changed.")
                if current.lifecycle == "retired":
                    raise CollaborationConflict("Retired participant cannot be reconfigured.")
                superseded_histories = (
                    (
                        "configurations",
                        current.reference.participant_id,
                        current.configuration_revision,
                    ),
                )
                participants = (
                    current.model_copy(
                        update={
                            "configuration": request.configuration,
                            "configuration_revision": current.configuration_revision + 1,
                        }
                    ),
                )
                event_type = "configured"
            else:
                assert isinstance(request, ParticipantAliasChange)
                raw = await tx.get("aliases", (request.alias,))
                current_alias = (
                    None
                    if raw is None
                    else prepare_contract(ParticipantAlias, raw, redactor=redactor)
                )
                if (
                    revision != request.expected_alias_revision
                    or (None if current_alias is None else current_alias.target)
                    != request.expected_target
                ):
                    raise CollaborationConflict("Alias revision or expected target changed.")
                refs = tuple(
                    ref for ref in (request.expected_target, request.target) if ref is not None
                )
                participants = tuple(
                    [await self._participant(tx, ref, initialized.owner, redactor) for ref in refs]
                )
                if request.target is not None and participants[-1].lifecycle == "retired":
                    raise CollaborationConflict("Alias cannot bind a retired participant.")
                revision += 1
                alias_delta = int(request.target is not None) - int(current_alias is not None)
                alias = (
                    None
                    if request.target is None
                    else ParticipantAlias(
                        alias=request.alias, target=request.target, revision=revision
                    )
                )
                event_type = "alias_changed"
            event = ParticipantEvent(
                id=uuid4().hex,
                sequence=anchor.event_sequence + 1,
                operation=expected.operation,
                type=event_type,
                participants=tuple(p.reference for p in participants),
            )
            receipt = prepare_contract(
                ParticipantReceipt,
                ParticipantReceipt(
                    expected=expected,
                    participants=participants,
                    alias=alias,
                    alias_revision=revision,
                    event=event,
                ),
                redactor=redactor,
            )
            # Count canonical retained document payloads once. Mutable replacements
            # contribute their delta; unreferenced history is reclaimed below.
            charge = len(contract_bytes(receipt, redactor=redactor)) + len(
                contract_bytes(event, redactor=redactor)
            )
            if isinstance(request, ParticipantCreate):
                charge += len(
                    contract_bytes(
                        ParticipantLifecycleEvidence.from_snapshot(participants[0]),
                        redactor=redactor,
                    )
                ) + len(
                    contract_bytes(
                        ParticipantPermitState(
                            reference=participants[0].reference, issued_frontier=0, outstanding=0
                        ),
                        redactor=redactor,
                    )
                )
            if not isinstance(request, ParticipantAliasChange):
                snapshot = participants[0]
                charge += len(contract_bytes(snapshot, redactor=redactor)) + len(
                    contract_bytes(
                        ParticipantConfigurationEvidence.from_snapshot(snapshot),
                        redactor=redactor,
                    )
                )
                previous = await tx.get("participants", (snapshot.reference.participant_id,))
                if previous is not None:
                    charge -= len(
                        contract_bytes(
                            prepare_contract(ParticipantSnapshot, previous, redactor=redactor),
                            redactor=redactor,
                        )
                    )
            if (
                isinstance(request, (ParticipantCreate, ParticipantAliasChange))
                and request.alias is not None
            ):
                previous_alias = await tx.get("aliases", (request.alias,))
                if previous_alias is not None:
                    charge -= len(
                        contract_bytes(
                            prepare_contract(ParticipantAlias, previous_alias, redactor=redactor),
                            redactor=redactor,
                        )
                    )
                if alias is not None:
                    charge += len(contract_bytes(alias, redactor=redactor))
            updated = anchor.model_copy(
                update={
                    "participant_count": anchor.participant_count + added_participant,
                    "alias_count": anchor.alias_count + alias_delta,
                    "operation_count": anchor.operation_count + 1,
                    "event_count": anchor.event_count + 1,
                    "event_sequence": event.sequence,
                    "retained_bytes": anchor.retained_bytes + charge,
                    "alias_revision": revision,
                }
            )
            updated = prepare_contract(_Anchor, updated, redactor=redactor)
            if not isinstance(request, ParticipantAliasChange):
                participant = receipt.participants[0]
                await tx.put(
                    "participants",
                    (participant.reference.participant_id,),
                    participant,
                    insert=isinstance(request, ParticipantCreate),
                )
                await tx.put(
                    "configurations",
                    (participant.reference.participant_id, participant.configuration_revision),
                    ParticipantConfigurationEvidence.from_snapshot(participant),
                    insert=True,
                )
                if isinstance(request, ParticipantCreate):
                    await tx.put(
                        "lifecycle_history",
                        (participant.reference.participant_id, 1),
                        ParticipantLifecycleEvidence.from_snapshot(participant),
                        insert=True,
                    )
                    await tx.put(
                        "participant_permits",
                        (participant.reference.participant_id,),
                        ParticipantPermitState(
                            reference=participant.reference, issued_frontier=0, outstanding=0
                        ),
                        insert=True,
                    )
            if isinstance(request, ParticipantAliasChange) or (
                isinstance(request, ParticipantCreate) and request.alias is not None
            ):
                if receipt.alias is None:
                    assert request.alias is not None
                    await tx.delete("aliases", (request.alias,))
                else:
                    await tx.put("aliases", (receipt.alias.alias,), receipt.alias, insert=False)
            await tx.put("operations", _key(expected), receipt, insert=True)
            await tx.put("events", (event.sequence,), receipt.event, insert=True)
            if superseded_histories:
                from cayu.collaboration._retention_store import release_unused_history

                released = await release_unused_history(tx, superseded_histories, redactor)
                updated = prepare_contract(
                    _Anchor,
                    updated.model_copy(
                        update={
                            "retained_bytes": updated.retained_bytes - released,
                        }
                    ),
                    redactor=redactor,
                )
            require_capacity(updated, ordinary=True)
            await tx.put("anchors", (), updated, insert=False)
            return receipt

    async def inspect_namespace(
        self, initialized: CollaborationInitialization, *, redactor: SecretRedactor
    ) -> NamespaceInspection:
        from cayu.collaboration._namespace_store import inspect_namespace

        initialized = prepare_contract(CollaborationInitialization, initialized, redactor=redactor)
        return await inspect_namespace(self, initialized, redactor)

    async def inspect_retirement(
        self,
        initialized: CollaborationInitialization,
        namespace: NamespaceRef,
        *,
        redactor: SecretRedactor,
    ) -> NamespaceRetirementEvidence | None:
        from cayu.collaboration._namespace_store import inspect_retirement

        initialized = prepare_contract(CollaborationInitialization, initialized, redactor=redactor)
        namespace = prepare_contract(NamespaceRef, namespace, redactor=redactor)
        return await inspect_retirement(self, initialized, namespace, redactor)

    async def _register_permit(
        self,
        initialized: CollaborationInitialization,
        expected: PermitCommand,
        *,
        redactor: SecretRedactor,
    ) -> PermitReceipt:
        """Trusted owner integration only; registration never dispatches execution."""
        from cayu.collaboration._permit_store import prepare_permit, register_permit

        initialized = prepare_contract(CollaborationInitialization, initialized, redactor=redactor)
        expected = prepare_permit(initialized, expected, redactor)
        return await self._owners.run(
            partial(register_permit, self, initialized, expected, redactor),
            key=("mutation", initialized.binding.application_scope, *_key(expected)),
            expectation=contract_bytes(expected, redactor=redactor),
            redactor=redactor,
        )

    async def _lookup_registered_permit(
        self,
        initialized: CollaborationInitialization,
        operation: OperationRef,
        *,
        redactor: SecretRedactor,
    ) -> PermitReceipt | None:
        """Read one retained registration for acknowledgement-loss recovery."""
        from cayu.collaboration._permit_store import prepare_permit_record, registered_receipt

        initialized = prepare_contract(CollaborationInitialization, initialized, redactor=redactor)
        operation = prepare_contract(OperationRef, operation, redactor=redactor)
        async with self._transaction(initialized.binding.application_scope, write=False) as tx:
            anchor = await self._anchor(tx, initialized, redactor)
            raw = await tx.get(
                "operations",
                (operation.namespace_incarnation, operation.generation, operation.caller_key),
            )
            if raw is None:
                return None
            from cayu.collaboration.base import _stored_mode

            if _stored_mode(raw) != "permit":
                raise CollaborationConflict("Operation key already has different intent.")
            record = prepare_permit_record(raw, redactor)
            if not isinstance(record, PermitReceipt):
                return None
            result = await registered_receipt(tx, record.expected, redactor)
            if result is None or result.expected.operation != operation:
                raise CollaborationUnavailable("Retained permit identity conflicts.")
            del anchor
            return result

    async def scan_obligations(
        self,
        initialized: CollaborationInitialization,
        participant: ParticipantRef,
        *,
        after: int,
        limit: int,
        pending_only: bool,
        retention_revision: int | None,
        redactor: SecretRedactor,
    ) -> tuple[int, tuple[ParticipantObligation, ...]]:
        initialized = prepare_contract(CollaborationInitialization, initialized, redactor=redactor)
        participant = prepare_contract(ParticipantRef, participant, redactor=redactor)
        if (
            type(after) is not int
            or not 0 <= after <= 2**53 - 1
            or type(limit) is not int
            or not 1 <= limit <= 64
            or type(pending_only) is not bool
            or (
                retention_revision is not None
                and (
                    type(retention_revision) is not int or not 1 <= retention_revision <= 2**53 - 1
                )
            )
        ):
            raise ValueError("Invalid obligation query bounds.")
        async with self._transaction(initialized.binding.application_scope, write=False) as tx:
            anchor = await self._anchor(tx, initialized, redactor)
            await self._participant(tx, participant, initialized.owner, redactor)
            if retention_revision is not None and retention_revision != anchor.retention_revision:
                raise CollaborationHistoryUnavailable(
                    "Obligation cursor predates retained history."
                )
            records = await tx.scan_permits(
                participant.participant_id, after=after, limit=limit, pending_only=pending_only
            )
            results = []
            previous = after
            for raw in records:
                value = prepare_contract(PermitSnapshot, raw, redactor=redactor)
                request = value.expected.intent.request
                if (
                    request.participant != participant
                    or value.position <= previous
                    or (pending_only and value.state != "pending")
                ):
                    raise CollaborationUnavailable("Obligation query evidence conflicts.")
                results.append(
                    prepare_contract(
                        ParticipantObligation,
                        ParticipantObligation(
                            participant=request.participant,
                            position=value.position,
                            operation=request.operation,
                            source_operation=request.source_operation,
                            settlement_operation=request.settlement_operation,
                            admission_generation=request.admission_generation,
                            expected_configuration_revision=request.expected_configuration_revision,
                            admission_commitment=request.admission_commitment,
                            target=request.target,
                            target_state=request.target_state,
                            effect_scope=request.effect_scope,
                            required_settlement=request.required_settlement,
                            state=value.state,
                            outcome=None if value.settlement is None else value.settlement.outcome,
                        ),
                        redactor=redactor,
                    )
                )
                previous = value.position
            return anchor.retention_revision, tuple(results)

    async def _settle_permit(
        self,
        initialized: CollaborationInitialization,
        expected: PermitCommand,
        *,
        reader: PermitSettlementReader,
        redactor: SecretRedactor,
    ) -> PermitSettlement:
        """Consume trusted receiving-owner readback, never a caller-supplied proof."""
        from cayu.collaboration._permit_store import prepare_permit, settle_permit

        initialized = prepare_contract(CollaborationInitialization, initialized, redactor=redactor)
        expected = prepare_permit(initialized, expected, redactor)
        operation = expected.intent.request.settlement_operation
        return await self._owners.run(
            partial(settle_permit, self, initialized, expected, reader, redactor),
            key=(
                "mutation",
                initialized.binding.application_scope,
                operation.namespace_incarnation,
                operation.generation,
                operation.caller_key,
            ),
            expectation=contract_bytes(expected, redactor=redactor),
            redactor=redactor,
        )

    async def _exclude_permit(
        self,
        initialized: CollaborationInitialization,
        expected: PermitCommand,
        *,
        reader: PermitSettlementReader,
        redactor: SecretRedactor,
    ) -> PermitExclusion | PermitSettlement | RetiredPermitExclusion:
        """Qualified receiver exclusion, not a caller-supplied abort assertion."""
        from cayu.collaboration._permit_store import exclude_permit, prepare_permit

        initialized = prepare_contract(CollaborationInitialization, initialized, redactor=redactor)
        expected = prepare_permit(initialized, expected, redactor)
        return await self._owners.run(
            partial(exclude_permit, self, initialized, expected, reader, redactor),
            key=("permit_exclusion", initialized.binding.application_scope, *_key(expected)),
            expectation=contract_bytes(expected, redactor=redactor),
            redactor=redactor,
        )

    async def apply_lifecycle(
        self,
        initialized: CollaborationInitialization,
        expected: LifecycleCommand,
        *,
        redactor: SecretRedactor,
    ) -> LifecycleReceipt:
        from cayu.collaboration._namespace_store import apply_lifecycle, prepare_lifecycle

        initialized = prepare_contract(CollaborationInitialization, initialized, redactor=redactor)
        expected = prepare_lifecycle(initialized, expected, redactor)
        return await self._owners.run(
            partial(apply_lifecycle, self, initialized, expected, redactor),
            key=("mutation", initialized.binding.application_scope, *_key(expected)),
            expectation=contract_bytes(expected, redactor=redactor),
            redactor=redactor,
        )

    async def lookup_lifecycle(
        self,
        initialized: CollaborationInitialization,
        expected: LifecycleCommand,
        *,
        redactor: SecretRedactor,
    ) -> ExactLookup[LifecycleReceipt]:
        from cayu.collaboration._namespace_store import prepare_lifecycle

        initialized = prepare_contract(CollaborationInitialization, initialized, redactor=redactor)
        try:
            expected = prepare_lifecycle(initialized, expected, redactor)
        except CollaborationConflict:
            return ExactConflict()
        material = contract_bytes(expected, redactor=redactor)
        return await self._owners.run(
            partial(self._lookup_lifecycle, initialized, expected, redactor=redactor),
            key=("readback", initialized.binding.application_scope, *_key(expected), material),
            expectation=material,
            redactor=redactor,
        )

    async def _lookup_lifecycle(
        self,
        initialized: CollaborationInitialization,
        expected: LifecycleCommand,
        *,
        redactor: SecretRedactor,
    ) -> ExactLookup[LifecycleReceipt]:
        from cayu.collaboration._namespace_store import lifecycle_replay

        try:
            async with self._transaction(initialized.binding.application_scope, write=False) as tx:
                anchor = await self._anchor(tx, initialized, redactor)
                receipt = await lifecycle_replay(self, tx, expected, redactor)
                if receipt is not None:
                    return ExactMatch[LifecycleReceipt](receipt=receipt)
                return await self._missing_operation(
                    tx, anchor, expected.operation.generation, redactor
                )
        except CollaborationConflict:
            return ExactConflict()
        except (CollaborationUnavailable, CollaborationContractError):
            return ExactUnavailable()

    async def inspect(
        self,
        initialized: CollaborationInitialization,
        participant: ParticipantRef,
        *,
        redactor: SecretRedactor,
    ) -> ParticipantInspection:
        initialized = prepare_contract(CollaborationInitialization, initialized, redactor=redactor)
        participant = prepare_contract(ParticipantRef, participant, redactor=redactor)
        async with self._transaction(initialized.binding.application_scope, write=False) as tx:
            anchor = await self._anchor(tx, initialized, redactor)
            snapshot = await self._participant(tx, participant, initialized.owner, redactor)
            permits = await self._permit_state(tx, participant, redactor)
            return ParticipantInspection(
                participant=snapshot,
                alias_revision=anchor.alias_revision,
                issued_permit_frontier=permits.issued_frontier,
                outstanding_obligations=permits.outstanding,
                settlement="unsettled" if permits.outstanding else "settled",
            )

    async def resolve_alias(
        self,
        initialized: CollaborationInitialization,
        alias: str,
        *,
        redactor: SecretRedactor,
    ) -> ParticipantAlias | None:
        from cayu.collaboration._contracts import Identifier

        class AliasQuery(ContractValue):
            name: Identifier

        initialized = prepare_contract(CollaborationInitialization, initialized, redactor=redactor)
        alias = prepare_contract(AliasQuery, {"name": alias}, redactor=redactor).name
        async with self._transaction(initialized.binding.application_scope, write=False) as tx:
            await self._anchor(tx, initialized, redactor)
            raw = await tx.get("aliases", (alias,))
            if raw is None:
                return None
            binding = prepare_contract(ParticipantAlias, raw, redactor=redactor)
            if binding.alias != alias:
                raise CollaborationUnavailable("Alias index conflicts with its binding.")
            await self._participant(tx, binding.target, initialized.owner, redactor)
            return binding

    async def scan(
        self,
        initialized: CollaborationInitialization,
        *,
        table: Literal["participants", "events"],
        after: str | int,
        limit: int,
        allowed: tuple[ParticipantRef, ...] | None,
        redactor: SecretRedactor,
    ) -> tuple[ParticipantSnapshot | ParticipantEvent, ...]:
        _, records = await self._scan_page(
            initialized,
            table=table,
            after=after,
            limit=limit,
            allowed=allowed,
            retention_revision=None,
            redactor=redactor,
        )
        return records

    async def scan_events(
        self,
        initialized: CollaborationInitialization,
        *,
        after: int,
        limit: int,
        allowed: tuple[ParticipantRef, ...] | None,
        retention_revision: int | None,
        redactor: SecretRedactor,
    ) -> tuple[int, tuple[ParticipantEvent, ...]]:
        revision, records = await self._scan_page(
            initialized,
            table="events",
            after=after,
            limit=limit,
            allowed=allowed,
            retention_revision=retention_revision,
            redactor=redactor,
        )
        return revision, cast("tuple[ParticipantEvent, ...]", records)

    async def _scan_page(
        self,
        initialized: CollaborationInitialization,
        *,
        table: Literal["participants", "events"],
        after: str | int,
        limit: int,
        allowed: tuple[ParticipantRef, ...] | None,
        retention_revision: int | None,
        redactor: SecretRedactor,
    ) -> tuple[int, tuple[ParticipantSnapshot | ParticipantEvent, ...]]:
        initialized = prepare_contract(CollaborationInitialization, initialized, redactor=redactor)
        if retention_revision is not None and (
            type(retention_revision) is not int or not 1 <= retention_revision <= 2**53 - 1
        ):
            raise ValueError("Invalid retention revision.")
        if (
            type(limit) is not int
            or not 1 <= limit <= 64
            or table not in ("participants", "events")
        ):
            raise ValueError("Invalid participant page.")
        if (table == "participants" and type(after) is not str) or (
            table == "events" and (type(after) is not int or after < 0)
        ):
            raise ValueError("Invalid participant cursor.")
        if allowed is not None:
            if type(allowed) is not tuple or len(allowed) > 64:
                raise ValueError("Invalid participant selection.")
            allowed = tuple(
                prepare_contract(ParticipantRef, ref, redactor=redactor) for ref in allowed
            )
            if any(ref.owner != initialized.owner for ref in allowed):
                raise CollaborationConflict("Participant selection belongs to another owner.")
        async with self._transaction(initialized.binding.application_scope, write=False) as tx:
            anchor = await self._anchor(tx, initialized, redactor)
            if retention_revision is not None and retention_revision != anchor.retention_revision:
                raise CollaborationHistoryUnavailable("Event cursor predates retained history.")
            ids = None if allowed is None else tuple(ref.participant_id for ref in allowed)
            rows = await tx.scan(table, after=after, limit=limit, allowed=ids)
            schema = ParticipantSnapshot if table == "participants" else ParticipantEvent
            values = tuple(prepare_contract(schema, row, redactor=redactor) for row in rows)
            for value in values:
                refs = (
                    (value.reference,)
                    if isinstance(value, ParticipantSnapshot)
                    else value.participants
                )
                if any(
                    ref.owner != initialized.owner or (allowed is not None and ref not in allowed)
                    for ref in refs
                ):
                    raise CollaborationUnavailable(
                        "Participant query evidence conflicts with scope."
                    )
            return anchor.retention_revision, values
