"""Typed identity owner and backend-neutral transaction rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from functools import partial
from typing import ClassVar, Literal, Protocol
from uuid import uuid4

from cayu.collaboration._capabilities import CapabilityDescriptor, FamilyVersion
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
    OwnerRef,
)
from cayu.collaboration._ownership import _MutationOwners
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
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
    ParticipantConfigure,
    ParticipantCreate,
    ParticipantEvent,
    ParticipantInspection,
    ParticipantIntent,
    ParticipantReceipt,
    ParticipantRef,
    ParticipantSnapshot,
)
from cayu.vaults.redaction import SecretRedactor

Table = Literal["anchors", "participants", "configurations", "aliases", "operations", "events"]
Key = tuple[str | int, ...]
IDENTITY_FAMILY = FamilyVersion(family="participant.identity", version=1)
# Reserve the maximum bounded anchor envelope once. Its counters can grow
# without changing the admission decision that those same counters describe.
_ANCHOR_BYTES = 64 * 1024


class _Repository(Protocol):
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


class _Anchor(ContractValue):
    initialization: CollaborationInitialization
    participant_count: Counter = 0
    alias_count: Counter = 0
    operation_count: Counter = 0
    event_count: Counter = 1
    retained_bytes: Counter
    alias_revision: Counter = 0


def _key(expected: ExpectedOperation[ParticipantIntent]) -> Key:
    op = expected.operation
    return op.namespace_incarnation, op.generation, op.caller_key


class CollaborationStore(ABC):
    """Trusted owner API. Public authentication belongs to the application boundary.

    Implementations must provide atomic native transactions, exact immutable
    receipts and cancellation settlement. No raw record write API is public.
    """

    identity_contract_version: ClassVar[int] = 1
    _owners: _MutationOwners

    @abstractmethod
    def _transaction(
        self, scope: str, *, write: bool
    ) -> AbstractAsyncContextManager[_Repository]: ...

    @abstractmethod
    async def close(self) -> None: ...

    def capabilities(self, owner: OwnerRef) -> CapabilityDescriptor:
        return CapabilityDescriptor(
            owner=owner, mutations=(IDENTITY_FAMILY,), readbacks=(IDENTITY_FAMILY,)
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
            size = _ANCHOR_BYTES + len(contract_bytes(event, redactor=redactor))
            limits = binding.limits
            if (
                size > limits.retained_bytes - limits.control_bytes
                or limits.events - limits.control_events < 1
            ):
                raise CollaborationCapacityExceeded("Bootstrap exceeds ordinary evidence capacity.")
            await tx.put(
                "anchors",
                (),
                _Anchor(initialization=initialization, retained_bytes=size),
                insert=True,
            )
            await tx.put("events", (1,), event, insert=True)
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
            or expected.operation.generation != initialized.generation
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
            stored = await tx.get(
                "configurations",
                (snapshot.reference.participant_id, snapshot.configuration_revision),
            )
            if stored is None:
                raise CollaborationUnavailable("Participant receipt configuration is unavailable.")
            require_exact_contract(
                snapshot,
                prepare_contract(ParticipantSnapshot, stored, redactor=redactor),
                redactor=redactor,
            )
        return ExactMatch[ParticipantReceipt](receipt=receipt)

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
                await self._anchor(tx, initialized, redactor)
                return await self._receipt(tx, expected, redactor)
        except CollaborationConflict:
            return ExactConflict()
        except (CollaborationUnavailable, CollaborationContractError):
            return ExactUnavailable()

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
            request = expected.intent.request
            participants: tuple[ParticipantSnapshot, ...]
            alias: ParticipantAlias | None = None
            added_participant = 0
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
                participants = (
                    ParticipantSnapshot(
                        reference=current.reference,
                        configuration=request.configuration,
                        configuration_revision=current.configuration_revision + 1,
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
                sequence=anchor.event_count + 1,
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
            # Count canonical retained document payloads once. Immutable history
            # is never removed; mutable replacements contribute only their delta.
            charge = len(contract_bytes(receipt, redactor=redactor)) + len(
                contract_bytes(event, redactor=redactor)
            )
            if not isinstance(request, ParticipantAliasChange):
                snapshot = participants[0]
                charge += 2 * len(contract_bytes(snapshot, redactor=redactor))
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
            limits = initialized.binding.limits
            updated = _Anchor(
                initialization=initialized,
                participant_count=anchor.participant_count + added_participant,
                alias_count=anchor.alias_count + alias_delta,
                operation_count=anchor.operation_count + 1,
                event_count=event.sequence,
                retained_bytes=anchor.retained_bytes + charge,
                alias_revision=revision,
            )
            if (
                updated.participant_count > limits.participants
                or updated.alias_count > limits.aliases
                or updated.operation_count > limits.operations - limits.control_operations
                or updated.event_count > limits.events - limits.control_events
                or updated.retained_bytes > limits.retained_bytes - limits.control_bytes
            ):
                raise CollaborationCapacityExceeded(
                    "Participant admission exceeds retained capacity."
                )
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
                    participant,
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
            await tx.put("anchors", (), updated, insert=False)
            return receipt

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
            return ParticipantInspection(
                participant=await self._participant(tx, participant, initialized.owner, redactor),
                alias_revision=anchor.alias_revision,
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
        initialized = prepare_contract(CollaborationInitialization, initialized, redactor=redactor)
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
            await self._anchor(tx, initialized, redactor)
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
            return values
