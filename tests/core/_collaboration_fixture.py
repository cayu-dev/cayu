"""Deliberately limited in-process receiver, never production store evidence."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import ClassVar, Literal

from pydantic import StrictBool  # noqa: TC002 - Pydantic resolves this annotation at runtime.
from tests.core.collaboration_conformance import AsyncFaultBarrier, FaultPoint, ReceiverState

from cayu.collaboration._capabilities import CapabilityDescriptor, FamilyVersion
from cayu.collaboration._contracts import (
    CollaborationConflict,
    ContractValue,
    ExactConflict,
    ExactLookup,
    ExactMatch,
    ExactNotFound,
    ExactUnavailable,
    ExpectedOperation,
    Generation,
    HandoffIntent,
    HandoffSlot,
    Identifier,
    InitiatorBinding,
    OperationRef,
    OwnerRef,
)
from cayu.collaboration._preparation import prepare_contract, require_exact_contract
from cayu.vaults.redaction import SecretRedactor


class ProbeIntent(ContractValue):
    unordered_fields: ClassVar[frozenset[str]] = frozenset({"references"})
    target: Identifier
    references: tuple[Identifier, ...] = ()
    ordered: tuple[Identifier, ...] = ()
    policy: Identifier = "policy-1"
    enabled: StrictBool = True
    threshold: Generation = 1
    selected: Identifier | None = None


ProbeCommand = ExpectedOperation[ProbeIntent]


class ProbeReceipt(ContractValue):
    expected: ProbeCommand
    stage: Literal["applied", "excluded"]
    selection: Identifier


def command(*, key: str = "key", scope: str = "app") -> ProbeCommand:
    source = OwnerRef(application_scope=scope, owner_id="source", incarnation="one")
    return ProbeCommand(
        operation=OperationRef(
            application_scope=scope,
            namespace_incarnation="ns",
            generation=1,
            caller_key=key,
        ),
        kind="probe",
        schema_version=1,
        mode="send",
        source=source,
        destination=OwnerRef(application_scope=scope, owner_id="destination", incarnation="one"),
        initiator=InitiatorBinding(
            issuer=source,
            principal="principal",
            participant=None,
            mandate=None,
            invocation_id=None,
            interaction_id=None,
        ),
        receipt_stage="applied",
        intent=ProbeIntent(target="B"),
    )


def slot(cmd: ProbeCommand, name: str = "delivery") -> HandoffSlot:
    return HandoffSlot(source=cmd.source, parent=cmd.operation, slot=name)


class InjectedFault(ConnectionError):
    pass


@dataclass
class TestState:
    __test__ = False
    effects: int = 0
    events: int = 0


class LimitedReceiver:
    """Models responsibility and atomic election; no backend/restart claims."""

    def __init__(self, *, defect: str | None = None) -> None:
        self.authority = object()
        self.redactor = SecretRedactor()
        self._lock = asyncio.Lock()
        self._intents: dict[HandoffSlot, ProbeCommand] = {}
        self._receipts: dict[OperationRef, ProbeReceipt] = {}
        self._pending: set[OperationRef] = set()
        self._cleanup: set[OperationRef] = set()
        self.state = TestState()
        self.fault: FaultPoint | None = None
        self.barrier: AsyncFaultBarrier | None = None
        self.available = True
        self.defect = defect

    def _authorize(self, authority: object) -> None:
        if authority is not self.authority:
            raise PermissionError("Fixture authority required")

    def _fault(self, point: FaultPoint) -> None:
        if self.fault is point:
            self.fault = None
            raise InjectedFault("Injected boundary failure")

    async def prepare(
        self,
        position: HandoffSlot,
        proposed: ProbeCommand,
        *,
        authority: object,
    ) -> ProbeCommand:
        self._authorize(authority)
        position = prepare_contract(HandoffSlot, position, redactor=self.redactor)
        proposed = prepare_contract(ProbeCommand, proposed, redactor=self.redactor)
        prepare_contract(
            HandoffIntent[ProbeIntent],
            {"slot": position, "child": proposed},
            redactor=self.redactor,
        )
        self._fault(FaultPoint.BEFORE_ACCEPT)
        async with self._lock:
            existing = self._intents.get(position)
            if existing is not None:
                # Source election selects a key once; a losing candidate adopts it.
                comparison = proposed.model_copy(
                    update={
                        "operation": proposed.operation.model_copy(
                            update={
                                "caller_key": existing.operation.caller_key,
                            }
                        ),
                    }
                )
                require_exact_contract(existing, comparison, redactor=self.redactor)
                return existing
            if any(item.operation == proposed.operation for item in self._intents.values()):
                raise CollaborationConflict("Child key already belongs to another slot")
            self._intents[position] = proposed
            self._pending.add(proposed.operation)
        self._fault(FaultPoint.AFTER_PREPARE)
        return proposed

    async def apply(self, cmd: ProbeCommand, *, authority: object) -> ProbeReceipt:
        self._authorize(authority)
        cmd = prepare_contract(ProbeCommand, cmd, redactor=self.redactor)
        self._fault(FaultPoint.UNKNOWN_ACK)
        if self.barrier is not None:
            await self.barrier.pause()
        async with self._lock:
            prior = self._receipts.get(cmd.operation)
            if prior is not None:
                if self.defect == "apply_conflict":
                    return prior
                if self.defect == "mutate_before_conflict" and prior.expected != cmd:
                    self.state.effects += 1
                require_exact_contract(prior.expected, cmd, redactor=self.redactor)
                if self.defect == "duplicate":
                    self.state.effects += 1
                return prior
            receipt = ProbeReceipt(expected=cmd, stage="applied", selection="frozen-selection")
            self._receipts[cmd.operation] = receipt
            self.state.effects += 1
            self.state.events += 1
        self._fault(FaultPoint.AFTER_COMMIT)
        return receipt

    async def lookup(
        self, expected: ProbeCommand, *, authority: object
    ) -> ExactLookup[ProbeReceipt]:
        self._authorize(authority)
        expected = prepare_contract(ProbeCommand, expected, redactor=self.redactor)
        if not self.available:
            return ExactUnavailable()
        receipt = self._receipts.get(expected.operation)
        if receipt is None:
            if self.defect == "absence":
                return ExactMatch[ProbeReceipt](
                    receipt=ProbeReceipt(
                        expected=expected,
                        stage="excluded",
                        selection="fabricated",
                    )
                )
            return ExactNotFound()
        try:
            require_exact_contract(receipt.expected, expected, redactor=self.redactor)
        except CollaborationConflict:
            if self.defect != "conflict":
                return ExactConflict()
        return ExactMatch[ProbeReceipt](receipt=receipt)

    async def exclude(self, cmd: ProbeCommand, *, authority: object) -> ProbeReceipt:
        self._authorize(authority)
        cmd = prepare_contract(ProbeCommand, cmd, redactor=self.redactor)
        async with self._lock:
            receipt = self._receipts.get(cmd.operation)
            if receipt is not None:
                require_exact_contract(receipt.expected, cmd, redactor=self.redactor)
                return receipt
            receipt = ProbeReceipt(expected=cmd, stage="excluded", selection="not-selected")
            self._receipts[cmd.operation] = receipt
            self.state.events += 1
            return receipt

    async def acknowledge(self, cmd: ProbeCommand, *, authority: object) -> None:
        found = await self.lookup(cmd, authority=authority)
        if not isinstance(found, ExactMatch):
            raise CollaborationConflict("Positive receipt required for transfer")
        self._pending.discard(cmd.operation)
        self._cleanup.add(cmd.operation)
        self._fault(FaultPoint.AFTER_TRANSFER)

    async def cleanup(self, cmd: ProbeCommand, *, authority: object) -> None:
        self._authorize(authority)
        self._fault(FaultPoint.CLEANUP)
        self._cleanup.discard(cmd.operation)

    async def inspect(self) -> ReceiverState:
        return ReceiverState(
            prepared=len(self._intents),
            effects=self.state.effects,
            events=self.state.events,
            receipts=len(self._receipts),
            pending=len(self._pending),
            cleanup_pending=len(self._cleanup),
        )


class ReceiverWrapper:
    """Test-only forwarding adapter with an optionally dishonest declaration."""

    def __init__(self, inner: LimitedReceiver, *, drop_readback: bool = False) -> None:
        self.inner = inner
        self.drop_readback = drop_readback
        self.family = FamilyVersion(family="fixture.apply", version=1)
        self.capability = CapabilityDescriptor(
            owner=command().destination,
            mutations=(self.family,),
            readbacks=(self.family,),
        )

    async def prepare(
        self, position: HandoffSlot, proposed: ProbeCommand, *, authority: object
    ) -> ProbeCommand:
        return await self.inner.prepare(position, proposed, authority=authority)

    async def apply(self, cmd: ProbeCommand, *, authority: object) -> ProbeReceipt:
        return await self.inner.apply(cmd, authority=authority)

    async def lookup(
        self, expected: ProbeCommand, *, authority: object
    ) -> ExactLookup[ProbeReceipt]:
        result = await self.inner.lookup(expected, authority=authority)
        # Simulates losing a guarantee while forwarding the full declaration.
        return ExactNotFound() if self.drop_readback else result

    async def exclude(self, cmd: ProbeCommand, *, authority: object) -> ProbeReceipt:
        return await self.inner.exclude(cmd, authority=authority)

    async def acknowledge(self, cmd: ProbeCommand, *, authority: object) -> None:
        await self.inner.acknowledge(cmd, authority=authority)

    async def cleanup(self, cmd: ProbeCommand, *, authority: object) -> None:
        await self.inner.cleanup(cmd, authority=authority)

    async def inspect(self) -> ReceiverState:
        return await self.inner.inspect()
