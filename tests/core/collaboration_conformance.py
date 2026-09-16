"""Repository-private typed assertions for future receiving-owner adapters.

Adapters retain their own persistence and authority. The limited self-test
receiver is not a Cayu backend, and reopen/reset is not process-loss evidence.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, Protocol, TypeVar

from cayu.collaboration._contracts import (
    CollaborationConflict,
    ContractValue,
    ExactLookup,
    ExactMatch,
    ExactNotFound,
    ExpectedOperation,
    HandoffSlot,
)

IntentT = TypeVar("IntentT", bound=ContractValue)
ReceiptT = TypeVar("ReceiptT", bound=ContractValue)


class FaultPoint(StrEnum):
    BEFORE_ACCEPT = "before_accept"
    AFTER_PREPARE = "after_prepare"
    UNKNOWN_ACK = "unknown_ack"
    AFTER_COMMIT = "after_commit"
    AFTER_TRANSFER = "after_transfer"
    CLEANUP = "cleanup"


@dataclass(frozen=True)
class ReceiverState:
    prepared: int
    effects: int
    events: int
    receipts: int
    pending: int
    cleanup_pending: int


class CollaborationFixture(Protocol[IntentT, ReceiptT]):
    """One concrete command/receipt family and a fixture-owned authority handle."""

    async def prepare(
        self,
        slot: HandoffSlot,
        proposed: ExpectedOperation[IntentT],
        *,
        authority: object,
    ) -> ExpectedOperation[IntentT]: ...

    async def apply(
        self,
        command: ExpectedOperation[IntentT],
        *,
        authority: object,
    ) -> ReceiptT: ...

    async def lookup(
        self,
        expected: ExpectedOperation[IntentT],
        *,
        authority: object,
    ) -> ExactLookup[ReceiptT]: ...

    async def exclude(
        self,
        command: ExpectedOperation[IntentT],
        *,
        authority: object,
    ) -> ReceiptT: ...

    async def acknowledge(
        self,
        command: ExpectedOperation[IntentT],
        *,
        authority: object,
    ) -> None: ...

    async def cleanup(
        self,
        command: ExpectedOperation[IntentT],
        *,
        authority: object,
    ) -> None: ...

    async def inspect(self) -> ReceiverState: ...


class ReopenableFixture(CollaborationFixture[IntentT, ReceiptT], Protocol):
    """Only real persistent adapters may advertise this qualification hook."""

    async def reopen(self) -> CollaborationFixture[IntentT, ReceiptT]: ...


class IndependentWorkerFixture(CollaborationFixture[IntentT, ReceiptT], Protocol):
    async def independent_worker(self) -> CollaborationFixture[IntentT, ReceiptT]: ...


class AsyncFaultBarrier:
    """Bounded async-only receiver barrier; session commit guards use their harness."""

    def __init__(self) -> None:
        self._entered = asyncio.Event()
        self._release = asyncio.Event()

    async def pause(self) -> None:
        self._entered.set()
        await asyncio.wait_for(self._release.wait(), timeout=5)

    async def entered(self) -> None:
        await asyncio.wait_for(self._entered.wait(), timeout=5)

    def release(self) -> None:
        self._release.set()


@dataclass(frozen=True)
class ConformanceCase(Generic[IntentT]):
    slot: HandoffSlot
    original: ExpectedOperation[IntentT]
    changed: ExpectedOperation[IntentT]
    authority: object


async def assert_exact_replay(
    fixture: CollaborationFixture[IntentT, ReceiptT],
    case: ConformanceCase[IntentT],
) -> None:
    before = await fixture.inspect()
    command = await fixture.prepare(case.slot, case.original, authority=case.authority)
    first = await fixture.apply(command, authority=case.authority)
    second = await fixture.apply(command, authority=case.authority)
    found = await fixture.lookup(command, authority=case.authority)
    assert isinstance(found, ExactMatch), "Exact committed receipt is missing"
    assert found.receipt == first == second, "Exact replay changed its receipt"
    after = await fixture.inspect()
    assert after.effects - before.effects == 1, "Duplicate mutation"
    assert after.events - before.events == 1, "Duplicate publication"
    conflict = await fixture.lookup(case.changed, authority=case.authority)
    assert conflict.status == "conflict", "Changed-input replay was accepted"
    try:
        await fixture.apply(case.changed, authority=case.authority)
    except CollaborationConflict:
        pass
    else:
        raise AssertionError("Changed-input mutation replay was accepted")
    assert await fixture.inspect() == after, "Conflicting mutation changed retained state"
    replay = await fixture.lookup(command, authority=case.authority)
    assert isinstance(replay, ExactMatch), "Conflict discarded the committed receipt"
    assert replay.receipt == first, "Conflict replaced the committed receipt"


async def assert_absence_is_not_exclusion(
    fixture: CollaborationFixture[IntentT, ReceiptT],
    case: ConformanceCase[IntentT],
) -> None:
    absent = await fixture.lookup(case.original, authority=case.authority)
    assert isinstance(absent, ExactNotFound), "Absence was represented as positive evidence"
    command = await fixture.prepare(case.slot, case.original, authority=case.authority)
    state = await fixture.inspect()
    assert state.pending == 1 and state.effects == 0
    absent = await fixture.lookup(command, authority=case.authority)
    assert isinstance(absent, ExactNotFound)
    # No acknowledgement or release is authorized by this observation.
    assert (await fixture.inspect()).pending == 1
