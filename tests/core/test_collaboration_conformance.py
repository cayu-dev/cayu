from __future__ import annotations

import asyncio

import pytest
from tests.core._collaboration_fixture import (
    InjectedFault,
    LimitedReceiver,
    ProbeCommand,
    ReceiverWrapper,
    command,
    slot,
)
from tests.core.collaboration_conformance import (
    AsyncFaultBarrier,
    ConformanceCase,
    FaultPoint,
    assert_absence_is_not_exclusion,
    assert_exact_replay,
)

from cayu.collaboration._capabilities import require_capability
from cayu.collaboration._contracts import (
    CollaborationConflict,
    CollaborationContractError,
    ExactMatch,
    ExactNotFound,
)


def test_source_slot_mismatch_rejects_before_retention() -> None:
    async def run():
        receiver = LimitedReceiver()
        cmd = command()
        foreign = slot(command(scope="foreign"))
        with pytest.raises(CollaborationContractError):
            await receiver.prepare(foreign, cmd, authority=receiver.authority)
        state = await receiver.inspect()
        assert state.prepared == state.pending == state.effects == state.events == 0

    asyncio.run(run())


def case(receiver: LimitedReceiver) -> ConformanceCase:
    original = command()
    return ConformanceCase(
        slot=slot(original),
        original=original,
        changed=original.model_copy(update={"mode": "other"}),
        authority=receiver.authority,
    )


def test_positive_receiver() -> None:
    async def run():
        receiver = LimitedReceiver()
        await assert_absence_is_not_exclusion(receiver, case(receiver))
        await assert_exact_replay(receiver, case(receiver))

    asyncio.run(run())


@pytest.mark.parametrize("drop_readback", [False, True])
def test_wrapper_declaration_requires_behavioral_conformance(drop_readback: bool) -> None:
    async def run():
        receiver = LimitedReceiver()
        wrapper = ReceiverWrapper(receiver, drop_readback=drop_readback)
        require_capability(
            wrapper.capability,
            expected_owner=command().destination,
            required=wrapper.family,
            supported=(wrapper.family,),
            access="mutation",
            redactor=receiver.redactor,
        )
        if drop_readback:
            with pytest.raises(AssertionError, match="Exact committed receipt is missing"):
                await assert_exact_replay(wrapper, case(receiver))
            # The real destination committed; only the wrapper lost its readback.
            result = await receiver.lookup(command(), authority=receiver.authority)
            assert isinstance(result, ExactMatch)
            state = await receiver.inspect()
            assert state.effects == state.events == state.receipts == 1
            assert state.pending == 1
        else:
            await assert_exact_replay(wrapper, case(receiver))

    asyncio.run(run())


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("duplicate", "Duplicate mutation"),
        ("conflict", "Changed-input replay"),
        ("apply_conflict", "Changed-input mutation replay"),
        ("mutate_before_conflict", "Conflicting mutation changed retained state"),
        ("absence", "Absence was represented"),
    ],
)
def test_harness_detects_defective_receivers(defect, message) -> None:
    async def run():
        receiver = LimitedReceiver(defect=defect)
        assertion = assert_absence_is_not_exclusion if defect == "absence" else assert_exact_replay
        with pytest.raises(AssertionError, match=message):
            await assertion(receiver, case(receiver))

    asyncio.run(run())


@pytest.mark.parametrize("point", list(FaultPoint))
def test_six_failure_windows(point: FaultPoint) -> None:
    async def run():
        receiver = LimitedReceiver()
        receiver.fault = point
        cmd = command()
        auth = receiver.authority
        with pytest.raises(InjectedFault):
            await receiver.prepare(slot(cmd), cmd, authority=auth)
            await receiver.apply(cmd, authority=auth)
            await receiver.acknowledge(cmd, authority=auth)
            await receiver.cleanup(cmd, authority=auth)
        state = await receiver.inspect()
        if point is FaultPoint.BEFORE_ACCEPT:
            assert state.prepared == state.effects == state.events == state.pending == 0
        elif point in (FaultPoint.AFTER_PREPARE, FaultPoint.UNKNOWN_ACK):
            assert state.prepared == state.pending == 1
            assert state.effects == state.events == state.receipts == 0
            assert isinstance(await receiver.lookup(cmd, authority=auth), ExactNotFound)
            with pytest.raises(CollaborationConflict):
                await receiver.acknowledge(cmd, authority=auth)
        else:
            assert state.effects == state.events == state.receipts == 1
            assert isinstance(await receiver.lookup(cmd, authority=auth), ExactMatch)
            assert state.pending == (1 if point is FaultPoint.AFTER_COMMIT else 0)
            assert state.cleanup_pending == (0 if point is FaultPoint.AFTER_COMMIT else 1)
        # Recovery resumes from the same intent/receipt, never a fresh operation.
        retained = await receiver.prepare(slot(cmd), cmd, authority=auth)
        result = await receiver.apply(retained, authority=auth)
        await receiver.acknowledge(retained, authority=auth)
        await receiver.cleanup(retained, authority=auth)
        end = await receiver.inspect()
        assert end.effects == end.events == end.receipts == 1
        assert end.pending == end.cleanup_pending == 0
        assert result.selection == "frozen-selection"

    asyncio.run(run())


def test_concurrent_election_scope_and_partial_batch() -> None:
    async def run():
        receiver = LimitedReceiver()
        first, other = command(), command(key="different-candidate")
        chosen = await asyncio.gather(
            *(
                receiver.prepare(slot(first), item, authority=receiver.authority)
                for item in (first, other)
            )
        )
        assert chosen[0] == chosen[1]
        await asyncio.gather(
            *(receiver.apply(item, authority=receiver.authority) for item in chosen)
        )
        changed = first.model_copy(
            update={"intent": first.intent.model_copy(update={"target": "C"})}
        )
        with pytest.raises(CollaborationConflict):
            await receiver.prepare(slot(first), changed, authority=receiver.authority)
        with pytest.raises(CollaborationConflict):
            await receiver.prepare(slot(first, "another-slot"), first, authority=receiver.authority)
        second = command(key="second-item")
        await receiver.prepare(slot(second), second, authority=receiver.authority)
        receiver.fault = FaultPoint.UNKNOWN_ACK
        with pytest.raises(InjectedFault):
            await receiver.apply(second, authority=receiver.authority)
        await receiver.apply(first, authority=receiver.authority)
        await receiver.apply(second, authority=receiver.authority)
        independent = command(scope="other-app")
        await receiver.prepare(slot(independent), independent, authority=receiver.authority)
        await receiver.apply(independent, authority=receiver.authority)
        assert (await receiver.inspect()).effects == 3

    asyncio.run(run())


def test_exclusion_competes_with_apply() -> None:
    async def run():
        for exclusion_first in (False, True):
            receiver = LimitedReceiver()
            cmd = command()
            first, second = (
                (receiver.exclude, receiver.apply)
                if exclusion_first
                else (receiver.apply, receiver.exclude)
            )
            a, b = await asyncio.gather(
                first(cmd, authority=receiver.authority), second(cmd, authority=receiver.authority)
            )
            assert a == b
            state = await receiver.inspect()
            assert state.receipts == state.events == 1
            assert state.effects == (0 if a.stage == "excluded" else 1)

    asyncio.run(run())


def test_equal_shaped_values_do_not_authenticate_and_readback_is_exact() -> None:
    async def run():
        receiver = LimitedReceiver()
        cmd = command()
        reconstructed = ProbeCommand.model_validate_json(cmd.model_dump_json())
        with pytest.raises(PermissionError):
            await receiver.prepare(slot(cmd), reconstructed, authority=cmd.initiator)
        assert (await receiver.inspect()).prepared == 0
        await receiver.prepare(slot(cmd), reconstructed, authority=receiver.authority)
        receipt = await receiver.apply(reconstructed, authority=receiver.authority)
        assert (
            await receiver.lookup(reconstructed, authority=receiver.authority)
        ).receipt == receipt
        with pytest.raises(PermissionError):
            await receiver.lookup(reconstructed, authority=object())
        receiver.available = False
        assert (
            await receiver.lookup(reconstructed, authority=receiver.authority)
        ).status == "unavailable"
        assert (await receiver.inspect()).pending == 1

    asyncio.run(run())


def test_observer_cancellation_does_not_prove_dispatch_stopped() -> None:
    async def run():
        receiver = LimitedReceiver()
        cmd = command()
        await receiver.prepare(slot(cmd), cmd, authority=receiver.authority)
        barrier = receiver.barrier = AsyncFaultBarrier()
        owned = asyncio.create_task(receiver.apply(cmd, authority=receiver.authority))

        async def observe():
            return await asyncio.shield(owned)

        observer = asyncio.create_task(observe())
        try:
            await barrier.entered()
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
            assert observer.cancelled() and observer.cancelling() == 1
            assert not owned.done()
            assert (await receiver.inspect()).pending == 1
            assert isinstance(
                await receiver.lookup(cmd, authority=receiver.authority), ExactNotFound
            )
        finally:
            barrier.release()
            await asyncio.wait_for(owned, timeout=5)
            if not observer.done():
                await observer
        assert (await receiver.lookup(cmd, authority=receiver.authority)).status == "match"
        await receiver.acknowledge(cmd, authority=receiver.authority)
        assert (await receiver.inspect()).pending == 0

    asyncio.run(run())


def test_fatal_signal_is_not_unavailable() -> None:
    class FatalSignal(BaseException):
        pass

    class FatalReceiver(LimitedReceiver):
        async def lookup(self, expected, *, authority):
            raise FatalSignal()

    async def run():
        receiver = FatalReceiver()
        with pytest.raises(FatalSignal):
            await assert_absence_is_not_exclusion(receiver, case(receiver))

    asyncio.run(run())
