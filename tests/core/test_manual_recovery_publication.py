"""Manual recovery keeps durable evidence ahead of delivery and cleanup failures."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from cayu.events import Event, EventType
from cayu.runtime._manual_recovery_publication import (
    ManualRecoveryPublication,
    reconcile_manual_recovery_persistence,
)
from cayu.vaults.redaction import SecretRedactor


def _event():
    return Event(type=EventType.TOOL_CALL_COMPLETED, session_id="manual-recovery")


def test_successful_append_retains_evidence_without_reading_it_again():
    async def scenario():
        event = _event()
        emitted = [event]

        async def persist(session_id, events):
            assert session_id == event.session_id and events == emitted
            return emitted

        async def read(_event):
            raise AssertionError("Acknowledged publication must not be re-read")

        publication = ManualRecoveryPublication(
            SimpleNamespace(persist_many=persist, is_persisted=read)
        )
        publication.event = event
        assert await publication.persist(event.session_id, emitted) is emitted
        assert publication.persisted
        reconciliation = await publication.reconcile()
        assert reconciliation.failure_payload(redactor=SecretRedactor()) == {
            "manual_recovery_persisted": True
        }

    asyncio.run(scenario())


@pytest.mark.parametrize("committed", [False, True])
def test_lost_append_acknowledgement_reads_the_preassigned_event(committed):
    async def scenario():
        event = _event()
        error = RuntimeError("append acknowledgement lost")
        calls = []

        async def persist(_session_id, _events):
            raise error

        async def read(candidate):
            calls.append(candidate)
            return committed

        publication = ManualRecoveryPublication(
            SimpleNamespace(persist_many=persist, is_persisted=read)
        )
        publication.event = event
        with pytest.raises(RuntimeError) as caught:
            await publication.persist(event.session_id, [event])
        assert caught.value is error and not publication.persisted
        reconciliation = await publication.reconcile()
        payload = reconciliation.failure_payload(redactor=SecretRedactor())
        assert payload == ({"manual_recovery_persisted": True} if committed else None)
        assert reconciliation.persisted is committed
        assert not publication.persisted
        assert calls == [event]

    asyncio.run(scenario())


def test_failed_read_retains_unknown_persistence_without_error_message():
    async def scenario():
        async def read(_event):
            raise RuntimeError("private storage diagnostics")

        publication = ManualRecoveryPublication(SimpleNamespace(is_persisted=read))
        publication.event = _event()
        reconciliation = await publication.reconcile()
        payload = reconciliation.failure_payload(redactor=SecretRedactor())
        assert payload == {
            "manual_recovery_persistence_unknown": True,
            "persistence_reconciliation_error_type": "RuntimeError",
        }
        assert not publication.persisted

    asyncio.run(scenario())


def test_failure_before_an_event_is_prepared_does_not_read_persistence():
    async def scenario():
        async def read(_event):
            raise AssertionError("No append identity exists")

        publication = ManualRecoveryPublication(SimpleNamespace(is_persisted=read))
        reconciliation = await publication.reconcile()
        assert reconciliation.persisted is False
        assert reconciliation.failure_payload(redactor=SecretRedactor()) is None
        assert not publication.persisted

    asyncio.run(scenario())


@pytest.mark.parametrize("committed", [False, True])
def test_caller_cancellation_waits_for_reconciliation_without_accepting_read_evidence(committed):
    async def scenario():
        entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def read(_event):
            entered.set()
            await release.wait()
            finished.set()
            return committed

        publication = ManualRecoveryPublication(SimpleNamespace(is_persisted=read))
        publication.event = _event()
        task = asyncio.create_task(publication.reconcile())
        try:
            await asyncio.wait_for(entered.wait(), 5)
            task.cancel("caller cancelled recovery")
            await asyncio.sleep(0)
            assert not task.done() and not finished.is_set()
            release.set()
            reconciliation = await asyncio.wait_for(task, 5)
            assert isinstance(reconciliation.cancellation, asyncio.CancelledError)
            assert str(reconciliation.cancellation) == "caller cancelled recovery"
            assert reconciliation.persisted is committed
            assert finished.is_set()
            # Cancellation is handled before the caller accepts read evidence.
            assert not publication.persisted
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_child_read_cancellation_is_preserved_without_cancelling_the_parent():
    async def scenario():
        cancellation = asyncio.CancelledError("storage read cancelled")

        async def read(_event):
            raise cancellation

        publication = ManualRecoveryPublication(SimpleNamespace(is_persisted=read))
        publication.event = _event()
        reconciliation = await publication.reconcile()
        assert reconciliation.cancellation is cancellation
        assert reconciliation.persisted is None and reconciliation.error is None
        assert asyncio.current_task().cancelling() == 0

    asyncio.run(scenario())


def test_fatal_read_failure_is_not_reclassified_as_unknown_persistence():
    class FatalReadFailure(BaseException):
        pass

    async def scenario():
        failure = BaseExceptionGroup("fatal read", [FatalReadFailure("stop")])

        async def read(_event):
            raise failure

        with pytest.raises(BaseExceptionGroup) as caught:
            await reconcile_manual_recovery_persistence(
                SimpleNamespace(is_persisted=read), _event()
            )
        assert caught.value is failure

    asyncio.run(scenario())
