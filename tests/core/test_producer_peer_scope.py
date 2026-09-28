"""Provenance lifetime characterization; public journeys qualify store mutation."""

import asyncio
from types import SimpleNamespace as Value

import pytest
from tests.core.test_peer_content import _request

from cayu.collaboration._contracts import OwnerRef
from cayu.collaboration._producer_peer_scope import (
    permits_producer_wait_delivery,
    producer_delivery_scope,
)
from cayu.collaboration.waits import request_object_ref
from cayu.vaults.redaction import SecretRedactor


def values():
    reference = Value(
        owner=OwnerRef(application_scope="scope", owner_id="owner", incarnation="one"),
        request_id="request",
        incarnation="one",
    )
    command = Value(
        admission=Value(expected=Value(intent=Value(selection=Value(reference=reference))))
    )
    record = Value(
        ticket=Value(
            purpose="explicit_execution_wait",
            execution_admission_sha256="a" * 64,
            collaboration_wait_sha256="b" * 64,
            targets=(request_object_ref(reference),),
        ),
        latch=None,
    )
    return command, _request(), record


@pytest.mark.parametrize("latched", [False, True])
def test_live_scope_requires_exact_append_and_selected_wait_target(latched):
    command, append, record = values()
    if latched:
        record.latch = Value(selected_manifest=record.ticket.targets)
    assert not permits_producer_wait_delivery(append, record)
    with producer_delivery_scope(command, append, redactor=SecretRedactor()):
        assert permits_producer_wait_delivery(append, record)
        changed = append.model_copy(update={"operation_key": "another-append"})
        assert not permits_producer_wait_delivery(changed, record)
        if latched:
            record.latch.selected_manifest = ()
        else:
            record.ticket.targets = ()
        assert not permits_producer_wait_delivery(append, record)
    assert not permits_producer_wait_delivery(append, record)


@pytest.mark.parametrize(
    "field,value",
    [
        ("purpose", "other-purpose"),
        ("execution_admission_sha256", None),
        ("collaboration_wait_sha256", None),
        ("targets", ()),
    ],
)
def test_producer_scope_does_not_infer_a_native_wait_from_incomplete_evidence(field, value):
    command, append, record = values()
    with producer_delivery_scope(command, append, redactor=SecretRedactor()):
        assert permits_producer_wait_delivery(append, record)
        setattr(record.ticket, field, value)
        assert not permits_producer_wait_delivery(append, record)


@pytest.mark.anyio
async def test_child_context_cannot_reuse_closed_producer_provenance():
    command, append, record = values()
    entered, release = asyncio.Event(), asyncio.Event()

    async def inherited():
        entered.set()
        await release.wait()
        return permits_producer_wait_delivery(append, record)

    with producer_delivery_scope(command, append, redactor=SecretRedactor()):
        child = asyncio.create_task(inherited())
        await entered.wait()
    release.set()
    assert not await child


@pytest.mark.anyio
async def test_cancelled_owner_revokes_inherited_delivery_scope_without_swallowing_signal():
    command, append, record = values()
    entered, release = asyncio.Event(), asyncio.Event()
    children = []
    handled = []

    async def inherited():
        await release.wait()
        return permits_producer_wait_delivery(append, record)

    async def owner():
        try:
            with producer_delivery_scope(command, append, redactor=SecretRedactor()):
                children.append(asyncio.create_task(inherited()))
                entered.set()
                await asyncio.Event().wait()
        except asyncio.CancelledError:
            handled.append(True)
            raise

    task = asyncio.create_task(owner())
    await entered.wait()
    task.cancel()
    task.cancel()
    assert task.cancelling() == 2
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled() and task.cancelling() == 2
    assert handled == [True]
    release.set()
    assert not await children[0]
