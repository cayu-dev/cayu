"""Delivery responsibility reconstruction, not public receiving authorization."""

import asyncio
import json
import sqlite3
import time
from contextlib import asynccontextmanager
from hashlib import sha256
from types import SimpleNamespace

import pytest
from tests.core.test_clarification_contracts import fixture, operation
from tests.core.test_clarification_schema import _install
from tests.core.test_participant_identity import stores
from tests.core.test_peer_content import _delivery_request

from cayu.collaboration._clarification_deliveries import (
    ClarificationDeliveryIntent,
    ClarificationDeliveryRecord,
)
from cayu.collaboration._contracts import CollaborationContractError, ObjectRef, OwnerRef
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration.clarifications import ClarificationDueCursor
from cayu.collaboration.exports import SessionExportRequest
from cayu.collaboration.participants import ParticipantRef
from cayu.collaboration.peer_content import PeerContentPayload, PeerContentReceipt
from cayu.storage._collaboration_repository import _SQLRepository
from cayu.vaults.redaction import SecretRedactor

__all__ = ["stores"]


def delivery_record(
    suffix="delivery", *, deadline=190, question=None, sender=None, delivery_operation=None
):
    question = fixture()[0].question if question is None else question
    recipient = question.responder
    sender = sender or ParticipantRef(
        owner=recipient.owner, participant_id="sender", incarnation="one"
    )
    delivery_operation = delivery_operation or operation(suffix)
    text = "Which format should the answer use?"
    payload = PeerContentPayload(
        text=text,
        content_sha256=sha256(
            json.dumps(
                {"text": text, "artifact_commitments": []},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    )
    source = question.source.model_copy(
        update={
            "producer": ObjectRef(
                owner=sender.owner,
                kind="session_export",
                object_id="export-receipt",
                incarnation=question.source.export.session_instance_id,
                revision=1,
            ),
            "content_sha256": sha256(text.encode()).hexdigest(),
            "content_bytes": len(text.encode()),
            "audience": ObjectRef(
                owner=recipient.owner,
                kind="participant",
                object_id=recipient.participant_id,
                incarnation=recipient.incarnation,
            ),
        }
    )
    question = question.model_copy(update={"source": source})
    request = _delivery_request(
        source=SimpleNamespace(
            id=source.export.session_id, instance_id=source.export.session_instance_id
        ),
        target=SimpleNamespace(id="target", instance_id="target-one", run_epoch=1),
        sender=sender,
        consumer=recipient,
        suffix=suffix,
        payload=payload,
    )
    key = request.append_key.model_copy(
        update={
            "collaboration_namespace": delivery_operation.namespace_incarnation,
            "collaboration_generation": delivery_operation.generation,
        }
    )
    request = request.model_copy(
        update={
            "append_key": key,
            "attempt_key": request.attempt_key.model_copy(
                update={"append_key": key, "deadline_at_ms": deadline}
            ),
        }
    )
    return prepare_contract(
        ClarificationDeliveryRecord,
        ClarificationDeliveryRecord(
            intent=ClarificationDeliveryIntent(
                operation=delivery_operation,
                initiator=question.initiator,
                question=question,
                sender=sender,
                recipient=recipient,
                export=SessionExportRequest(
                    ref=source.export,
                    source_indices=(0,),
                    source_selection=source.selection,
                    audience=OwnerRef(
                        application_scope=recipient.owner.application_scope,
                        owner_id=recipient.participant_id,
                        incarnation=recipient.incarnation,
                    ),
                    projector=source.projector,
                    policy=source.policy,
                ),
                append=request,
            )
        ),
        redactor=SecretRedactor(),
    )


def settled_record(record, status="appended"):
    request = record.intent.append
    return ClarificationDeliveryRecord(
        intent=record.intent,
        state="settled",
        receipt=PeerContentReceipt(
            operation_key=request.operation_key,
            append_key=request.append_key,
            attempt_generation=request.attempt_key.attempt_generation,
            status=status,
            occurrence=request.occurrence if status == "appended" else None,
            queue_id="queue" if status == "appended" else None,
            target_session_id=request.append_key.target_session_id,
            target_session_instance_id=request.append_key.target_session_instance_id,
            reason=None if status == "appended" else "delivery_deadline_expired",
        ),
    )


def test_delivery_reconstruction_preserves_complete_tuple():
    for record in (delivery_record(), settled_record(delivery_record())):
        assert ClarificationDeliveryRecord.model_validate_json(record.model_dump_json()) == record


@pytest.mark.parametrize("status", ["pending", "not_exposed"])
def test_nonterminal_or_exposure_receipts_cannot_settle_delivery(status):
    with pytest.raises(ValueError, match="exact append attempt"):
        settled_record(delivery_record(), status)


@pytest.mark.parametrize(
    "field,value",
    [
        ("operation_key", "another"),
        ("attempt_generation", 2),
        ("target_session_id", "another"),
        ("target_session_instance_id", "replaced"),
        ("disclosure", "withheld"),
    ],
)
def test_settlement_rejects_conflicting_receiving_evidence(field, value):
    record = settled_record(delivery_record())
    raw = record.model_dump(mode="json")
    raw["receipt"][field] = value
    with pytest.raises(ValueError):
        ClarificationDeliveryRecord.model_validate(raw)


@pytest.mark.parametrize("case", ["wake", "deadline", "content", "selection", "source", "sender"])
def test_delivery_rejects_cross_boundary_substitution(case):
    raw = delivery_record().model_dump(mode="json")
    intent = raw["intent"]
    if case == "wake":
        intent["append"]["wake_policy"] = "ordinary_continuation"
    elif case == "deadline":
        intent["append"]["attempt_key"]["deadline_at_ms"] = 201
    elif case == "content":
        intent["question"]["source"]["content_sha256"] = "f" * 64
    elif case == "selection":
        intent["export"]["source_selection"] = "whole_records"
    elif case == "source":
        intent["export"]["ref"]["session_instance_id"] = "replaced"
    else:
        intent["sender"]["incarnation"] = "replaced"
    with pytest.raises(ValueError):
        ClarificationDeliveryRecord.model_validate(raw)


@pytest.mark.anyio
async def test_native_pending_delivery_restart_discovery_and_settlement(stores):
    store = stores()
    records = [delivery_record(f"delivery-{index}", deadline=180 + index) for index in range(3)]
    keys = [("namespace", 1, row.intent.operation.caller_key) for row in records]
    try:
        async with store._transaction("app", write=True) as tx:
            for key, record in zip(keys, records, strict=True):
                await tx.put("clarification_deliveries", key, record, insert=True)
        reopened = stores()
        async with reopened._transaction("app", write=False) as tx:
            first = await tx.scan_pending_clarification_deliveries(after=None, limit=1)
            assert ClarificationDeliveryRecord.model_validate(first[0]) == records[0]
            # All deadlines are in the past: expiry does not erase responsibility.
            cursor = ClarificationDueCursor(
                deadline_at_ms=180, operation=records[0].intent.operation
            )
            rest = await tx.scan_pending_clarification_deliveries(after=cursor, limit=64)
            assert [ClarificationDeliveryRecord.model_validate(row) for row in rest] == records[1:]
            for limit in (0, True, 65):
                with pytest.raises(CollaborationContractError):
                    await tx.scan_pending_clarification_deliveries(after=None, limit=limit)
        terminal = settled_record(records[0])
        async with reopened._transaction("app", write=True) as tx:
            await tx.put("clarification_deliveries", keys[0], terminal, insert=False)
        async with store._transaction("app", write=False) as tx:
            assert (
                ClarificationDeliveryRecord.model_validate(
                    await tx.get("clarification_deliveries", keys[0])
                )
                == terminal
            )
            pending = await tx.scan_pending_clarification_deliveries(after=None, limit=64)
            assert len(pending) == 2
    finally:
        async with store._transaction("app", write=True) as tx:
            for key in keys:
                await tx.delete("clarification_deliveries", key)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "column,value",
    [
        ("participant_id", "other"),
        ("state", "settled"),
        ("next_due_at_ms", 900),
        ("request_id", "other"),
        ("request_incarnation", "replaced"),
    ],
)
async def test_delivery_index_corruption_is_rejected(column, value):
    connection = sqlite3.connect(":memory:")
    try:
        _install(connection)
        tx = _SQLRepository(connection, "app", postgres=False)
        await tx.put(
            "clarification_deliveries", ("namespace", 1, "delivery"), delivery_record(), insert=True
        )
        connection.execute(
            f"UPDATE cayu_collaboration_clarification_deliveries SET {column}=?", (value,)
        )
        with pytest.raises(CollaborationContractError):
            await tx.get("clarification_deliveries", ("namespace", 1, "delivery"))
        # Changing pending to settled hides it from pending discovery, but exact
        # reconciliation must still reject the contradictory indexed state.
        if column != "state":
            with pytest.raises(CollaborationContractError):
                await tx.scan_pending_clarification_deliveries(after=None, limit=1)
    finally:
        connection.close()


async def registered_question(store, *, max_pending=None):
    from tests.core.test_clarification_transactions import open_question, preparation

    initial, opening = await preparation(store)
    if max_pending is not None:
        opening = opening.model_copy(
            update={
                "question": opening.question.model_copy(
                    update={
                        "policy": opening.question.policy.model_copy(
                            update={"max_pending": max_pending}
                        )
                    }
                )
            }
        )
    record = delivery_record(
        question=opening.question,
        sender=opening.expected.intent.selection.recipient.reference,
        delivery_operation=initial.operation("delivery"),
        deadline=opening.question.deadline_at_ms,
    )
    opening = opening.model_copy(update={"question": record.intent.question})
    await open_question(store, initial, opening)
    return initial, record.intent


@pytest.mark.anyio
async def test_native_delivery_reservation_is_atomic_exact_and_bounded(stores):
    from cayu.collaboration._clarification_delivery_store import (
        DELIVERY_SETTLEMENT_BYTES,
        discover_deliveries_in_transaction,
        load_delivery_in_transaction,
        register_delivery_in_transaction,
    )
    from cayu.collaboration._contracts import CollaborationConflict
    from cayu.collaboration._request_store import operation_key

    store = stores()
    initial, intent = await registered_question(store, max_pending=2)
    scope = initial.binding.application_scope
    redactor = SecretRedactor()

    async def register(instance, candidate=intent, *, fail=False):
        async with instance._transaction(scope, write=True) as tx:
            result = await register_delivery_in_transaction(
                instance,
                tx,
                initial,
                candidate,
                authority_expires_at_ms=time.time_ns() // 1_000_000 + 300_000,
                redactor=redactor,
            )
            if fail:
                raise OSError("before commit")
            return result

    async with store._transaction(scope, write=False) as tx:
        before = await store._anchor(tx, initial, redactor)
    with pytest.raises(OSError, match="before commit"):
        await register(store, fail=True)
    async with store._transaction(scope, write=False) as tx:
        assert await store._anchor(tx, initial, redactor) == before
        assert (
            await load_delivery_in_transaction(store, tx, initial, intent, redactor=redactor)
            is None
        )
    first, second = await asyncio.gather(register(store), register(stores()))
    assert first == second
    async with store._transaction(scope, write=False) as tx:
        after = await store._anchor(tx, initial, redactor)
        assert await discover_deliveries_in_transaction(
            store, tx, initial, after=None, limit=1, redactor=redactor
        ) == (first,)
        assert after.operation_count == before.operation_count + 1
        assert after.reserved_bytes == before.reserved_bytes + DELIVERY_SETTLEMENT_BYTES
        operations = await tx.scan_operations(
            intent.operation.namespace_incarnation, intent.operation.generation, limit=100
        )
        assert sum(row.get("mode") == "clarification_delivery" for row in operations) == 1
    assert await register(stores()) == first
    changed = intent.model_copy(
        update={"export": intent.export.model_copy(update={"source_indices": (1,)})}
    )
    with pytest.raises(CollaborationConflict):
        await register(stores(), changed)
    # The one open question and this handoff jointly reach the ceiling. A new
    # delivery cannot look only at delivery rows and miss the decision debt.
    over_capacity = intent.model_copy(update={"operation": initial.operation("over-capacity")})
    with pytest.raises(CollaborationContractError):
        await register(stores(), over_capacity)
    async with store._transaction(scope, write=False) as tx:
        assert await store._anchor(tx, initial, redactor) == after
        retained = await tx.get("clarification_lineages", operation_key(intent.question.lineage))
        assert retained["usage"]["pending"] == 2  # decision + delivery, not duplicate workers


@pytest.mark.anyio
async def test_delivery_cancelled_readback_stays_pending_until_native_exclusion(
    stores, monkeypatch
):
    from cayu.collaboration._clarification_delivery_store import (
        DELIVERY_SETTLEMENT_BYTES,
        load_delivery_in_transaction,
        reconcile_delivery,
        register_delivery_in_transaction,
    )
    from cayu.sessions.base import InMemorySessionStore

    store = stores()
    initial, intent = await registered_question(store)
    scope = initial.binding.application_scope
    redactor = SecretRedactor()
    async with store._transaction(scope, write=True) as tx:
        pending = await register_delivery_in_transaction(
            store,
            tx,
            initial,
            intent,
            authority_expires_at_ms=time.time_ns() // 1_000_000 + 300_000,
            redactor=redactor,
        )
        reserved = await store._anchor(tx, initial, redactor)
    receiver = InMemorySessionStore()
    assert (
        await reconcile_delivery(stores(), initial, intent, receiver, redactor=redactor) == pending
    )
    # Real native exclusion: these targets do not exist. This does not claim to
    # qualify authorized public append or successful provider exposure.
    receipt = await receiver.append_peer_content(intent.append)
    assert receipt.status == "excluded"
    read = receiver.read_peer_content_attempt
    entered = asyncio.Event()

    async def interrupted(expected):
        result = await read(expected)
        entered.set()
        await asyncio.Event().wait()
        return result

    with monkeypatch.context() as patch:
        patch.setattr(receiver, "read_peer_content_attempt", interrupted)
        task = asyncio.create_task(
            reconcile_delivery(stores(), initial, intent, receiver, redactor=redactor)
        )
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled() and task.cancelling() == 1
    async with store._transaction(scope, write=False) as tx:
        assert (
            await load_delivery_in_transaction(store, tx, initial, intent, redactor=redactor)
            == pending
        )
        assert await store._anchor(tx, initial, redactor) == reserved
    transaction = store._transaction

    @asynccontextmanager
    async def lost_settlement_ack(scope, *, write):
        async with transaction(scope, write=write) as tx:
            yield tx
        if write:
            raise ConnectionError("settlement committed but acknowledgement was lost")

    with monkeypatch.context() as patch:
        patch.setattr(store, "_transaction", lost_settlement_ack)
        with pytest.raises(ConnectionError, match="acknowledgement was lost"):
            await reconcile_delivery(store, initial, intent, receiver, redactor=redactor)
    first, second = await asyncio.gather(
        reconcile_delivery(stores(), initial, intent, receiver, redactor=redactor),
        reconcile_delivery(stores(), initial, intent, receiver, redactor=redactor),
    )
    assert first == second and first.state == "settled" and first.receipt.status == "excluded"
    assert await reconcile_delivery(stores(), initial, intent, receiver, redactor=redactor) == first
    async with store._transaction(scope, write=False) as tx:
        after = await store._anchor(tx, initial, redactor)
        assert after.reserved_bytes == reserved.reserved_bytes - DELIVERY_SETTLEMENT_BYTES
        assert after.operation_count == reserved.operation_count
