"""Bounded native dependency enumeration; pruning authorization is separate."""

import pytest
from tests.core.test_clarification_deliveries import delivery_record, settled_record
from tests.core.test_participant_identity import stores as stores

from cayu.collaboration._clarification_retention import (
    operation_retains_clarification,
    prune_delivery_record,
)
from cayu.collaboration._contracts import CollaborationContractError
from cayu.collaboration._preparation import contract_bytes
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.lifecycle import NamespaceRef
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.vaults.redaction import SecretRedactor


@pytest.mark.anyio
async def test_request_delivery_handoffs_are_bounded_and_reconstructed(stores):
    store = stores()
    rows = [delivery_record(f"retention-{index:03}") for index in range(65)]
    rows[0] = settled_record(rows[0])
    request = rows[0].intent.question.request
    keys = [operation_key(item.intent.operation) for item in rows]
    try:
        async with store._transaction("app", write=True) as tx:
            for key, record in zip(keys, rows, strict=True):
                await tx.put("clarification_deliveries", key, record, insert=True)
        reopened = stores()
        async with reopened._transaction("app", write=False) as tx:
            for limit in (1, 63, 64):
                actual = await tx.scan_clarification_request_handoffs(
                    request, family="clarification_deliveries", limit=limit
                )
                assert actual == [row.model_dump(mode="json") for row in rows[:limit]]
            pending = await tx.scan_clarification_request_handoffs(
                request, family="clarification_deliveries", limit=1, pending_only=True
            )
            assert pending == [rows[1].model_dump(mode="json")]
            with pytest.raises(CollaborationContractError):
                await tx.scan_clarification_request_handoffs(
                    request, family="clarification_deliveries", limit=1, pending_only=1
                )
            assert (
                await tx.scan_clarification_request_handoffs(
                    request.model_copy(update={"incarnation": "unrelated"}),
                    family="clarification_deliveries",
                    limit=64,
                )
                == []
            )
            for limit in (0, True, 65):
                with pytest.raises(CollaborationContractError):
                    await tx.scan_clarification_request_handoffs(
                        request, family="clarification_deliveries", limit=limit
                    )
            with pytest.raises(CollaborationContractError):
                await tx.scan_clarification_request_handoffs(request, family="operations", limit=1)
    finally:
        async with store._transaction("app", write=True) as tx:
            for key in keys:
                await tx.delete("clarification_deliveries", key)


@pytest.mark.anyio
async def test_retirement_uses_current_delivery_debt_not_initial_receipt(stores):
    store = stores()
    record = delivery_record("retirement-debt")
    key = operation_key(record.intent.operation)
    namespace = NamespaceRef(
        owner=record.intent.question.request.owner,
        namespace_incarnation=record.intent.operation.namespace_incarnation,
        generation=record.intent.operation.generation,
    )
    try:
        async with store._transaction("app", write=True) as tx:
            await tx.put("clarification_deliveries", key, record, insert=True)
        reopened = stores()
        async with reopened._transaction("app", write=False) as tx:
            assert await operation_retains_clarification(
                tx, record.model_dump(mode="json"), namespace, SecretRedactor()
            )
        async with store._transaction("app", write=True) as tx:
            await tx.put("clarification_deliveries", key, settled_record(record), insert=False)
        async with reopened._transaction("app", write=False) as tx:
            assert not await operation_retains_clarification(
                tx, record.model_dump(mode="json"), namespace, SecretRedactor()
            )
    finally:
        async with store._transaction("app", write=True) as tx:
            await tx.delete("clarification_deliveries", key)


@pytest.mark.anyio
async def test_delivery_prune_requires_settlement_and_transactional_bundle(stores):
    store = stores()
    registration = delivery_record("retention-bundle")
    key = operation_key(registration.intent.operation)
    redactor = SecretRedactor()
    try:
        async with store._transaction("app", write=True) as tx:
            await tx.put("clarification_deliveries", key, registration, insert=True)
            await tx.put("operations", key, registration, insert=True)
        with pytest.raises(CollaborationUnavailable):
            async with store._transaction("app", write=True) as tx:
                await prune_delivery_record(tx, registration, redactor)
        terminal = settled_record(registration)
        async with store._transaction("app", write=True) as tx:
            await tx.put("clarification_deliveries", key, terminal, insert=False)
        with pytest.raises(OSError):
            async with store._transaction("app", write=True) as tx:
                await prune_delivery_record(tx, registration, redactor)
                raise OSError("after native delete before commit")
        reopened = stores()
        async with reopened._transaction("app", write=True) as tx:
            assert await tx.get("operations", key) == registration.model_dump(mode="json")
            assert await tx.get("clarification_deliveries", key) == terminal.model_dump(mode="json")
            released = await prune_delivery_record(tx, registration, redactor)
            assert released == sum(
                len(contract_bytes(item, redactor=redactor)) for item in (registration, terminal)
            )
        async with store._transaction("app", write=False) as tx:
            assert await tx.get("operations", key) is None
            assert await tx.get("clarification_deliveries", key) is None
    finally:
        async with store._transaction("app", write=True) as tx:
            await tx.delete("operations", key)
            await tx.delete("clarification_deliveries", key)
