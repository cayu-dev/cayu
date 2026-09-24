"""Native shared-lineage reclamation; public journey is qualified separately."""

import pytest
from tests.core.test_clarification_contracts import fixture, operation
from tests.core.test_clarification_records import records
from tests.core.test_participant_identity import stores as stores

from cayu.collaboration._clarification_pruning import prune_question_material
from cayu.collaboration._preparation import contract_bytes
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.vaults.redaction import SecretRedactor


@pytest.mark.anyio
async def test_shared_lineage_survives_until_last_terminal_question(stores):
    store = stores()
    material = records()
    first = material[0][2]
    lineage = material[2][2]
    sibling = fixture()[0]
    sibling = sibling.model_copy(
        update={
            "question": sibling.question.model_copy(
                update={
                    "operation": operation("sibling-question"),
                    "request": sibling.question.request.model_copy(
                        update={"request_id": "sibling"}
                    ),
                }
            ),
        }
    )
    sibling_key = operation_key(sibling.question.operation)
    lineage_key = operation_key(lineage.operation)
    redactor = SecretRedactor()
    async with store._transaction("app", write=True) as tx:
        for family, key, value in material:
            await tx.put(family, key, value, insert=True)
        await tx.put("clarification_questions", sibling_key, sibling, insert=True)
    reopened = stores()
    async with reopened._transaction("app", write=True) as tx:
        released = await prune_question_material(tx, (first,), redactor)
        assert released == sum(
            len(contract_bytes(value, redactor=redactor)) for value in (first, first.input)
        )
        assert await tx.get("clarification_lineages", lineage_key) == lineage.model_dump(
            mode="json"
        )
        assert await tx.get("clarification_questions", sibling_key) == sibling.model_dump(
            mode="json"
        )
    closed = sibling.model_copy(
        update={
            "state": "cancelled",
            "closed_at_ms": 150,
            "closure_operation": operation("sibling-close"),
        }
    )
    async with reopened._transaction("app", write=True) as tx:
        await tx.put("clarification_questions", sibling_key, closed, insert=False)
    # A missing sibling reference is not proof that all responsibility settled.
    with pytest.raises(CollaborationUnavailable):
        async with reopened._transaction("app", write=True) as tx:
            await prune_question_material(tx, (closed,), redactor)
    async with reopened._transaction("app", write=True) as tx:
        assert await tx.get("clarification_questions", sibling_key) == closed.model_dump(
            mode="json"
        )
        settled = lineage.model_copy(
            update={"usage": lineage.usage.model_copy(update={"pending": 0})}
        )
        await tx.put("clarification_lineages", lineage_key, settled, insert=False)
        released = await prune_question_material(tx, (closed,), redactor)
        assert released == sum(
            len(contract_bytes(value, redactor=redactor)) for value in (closed, settled)
        )
        assert await tx.get("clarification_lineages", lineage_key) is None
        assert await tx.get("clarification_questions", sibling_key) is None
